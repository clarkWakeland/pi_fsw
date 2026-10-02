import dataclasses
import logging
import os
import select
import termios
import threading
import time
from typing import Callable, Optional


logger = logging.getLogger(__name__)

# The v1 controller currently connected through the first xHCI path drives the
# physical tilt motor. Keep these mappings tied to the wiring, rather than the
# ESP32's generic AXIS1 identifier.
DEFAULT_TILT_PORT = "/dev/serial/by-path/platform-xhci-hcd.0-usb-0:2:1.0-port0"
DEFAULT_PAN_PORT = "/dev/serial/by-path/platform-xhci-hcd.1-usb-0:2:1.0-port0"
SERIAL_BAUD = 115200
RECONNECT_INTERVAL_SECONDS = 1.0
PROTOCOL_RETRY_INTERVAL_SECONDS = 30.0
PROTOCOL_HANDSHAKE_TIMEOUT_SECONDS = 2.0
PING_INTERVAL_SECONDS = 0.5
STATUS_RETRY_INTERVAL_SECONDS = 2.0
PENDING_TARGET_MAX_AGE_SECONDS = 0.5


class ProtocolMismatchError(OSError):
    pass


@dataclasses.dataclass(frozen=True)
class GimbalState:
    connected: bool = False
    sequence: int = 0
    device_milliseconds: int = 0
    reported_axis: Optional[str] = None
    controller_state: str = "DISCONNECTED"
    angle_degrees: Optional[float] = None
    target_degrees: Optional[float] = None
    error_degrees: Optional[float] = None
    motor_velocity: Optional[float] = None
    encoder_ok: bool = False
    driver_fault_ok: bool = False
    magnet_weak: bool = False
    zero_persisted: bool = False
    i2c_errors: int = 0


def _configure_serial(fd):
    attributes = termios.tcgetattr(fd)
    attributes[0] = 0
    attributes[1] = 0
    attributes[2] = termios.CLOCAL | termios.CREAD | termios.CS8
    attributes[3] = 0
    attributes[4] = termios.B115200
    attributes[5] = termios.B115200
    attributes[6][termios.VMIN] = 0
    attributes[6][termios.VTIME] = 0
    termios.tcsetattr(fd, termios.TCSANOW, attributes)
    termios.tcflush(fd, termios.TCIOFLUSH)


class GimbalAxis:
    """Persistent, reconnecting link to one ESP32 gimbal controller."""

    def __init__(
        self,
        axis_name,
        device_path,
        state_callback: Optional[Callable[[str, GimbalState], None]] = None,
        autostart=True,
    ):
        self.axis_name = str(axis_name)
        self.device_path = str(device_path)
        self.state_callback = state_callback
        self._lock = threading.Lock()
        self._write_lock = threading.Lock()
        self._state = GimbalState()
        self._sequence = int(time.time() * 1000) & 0xFFFFFFFF
        self._pending_target = None
        self._pending_target_time = 0.0
        self._stop_event = threading.Event()
        self._thread = None
        self._fd = None
        self._receive_buffer = bytearray()
        self._last_transmit_time = 0.0
        self._last_connect_warning = 0.0
        self._protocol_ready = False
        self._connected_at = 0.0
        if autostart:
            self.start()

    @property
    def state(self):
        with self._lock:
            return self._state

    def start(self):
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._run,
            name=f"gimbal-{self.axis_name.lower()}",
            daemon=True,
        )
        self._thread.start()

    def close(self):
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        self._disconnect()

    def set_target(self, angle_degrees):
        with self._lock:
            self._pending_target = float(angle_degrees)
            self._pending_target_time = time.monotonic()

    def stop_motor(self):
        self._queue_immediate("STOP")

    def request_status(self):
        self._queue_immediate("STATUS")

    def _next_sequence(self):
        with self._lock:
            self._sequence = (self._sequence + 1) & 0xFFFFFFFF
            return self._sequence

    def _queue_immediate(self, command):
        # Immediate commands are best-effort. The worker will reconnect and
        # request status automatically if the device is currently unavailable.
        fd = self._fd
        if fd is None:
            return False
        sequence = self._next_sequence()
        try:
            self._write_line(f"{command},{sequence}")
            return True
        except OSError:
            self._disconnect()
            return False

    def _take_pending_target(self, now):
        with self._lock:
            target = self._pending_target
            target_time = self._pending_target_time
            self._pending_target = None
        if target is None or now - target_time > PENDING_TARGET_MAX_AGE_SECONDS:
            return None
        return target

    def _connect(self):
        fd = os.open(self.device_path, os.O_RDWR | os.O_NOCTTY | os.O_NONBLOCK)
        try:
            _configure_serial(fd)
        except Exception:
            os.close(fd)
            raise
        self._fd = fd
        self._receive_buffer.clear()
        self._last_transmit_time = 0.0
        self._protocol_ready = False
        self._connected_at = time.monotonic()
        logger.info("Connected %s gimbal controller at %s", self.axis_name, self.device_path)
        self._write_line(f"STATUS,{self._next_sequence()}")

    def _disconnect(self):
        with self._write_lock:
            fd = self._fd
            self._fd = None
            self._protocol_ready = False
            if fd is not None:
                try:
                    os.close(fd)
                except OSError:
                    pass
        with self._lock:
            self._pending_target = None
            previous = self._state
            self._state = dataclasses.replace(
                previous,
                connected=False,
                controller_state="DISCONNECTED",
            )
            state = self._state
        if fd is not None:
            if self._stop_event.is_set():
                logger.info("Closed %s gimbal controller", self.axis_name)
            else:
                logger.warning("Disconnected %s gimbal controller", self.axis_name)
        self._notify_state(state)

    def _write_line(self, line):
        with self._write_lock:
            fd = self._fd
            if fd is None:
                raise OSError("serial device is not connected")
            payload = (line + "\n").encode("ascii")
            offset = 0
            while offset < len(payload):
                written = os.write(fd, payload[offset:])
                if written <= 0:
                    raise OSError("serial write returned no data")
                offset += written
            self._last_transmit_time = time.monotonic()

    def _read_available(self):
        fd = self._fd
        if fd is None:
            return
        readable, _, _ = select.select([fd], [], [], 0.05)
        if not readable:
            return
        data = os.read(fd, 4096)
        if not data:
            raise OSError("serial device closed")
        self._receive_buffer.extend(data)
        if len(self._receive_buffer) > 16384:
            self._receive_buffer.clear()
            raise ProtocolMismatchError("serial stream has no valid line framing")
        while b"\n" in self._receive_buffer:
            raw_line, _, remaining = self._receive_buffer.partition(b"\n")
            self._receive_buffer = bytearray(remaining)
            line = raw_line.decode("ascii", errors="replace").strip()
            if line:
                self._handle_line(line)

    def _handle_line(self, line):
        fields = line.split(",")
        if fields[0] == "STATE" and len(fields) == 14:
            try:
                state = GimbalState(
                    connected=True,
                    sequence=int(fields[1]),
                    device_milliseconds=int(fields[2]),
                    reported_axis=fields[3],
                    controller_state=fields[4],
                    angle_degrees=float(fields[5]),
                    target_degrees=float(fields[6]),
                    error_degrees=float(fields[7]),
                    motor_velocity=float(fields[8]),
                    encoder_ok=fields[9] == "1",
                    driver_fault_ok=fields[10] == "1",
                    magnet_weak=fields[11] == "1",
                    zero_persisted=fields[12] == "1",
                    i2c_errors=int(fields[13]),
                )
            except ValueError:
                logger.warning("Malformed STATE from %s: %r", self.axis_name, line)
                return
            with self._lock:
                self._state = state
            self._protocol_ready = True
            self._notify_state(state)
            return
        if fields[0] == "NACK":
            logger.warning("%s gimbal rejected command: %s", self.axis_name, line)
        elif fields[0] == "EVENT":
            logger.info("%s gimbal event: %s", self.axis_name, line)
        elif fields[0] == "HELLO":
            logger.info("%s gimbal handshake: %s", self.axis_name, line)

    def _notify_state(self, state):
        if self.state_callback is not None:
            try:
                self.state_callback(self.axis_name, state)
            except Exception:
                logger.exception("Gimbal state callback failed for %s", self.axis_name)

    def _run(self):
        while not self._stop_event.is_set():
            if self._fd is None:
                try:
                    self._connect()
                except OSError as exc:
                    self._disconnect()
                    now = time.monotonic()
                    if now - self._last_connect_warning >= 30.0:
                        logger.warning(
                            "Cannot connect %s gimbal controller at %s: %s",
                            self.axis_name,
                            self.device_path,
                            exc,
                        )
                        self._last_connect_warning = now
                    self._stop_event.wait(RECONNECT_INTERVAL_SECONDS)
                    continue

            try:
                self._read_available()
                now = time.monotonic()
                if not self._protocol_ready:
                    if now - self._connected_at >= PROTOCOL_HANDSHAKE_TIMEOUT_SECONDS:
                        raise ProtocolMismatchError(
                            "controller did not complete the v1 protocol handshake"
                        )
                    if now - self._last_transmit_time >= STATUS_RETRY_INTERVAL_SECONDS:
                        self._write_line(f"STATUS,{self._next_sequence()}")
                    continue
                target = self._take_pending_target(now)
                if target is not None:
                    self._write_line(
                        f"TARGET,{self._next_sequence()},{target:.3f}"
                    )
                elif now - self._last_transmit_time >= PING_INTERVAL_SECONDS:
                    self._write_line(f"PING,{self._next_sequence()}")
            except ProtocolMismatchError as exc:
                logger.warning(
                    "%s gimbal controller is not running the v1 interface: %s",
                    self.axis_name,
                    exc,
                )
                self._disconnect()
                self._stop_event.wait(PROTOCOL_RETRY_INTERVAL_SECONDS)
            except OSError as exc:
                logger.warning("%s gimbal serial error: %s", self.axis_name, exc)
                self._disconnect()
                self._stop_event.wait(RECONNECT_INTERVAL_SECONDS)

        self._disconnect()
