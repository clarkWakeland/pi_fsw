# pi_fsw
repo for sw running on pi

## Device telemetry

The camera exposes a local JSON status snapshot at `GET /telemetry` on port 5001.
The Qwatercam workstation application polls this endpoint and relays the metrics to
Grafana. The camera itself does not need internet access or Grafana credentials.

Example device metadata:

```json
{
  "serial_number": "0002"
}
```

Example request:

```bash
curl http://192.168.4.2:5001/telemetry
```

Every payload includes the validated device serial number, firmware version,
observation time, scalar health values, and systemd service state. The serial
number in `/opt/device_metadata.json` is the source of truth.

## ESP32 gimbal control

The pan and tilt motors use persistent USB serial connections to their ESP32
controllers. Defaults can be overridden for development:

```sh
export QCAM_TILT_SERIAL_PORT=/dev/serial/by-path/platform-xhci-hcd.0-usb-0:2:1.0-port0
export QCAM_PAN_SERIAL_PORT=/dev/serial/by-path/platform-xhci-hcd.1-usb-0:2:1.0-port0
```

`servo_control.MotorControl` preserves the existing tracking and manual-control
API, but now turns each step into an absolute target angle. Serial workers keep
only the newest target, reconnect automatically, and send periodic keepalives.
The device remains available for streaming if either controller is absent.
