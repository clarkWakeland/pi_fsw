from picamera2 import Picamera2
from libcamera import Transform
from picamera2.encoders import H264Encoder
from picamera2.outputs import FileOutput
import av
import cv2
import time
from control import PersonTracking
import websockets
import asyncio
import json
import subprocess
import logging
import threading
from stream_settings import (
    bitrate_settings_payload,
    load_bitrate_bps,
    parse_bitrate_bps,
    save_bitrate_bps,
)

class Websocket_handler():
    def __init__(self, person_tracking_instance, camera_instance):
        self.tracker = person_tracking_instance
        self.camera = camera_instance
        self.runML_pi = False
        self.websockets = set()
        self.loop = None

        self.message_handlers = {
            "box-drawn": self.handle_box_draw,
            "canvas-click": self.handle_canvas_click,
            "toggle-tracking": self.handle_toggle_tracking,
            "get-tracking-status": self.handle_get_tracking_status,
            "get-stream-settings": self.handle_get_stream_settings,
            "set-stream-bitrate": self.handle_set_stream_bitrate,
            "manual-control": self.handle_manual_control,
            "autofocus": self.handle_autofocus,
            "toggle-auto-acquire": self.handle_auto_acquire
        }

    async def handle_box_draw(self, data, websocket):
        print(data) # TODO: implement
        box = data['box']
        box_data = [box['x'], box['y'], box['width'], box['height']]
        self.tracker.user_intent.set_ROI(box_data)  # box is [x, y, w, h]

    async def handle_canvas_click(self, data, websocket):
        print("handling_canvas_click")

        coords = data["position"]
        x = float(coords['x'])
        y = float(coords['y'])
        self.tracker.user_intent.set_click_coordinates(x, y)
        self.tracker.user_intent.clear_ROI()

    async def handle_toggle_tracking(self, data, websocket):
        run_ml = self.tracker.toggle_tracking()
        await websocket.send(json.dumps({"tracking_primed": run_ml}))
        await websocket.send(json.dumps({
            "type": "tracking-state",
            "payload": self.tracker.get_tracking_status()
        }))

    async def handle_get_tracking_status(self, data, websocket):
        status = self.tracker.get_tracking_status()
        await websocket.send(json.dumps({"tracking_primed": status["run_ml"]}))
        await websocket.send(json.dumps({"type": "tracking-state", "payload": status}))

    async def handle_get_stream_settings(self, data, websocket):
        await websocket.send(json.dumps({
            "type": "stream-settings",
            "payload": self.camera.get_stream_settings(),
        }))

    async def handle_set_stream_bitrate(self, data, websocket):
        try:
            if "bitrate_bps" in data:
                bitrate_bps = parse_bitrate_bps(data.get("bitrate_bps"))
            elif "bitrate_mbps" in data:
                bitrate_bps = parse_bitrate_bps(float(data.get("bitrate_mbps")) * 1_000_000)
            else:
                raise ValueError("bitrate_bps is required")

            loop = asyncio.get_running_loop()
            payload = await loop.run_in_executor(None, self.camera.set_bitrate, bitrate_bps)
            await self.broadcast({"type": "stream-settings", "payload": payload})
        except Exception as exc:
            await websocket.send(json.dumps({
                "type": "stream-settings-error",
                "payload": {"message": str(exc)},
            }))

    async def handle_manual_control(self, data, websocket):
        analog = data.get('analog', {})
        if not isinstance(analog, dict):
            analog = {}

        if "x" not in analog or "y" not in analog:
            direction = data.get('direction')
            magnitude = float(analog.get('magnitude', 1.0 if direction else 0.0))
            legacy_axes = {
                "UP": (0.0, -magnitude),
                "DOWN": (0.0, magnitude),
                "LEFT": (-magnitude, 0.0),
                "RIGHT": (magnitude, 0.0),
            }
            legacy_x, legacy_y = legacy_axes.get(direction, (0.0, 0.0))
            analog = {
                "x": legacy_x,
                "y": legacy_y,
                "magnitude": min(1.0, (legacy_x ** 2 + legacy_y ** 2) ** 0.5),
                "source": "legacy-direction",
            }

        self.tracker.manual_control(analog)

    async def handle_autofocus(self, data, websocket):
        self.camera.picam2.autofocus_cycle(wait = False)
        print("autofocus command received")

    async def handle_auto_acquire(self, data, websocket):
        self.tracker.user_intent.auto_acquire = not self.tracker.user_intent.auto_acquire
        print(f"auto acquire set to {self.tracker.user_intent.auto_acquire}")
        self.tracker.emit_tracking_state(force=True)

    async def main(self):
        self.loop = asyncio.get_running_loop()
        async with websockets.serve(self.handle, "0.0.0.0", port=5000):
            print("websocket server started on port 5000")
            await asyncio.Future() # run forever

    async def broadcast(self, message):
        if not self.websockets:
            return

        data = json.dumps(message)
        disconnected = []
        for ws in list(self.websockets):
            try:
                await ws.send(data)
            except Exception:
                disconnected.append(ws)

        for ws in disconnected:
            self.websockets.discard(ws)

    def send_from_thread(self, message):
        if self.loop is None:
            return
        asyncio.run_coroutine_threadsafe(self.broadcast(message), self.loop)

    async def handle(self, websocket):
        self.websockets.add(websocket)
        try:
            await websocket.send(json.dumps({
                "type": "tracking-state",
                "payload": self.tracker.get_tracking_status()
            }))
            async for message in websocket:
                message_dict = json.loads(message)
                msg_type = message_dict.get("type")
                func  = self.message_handlers.get(msg_type)
                if func:
                    await func(message_dict, websocket)
                else:
                    print(f"Unknown message type: {msg_type}")
        finally:
            self.websockets.discard(websocket)


class LowLatencyH264Encoder(H264Encoder):
    """Configure Picamera2's libx264 encoder for bounded live-stream latency."""

    def _start(self):
        super()._start()

        try:
            codec_context = self._stream.codec_context
            codec_context.thread_count = 0
            codec_context.thread_type = av.codec.context.ThreadType.SLICE
            codec_context.max_b_frames = 0
            codec_context.options.update({
                "preset": "ultrafast",
                "tune": "zerolatency",
                "slices": "4",
                "refs": "1",
            })
        except (AttributeError, TypeError, ValueError) as exc:
            raise RuntimeError(
                "Installed Picamera2/PyAV does not support the required "
                "low-latency H.264 encoder settings"
            ) from exc

        logging.info(
            "H.264 encoder configured for low latency: slice threads, "
            "4 slices, no B-frames, 1 reference frame"
        )


class CameraStreamer:
    ML_STREAM_NAME = "lores"
    ML_STREAM_SIZE = (640, 640)

    def __init__(self):
        self.mediamtx_process = subprocess.Popen(["./mediamtx"], cwd="/home/clark64/Downloads", )
        self.picam2 = Picamera2()
        self._ml_capture_fallback_logged = False
        self._using_lores = False
        self.stream_lock = threading.Lock()
        self.bitrate_bps = load_bitrate_bps()
        self.encoder = None
        self.ffmpeg_process = None

        try:
            self.picam2.configure(self._create_video_configuration(use_lores=True))
            self._using_lores = True
        except Exception as exc:
            logging.warning(
                "Falling back to main-only camera configuration: %s",
                exc
            )
            self._ml_capture_fallback_logged = True
            self.picam2.configure(self._create_video_configuration(use_lores=False))

        self.start_stream()

        # wait for camera
        time.sleep(2)
        self.picam2.autofocus_cycle(wait = False)

    def _start_ffmpeg(self):
        return subprocess.Popen([
            'ffmpeg',
            '-nostats',
            '-loglevel', 'warning',
            '-f', 'h264',
            '-framerate', '30',
            '-use_wallclock_as_timestamps', '1',
            '-fflags', 'nobuffer',
            '-i', 'pipe:0',
            '-c:v', 'copy',
            '-max_delay', '0',
            '-muxdelay', '0',
            '-flush_packets', '1',
            '-f', 'rtsp',
            '-rtsp_transport', 'tcp',
            'rtsp://127.0.0.1:8554/live.stream'  # Output to mediamtx server
        ], stdin=subprocess.PIPE)

    def _create_encoder(self):
        return LowLatencyH264Encoder(
            bitrate=self.bitrate_bps,
            iperiod=30,
            framerate=30,
            profile="baseline",
        )

    def start_stream(self):
        with self.stream_lock:
            self.encoder = self._create_encoder()
            self.ffmpeg_process = self._start_ffmpeg()
            print(f"started ffmpeg process at bitrate {self.bitrate_bps} bps")
            self.picam2.start_recording(
            self.encoder,
            FileOutput(self.ffmpeg_process.stdin)
            )

    def stop_stream(self):
        try:
            self.picam2.stop_recording()
        except Exception as exc:
            logging.warning("stop_recording failed or stream was already stopped: %s", exc)

        if self.ffmpeg_process is None:
            return

        try:
            if self.ffmpeg_process.stdin:
                self.ffmpeg_process.stdin.close()
        except Exception:
            pass

        self.ffmpeg_process.terminate()
        try:
            self.ffmpeg_process.wait(timeout=2)
        except subprocess.TimeoutExpired:
            self.ffmpeg_process.kill()
            self.ffmpeg_process.wait(timeout=2)
        finally:
            self.ffmpeg_process = None
            self.encoder = None

    def restart_stream(self):
        with self.stream_lock:
            self.stop_stream()
            self.encoder = self._create_encoder()
            self.ffmpeg_process = self._start_ffmpeg()
            print(f"restarted ffmpeg process at bitrate {self.bitrate_bps} bps")
            self.picam2.start_recording(
            self.encoder,
            FileOutput(self.ffmpeg_process.stdin)
            )

    def get_stream_settings(self):
        return bitrate_settings_payload(self.bitrate_bps)

    def set_bitrate(self, bitrate_bps):
        bitrate_bps = parse_bitrate_bps(bitrate_bps)
        if bitrate_bps == self.bitrate_bps:
            save_bitrate_bps(bitrate_bps)
            return self.get_stream_settings()

        previous_bitrate_bps = self.bitrate_bps
        self.bitrate_bps = bitrate_bps
        try:
            self.restart_stream()
        except Exception:
            logging.exception("Failed to restart stream at bitrate %s", bitrate_bps)
            self.bitrate_bps = previous_bitrate_bps
            try:
                self.restart_stream()
            except Exception:
                logging.exception("Failed to restore previous bitrate %s", previous_bitrate_bps)
            raise

        save_bitrate_bps(bitrate_bps)
        return self.get_stream_settings()

    def _create_video_configuration(self, use_lores):
        if not use_lores:
            return self.picam2.create_video_configuration(
                main={"format": 'YUV420', "size": (1920, 1080)},
                transform=Transform(hflip=1, vflip=1)
            )

        return self.picam2.create_video_configuration(
            main={"format": 'YUV420', "size": (1920, 1080)},
            lores={"format": 'BGR888', "size": self.ML_STREAM_SIZE},
            transform=Transform(hflip=1, vflip=1)
        )
    
    def capture_array(self):
        return self.picam2.capture_array()

    def _capture_main_bgr_array(self):
        frame = self.capture_array()
        if frame is None:
            return None
        return cv2.cvtColor(frame, cv2.COLOR_YUV2BGR_I420)

    def capture_ml_array(self):
        if not self._using_lores:
            return self._capture_main_bgr_array()

        try:
            return self.picam2.capture_array(self.ML_STREAM_NAME)
        except Exception as exc:
            if not self._ml_capture_fallback_logged:
                logging.warning(
                    "Falling back to main camera stream for ML capture: %s",
                    exc
                )
                self._ml_capture_fallback_logged = True
            self._using_lores = False
            return self._capture_main_bgr_array()

def send_ws_message(message):
    if "wsHandler" not in globals():
        return
    wsHandler.send_from_thread(message)
    logging.debug("websocket broadcast: %s", message)

def main():
    global camera
    global pTrack
    global wsHandler

    camera = CameraStreamer()
    pTrack = PersonTracking(send_ws_message, camera)
    wsHandler = Websocket_handler(pTrack, camera)

    try:
        asyncio.run(wsHandler.main())
    except KeyboardInterrupt:
        print("Shutting down...")


if __name__ == '__main__':
    main()

    
