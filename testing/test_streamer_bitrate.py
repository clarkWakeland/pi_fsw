#!/usr/bin/env python3
import asyncio
import json
import sys
import types
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


class H264EncoderStub:
    def __init__(self, bitrate=None, iperiod=None, framerate=None, profile=None):
        self.bitrate = bitrate
        self.iperiod = iperiod
        self.framerate = framerate
        self.profile = profile

    def _start(self):
        self._stream = types.SimpleNamespace(
            codec_context=types.SimpleNamespace(
                thread_count=None,
                thread_type="FRAME",
                max_b_frames=-1,
                options={},
            )
        )


class FileOutputStub:
    def __init__(self, output):
        self.output = output


picamera2_module = types.ModuleType("picamera2")
picamera2_module.Picamera2 = object
picamera2_devices_module = types.ModuleType("picamera2.devices")
picamera2_devices_module.Hailo = object
picamera2_encoders_module = types.ModuleType("picamera2.encoders")
picamera2_encoders_module.H264Encoder = H264EncoderStub
picamera2_encoders_module.Quality = types.SimpleNamespace(VERY_HIGH="very-high")
picamera2_outputs_module = types.ModuleType("picamera2.outputs")
picamera2_outputs_module.FileOutput = FileOutputStub
sys.modules["picamera2"] = picamera2_module
sys.modules["picamera2.devices"] = picamera2_devices_module
sys.modules["picamera2.encoders"] = picamera2_encoders_module
sys.modules["picamera2.outputs"] = picamera2_outputs_module

av_module = types.ModuleType("av")
av_module.codec = types.SimpleNamespace(
    context=types.SimpleNamespace(
        ThreadType=types.SimpleNamespace(SLICE="SLICE"),
    )
)
sys.modules["av"] = av_module

libcamera_module = types.ModuleType("libcamera")
libcamera_module.Transform = lambda **kwargs: kwargs
sys.modules["libcamera"] = libcamera_module

control_module = types.ModuleType("control")
control_module.PersonTracking = object
sys.modules["control"] = control_module

websockets_module = types.ModuleType("websockets")
websockets_module.serve = object
sys.modules["websockets"] = websockets_module

import streamer as streamer_module
from streamer import CameraStreamer, LowLatencyH264Encoder, Websocket_handler


class CameraStub:
    def __init__(self):
        self.set_calls = []

    def get_stream_settings(self):
        return {
            "bitrate_bps": 15_000_000,
            "bitrate_mbps": 15.0,
            "allowed_presets_mbps": [15, 30, 60],
            "custom_min_mbps": 1.0,
            "custom_max_mbps": 100.0,
        }

    def set_bitrate(self, bitrate_bps):
        self.set_calls.append(bitrate_bps)
        return {
            "bitrate_bps": bitrate_bps,
            "bitrate_mbps": bitrate_bps / 1_000_000,
            "allowed_presets_mbps": [15, 30, 60],
            "custom_min_mbps": 1.0,
            "custom_max_mbps": 100.0,
        }


class WebsocketStub:
    def __init__(self):
        self.sent = []

    async def send(self, data):
        self.sent.append(json.loads(data))


class PicameraStub:
    def __init__(self):
        self.recording_calls = []
        self.configuration_calls = []

    def start_recording(self, *args, **kwargs):
        self.recording_calls.append((args, kwargs))

    def create_video_configuration(self, **kwargs):
        self.configuration_calls.append(kwargs)
        return kwargs


class ProcessStub:
    stdin = object()


class LockStub:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


def make_handler(camera):
    tracker = types.SimpleNamespace(get_tracking_status=lambda: {"run_ml": False})
    handler = Websocket_handler(tracker, camera)
    handler.websockets = set()
    return handler


def make_camera_streamer():
    camera = object.__new__(CameraStreamer)
    camera.stream_lock = LockStub()
    camera.bitrate_bps = 60_000_000
    camera.picam2 = PicameraStub()
    camera.ffmpeg_process = None
    camera.encoder = None
    camera._start_ffmpeg = lambda: ProcessStub()
    return camera


def test_camera_streamer_preserves_explicit_encoder_bitrate():
    camera = make_camera_streamer()

    camera.start_stream()

    args, kwargs = camera.picam2.recording_calls[0]
    encoder = args[0]
    assert isinstance(encoder, LowLatencyH264Encoder)
    assert encoder.bitrate == 60_000_000
    assert encoder.iperiod == 30
    assert encoder.framerate == 30
    assert encoder.profile == "baseline"
    assert "quality" not in kwargs


def test_low_latency_encoder_uses_slice_threads_without_b_frames():
    encoder = LowLatencyH264Encoder(
        bitrate=15_000_000,
        iperiod=30,
        framerate=30,
        profile="baseline",
    )

    encoder._start()

    codec_context = encoder._stream.codec_context
    assert codec_context.thread_count == 0
    assert codec_context.thread_type == "SLICE"
    assert codec_context.max_b_frames == 0
    assert codec_context.options == {
        "preset": "ultrafast",
        "tune": "zerolatency",
        "slices": "4",
        "refs": "1",
    }


def test_low_latency_encoder_fails_clearly_when_codec_context_is_unsupported(monkeypatch):
    class UnsupportedCodecContext:
        __slots__ = ()

    def start_with_unsupported_context(encoder):
        encoder._stream = types.SimpleNamespace(codec_context=UnsupportedCodecContext())

    monkeypatch.setattr(H264EncoderStub, "_start", start_with_unsupported_context)
    encoder = LowLatencyH264Encoder(bitrate=15_000_000)

    try:
        encoder._start()
    except RuntimeError as exc:
        assert "does not support the required low-latency" in str(exc)
    else:
        raise AssertionError("Expected incompatible codec context to fail")


def test_camera_streamer_starts_ffmpeg_with_live_timestamps_and_no_mux_delay(monkeypatch):
    popen_calls = []

    def popen_stub(command, **kwargs):
        popen_calls.append((command, kwargs))
        return ProcessStub()

    monkeypatch.setattr(streamer_module.subprocess, "Popen", popen_stub)
    camera = object.__new__(CameraStreamer)

    camera._start_ffmpeg()

    command, kwargs = popen_calls[0]
    assert command == [
        "ffmpeg",
        "-nostats",
        "-loglevel", "warning",
        "-f", "h264",
        "-framerate", "30",
        "-use_wallclock_as_timestamps", "1",
        "-fflags", "nobuffer",
        "-i", "pipe:0",
        "-c:v", "copy",
        "-max_delay", "0",
        "-muxdelay", "0",
        "-flush_packets", "1",
        "-f", "rtsp",
        "-rtsp_transport", "tcp",
        "rtsp://127.0.0.1:8554/live.stream",
    ]
    assert kwargs == {"stdin": streamer_module.subprocess.PIPE}


def test_camera_streamer_encodes_yuv_main_and_preserves_bgr_ml_stream():
    camera = make_camera_streamer()

    configuration = camera._create_video_configuration(use_lores=True)

    assert configuration["main"] == {"format": "YUV420", "size": (1920, 1080)}
    assert configuration["lores"] == {"format": "BGR888", "size": (640, 640)}


def test_camera_streamer_main_only_fallback_remains_yuv():
    camera = make_camera_streamer()

    configuration = camera._create_video_configuration(use_lores=False)

    assert configuration["main"] == {"format": "YUV420", "size": (1920, 1080)}
    assert "lores" not in configuration


def test_camera_streamer_converts_main_yuv_to_bgr_for_ml(monkeypatch):
    camera = make_camera_streamer()
    camera._using_lores = False
    camera.picam2.capture_array = lambda: "yuv-frame"
    monkeypatch.setattr(
        streamer_module.cv2,
        "cvtColor",
        lambda frame, conversion: (frame, conversion),
    )

    assert camera.capture_ml_array() == (
        "yuv-frame",
        streamer_module.cv2.COLOR_YUV2BGR_I420,
    )


def test_get_stream_settings_replies_with_camera_payload():
    camera = CameraStub()
    handler = make_handler(camera)
    websocket = WebsocketStub()

    asyncio.run(handler.handle_get_stream_settings({}, websocket))

    assert websocket.sent == [{"type": "stream-settings", "payload": camera.get_stream_settings()}]


def test_set_stream_bitrate_accepts_mbps_and_broadcasts_payload():
    camera = CameraStub()
    handler = make_handler(camera)
    websocket = WebsocketStub()
    handler.websockets.add(websocket)

    asyncio.run(handler.handle_set_stream_bitrate({"bitrate_mbps": 30}, websocket))

    assert camera.set_calls == [30_000_000]
    assert websocket.sent == [{
        "type": "stream-settings",
        "payload": {
            "bitrate_bps": 30_000_000,
            "bitrate_mbps": 30.0,
            "allowed_presets_mbps": [15, 30, 60],
            "custom_min_mbps": 1.0,
            "custom_max_mbps": 100.0,
        },
    }]


def test_set_stream_bitrate_rejects_invalid_value():
    camera = CameraStub()
    handler = make_handler(camera)
    websocket = WebsocketStub()

    asyncio.run(handler.handle_set_stream_bitrate({"bitrate_bps": 0}, websocket))

    assert camera.set_calls == []
    assert websocket.sent[0]["type"] == "stream-settings-error"


if __name__ == "__main__":
    test_camera_streamer_preserves_explicit_encoder_bitrate()
    test_camera_streamer_encodes_yuv_main_and_preserves_bgr_ml_stream()
    test_camera_streamer_main_only_fallback_remains_yuv()
    test_get_stream_settings_replies_with_camera_payload()
    test_set_stream_bitrate_accepts_mbps_and_broadcasts_payload()
    test_set_stream_bitrate_rejects_invalid_value()
