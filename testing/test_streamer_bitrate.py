#!/usr/bin/env python3
import asyncio
import json
import sys
import types
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


class H264EncoderStub:
    def __init__(self, bitrate=None, iperiod=None):
        self.bitrate = bitrate
        self.iperiod = iperiod


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

libcamera_module = types.ModuleType("libcamera")
libcamera_module.Transform = object
sys.modules["libcamera"] = libcamera_module

smbus2_module = types.ModuleType("smbus2")
smbus2_module.SMBus = lambda _bus: types.SimpleNamespace(read_word_data=lambda *_args: 0)
sys.modules["smbus2"] = smbus2_module

control_module = types.ModuleType("control")
control_module.PersonTracking = object
sys.modules["control"] = control_module

websockets_module = types.ModuleType("websockets")
websockets_module.serve = object
sys.modules["websockets"] = websockets_module

from streamer import CameraStreamer, Websocket_handler


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

    def start_recording(self, *args, **kwargs):
        self.recording_calls.append((args, kwargs))


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
    assert encoder.bitrate == 60_000_000
    assert "quality" not in kwargs


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
    test_get_stream_settings_replies_with_camera_payload()
    test_set_stream_bitrate_accepts_mbps_and_broadcasts_payload()
    test_set_stream_bitrate_rejects_invalid_value()
