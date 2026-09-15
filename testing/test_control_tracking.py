#!/usr/bin/env python3
import sys
import threading
import types
from pathlib import Path

import numpy as np


class TrackStub:
    def __init__(self, track_id, tlbr, score=0.9):
        self.track_id = track_id
        self.tlbr = np.asarray(tlbr, dtype=np.float32)
        self.score = score


class PanTiltStub:
    def pan(self, value):
        pass

    def tilt(self, value):
        pass


class MotorStub:
    def __init__(self, ws_callback=None):
        self.reset_called = False
        self.manual_calls = []

    def reset_tracking_steps(self):
        self.reset_called = True

    def set_manual_input(self, x_input, y_input, max_step_change=None):
        self.manual_calls.append((x_input, y_input, max_step_change))


picamera2_module = types.ModuleType("picamera2")
picamera2_devices_module = types.ModuleType("picamera2.devices")
picamera2_devices_module.Hailo = object
picamera2_module.devices = picamera2_devices_module
sys.modules["picamera2"] = picamera2_module
sys.modules["picamera2.devices"] = picamera2_devices_module
sys.modules["pantilthat"] = PanTiltStub()
yolox_module = types.ModuleType("yolox")
yolox_tracker_module = types.ModuleType("yolox.tracker")
yolox_byte_tracker_module = types.ModuleType("yolox.tracker.byte_tracker")
yolox_byte_tracker_module.BYTETracker = object
yolox_tracker_module.byte_tracker = yolox_byte_tracker_module
yolox_module.tracker = yolox_tracker_module
sys.modules["yolox"] = yolox_module
sys.modules["yolox.tracker"] = yolox_tracker_module
sys.modules["yolox.tracker.byte_tracker"] = yolox_byte_tracker_module
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import control as control_module
from control import PersonTracking
from userIntent import UserIntent


def make_tracker():
    tracker = object.__new__(PersonTracking)
    tracker.user_intent = UserIntent()
    tracker.status_lock = threading.Lock()
    tracker.tracking_state = "IDLE"
    tracker.state_seq = 0
    tracker.state_changed_at = 0.0
    tracker.last_emitted_state = None
    tracker.last_emitted_seq = -1
    tracker.ws_events = []
    tracker.ws_callback = tracker.ws_events.append
    tracker.tracking_object = None
    tracker.lost_frames = 0
    tracker.lost_weight = 1
    tracker.x_delta = 0
    tracker.y_delta = 0
    tracker.target_confirm_frames = 0
    tracker.tracking_motion_enabled = False
    tracker.tracking_motion_enabled_at = None
    tracker.last_target_box = None
    tracker.manual_lock = threading.Lock()
    tracker.manual_x = 0.0
    tracker.manual_y = 0.0
    tracker.manual_updated_at = 0.0
    tracker.manual_input_active = False
    tracker.hailo = object()
    tracker.ml_available = True
    tracker.ml_error = None
    tracker.mc = MotorStub()
    return tracker


def test_hailo_startup_failure_keeps_tracking_controller_alive(monkeypatch):
    started_threads = []

    class ThreadStub:
        def __init__(self, target, daemon):
            self.target = target
            self.daemon = daemon

        def start(self):
            started_threads.append(self.target.__name__)

    class ByteTrackerStub:
        def __init__(self, args):
            self.args = args

    def raise_hailo_error(_model_path):
        raise RuntimeError("hailort driver ioctl failed")

    monkeypatch.setattr(control_module, "Hailo", raise_hailo_error)
    monkeypatch.setattr(control_module, "MotorControl", MotorStub)
    monkeypatch.setattr(control_module, "BYTETracker", ByteTrackerStub)
    monkeypatch.setattr(control_module.threading, "Thread", ThreadStub)
    monkeypatch.setattr(sys, "argv", ["test-control"])

    tracker = PersonTracking()
    status = tracker.get_tracking_status()

    assert tracker.hailo is None
    assert not status["ml_available"]
    assert status["ml_error"] == "RuntimeError: hailort driver ioctl failed"
    assert not status["run_ml"]
    assert isinstance(tracker.mc, MotorStub)
    assert isinstance(tracker.BYTEtracker, ByteTrackerStub)
    assert started_threads == [
        "tracking_servo",
        "manual_servo_loop",
        "ml_loop",
        "state_heartbeat_loop",
    ]


def test_tracking_cannot_be_enabled_when_hailo_is_unavailable():
    tracker = make_tracker()
    tracker.hailo = None
    tracker.ml_available = False
    tracker.ml_error = "RuntimeError: Hailo unavailable"

    enabled = tracker.set_tracking_enabled(True)

    assert not enabled
    assert not tracker.user_intent.runML
    assert tracker.tracking_state == "IDLE"
    assert tracker.ws_events[-1]["payload"]["ml_available"] is False


def test_runtime_hailo_failure_disables_tracking_but_preserves_manual_control():
    tracker = make_tracker()
    tracker.set_tracking_enabled(True)

    tracker._mark_ml_unavailable(RuntimeError("device disconnected"))
    tracker.manual_control({"x": 0.25, "y": -0.5, "source": "test"})

    assert not tracker.ml_available
    assert tracker.ml_error == "RuntimeError: device disconnected"
    assert not tracker.user_intent.runML
    assert tracker.tracking_state == "IDLE"
    assert tracker.manual_input_active


def test_click_target_requires_three_observed_frames_before_motion():
    tracker = make_tracker()
    track = TrackStub(10, [50.0, 50.0, 150.0, 150.0])
    tracker.user_intent.click_x = 100
    tracker.user_intent.click_y = 100

    tracker.update_tracking_object([track])

    assert tracker.tracking_object is track
    assert tracker.tracking_state == "ACQUIRING"
    assert tracker.target_confirm_frames == 1
    assert not tracker.tracking_motion_enabled

    tracker.update_tracking_object([track])

    assert tracker.tracking_state == "ACQUIRING"
    assert tracker.target_confirm_frames == 2
    assert not tracker.tracking_motion_enabled

    tracker.update_tracking_object([track])

    assert tracker.tracking_state == "TRACKING"
    assert tracker.target_confirm_frames == 3
    assert tracker.tracking_motion_enabled


def test_tracking_reacquires_nearby_target_with_new_track_id():
    tracker = make_tracker()
    original = TrackStub(10, [200.0, 200.0, 280.0, 300.0])
    reacquired = TrackStub(22, [210.0, 205.0, 290.0, 305.0], score=0.65)

    tracker._start_target_confirmation(original)
    tracker.target_confirm_frames = 3
    tracker.tracking_motion_enabled = True

    tracker.update_tracking_object([reacquired])

    assert tracker.tracking_object is reacquired
    assert tracker.tracking_state == "TRACKING"
    assert tracker.lost_frames == 0


def test_adjust_delta_smooths_target_center_and_applies_soft_deadband():
    tracker = make_tracker()
    tracker.smoothed_target_center = None

    tracker.adjust_delta([300.0, 300.0, 340.0, 340.0])
    tracker.adjust_delta([300.0, 320.0, 340.0, 360.0])

    assert tracker.x_delta == 0.0
    assert np.isclose(tracker.y_delta, 74.28571428571429)


def test_new_target_confirmation_resets_servo_step_limiter():
    tracker = make_tracker()

    tracker._start_target_confirmation(TrackStub(10, [50.0, 50.0, 150.0, 150.0]))

    assert tracker.mc.reset_called


def test_manual_control_stores_latest_input_without_immediate_servo_move():
    tracker = make_tracker()

    tracker.manual_control({"x": 0.25, "y": -0.5, "magnitude": 0.6, "source": "test"})

    assert tracker.manual_x == 0.25
    assert tracker.manual_y == -0.5
    assert tracker.manual_input_active
    assert tracker.mc.manual_calls == []


def test_manual_control_step_applies_latest_input_with_acceleration_limit():
    tracker = make_tracker()
    tracker.manual_x = 0.25
    tracker.manual_y = -0.5
    tracker.manual_updated_at = 10.0
    tracker.manual_input_active = True

    tracker._apply_manual_control_step(now=10.02)

    assert tracker.mc.manual_calls == [(0.25, -0.5, 0.3)]


if __name__ == "__main__":
    test_click_target_requires_three_observed_frames_before_motion()
    test_tracking_reacquires_nearby_target_with_new_track_id()
    test_adjust_delta_smooths_target_center_and_applies_soft_deadband()
    test_new_target_confirmation_resets_servo_step_limiter()
    test_manual_control_stores_latest_input_without_immediate_servo_move()
    test_manual_control_step_applies_latest_input_with_acceleration_limit()
