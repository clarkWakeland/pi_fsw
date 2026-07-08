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
    def __init__(self):
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
    tracker.mc = MotorStub()
    return tracker


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
