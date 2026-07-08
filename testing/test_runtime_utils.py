#!/usr/bin/env python3
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from runtime_utils import (
    ML_INPUT_SIZE,
    apply_soft_deadband,
    deadline_sleep_seconds,
    find_reacquire_track,
    limit_step_acceleration,
    prepare_hailo_detections,
    servo_ramp_step_limit,
    smooth_point,
    tracking_motion_confirmed,
)


class TrackStub:
    def __init__(self, track_id, tlbr, score=0.9):
        self.track_id = track_id
        self.tlbr = np.asarray(tlbr, dtype=np.float32)
        self.score = score


def test_prepare_hailo_detections_scales_boxes_and_suppresses_overlaps():
    raw = np.array([
        [0.10, 0.10, 0.30, 0.30, 0.90],
        [0.11, 0.11, 0.31, 0.31, 0.80],
        [0.60, 0.60, 0.80, 0.80, 0.70],
    ], dtype=np.float32)

    detections = prepare_hailo_detections(raw, image_size=ML_INPUT_SIZE, iou_threshold=0.7)

    assert detections.shape == (2, 5)
    np.testing.assert_allclose(detections[0], [64.0, 64.0, 192.0, 192.0, 0.90], rtol=1e-5)
    np.testing.assert_allclose(detections[1], [384.0, 384.0, 512.0, 512.0, 0.70], rtol=1e-5)


def test_prepare_hailo_detections_handles_empty_results():
    detections = prepare_hailo_detections([], image_size=ML_INPUT_SIZE, iou_threshold=0.7)

    assert detections.shape == (0, 5)


def test_deadline_sleep_seconds_only_sleeps_until_remaining_deadline():
    assert np.isclose(deadline_sleep_seconds(now=1.00, deadline=1.05), 0.05)
    assert deadline_sleep_seconds(now=1.07, deadline=1.05) == 0.0


def test_find_reacquire_track_prefers_nearby_candidate_with_new_id():
    last_box = np.array([200.0, 200.0, 280.0, 300.0], dtype=np.float32)
    nearby = TrackStub(22, [210.0, 205.0, 290.0, 305.0], score=0.65)
    far = TrackStub(31, [420.0, 420.0, 500.0, 520.0], score=0.99)

    selected = find_reacquire_track(last_box, [far, nearby])

    assert selected is nearby


def test_find_reacquire_track_rejects_far_candidate():
    last_box = np.array([200.0, 200.0, 280.0, 300.0], dtype=np.float32)
    far = TrackStub(31, [420.0, 420.0, 500.0, 520.0], score=0.99)

    selected = find_reacquire_track(last_box, [far])

    assert selected is None


def test_servo_ramp_step_limit_starts_slow_and_reaches_full_step():
    assert servo_ramp_step_limit(now=10.0, started_at=10.0) == 0.75
    assert np.isclose(servo_ramp_step_limit(now=10.25, started_at=10.0), 1.875)
    assert servo_ramp_step_limit(now=10.50, started_at=10.0) == 3.0


def test_tracking_motion_requires_three_confirmed_frames():
    assert not tracking_motion_confirmed(1)
    assert not tracking_motion_confirmed(2)
    assert tracking_motion_confirmed(3)


def test_smooth_point_initializes_and_blends_center_measurements():
    first = smooth_point(None, np.array([100.0, 200.0], dtype=np.float32), alpha=0.35)
    blended = smooth_point(first, np.array([200.0, 100.0], dtype=np.float32), alpha=0.35)

    np.testing.assert_allclose(first, [100.0, 200.0], rtol=1e-5)
    np.testing.assert_allclose(blended, [135.0, 165.0], rtol=1e-5)


def test_apply_soft_deadband_removes_jitter_and_ramps_response():
    assert apply_soft_deadband(10.0, deadband=15.0, full_response=120.0) == 0.0
    assert apply_soft_deadband(-10.0, deadband=15.0, full_response=120.0) == 0.0
    assert np.isclose(apply_soft_deadband(67.5, deadband=15.0, full_response=120.0), 60.0)
    assert apply_soft_deadband(-120.0, deadband=15.0, full_response=120.0) == -120.0
    assert apply_soft_deadband(150.0, deadband=15.0, full_response=120.0) == 150.0


def test_limit_step_acceleration_bounds_command_changes():
    assert limit_step_acceleration(3.0, previous_step=0.0, max_step_change=0.4) == 0.4
    assert limit_step_acceleration(-3.0, previous_step=0.4, max_step_change=0.4) == 0.0
    assert limit_step_acceleration(0.6, previous_step=0.4, max_step_change=0.4) == 0.6


if __name__ == "__main__":
    test_prepare_hailo_detections_scales_boxes_and_suppresses_overlaps()
    test_prepare_hailo_detections_handles_empty_results()
    test_deadline_sleep_seconds_only_sleeps_until_remaining_deadline()
    test_find_reacquire_track_prefers_nearby_candidate_with_new_id()
    test_find_reacquire_track_rejects_far_candidate()
    test_servo_ramp_step_limit_starts_slow_and_reaches_full_step()
    test_tracking_motion_requires_three_confirmed_frames()
    test_smooth_point_initializes_and_blends_center_measurements()
    test_apply_soft_deadband_removes_jitter_and_ramps_response()
    test_limit_step_acceleration_bounds_command_changes()
