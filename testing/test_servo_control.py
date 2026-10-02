#!/usr/bin/env python3
from pathlib import Path
from types import SimpleNamespace
import sys

import numpy as np


class AxisStub:
    def __init__(self, angle=0.0):
        self.targets = []
        self.state = SimpleNamespace(connected=True, angle_degrees=angle)
        self.closed = False

    def set_target(self, value):
        self.targets.append(value)

    def close(self):
        self.closed = True


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from servo_control import MotorControl


def make_motor(pan_angle=0.0, tilt_angle=0.0):
    pan = AxisStub(pan_angle)
    tilt = AxisStub(tilt_angle)
    motor = MotorControl(axis_controllers={"x": pan, "y": tilt})
    return motor, pan, tilt


def test_tracking_proportional_gain_is_reduced_to_limit_overshoot():
    motor, _, _ = make_motor()

    assert motor.PROPORTIONAL_GAIN == 0.0115
    assert motor.DERIVATIVE_GAIN == 0.0005


def test_set_angle_can_limit_single_servo_step():
    motor, pan, _ = make_motor()

    motor.set_angle("x", 200.0, max_step=0.75)

    assert motor.virtual_pan_angle == 0.75
    assert pan.targets[-1] == 0.75


def test_set_angle_can_limit_servo_acceleration_between_steps():
    motor, pan, _ = make_motor()

    motor.set_angle("x", 200.0, max_step=3.0, max_step_change=0.4)
    motor.set_angle("x", 200.0, max_step=3.0, max_step_change=0.4)

    assert np.isclose(motor.virtual_pan_angle, 1.2)
    np.testing.assert_allclose(pan.targets[-2:], [0.4, 1.2], rtol=1e-5)


def test_shared_motor_boundary_preserves_both_servo_axis_directions():
    motor, pan, tilt = make_motor()

    motor._apply_axis_step("x", 1.0)
    motor._apply_axis_step("y", -1.0)

    assert motor.virtual_pan_angle == 1.0
    assert motor.virtual_tilt_angle == -1.0
    assert pan.targets[-1] == 1.0
    assert tilt.targets[-1] == -1.0


def test_pitch_range_runs_from_negative_five_to_positive_ninety():
    motor, _, tilt = make_motor()

    motor._apply_axis_step("y", 90.0)
    assert motor.virtual_tilt_angle == 90.0
    assert tilt.targets[-1] == 90.0

    motor._apply_axis_step("y", 1.0)
    assert motor.virtual_tilt_angle == 90.0
    assert tilt.targets[-1] == 90.0

    motor.virtual_tilt_angle = 0.0
    motor._apply_axis_step("y", -5.0)
    assert motor.virtual_tilt_angle == -5.0
    assert tilt.targets[-1] == -5.0

    motor._apply_axis_step("y", -1.0)
    assert motor.virtual_tilt_angle == -5.0
    assert tilt.targets[-1] == -5.0


def test_tracking_step_limiter_can_be_reset_for_new_target():
    motor, _, _ = make_motor()

    motor.set_angle("x", 200.0, max_step=3.0, max_step_change=0.4)
    motor.set_angle("y", 200.0, max_step=3.0, max_step_change=0.4)
    motor.reset_tracking_steps()

    assert motor.last_x_step == 0.0
    assert motor.last_y_step == 0.0


def test_manual_full_stick_uses_reduced_max_step():
    motor, _, _ = make_motor()

    assert motor._manual_axis_to_step(1.0) == 0.28
    assert motor._manual_axis_to_step(-1.0) == -0.28


def test_manual_precision_band_uses_reduced_max_step():
    motor, _, _ = make_motor()

    assert motor._manual_axis_to_step(0.5) == 0.08
    assert motor._manual_axis_to_step(-0.5) == -0.08


def test_manual_precision_band_enforces_minimum_nonzero_step():
    motor, _, _ = make_motor()

    assert motor._manual_axis_to_step(0.0) == 0.0
    assert motor._manual_axis_to_step(0.08) == 0.0
    assert motor._manual_axis_to_step(0.1) == 0.02
    assert motor._manual_axis_to_step(-0.1) == -0.02


def test_manual_input_can_limit_acceleration_between_steps():
    motor, pan, _ = make_motor()

    motor.set_manual_input(1.0, 0.0, max_step_change=0.3)
    motor.set_manual_input(1.0, 0.0, max_step_change=0.3)

    assert np.isclose(motor.virtual_pan_angle, -0.56)
    np.testing.assert_allclose(pan.targets[-2:], [-0.28, -0.56], rtol=1e-5)


def test_initial_target_starts_from_esp32_feedback():
    motor, pan, _ = make_motor(pan_angle=12.5)

    motor._apply_axis_step("x", 1.0)

    assert motor.virtual_pan_angle == 13.5
    assert pan.targets == [13.5]


def test_close_closes_both_axis_links():
    motor, pan, tilt = make_motor()

    motor.close()

    assert pan.closed
    assert tilt.closed


if __name__ == "__main__":
    test_set_angle_can_limit_single_servo_step()
    test_set_angle_can_limit_servo_acceleration_between_steps()
    test_shared_motor_boundary_preserves_both_servo_axis_directions()
    test_pitch_range_runs_from_negative_five_to_positive_ninety()
    test_tracking_step_limiter_can_be_reset_for_new_target()
    test_manual_full_stick_uses_reduced_max_step()
    test_manual_precision_band_uses_reduced_max_step()
    test_manual_precision_band_enforces_minimum_nonzero_step()
    test_manual_input_can_limit_acceleration_between_steps()
    test_initial_target_starts_from_esp32_feedback()
    test_close_closes_both_axis_links()
