#!/usr/bin/env python3
import sys
import types
from pathlib import Path

import numpy as np


class PanTiltStub:
    def __init__(self):
        self.pan_values = []
        self.tilt_values = []

    def pan(self, value):
        self.pan_values.append(value)

    def tilt(self, value):
        self.tilt_values.append(value)


stub = PanTiltStub()
module = types.SimpleNamespace(pan=stub.pan, tilt=stub.tilt)
sys.modules["pantilthat"] = module
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from servo_control import MotorControl


def test_set_angle_can_limit_single_servo_step():
    motor = MotorControl()

    motor.set_angle("x", 200.0, max_step=0.75)

    assert motor.virtual_pan_angle == 0.75
    assert stub.pan_values[-1] == 0.75


def test_set_angle_can_limit_servo_acceleration_between_steps():
    motor = MotorControl()

    motor.set_angle("x", 200.0, max_step=3.0, max_step_change=0.4)
    motor.set_angle("x", 200.0, max_step=3.0, max_step_change=0.4)

    assert np.isclose(motor.virtual_pan_angle, 1.2)
    np.testing.assert_allclose(stub.pan_values[-2:], [0.4, 1.2], rtol=1e-5)


def test_tracking_step_limiter_can_be_reset_for_new_target():
    motor = MotorControl()

    motor.set_angle("x", 200.0, max_step=3.0, max_step_change=0.4)
    motor.set_angle("y", 200.0, max_step=3.0, max_step_change=0.4)
    motor.reset_tracking_steps()

    assert motor.last_x_step == 0.0
    assert motor.last_y_step == 0.0


def test_manual_full_stick_uses_reduced_max_step():
    motor = MotorControl()

    assert motor._manual_axis_to_step(1.0) == 1.6
    assert motor._manual_axis_to_step(-1.0) == -1.6


def test_manual_precision_band_uses_reduced_max_step():
    motor = MotorControl()

    assert motor._manual_axis_to_step(0.5) == 0.55
    assert motor._manual_axis_to_step(-0.5) == -0.55


def test_manual_input_can_limit_acceleration_between_steps():
    motor = MotorControl()

    motor.set_manual_input(1.0, 0.0, max_step_change=0.3)
    motor.set_manual_input(1.0, 0.0, max_step_change=0.3)

    assert np.isclose(motor.virtual_pan_angle, -0.9)
    np.testing.assert_allclose(stub.pan_values[-2:], [-0.3, -0.9], rtol=1e-5)


if __name__ == "__main__":
    test_set_angle_can_limit_single_servo_step()
    test_set_angle_can_limit_servo_acceleration_between_steps()
    test_tracking_step_limiter_can_be_reset_for_new_target()
    test_manual_full_stick_uses_reduced_max_step()
    test_manual_precision_band_uses_reduced_max_step()
    test_manual_input_can_limit_acceleration_between_steps()
