import logging
import os
import threading
import time
import numpy as np
from gimbal_serial import DEFAULT_PAN_PORT, DEFAULT_TILT_PORT, GimbalAxis
from runtime_utils import limit_step_acceleration


logger = logging.getLogger(__name__)


class MotorControl:
   
    def __init__(self, ws_callback=None, axis_controllers=None):
        self.PROPORTIONAL_GAIN = 0.0115  # Reduced tracking response to limit overshoot.
        self.DERIVATIVE_GAIN = 0.0005   # Experimental constants, deviation from these can result in oscillation
                                        # or sluggish movement, but can probably be tuned more
        self.last_x_delta = 0
        self.last_y_delta = 0
        self.last_x_time = time.time()
        self.last_y_time = time.time()
        self.last_x_step = 0.0
        self.last_y_step = 0.0
        self.last_manual_x_step = 0.0
        self.last_manual_y_step = 0.0
        self.ws_callback = ws_callback
        self.HIGH_CLAMP_CONTROL = 3
        self.LOW_CLAMP_CONTROL = -3
        self.last_limit_event_time = 0.0
        self.limit_event_cooldown_s = 1.0
        self.X_MIN_ANGLE = -90
        self.X_MAX_ANGLE = 90
        self.Y_MIN_ANGLE = 0
        self.Y_MAX_ANGLE = 90
        self.angle_lock = threading.Lock()
        self.angle_initialized = {"x": False, "y": False}

        # Manual target-angle tuning. The ESP32 performs the physical velocity and
        # acceleration limiting, so these steps only advance its absolute target.
        self.MANUAL_DEADZONE = 0.08
        self.MANUAL_MIN_STEP = 0.02
        self.MANUAL_PRECISION_BAND_MAX = 0.5
        self.MANUAL_LOW_BAND_MAX_STEP = 0.08
        self.MANUAL_LOW_BAND_EXPO = 1.4
        self.MANUAL_HIGH_BAND_EXPO = 1.25
        self.MANUAL_MAX_STEP = 0.28

        self.virtual_pan_angle = 0.0
        self.virtual_tilt_angle = 0.0

        if axis_controllers is None:
            axis_controllers = {
                "x": GimbalAxis(
                    "PAN",
                    os.environ.get("QCAM_PAN_SERIAL_PORT", DEFAULT_PAN_PORT),
                    state_callback=self._handle_axis_state,
                ),
                "y": GimbalAxis(
                    "TILT",
                    os.environ.get("QCAM_TILT_SERIAL_PORT", DEFAULT_TILT_PORT),
                    state_callback=self._handle_axis_state,
                ),
            }
        self.axis_controllers = axis_controllers
        logger.info("ESP32 gimbal motor control initialized")

    def _handle_axis_state(self, axis_name, state):
        axis = "x" if axis_name.upper() == "PAN" else "y"
        with self.angle_lock:
            if not state.connected:
                self.angle_initialized[axis] = False
                return
            if state.angle_degrees is None:
                return
            if not self.angle_initialized[axis] or state.controller_state == "DISABLED":
                if axis == "x":
                    self.virtual_pan_angle = float(state.angle_degrees)
                else:
                    self.virtual_tilt_angle = float(state.angle_degrees)
                self.angle_initialized[axis] = True

    def _initialize_axis_angle(self, axis):
        if self.angle_initialized[axis]:
            return True
        controller = self.axis_controllers.get(axis)
        if controller is None:
            return False
        state = controller.state
        if not state.connected or state.angle_degrees is None:
            return False
        if axis == "x":
            self.virtual_pan_angle = float(state.angle_degrees)
        else:
            self.virtual_tilt_angle = float(state.angle_degrees)
        self.angle_initialized[axis] = True
        return True

    def reset_tracking_steps(self):
        self.last_x_step = 0.0
        self.last_y_step = 0.0

    def reset_manual_steps(self):
        self.last_manual_x_step = 0.0
        self.last_manual_y_step = 0.0

    def emit_servo_limit(self, axis, requested_angle, min_angle, max_angle):
        now = time.time()
        if now - self.last_limit_event_time < self.limit_event_cooldown_s:
            return

        self.last_limit_event_time = now
        if self.ws_callback is None:
            return

        self.ws_callback({
            "servo_at_max_angle": {
                "axis": axis,
                "requested_angle": float(requested_angle),
                "min_angle": float(min_angle),
                "max_angle": float(max_angle),
            }
        })

    def calc_derivative(self, delta, last_delta, time_diff):
        if time_diff <= 0:
            return 0
        d = (delta - last_delta) / time_diff
        return d * self.DERIVATIVE_GAIN

    def _apply_axis_step(self, axis, step):
        axis = axis.lower()
        if axis not in self.axis_controllers:
            logger.warning("Ignoring command for unavailable gimbal axis %s", axis)
            return

        with self.angle_lock:
            if not self._initialize_axis_angle(axis):
                logger.debug("Waiting for initial %s gimbal angle", axis)
                return

            if axis == "x":
                current_angle = self.virtual_pan_angle
                min_angle = self.X_MIN_ANGLE
                max_angle = self.X_MAX_ANGLE
            else:
                current_angle = self.virtual_tilt_angle
                min_angle = self.Y_MIN_ANGLE
                max_angle = self.Y_MAX_ANGLE

            requested_angle = current_angle + step
            if requested_angle < min_angle or requested_angle > max_angle:
                logger.info("Gimbal axis %s is at its configured angle limit", axis)
                self.emit_servo_limit(axis, requested_angle, min_angle, max_angle)
                return

            if axis == "x":
                self.virtual_pan_angle = float(requested_angle)
            else:
                self.virtual_tilt_angle = float(requested_angle)

        self.axis_controllers[axis].set_target(requested_angle)
        
    def set_angle(self, axis, delta, max_step=None, max_step_change=None):
        axis = axis.lower()
        now = time.time()

        if axis == "x":
            time_diff = now - self.last_x_time
            p = delta * self.PROPORTIONAL_GAIN
            d = self.calc_derivative(delta, self.last_x_delta, time_diff)
            control_output = self.clamp_control(p + d, max_step=max_step)
            if max_step_change is not None:
                control_output = limit_step_acceleration(control_output, self.last_x_step, max_step_change)
            self._apply_axis_step("x", control_output)
            self.last_x_step = control_output
            self.last_x_delta = delta
            self.last_x_time = now
            return

        if axis == "y":
            time_diff = now - self.last_y_time
            p = delta * self.PROPORTIONAL_GAIN
            d = self.calc_derivative(delta, self.last_y_delta, time_diff)
            control_output = self.clamp_control(p + d, max_step=max_step)
            if max_step_change is not None:
                control_output = limit_step_acceleration(control_output, self.last_y_step, max_step_change)
            self._apply_axis_step("y", control_output)
            self.last_y_step = control_output
            self.last_y_delta = delta
            self.last_y_time = now
            return

    def _manual_axis_to_step(self, axis_value):
        magnitude = abs(axis_value)
        if magnitude <= self.MANUAL_DEADZONE:
            return 0.0

        if magnitude < self.MANUAL_PRECISION_BAND_MAX:
            low_band_span = max(1e-6, self.MANUAL_PRECISION_BAND_MAX - self.MANUAL_DEADZONE)
            normalized = (magnitude - self.MANUAL_DEADZONE) / low_band_span
            curved = normalized ** self.MANUAL_LOW_BAND_EXPO
            step = curved * self.MANUAL_LOW_BAND_MAX_STEP
        else:
            high_band_span = max(1e-6, 1.0 - self.MANUAL_PRECISION_BAND_MAX)
            normalized = (magnitude - self.MANUAL_PRECISION_BAND_MAX) / high_band_span
            curved = normalized ** self.MANUAL_HIGH_BAND_EXPO
            step = self.MANUAL_LOW_BAND_MAX_STEP + curved * (self.MANUAL_MAX_STEP - self.MANUAL_LOW_BAND_MAX_STEP)

        step = max(step, self.MANUAL_MIN_STEP)
        return float(np.copysign(step, axis_value))

    def set_manual_input(self, x_input, y_input, max_step_change=None):
        x_input = max(-1.0, min(1.0, float(x_input)))
        y_input = max(-1.0, min(1.0, float(y_input)))

        # The brushless pan assembly is mounted opposite to the previous servo,
        # so preserve the joystick's sign for manual pan commands.
        x_step = self._manual_axis_to_step(x_input)
        y_step = self._manual_axis_to_step(-y_input)

        if max_step_change is not None:
            x_step = limit_step_acceleration(x_step, self.last_manual_x_step, max_step_change)
            y_step = limit_step_acceleration(y_step, self.last_manual_y_step, max_step_change)

        if y_step != 0.0:
            self._apply_axis_step("y", y_step)
        if x_step != 0.0:
            self._apply_axis_step("x", x_step)

        self.last_manual_x_step = x_step
        self.last_manual_y_step = y_step

    def clamp_control(self, angle, max_step=None):
        low = self.LOW_CLAMP_CONTROL
        high = self.HIGH_CLAMP_CONTROL
        if max_step is not None:
            max_step = abs(float(max_step))
            low = max(low, -max_step)
            high = min(high, max_step)
        return max(low, min(high, angle))

    def close(self):
        for controller in self.axis_controllers.values():
            controller.close()
