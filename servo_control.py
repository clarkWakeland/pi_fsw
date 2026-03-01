import time
import pantilthat
import numpy as np
class MotorControl:
   
    def __init__(self, ws_callback=None):
        self.X_SERVO_PIN = 0
        self.Y_SERVO_PIN = 1
        self.PROPORTIONAL_GAIN = 0.023  # I love numbers that I pulled from thin air
        self.DERIVATIVE_GAIN = 0.0005   # Experimental constants, deviation from these can result in oscillation
                                        # or sluggish movement, but can probably be tuned more
        self.last_x_delta = 0
        self.last_y_delta = 0
        self.last_x_time = time.time()
        self.last_y_time = time.time()
        self.ws_callback = ws_callback
        self.HIGH_CLAMP_CONTROL = 3
        self.LOW_CLAMP_CONTROL = -3
        self.last_limit_event_time = 0.0
        self.limit_event_cooldown_s = 1.0
        self.X_MIN_ANGLE = -90
        self.X_MAX_ANGLE = 90
        self.Y_MIN_ANGLE = -5
        self.Y_MAX_ANGLE = 90

        # Manual-control tuning with a dedicated precision band for small stick inputs.
        self.MANUAL_DEADZONE = 0.08
        self.MANUAL_PRECISION_BAND_MAX = 0.5
        # Keep low-band outputs above common stiction while preserving fine response.
        self.MANUAL_LOW_BAND_MAX_STEP = 0.9
        self.MANUAL_LOW_BAND_EXPO = 1.4
        self.MANUAL_HIGH_BAND_EXPO = 1.25
        self.MANUAL_MAX_STEP = 2.5

        # init angles
        pantilthat.pan(0)
        pantilthat.tilt(0)
        print('servo initialized')

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
        if axis == "x":
            current_angle = pantilthat.get_pan()
            requested_angle = current_angle + step
            if requested_angle < self.X_MIN_ANGLE or requested_angle > self.X_MAX_ANGLE:
                print('servo at max angle')
                self.emit_servo_limit("x", requested_angle, self.X_MIN_ANGLE, self.X_MAX_ANGLE)
                return
            pantilthat.pan(requested_angle)
            return

        if axis == "y":
            current_angle = pantilthat.get_tilt()
            requested_angle = current_angle + step
            if requested_angle < self.Y_MIN_ANGLE or requested_angle > self.Y_MAX_ANGLE:
                print('servo at max angle')
                self.emit_servo_limit("y", requested_angle, self.Y_MIN_ANGLE, self.Y_MAX_ANGLE)
                return
            pantilthat.tilt(requested_angle)
            return
        
    def set_angle(self, axis, delta):
        axis = axis.lower()
        now = time.time()

        if axis == "x":
            time_diff = now - self.last_x_time
            p = delta * self.PROPORTIONAL_GAIN
            d = self.calc_derivative(delta, self.last_x_delta, time_diff)
            control_output = self.clamp_control(p + d)
            self._apply_axis_step("x", control_output)
            self.last_x_delta = delta
            self.last_x_time = now
            return

        if axis == "y":
            time_diff = now - self.last_y_time
            p = delta * self.PROPORTIONAL_GAIN
            d = self.calc_derivative(delta, self.last_y_delta, time_diff)
            control_output = self.clamp_control(p + d)
            self._apply_axis_step("y", control_output)
            self.last_y_delta = delta
            self.last_y_time = now
            return

    def _manual_axis_to_step(self, axis_value):
        magnitude = abs(axis_value)
        if magnitude < self.MANUAL_DEADZONE:
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

        return float(np.copysign(step, axis_value))

    def set_manual_input(self, x_input, y_input):
        x_input = max(-1.0, min(1.0, float(x_input)))
        y_input = max(-1.0, min(1.0, float(y_input)))

        x_step = self._manual_axis_to_step(-x_input)
        y_step = self._manual_axis_to_step(-y_input)

        if y_step != 0.0:
            self._apply_axis_step("y", y_step)
        if x_step != 0.0:
            self._apply_axis_step("x", x_step)

    def clamp_control(self, angle):
        return max(self.LOW_CLAMP_CONTROL, min(self.HIGH_CLAMP_CONTROL, angle))
