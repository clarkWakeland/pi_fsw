import time
import pantilthat
import numpy as np
class MotorControl:
   
    def __init__(self, ws_callback=None):

        self.X_SERVO_PIN = 0
        self.Y_SERVO_PIN = 1
        self.PROPORTIONAL_GAIN = 0.023
        self.DERIVATIVE_GAIN = 0.0005
        self.last_x_delta = 0
        self.last_y_delta = 0
        self.last_x_time = time.time()
        self.last_y_time = time.time()
        self.min_delta = 5  # minimum delta to move servo
        self.ws_callback = ws_callback

        # init angles
        pantilthat.pan(0)
        pantilthat.tilt(0)
        print('servo initialized')

    def calc_derivative(self, delta, last_delta, last_time):
        current_time = time.time()
        time_diff = current_time - last_time

        if time_diff > 0:
            d = (delta - last_delta) / time_diff 
            d = d * self.DERIVATIVE_GAIN
            return d
        else:
            return 0
        
    def set_angle(self, axis, delta):

        if axis.lower() == "x":
            self.last_x_delta = delta

            # proportional term
            p = delta * self.PROPORTIONAL_GAIN

            # calculate derivative term
            d = self.calc_derivative(delta, self.last_x_delta, self.last_x_time)

            control_output = p + d
            new_angle = pantilthat.get_pan() + control_output
            if new_angle > 90 or new_angle < -90:
                print('servo at max angle')
                return
            
            pantilthat.pan(new_angle)
            
        elif axis.lower() == "y":
            self.last_y_delta = delta
            
            # proportional term
            p = delta * self.PROPORTIONAL_GAIN

            # calculate derivative term
            d = self.calc_derivative(delta, self.last_y_delta, self.last_y_time)
            control_output = p + d

            new_angle = pantilthat.get_tilt() + control_output
            if new_angle > 90 or new_angle < -90:
                print('servo at max angle')
                return
            
            pantilthat.tilt(new_angle)
