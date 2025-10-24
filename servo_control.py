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
        self.last_time = time.time()
        self.min_delta = 5  # minimum delta to move servo
        self.ws_callback = ws_callback

        # init angles
        pantilthat.pan(90)
        pantilthat.tilt(90)
        print('servo initialized')

    def set_angle(self, delta, axis):

        if axis.lower() == "x":
            
            self.last_time = current_time
            self.last_x_delta = delta

            # proportional term
            p = delta * self.PROPORTIONAL_GAIN

            # calculate time difference for derivative term
            current_time = time.time()
            time_diff = current_time - self.last_time
            if time_diff > 0:
                d = (delta - self.last_x_delta) / time_diff 
                d = d * self.DERIVATIVE_GAIN
            else:
                d = 0

            control_output = p + d
            new_angle = pantilthat.get_pan() + control_output
            if new_angle > 180 or new_angle < 0:
                print('servo at max angle')
                return
            
            pantilthat.pan(new_angle)
            
        elif axis.lower() == "y":
            control_output =  (delta / 30)  # bigger steps if further away
            new_angle = pantilthat.get_tilt() + control_output
            if new_angle > 180 or new_angle < 0:
                print('servo at max angle')
                return
            
            pantilthat.tilt(new_angle)
