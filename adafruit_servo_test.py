import time
from adafruit_servokit import ServoKit
import numpy as np
class MotorControl:
   
    def __init__(self, ws_callback=None):

        self.kit = ServoKit(channels=16)
        self.X_SERVO_PIN = 0
        self.Y_SERVO_PIN = 1
        self.PROPORTIONAL_GAIN = 0.023
        self.DERIVATIVE_GAIN = 0.0005
        # self.y_pos_error = []
        # self.x_pos_error = []
        # self.x_position = []
        # self.y_position = []
        self.last_x_delta = 0
        self.last_y_delta = 0
        self.last_time = time.time()
        self.min_delta = 5  # minimum delta to move servo
        self.ws_callback = ws_callback

        # init angles
        self.kit.servo[self.X_SERVO_PIN].angle = 90
        self.kit.servo[self.Y_SERVO_PIN].angle = 90
        print('servo initialized')
        print(self.kit.servo[self.Y_SERVO_PIN].angle)

    def set_angle(self, delta, axis):

        if not self.kit:
            print("Failed to connect to servo kit")
            return

        if axis.lower() == "x":
            current_time = time.time()
            time_diff = current_time - self.last_time
            self.last_time = current_time
            p = delta * self.PROPORTIONAL_GAIN
            d = (delta - self.last_x_delta) / time_diff * self.DERIVATIVE_GAIN if time_diff > 0 else 0
            self.last_x_delta = delta
            diff = p + d
            if self.kit.servo[self.X_SERVO_PIN].angle + diff > 180 or self.kit.servo[self.X_SERVO_PIN].angle + diff < 0:
                print('servo at max angle')
                return
            self.kit.servo[self.X_SERVO_PIN].angle += diff
            if self.ws_callback:
                self.ws_callback({"x_servo_angle": self.kit.servo[self.X_SERVO_PIN].angle, "x_servo_error": diff})
            
        elif axis.lower() == "y":
            diff =  (delta / 30)  # bigger steps if further away
            if self.kit.servo[self.Y_SERVO_PIN].angle + diff > 180 or self.kit.servo[self.Y_SERVO_PIN].angle + diff < 0:
                print('servo at max angle')
                return
            self.kit.servo[self.Y_SERVO_PIN].angle += diff
