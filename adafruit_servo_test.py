import time
from adafruit_servokit import ServoKit

class MotorControl:
    def __init__(self):

        self.kit = ServoKit(channels=16)
        self.X_SERVO_PIN = 0
        self.Y_SERVO_PIN = 1

        # init angles
        self.kit.servo[self.X_SERVO_PIN].angle = 90
        #self.kit.servo[self.Y_SERVO_PIN].angle = 90
        print('servo initialized')
        print(self.kit.servo[self.Y_SERVO_PIN].angle)

    def set_angle(self, delta, axis):

        if not self.kit:
            print("Failed to connect to servo kit")
            return

        if axis.lower() == "x":
            diff =  (delta / 30)  # bigger steps if further away
            if self.kit.servo[self.X_SERVO_PIN].angle + diff > 180 or self.kit.servo[self.X_SERVO_PIN].angle + diff < 0:
                print('servo at max angle')
                return
            self.kit.servo[self.X_SERVO_PIN].angle += diff
        elif axis.lower() == "y":
            diff =  (delta / 30)  # bigger steps if further away
            if self.kit.servo[self.Y_SERVO_PIN].angle + diff > 180 or self.kit.servo[self.Y_SERVO_PIN].angle + diff < 0:
                print('servo at max angle')
                return
            self.kit.servo[self.Y_SERVO_PIN].angle += diff

# for angle in range(0, 180, 1):  # Sweep from 0 to 180 degrees
#     set_angle(angle)
    #time.sleep(0.5)  # Wait for servo to reach position

#pi.set_servo_pulsewidth(SERVO_PIN, 0)  # Stop sending pulses
#pi.stop()  # Stop the pigpio daemon connection


