import time

class MotorControl:
    def __init__(self):
        self.SERVO_PIN = 18
        self.pi = pigpio.pi()
        
        # init angles
        self.pi.set_servo_pulsewidth(self.SERVO_PIN, 1500)
        self.y_command = self.pi.get_servo_pulsewidth(self.SERVO_PIN)
        print('servo initialized')
        print(self.y_command)
    
    def set_angle(self, delta):
        
        if not self.pi.connected:
            print("Failed to connect to pigpio daemon")
            return
        
        diff =  2 * (delta / 20)  # bigger steps if further away        
        if self.y_command + diff > 2500 or self.y_command + diff < 500:
            print('servo at max angle')
            return
        # Convert angle to pulse width (in microseconds)
        self.y_command = self.y_command + diff
        # print(f'commanded angle: {self.y_command}')
        self.pi.set_servo_pulsewidth(self.SERVO_PIN, self.y_command)
        #time.sleep(delta*-0.003+1.4)  # Allow time for servo to move
        # pi.set_servo_pulsewidth(SERVO_PIN, 0)  # Stop sending pulses

# for angle in range(0, 180, 1):  # Sweep from 0 to 180 degrees
#     set_angle(angle)
    #time.sleep(0.5)  # Wait for servo to reach position

#pi.set_servo_pulsewidth(SERVO_PIN, 0)  # Stop sending pulses
#pi.stop()  # Stop the pigpio daemon connection


