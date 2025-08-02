
import RPi.GPIO as GPIO
import time
import sys

SERVO_PIN = 18

def setup_servo():
    GPIO.setmode(GPIO.BCM)
    GPIO.setup(SERVO_PIN, GPIO.OUT)
    pwm = GPIO.PWM(SERVO_PIN, 50)  # 50Hz frequency
    pwm.start(7.5)  # Start with 50% duty cycle
    time.sleep(1)
    return pwm

def move_servo(pwm, angle):
    duty_cycle = 2.5 + (angle / 18.0)  # Convert angle to duty cycle
    pwm.ChangeDutyCycle(duty_cycle)
    time.sleep(1)  # Allow time for servo to move

def sweep_servo(pwm):
    try:
        for angle in range(0, 181, 10):  # Sweep from 0 to 180 degrees
            move_servo(pwm, angle)
        for angle in range(180, -1, -10):  # Sweep back from 180 to 0 degrees
            move_servo(pwm, angle)
    except KeyboardInterrupt:
        pass

def main():
    try:
        pwm = setup_servo()
        sweep_servo(pwm)
    finally:
        pwm.stop()
        GPIO.cleanup()

if __name__ == "__main__":
    main()
    sys.exit(0)