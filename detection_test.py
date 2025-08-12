from ultralytics import YOLO
from PIL import Image
import cv2
import numpy as np
from picamera2 import Picamera2
import time
import threading
from picamera2.devices import Hailo
from adafruit_servo_test import MotorControl

class PersonTracking:
    def __init__(self):

        self.frame = None
        self.running = True
        self.isTracking = False

        self.x_delta = 0
        self.y_delta = 0
        self.hailo = Hailo('/usr/share/hailo-models/yolov8s_h8l.hef')
        self.mc = MotorControl()
        self.ty = threading.Thread(target = self.tracking_y_servo, daemon = True)
        self.tx = threading.Thread(target = self.tracking_x_servo, daemon = True)
        self.ty.start()
        self.tx.start()
    
            
    def stop(self):
        print("Stopping person tracking...")
        self.running = False
        cv2.destroyAllWindows()

    def toggle_tracking(self):
        self.isTracking = not self.isTracking

    def basic_video(self, frame):
        if frame is not None:
            results = self.process_image(frame)
            if results is not None and self.isTracking:
                # print(f"Detected: {results}")
                if results[4] > 0.5: # Assuming class_id 0 is 'person'
                    box = results[:4] * 640
                    y1, x1, y2, x2 = map(int, box)
                    # cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    self.adjust_delta([x1, y1, x2, y2])
    
    def process_image(self, image):
        # Resize image to model input size
        # convert image to 640x640 array
        image = cv2.resize(image, (640, 640))

        # Run object detection
        results = self.hailo.run(image)[0]

        # return the one with the highest confidence
        if results.shape[0] > 0:
            return results[0]
        return None

    def adjust_delta(self, coords):
        # input list of [x1, y1, x2, y2]
        SCREEN_CENTER = (320, 400) # TODO: 240 for testing
        x_cent = (coords[2] + coords[0])/2
        y_cent = (coords[3] + coords[1])/2

        self.x_delta = SCREEN_CENTER[0] - x_cent
        self.y_delta = SCREEN_CENTER[1] - y_cent

    def tracking_y_servo(self):
        while True:
            # print(f"y_delta: {self.y_delta}")
            if abs(self.y_delta) > 20 and self.isTracking:
                self.mc.set_angle(self.y_delta * -1, 'y')
            time.sleep(0.05)

    def tracking_x_servo(self):
        while True:
            # print(f"x_delta: {self.x_delta}")
            if abs(self.x_delta) > 20 and self.isTracking:
                self.mc.set_angle(self.x_delta, 'x') # actually want to move in the other direction of our delta
            time.sleep(0.05)

    def manual_control(self, message):
        if self.isTracking:
            print("Manual control is disabled while tracking is on.")
            return
        match message:
            case "UP":
                self.mc.set_angle(-30, 'y')
            case "DOWN":
                self.mc.set_angle(30, 'y')
            case "LEFT":
                self.mc.set_angle(30, 'x')
            case "RIGHT":
                self.mc.set_angle(-30, 'x')

def main():
    tracker = PersonTracking()
    try:
        tracker.basic_video()
    except KeyboardInterrupt:
        tracker.stop()
    
if __name__ == "__main__":
    main()



