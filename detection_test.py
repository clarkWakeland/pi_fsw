from ultralytics import YOLO
from PIL import Image
import cv2
import numpy as np
from picamera2 import Picamera2
import time
import threading
from picamera2.devices import Hailo
from adafruit_servo_test import MotorControl
import matplotlib.pyplot as plt
from yolox.tracker.byte_tracker import BYTETracker
import argparse

class PersonTracking:
    def __init__(self, ws_callback=None):

        self.frame = None
        self.running = True
        self.isTracking = True
        self.click_x = None
        self.click_y = None
        self.tracking_object = None
        self.lost_object = False
        self.ws_callback = ws_callback
        self.x_delta = 0
        self.y_delta = 0
        self.hailo = Hailo('/usr/share/hailo-models/yolov8s_h8l.hef')
        self.mc = MotorControl(ws_callback=ws_callback)
        
        self.t = threading.Thread(target = self.tracking_servo, daemon = True)
        self.t.start()

        # Initialize the BYTETracker
        parser = argparse.ArgumentParser("basic args")
        parser.add_argument("--track_thresh", type=float, default=0.7, help="tracking confidence threshold")
        parser.add_argument("--track_buffer", type=int, default=60, help="the frames for keep lost tracks")
        parser.add_argument("--match_thresh", type=float, default=0.8, help="matching threshold for tracking")
        parser.add_argument('--min-box-area', type=float, default=10, help='filter out tiny boxes')
        parser.add_argument("--mot20", dest="mot20", default=True, action="store_true", help="test mot20.")

        self.tracker = BYTETracker(args=parser.parse_args())

    def start_tracking(self, x, y):
        self.isTracking = True
        self.click_x = x
        self.click_y = y
    
    def stop_tracking(self):
        self.isTracking = False

    def basic_video(self, frame):
        if frame is not None:
            self.process_image(frame)
            # cv2.imshow("frame", frame)
            # cv2.waitKey(1)
            if self.isTracking and self.tracking_object:
                # print(f"Detected: {results}")
                if self.tracking_object.score > 0.5: 
                    print(self.tracking_object.tlbr)
                    self.adjust_delta(self.tracking_object.tlbr * 640)
    
    def process_image(self, image):
        # Resize image to model input size
        # convert image to 640x640 array
        image = cv2.resize(image, (640, 640))

        # Run object detection
        results = self.hailo.run(image)[0]

        tracks = self.tracker.update(results, [640, 640], [640, 640])
        if self.click_x and self.click_y:
            x, y = self.click_x, self.click_y
            self.click_x, self.click_y = None, None
            print(tracks)
            for track in tracks:
                bbox = track.tlbr
                print(bbox)
                if bbox[0] <= y <= bbox[2] and bbox[1] <= x <= bbox[3]:
                    self.tracking_object = track
                    self.lost_object = False
                    self.ws_callback({"tracking": "Tracking Object"})
                    print(f"Started tracking object ID: {self.tracking_object.track_id}")
                    break

        elif self.tracking_object:
            #continue tracking the same object
            for track in tracks:
                if track.track_id == self.tracking_object.track_id:
                    self.tracking_object = track
                    break
                self.lost_object = True

            if self.lost_object:
                # case where object is lost
                self.isTracking = False
                self.ws_callback({"tracking": "Lost Object"})
                print("Lost track of object")


    def adjust_delta(self, coords):
        # input list of [x1, y1, x2, y2]
        SCREEN_CENTER = (320, 400) # TODO: 240 for testing
        y_cent = (coords[2] + coords[0])/2
        x_cent = (coords[3] + coords[1])/2

        self.x_delta = SCREEN_CENTER[0] - x_cent
        self.y_delta = SCREEN_CENTER[1] - y_cent

    def tracking_servo(self):
        while self.isTracking:
            
            if abs(self.y_delta) > 20:
                self.mc.set_angle(self.y_delta, 'y')

            if abs(self.x_delta) > 20:
                self.mc.set_angle(self.x_delta, 'x')
            time.sleep(0.05)

    def manual_control(self, message):
        if self.isTracking:
            print("Manual control is disabled while tracking is on.")
            return
        match message:
            case "UP":
                self.mc.set_angle(30, 'y')
            case "DOWN":
                self.mc.set_angle(-30, 'y')
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



