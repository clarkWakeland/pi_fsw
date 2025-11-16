# from ultralytics import YOLO
import cv2
import time
import threading
from picamera2.devices import Hailo
from servo_control import MotorControl
from yolox.tracker.byte_tracker import BYTETracker, STrack
import argparse
import logging
logging.basicConfig(
    level=logging.INFO, 
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger(__name__)

class PersonTracking:
    def __init__(self, ws_callback=None, camera=None):

        self.frame = None
        self.running = True
        self.runML = False
        self.click_x = None
        self.click_y = None
        self.tracking_object = None
        self.auto_acquire = False
        self.ws_callback = ws_callback
        self.camera = camera
        self.x_delta = 0
        self.y_delta = 0
        self.hailo = Hailo('small_data.hef')
        self.mc = MotorControl(ws_callback=ws_callback)

        threading.Thread(target = self.tracking_servo, daemon=True).start()
        threading.Thread(target = self.ml_loop, daemon=True).start()

        # Initialize the BYTETracker
        parser = argparse.ArgumentParser("basic args")
        parser.add_argument("--track_thresh", type=float, default=0.1, help="tracking confidence threshold")
        parser.add_argument("--track_buffer", type=int, default=60, help="the frames for keep lost tracks")
        parser.add_argument("--match_thresh", type=float, default=0.1, help="matching threshold for tracking")
        parser.add_argument('--min-box-area', type=float, default=10, help='filter out tiny boxes')
        parser.add_argument("--mot20", dest="mot20", default=True, action="store_true", help="test mot20.")

        self.BYTEtracker = BYTETracker(args=parser.parse_args())

    def start_tracking(self, x, y):
        logger.info(f"received {x} x and {y} y")
        self.runML = True
        self.click_x = x
        self.click_y = y
    
    def stop_tracking(self):
        self.runML = False

    def ml_loop(self):
        ''' 
        ML thread loop. 
        Only runs if user toggles tracking
        '''
        while True:
            if self.runML:
                logger.info("Processing images")
                frame = self.camera.capture_array()
                if frame is not None:
                    self.process_image(frame)
                    if self.runML and self.tracking_object:
                        if self.tracking_object.score > 0.5: 
                            logger.info(self.tracking_object.tlbr)
                            self.adjust_delta(self.tracking_object.tlbr * 640)

            time.sleep(0.1) # run inference 10 times/sec for now, can be adjusted
        
    def process_image(self, image):
        ''' 
        Process a single image, return object tracks
        '''
        image = cv2.resize(image, (640, 640))
        results = self.hailo.run(image)[0]
        logger.info(f"Detections: {results}")

        if len(results) != 0:
            results = results[0]  # pick highest confidence detection
            # Ignore byte tracker for now
            # tracks = self.BYTEtracker.update(results, [640, 640], [640, 640])
            track = STrack(tlwh=[results[0], results[1], results[2]-results[0], results[3]-results[1]], score=results[4])
            self.update_tracking_object([track])

    def update_tracking_object(self, tracks):
        ''' 
        logic for selecting and updating the tracked object
        '''

        if self.auto_acquire:
            highest_confidence = 0
            if not tracks:
                self.x_delta = 0
                self.y_delta = 0
                self.tracking_object = None
                return
            logger.info([track.score for track in tracks])
            for track in tracks:
                if track.score > highest_confidence:
                    logger.info(f"Auto acquired object ID: {track.track_id} with confidence {track.score}")
                    self.ws_callback({"object": {"box": {'x1': track.tlbr[0], 'y1': track.tlbr[1], 'x2': track.tlbr[2], 'y2': track.tlbr[3]}, "confidence": str(track.score)}})
                    highest_confidence = track.score
                    highest_conf = track

            self.tracking_object = highest_conf
            return

        if self.click_x and self.click_y: # user clicked, track new object if there is one at that location
            x, y = self.click_x, self.click_y
            self.click_x, self.click_y = None, None
            for track in tracks:
                x0, y0, x1, y1 = track.tlbr
                if y0 <= y <= y1 and x0 <= x <= x1:
                    self.tracking_object = track
                    self.ws_callback({"tracking": "Tracking Object"})
                    logger.info(f"Started tracking object ID: {self.tracking_object.track_id}")
                    return

        elif self.tracking_object: # continue tracking the same object
            object_still_there = any(track.track_id == self.tracking_object.track_id for track in tracks)
            if not object_still_there:
                self.tracking_object = None                
                self.runML = False # case where object is lost
                self.y_delta = 0
                self.x_delta = 0
                logger.info("Lost track of object")

    def adjust_delta(self, coords):
        # input list of [x0, y0, x1, y1]
        SCREEN_CENTER = (320, 400) 
        y_cent = (coords[2] + coords[0])/2
        x_cent = (coords[3] + coords[1])/2

        self.x_delta = SCREEN_CENTER[0] - x_cent
        self.y_delta = SCREEN_CENTER[1] - y_cent

    def tracking_servo(self):
        while True:
            if self.runML:
                if abs(self.y_delta) > 20:
                    self.mc.set_angle('y', self.y_delta)

                if abs(self.x_delta) > 20:
                    self.mc.set_angle('x', self.x_delta)

            time.sleep(0.05)

    def manual_control(self, message):
        if self.runML:
            logger.info("Manual control is disabled while tracking is on.")
            return
        match message:
            case "UP":
                self.mc.set_angle('y', -50)
            case "DOWN":
                self.mc.set_angle('y', 50)
            case "LEFT":
                self.mc.set_angle('x', 50)
            case "RIGHT":
                self.mc.set_angle('x', -50)