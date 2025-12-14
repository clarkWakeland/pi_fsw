# from ultralytics import YOLO
import cv2
import time
import threading
from picamera2.devices import Hailo
import torch
from servo_control import MotorControl
from yolox.tracker.byte_tracker import BYTETracker, STrack
import numpy as np
from userIntent import UserIntent
from torchvision.ops import nms
import argparse

import logging
logging.basicConfig(
    level=logging.INFO, 
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger(__name__)

class PersonTracking:
    def __init__(self, ws_callback=None, camera=None):
        
        self.user_intent = UserIntent()

        self.frame = None
        self.lost_frames = 0

        self.tracking_object = None

        self.ws_callback = ws_callback
        self.show_boxes = False
        self.camera = camera
        self.x_delta = 0
        self.y_delta = 0
        self.lost_weight = 1
        self.hailo = Hailo('hailo_models/best_train6.hef')
        self.mc = MotorControl(ws_callback=ws_callback)

        threading.Thread(target = self.tracking_servo, daemon=True).start()
        threading.Thread(target = self.ml_loop, daemon=True).start()

        # Initialize the BYTETracker
        parser = argparse.ArgumentParser("basic args")
        parser.add_argument("--track_thresh", type=float, default=0.5, help="tracking confidence threshold")
        parser.add_argument("--track_buffer", type=int, default=60, help="the frames for keep lost tracks")
        parser.add_argument("--match_thresh", type=float, default=0.80, help="matching threshold for tracking")
        parser.add_argument('--min-box-area', type=float, default=10, help='filter out tiny boxes')
        parser.add_argument("--mot20", dest="mot20", default=False, action="store_true", help="test mot20.")

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
            if self.user_intent.runML:
                logger.info("Processing images")
                frame = self.camera.capture_array()
                if frame is not None:
                    self.process_image(frame)
                    if self.tracking_object:
                        logger.info(self.tracking_object.tlbr)
                        self.adjust_delta(self.tracking_object.tlbr)

            time.sleep(0.1) # run inference 10 times/sec for now, can be adjusted
        
    def process_image(self, image):
        ''' 
        Process a single image, return object tracks
        '''
        image = cv2.resize(image, (640, 640))
        results = self.hailo.run(image)[0] #  returns normalized xyxy boxes with confidence
        results = np.array(results)
        results[:, :4] *= 640  # scale boxes to image size
        # run NMS
        results_ind = nms(torch.tensor(results)[:, :4], torch.tensor(results)[:,4], 0.7)
        results = [results[i] for i in results_ind]
        # logger.info(f"Detections: {results}")

        if len(results) != 0:
            tracks = self.BYTEtracker.update(np.array(results), [640, 640], [640, 640])
            self.update_tracking_object(tracks)

            if self.show_boxes:
                boxes = []
                for track in tracks:
                    currently_tracking = (self.tracking_object is not None and track.track_id == self.tracking_object.track_id)
                    boxes.append({'x1': track.tlbr[0], 'y1': track.tlbr[1], 'x2': track.tlbr[2], 'y2': track.tlbr[3], 'id': track.track_id, 'confidence': str(track.score), 'currently_tracking': currently_tracking})

                self.ws_callback({"boxes": boxes})

    def update_tracking_object(self, tracks):
        ''' 
        logic for selecting and updating the tracked object
        '''
        conf_thresh = 0.7
        if self.user_intent.auto_acquire:   # auto acquire mode
            if not tracks:
                self.x_delta = 0
                self.y_delta = 0
                self.tracking_object = None
                return
            logger.info([track.score for track in tracks])
            for track in tracks:
                if track.score > conf_thresh:
                    logger.info(f"Auto acquired object ID: {track.track_id} with confidence {track.score}")
                    self.ws_callback({"object": {"box": {'x1': track.tlbr[0], 'y1': track.tlbr[1], 'x2': track.tlbr[2], 'y2': track.tlbr[3]}, "confidence": str(track.score)}})
                    self.tracking_object = track
                    self.auto_acquire = False
                    return

        if self.user_intent.click_x and self.user_intent.click_y: # user clicked, track new object if there is one at that location
            x, y = self.user_intent.click_x, self.user_intent.click_y
            self.user_intent.clear_click_coordinates()
            print(tracks)
            for track in tracks:
                x0, y0, x1, y1 = track.tlbr
                logger.info(f"Checking track ID: {track.track_id} at box {track.tlbr}")
                logger.info(f"Click coordinates: ({x}, {y})")
                if y0 <= y <= y1 and x0 <= x <= x1:
                    self.tracking_object = track
                    self.ws_callback({"tracking": "Tracking Object"})
                    logger.info(f"Started tracking object ID: {self.tracking_object.track_id}")
                    return
        
        if self.user_intent.ROI_coords: # user defined ROI, track new object if there is one in that region
            y, x, h, w = self.user_intent.ROI_coords
            # do not clear ROI until we find an object
            # or user manually clears it

            for track in tracks:
                x0, y0, x1, y1 = track.tlbr
                # check if the track center is within the ROI
                x_cent = (x0 + x1) / 2
                y_cent = (y0 + y1) / 2
                if x <= x_cent <= x + w and y <= y_cent <= y + h:
                    self.tracking_object = track
                    self.ws_callback({"tracking": "Tracking Object"})
                    self.user_intent.clear_ROI()
                    logger.info(f"Started tracking object ID: {self.tracking_object.track_id} from ROI")
                    return

        elif self.tracking_object: # continue tracking the same object
            for track in tracks:
                if track.track_id == self.tracking_object.track_id:
                    matching_track = track
                    break
            else:
                matching_track = None
                print("No matching track found")

            if matching_track:
                print("Found matching track")
                self.tracking_object = matching_track
                self.lost_frames = 0
                self.lost_weight = 1
            else:                    # we lost the object
                self.lost_frames += 1
                # momentum decay
                self.lost_weight *= 0.5
                logger.info(f"Lost object ID: {self.tracking_object.track_id}, lost frames: {self.lost_frames}")

                if self.lost_frames > 20: # lost for more than 2 seconds
                    self.tracking_object = None                
                    self.runML = False
                    self.y_delta = 0
                    self.x_delta = 0
                    logger.info("Lost track of object")
                    self.ws_callback({"tracking_lost": "Lost Object"})

    def adjust_delta(self, coords):
        # input list of [x0, y0, x1, y1]
        SCREEN_CENTER = (320, 400) 
        print("coods:", coords)
        y_cent = (coords[2] + coords[0])/2
        x_cent = (coords[3] + coords[1])/2

        self.x_delta = (SCREEN_CENTER[0] - x_cent) * self.lost_weight
        self.y_delta = (SCREEN_CENTER[1] - y_cent) * self.lost_weight
        logger.info(f"x_delta: {self.x_delta}, y_delta: {self.y_delta}")

    def tracking_servo(self):
        while True:
            if self.tracking_object:
                if abs(self.y_delta) > 20:
                    self.mc.set_angle('y', self.y_delta)

                if abs(self.x_delta) > 20:
                    self.mc.set_angle('x', self.x_delta)

            time.sleep(0.075)

    def manual_control(self, message):
        if self.user_intent.runML:
            logger.info("Manual control is disabled while tracking is on.")
            return
        match message:
            case "UP":
                self.mc.set_angle('y', 50)
            case "DOWN":
                self.mc.set_angle('y', -50)
            case "LEFT":
                self.mc.set_angle('x', 50)
            case "RIGHT":
                self.mc.set_angle('x', -50)

