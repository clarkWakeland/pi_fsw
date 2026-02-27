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
        self.camera = camera
        self.x_delta = 0
        self.y_delta = 0
        self.lost_weight = 1
        self.status_lock = threading.Lock()
        self.tracking_state = "IDLE"
        self.state_seq = 0
        self.state_changed_at = time.time()
        self.last_emitted_state = None
        self.last_emitted_seq = -1
        self.hailo = Hailo('hailo_models/barometric_beta.hef')
        self.mc = MotorControl(ws_callback=ws_callback)

        threading.Thread(target = self.tracking_servo, daemon=True).start()
        threading.Thread(target = self.ml_loop, daemon=True).start()
        threading.Thread(target = self.state_heartbeat_loop, daemon=True).start()

        # Initialize the BYTETracker
        parser = argparse.ArgumentParser("basic args")
        parser.add_argument("--track_thresh", type=float, default=0.5, help="tracking confidence threshold")
        parser.add_argument("--track_buffer", type=int, default=60, help="the frames for keep lost tracks")
        parser.add_argument("--match_thresh", type=float, default=0.95, help="matching threshold for tracking")
        parser.add_argument('--min-box-area', type=float, default=10, help='filter out tiny boxes')
        parser.add_argument("--mot20", dest="mot20", default=False, action="store_true", help="test mot20.")

        self.BYTEtracker = BYTETracker(args=parser.parse_args())

    def get_tracking_status(self):
        with self.status_lock:
            target_box = None
            track_id = None
            confidence = None
            if self.tracking_object is not None:
                x1, y1, x2, y2 = self.tracking_object.tlbr
                target_box = {
                    "x1": float(x1),
                    "y1": float(y1),
                    "x2": float(x2),
                    "y2": float(y2),
                }
                track_id = int(self.tracking_object.track_id)
                confidence = float(self.tracking_object.score)

            return {
                "state": self.tracking_state,
                "run_ml": self.user_intent.runML,
                "auto_acquire": self.user_intent.auto_acquire,
                "track_id": track_id,
                "lost_frames": int(self.lost_frames),
                "target_box": target_box,
                "confidence": confidence,
                "x_delta": float(self.x_delta),
                "y_delta": float(self.y_delta),
                "seq": int(self.state_seq),
                "ts": float(self.state_changed_at),
            }

    def emit_tracking_state(self, force=False):
        if self.ws_callback is None:
            return

        payload = self.get_tracking_status()
        if not force:
            if payload["state"] == self.last_emitted_state and payload["seq"] == self.last_emitted_seq:
                return

        self.last_emitted_state = payload["state"]
        self.last_emitted_seq = payload["seq"]
        self.ws_callback({"type": "tracking-state", "payload": payload})

    def _set_state(self, new_state):
        with self.status_lock:
            if self.tracking_state != new_state:
                self.tracking_state = new_state
                self.state_seq += 1
                self.state_changed_at = time.time()
                should_emit = True
            else:
                should_emit = False

        if should_emit:
            self.emit_tracking_state()

    def set_tracking_enabled(self, enabled):
        self.user_intent.set_ML(enabled)
        if enabled:
            self._set_state("PRIMED")
            return

        with self.status_lock:
            self.tracking_object = None
            self.lost_frames = 0
            self.lost_weight = 1
            self.x_delta = 0
            self.y_delta = 0
        self._set_state("IDLE")

    def toggle_tracking(self):
        self.set_tracking_enabled(not self.user_intent.runML)
        return self.user_intent.runML

    def state_heartbeat_loop(self):
        while True:
            # Keep UI in sync even if no transitions happen.
            self.emit_tracking_state(force=True)
            time.sleep(1.0)

    def start_tracking(self, x, y):
        logger.info(f"received {x} x and {y} y")
        self.set_tracking_enabled(True)
        self.user_intent.set_click_coordinates(x, y)
    
    def stop_tracking(self):
        self.set_tracking_enabled(False)

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

            time.sleep(0.05) # run inference 20 times/sec for now, can be adjusted
        
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

        tracks = []
        if len(results) != 0:
            tracks = self.BYTEtracker.update(np.array(results), [640, 640], [640, 640])

        self.update_tracking_object(tracks)

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
            self._set_state("ACQUIRING")
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
                    self._set_state("TRACKING")
                    self.user_intent.auto_acquire = False
                    return

        if self.user_intent.click_x and self.user_intent.click_y: # user clicked, track new object if there is one at that location
            self._set_state("ACQUIRING")
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
                    self._set_state("TRACKING")
                    logger.info(f"Started tracking object ID: {self.tracking_object.track_id}")
                    return
        
        if self.user_intent.ROI_coords: # user defined ROI, track new object if there is one in that region
            self._set_state("ACQUIRING")
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
                    self._set_state("TRACKING")
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
                self._set_state("TRACKING")
            else:                    # we lost the object
                self.lost_frames += 1
                # momentum decay
                self.lost_weight *= 0.5
                self._set_state("LOST")
                logger.info(f"Lost object ID: {self.tracking_object.track_id}, lost frames: {self.lost_frames}")

                if self.lost_frames > 20: # lost for more than 2 seconds
                    self.tracking_object = None                
                    self.lost_frames = 0
                    self.lost_weight = 1
                    self.y_delta = 0
                    self.x_delta = 0
                    logger.info("Lost track of object")
                    self.ws_callback({"tracking_lost": "Lost Object"})
                    self._set_state("PRIMED")
        elif self.user_intent.runML:
            self._set_state("PRIMED")

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
                if abs(self.y_delta) > 15:
                    self.mc.set_angle('y', self.y_delta)

                if abs(self.x_delta) > 15:
                    self.mc.set_angle('x', self.x_delta)

            time.sleep(0.033)

    def manual_control(self, message_direction, analog):
        if self.user_intent.runML:
            logger.info("Manual control is disabled while tracking is on.")
            return
        
        mag = analog['magnitude']
        movement = mag * 75
        logger.info(f"Manual control: {message_direction}, magnitude: {mag}, analog: {analog}")
        match message_direction:
            case "UP":
                self.mc.set_angle('y', movement)
            case "DOWN":
                self.mc.set_angle('y', -movement)
            case "LEFT":
                self.mc.set_angle('x', movement)
            case "RIGHT":
                self.mc.set_angle('x', -movement)