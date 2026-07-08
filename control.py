import time
import threading
from picamera2.devices import Hailo
from servo_control import MotorControl
from yolox.tracker.byte_tracker import BYTETracker
import numpy as np
from userIntent import UserIntent
import argparse
from runtime_utils import (
    MANUAL_CONTROL_INTERVAL_SECONDS,
    MANUAL_INPUT_TIMEOUT_SECONDS,
    MANUAL_STEP_ACCEL_LIMIT,
    ML_INPUT_SIZE,
    ML_INTERVAL_SECONDS,
    SERVO_STEP_ACCEL_LIMIT,
    deadline_sleep_seconds,
    ensure_inference_size,
    apply_soft_deadband,
    find_reacquire_track,
    prepare_hailo_detections,
    servo_ramp_step_limit,
    smooth_point,
    tracking_motion_confirmed,
)

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
        self.target_confirm_frames = 0
        self.tracking_motion_enabled = False
        self.tracking_motion_enabled_at = None
        self.last_target_box = None
        self.smoothed_target_center = None
        self.manual_lock = threading.Lock()
        self.manual_x = 0.0
        self.manual_y = 0.0
        self.manual_updated_at = 0.0
        self.manual_input_active = False
        self.hailo = Hailo('hailo_models/barometric_beta.hef')
        self.mc = MotorControl(ws_callback=ws_callback)

        threading.Thread(target = self.tracking_servo, daemon=True).start()
        threading.Thread(target = self.manual_servo_loop, daemon=True).start()
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

    def _capture_ml_frame(self):
        if self.camera is None:
            return None

        if hasattr(self.camera, "capture_ml_array"):
            return self.camera.capture_ml_array()

        return self.camera.capture_array()

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
            self.target_confirm_frames = 0
            self.tracking_motion_enabled = False
            self.tracking_motion_enabled_at = None
            self.last_target_box = None
            self.smoothed_target_center = None
        self._clear_manual_input(reset_steps=True)
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

    def _start_target_confirmation(self, track):
        self.tracking_object = track
        self.target_confirm_frames = 1
        self.tracking_motion_enabled = False
        self.tracking_motion_enabled_at = None
        self.last_target_box = np.asarray(track.tlbr, dtype=np.float32).copy()
        self.smoothed_target_center = None
        self.mc.reset_tracking_steps()
        self.lost_frames = 0
        self.lost_weight = 1
        self.x_delta = 0
        self.y_delta = 0
        self._set_state("ACQUIRING")

    def _mark_target_observed(self, track):
        self.tracking_object = track
        self.last_target_box = np.asarray(track.tlbr, dtype=np.float32).copy()
        self.lost_frames = 0
        self.lost_weight = 1

        if self.tracking_motion_enabled:
            self._set_state("TRACKING")
            return

        self.target_confirm_frames += 1
        if tracking_motion_confirmed(self.target_confirm_frames):
            self.tracking_motion_enabled = True
            self.tracking_motion_enabled_at = time.monotonic()
            logger.info("Target confirmed; enabling servo tracking")
            self._set_state("TRACKING")
        else:
            self._set_state("ACQUIRING")

    def _servo_step_limit(self):
        return servo_ramp_step_limit(time.monotonic(), self.tracking_motion_enabled_at)

    def ml_loop(self):
        ''' 
        ML thread loop. 
        Only runs if user toggles tracking
        '''
        next_run = time.monotonic()
        while True:
            if self.user_intent.runML:
                next_run += ML_INTERVAL_SECONDS
                frame = self._capture_ml_frame()
                if frame is not None:
                    self.process_image(frame)
                    if self.tracking_object:
                        logger.debug("Tracking box: %s", self.tracking_object.tlbr)
                        self.adjust_delta(self.tracking_object.tlbr)

                now = time.monotonic()
                sleep_for = deadline_sleep_seconds(now, next_run)
                if sleep_for > 0:
                    time.sleep(sleep_for)
                else:
                    next_run = now
                continue

            next_run = time.monotonic()
            time.sleep(ML_INTERVAL_SECONDS)
        
    def process_image(self, image):
        ''' 
        Process a single image, return object tracks
        '''
        image = ensure_inference_size(image, ML_INPUT_SIZE)
        results = self.hailo.run(image)[0] #  returns normalized xyxy boxes with confidence
        results = prepare_hailo_detections(results, image_size=ML_INPUT_SIZE, iou_threshold=0.7)
        logger.debug("Detections after NMS: %s", results)

        tracks = []
        if len(results) != 0:
            tracks = self.BYTEtracker.update(results, [ML_INPUT_SIZE, ML_INPUT_SIZE], [ML_INPUT_SIZE, ML_INPUT_SIZE])

        self.update_tracking_object(tracks)

        boxes = []
        for track in tracks:
            currently_tracking = (self.tracking_object is not None and track.track_id == self.tracking_object.track_id)
            boxes.append({'x1': track.tlbr[0], 'y1': track.tlbr[1], 'x2': track.tlbr[2], 'y2': track.tlbr[3], 'id': track.track_id, 'confidence': str(track.score), 'currently_tracking': currently_tracking})

        if self.ws_callback is not None:
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
            logger.debug("Track scores: %s", [track.score for track in tracks])
            for track in tracks:
                if track.score > conf_thresh:
                    logger.info(f"Auto acquired object ID: {track.track_id} with confidence {track.score}")
                    self.ws_callback({"object": {"box": {'x1': track.tlbr[0], 'y1': track.tlbr[1], 'x2': track.tlbr[2], 'y2': track.tlbr[3]}, "confidence": str(track.score)}})
                    self._start_target_confirmation(track)
                    self.user_intent.auto_acquire = False
                    return

        if self.user_intent.click_x and self.user_intent.click_y: # user clicked, track new object if there is one at that location
            self._set_state("ACQUIRING")
            x, y = self.user_intent.click_x, self.user_intent.click_y
            self.user_intent.clear_click_coordinates()
            logger.debug("Candidate tracks: %s", tracks)
            for track in tracks:
                x0, y0, x1, y1 = track.tlbr
                logger.debug("Checking track ID %s at box %s", track.track_id, track.tlbr)
                logger.debug("Click coordinates: (%s, %s)", x, y)
                if y0 <= y <= y1 and x0 <= x <= x1:
                    self._start_target_confirmation(track)
                    self.ws_callback({"tracking": "Tracking Object"})
                    logger.info(f"Started target confirmation for object ID: {self.tracking_object.track_id}")
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
                    self._start_target_confirmation(track)
                    self.ws_callback({"tracking": "Tracking Object"})
                    self.user_intent.clear_ROI()
                    logger.info(f"Started target confirmation for object ID: {self.tracking_object.track_id} from ROI")
                    return

        elif self.tracking_object: # continue tracking the same object
            for track in tracks:
                if track.track_id == self.tracking_object.track_id:
                    matching_track = track
                    break
            else:
                matching_track = find_reacquire_track(self.last_target_box, tracks)
                if matching_track is not None:
                    logger.info(
                        "Re-acquired target as track ID %s after losing track ID %s",
                        matching_track.track_id,
                        self.tracking_object.track_id
                    )
                else:
                    logger.debug("No matching track found")

            if matching_track:
                logger.debug("Found matching track")
                self._mark_target_observed(matching_track)
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
                    self.target_confirm_frames = 0
                    self.tracking_motion_enabled = False
                    self.tracking_motion_enabled_at = None
                    self.last_target_box = None
                    self.smoothed_target_center = None
                    logger.info("Lost track of object")
                    self.ws_callback({"tracking_lost": "Lost Object"})
                    self._set_state("PRIMED")
        elif self.user_intent.runML:
            self._set_state("PRIMED")

    def adjust_delta(self, coords):
        # input list of [x0, y0, x1, y1]
        SCREEN_CENTER = (320, 400) 
        measured_center = np.array([
            (coords[2] + coords[0]) / 2,
            (coords[3] + coords[1]) / 2,
        ], dtype=np.float32)
        self.smoothed_target_center = smooth_point(self.smoothed_target_center, measured_center)
        y_cent = self.smoothed_target_center[0]
        x_cent = self.smoothed_target_center[1]

        self.x_delta = apply_soft_deadband(SCREEN_CENTER[0] - x_cent) * self.lost_weight
        self.y_delta = apply_soft_deadband(SCREEN_CENTER[1] - y_cent) * self.lost_weight
        logger.debug("x_delta: %s, y_delta: %s", self.x_delta, self.y_delta)

    def tracking_servo(self):
        while True:
            if self.tracking_object and self.tracking_motion_enabled:
                max_step = self._servo_step_limit()
                if self.y_delta != 0.0:
                    self.mc.set_angle('y', self.y_delta, max_step=max_step, max_step_change=SERVO_STEP_ACCEL_LIMIT)

                if self.x_delta != 0.0:
                    self.mc.set_angle('x', self.x_delta, max_step=max_step, max_step_change=SERVO_STEP_ACCEL_LIMIT)

            time.sleep(0.033)

    def _clear_manual_input(self, reset_steps=False):
        with self.manual_lock:
            self.manual_x = 0.0
            self.manual_y = 0.0
            self.manual_updated_at = 0.0
            self.manual_input_active = False

        if reset_steps and hasattr(self.mc, "reset_manual_steps"):
            self.mc.reset_manual_steps()

    def _apply_manual_control_step(self, now=None):
        if self.user_intent.runML:
            self._clear_manual_input(reset_steps=True)
            return

        if now is None:
            now = time.monotonic()

        with self.manual_lock:
            x = self.manual_x
            y = self.manual_y
            updated_at = self.manual_updated_at
            active = self.manual_input_active

        if not active:
            return

        if now - updated_at > MANUAL_INPUT_TIMEOUT_SECONDS:
            self._clear_manual_input(reset_steps=True)
            return

        self.mc.set_manual_input(x, y, max_step_change=MANUAL_STEP_ACCEL_LIMIT)

    def manual_servo_loop(self):
        while True:
            self._apply_manual_control_step()
            time.sleep(MANUAL_CONTROL_INTERVAL_SECONDS)

    def manual_control(self, analog):
        if self.user_intent.runML:
            logger.info("Manual control is disabled while tracking is on.")
            self._clear_manual_input(reset_steps=True)
            return

        x = float(analog.get('x', 0.0))
        y = float(analog.get('y', 0.0))
        magnitude = float(analog.get('magnitude', min(1.0, np.hypot(x, y))))
        source = analog.get('source', 'unknown')

        x = max(-1.0, min(1.0, x))
        y = max(-1.0, min(1.0, y))
        magnitude = max(0.0, min(1.0, magnitude))

        logger.debug(
            "Manual analog control source=%s x=%.3f y=%.3f magnitude=%.3f",
            source, x, y, magnitude
        )

        with self.manual_lock:
            self.manual_x = x
            self.manual_y = y
            self.manual_updated_at = time.monotonic()
            self.manual_input_active = True
