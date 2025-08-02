
import cv2
import numpy as np
import mediapipe as mp
from picamera2 import Picamera2
import threading
import time

class PersonTracking:
    print("Test")
    def __init__(self):
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(static_image_mode=False, model_complexity=2, enable_segmentation=True)
        self.picam2 = Picamera2()
        self.picam2.start()
        self.frame = None
        self.lock = threading.Lock()
        self.running = True