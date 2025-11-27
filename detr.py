from libcamera import Transform 
from ultralytics import RTDETR
from picamera2 import Picamera2
from picamera2.devices import Hailo
import torch
from torchvision.ops import nms
from yolox.tracker.byte_tracker import BYTETracker, STrack
import argparse
import cv2
import time
import numpy as np

# hailo = Hailo('detr_resnet50_v1_18_bn.hef')
hailo = Hailo('hailo_models/best_train6.hef')

# picam = Picamera2()
# # Just make the format RGB
# picam_config = picam.create_preview_configuration(main={"format": 'BGR888', "size": (1920, 1080)}, transform=Transform(hflip=1, vflip=1))
# picam.configure(picam_config)
# picam.start()
cap = cv2.VideoCapture('/home/clark64/Downloads/output_2.mp4')
# Initialize the BYTETracker
parser = argparse.ArgumentParser("basic args")
parser.add_argument("--track_thresh", type=float, default=0.5, help="tracking confidence threshold")
parser.add_argument("--track_buffer", type=int, default=60, help="the frames for keep lost tracks")
parser.add_argument("--match_thresh", type=float, default=0.90, help="matching threshold for tracking")
parser.add_argument('--min-box-area', type=float, default=10, help='filter out tiny boxes')
parser.add_argument("--mot20", dest="mot20", default=True, action="store_true", help="test mot20.")

btrack = BYTETracker(args=parser.parse_args())
frame_count = 0
tracks = []
useByteTrack = True
new_results = []
def process_image():
    global tracks, new_results
    results = hailo.run(resized_frame)[0]
    results_ind = nms(torch.tensor(results)[:, :4], torch.tensor(results)[:, 4], 0.6)

    print(results)
    new_results = [results[i] for i in results_ind]

    if len(new_results) == 0:
        cv2.imshow("Detections", resized_frame)
        cv2.waitKey(1)
        return
    tracks = btrack.update(np.array(new_results), [640, 640], [640, 640])

try:
    while True:
        # frame = picam.capture_array()
        # resized_frame = cv2.resize(frame, (640, 640))
        # start = time.time()
        # results = hailo.run(resized_frame)[0]
        # end = time.time()
        ret, frame = cap.read()
        # convert to BGR888
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        if not ret:
            break
        resized_frame = cv2.resize(frame, (640, 640))
        if frame_count % 3 == 0:
            process_image()
        
        if useByteTrack:

            for track in tracks:
                
                y1, x1, y2, x2 = track.tlbr # multiply by 640 to get original image coords
                x1 *= 640
                y1 *= 640
                x2 *= 640
                y2 *= 640
                conf = track.score
                id = track.track_id
                cv2.rectangle(resized_frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                # put track ID and confidence
                cv2.putText(resized_frame, f"ID: {id}", (int(x1), int(y1)-25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                cv2.putText(resized_frame, f"{conf:.2f}", (int(x1), int(y1)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        else:
            for result in new_results:
                y1, x1, y2, x2, conf = result
                x1 *= 640
                y1 *= 640
                x2 *= 640
                y2 *= 640
                cv2.rectangle(resized_frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                cv2.putText(resized_frame, f"{conf:.2f}", (int(x1), int(y1)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        cv2.imshow("Detections", resized_frame)
        cv2.waitKey(1)
        frame += 1


except KeyboardInterrupt:
    print("exiting")

        