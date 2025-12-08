from ultralytics import RTDETR, YOLO
import torch
from torchvision.ops import nms
from yolox.tracker.byte_tracker import BYTETracker, STrack
import argparse
import cv2
import time
import numpy as np

# hailo = Hailo('detr_resnet50_v1_18_bn.hef')
# hailo = Hailo('hailo_models/best_train6.hef')
model = YOLO("yolo_models/best_train6.pt")

# picam = Picamera2()
# # Just make the format RGB
# picam_config = picam.create_preview_configuration(main={"format": 'BGR888', "size": (1920, 1080)}, transform=Transform(hflip=1, vflip=1))
# picam.configure(picam_config)
# picam.start()
cap = cv2.VideoCapture('/home/clark/Desktop/dataset/validation_vid2.mp4')
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
BLUE_THRESHOLD = 87 # Adjust this threshold based on your requirements. 87 experimentally works well

def process_image():
    global tracks, new_results, resized_frame

    results = model(resized_frame)[0]
    results_ind = nms(results.boxes.xyxyn, results.boxes.conf, 0.6)
    # convert tensor to list
    results_list = results.boxes.xyxyn.tolist()
    scores_list = results.boxes.conf.tolist()

    new_results = [results_list[i] + [scores_list[i]] for i in results_ind]
     # append confidence to each box

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
        # frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        if not ret:
            break
        resized_frame = cv2.resize(frame, (640, 640))
        if frame_count % 3 == 0:
            process_image()
        
        blue_channel = resized_frame[:, :, 0]
        green_channel = resized_frame[:, :, 1]

        blue_relative_channel = cv2.subtract(blue_channel, green_channel)

        if useByteTrack:

            for track in tracks:
                
                x1, y1, x2, y2 = track.tlbr # multiply by 640 to get original image coords
                x1 *= 640
                y1 *= 640
                x2 *= 640
                y2 *= 640
                
                box_blue_relative = blue_relative_channel[int(y1):int(y2), int(x1):int(x2)]
                box_blue_relative = box_blue_relative[box_blue_relative < BLUE_THRESHOLD]
                mean_blue_relative = np.mean(box_blue_relative) if box_blue_relative.size > 0 else 0

                conf = track.score
                id = track.track_id
                x, y, a, h = track.tlwh_to_xyah(track.tlwh)
                # draw predicted rectangle from x, y, a, h and draw it in red
                if track.predicted is not None:
                    print(track.predicted)
                    x_pred, y_pred, a_pred, h_pred = track.predicted[:4]
                    x1_red = int(x_pred * 640 - a_pred * h_pred / 2 * 640)
                    y1_red = int(y_pred * 640 - h_pred / 2 * 640)
                    x2_red = int(x_pred * 640 + a_pred * h_pred / 2 * 640)
                    y2_red = int(y_pred * 640 + h_pred / 2 * 640)
                    cv2.rectangle(resized_frame, (x1_red, y1_red), (x2_red, y2_red), (0, 0, 255), 2)

                # draw actual detection box in green
                cv2.rectangle(resized_frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                # put track ID and confidence
                cv2.putText(resized_frame, f"ID: {id}", (int(x1), int(y1)-25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                cv2.putText(resized_frame, f"{conf:.2f}", (int(x1), int(y1)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                # put x, y, a, and h on bounding box
                cv2.putText(resized_frame, f"x:{x:.3f} y:{y:.3f} a:{a:.3f} h:{h:.3f}", (int(x1), int(y2)+15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                # cv2.putText(resized_frame, f"BlueRel: {mean_blue_relative:.2f}", (int(x1), int(y2)+15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
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
        input("Press enter to continue...")
        frame += 1


except KeyboardInterrupt:
    print("exiting")

        