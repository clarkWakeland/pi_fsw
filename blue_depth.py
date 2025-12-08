import cv2
import numpy as np
import torch
from ultralytics import YOLO
from yolox.tracker.byte_tracker import BYTETracker, STrack
import argparse
from torchvision.ops import nms

parser = argparse.ArgumentParser("basic args")
parser.add_argument("--track_thresh", type=float, default=0.5, help="tracking confidence threshold")
parser.add_argument("--track_buffer", type=int, default=60, help="the frames for keep lost tracks")
parser.add_argument("--match_thresh", type=float, default=0.8, help="matching threshold for tracking")
parser.add_argument('--min-box-area', type=float, default=10, help='filter out tiny boxes')
parser.add_argument("--mot20", dest="mot20", default=False, action="store_true", help="test mot20.")

btrack = BYTETracker(args=parser.parse_args())

cap = cv2.VideoCapture('/home/clark/Desktop/dataset/validation_vid.mp4')
model = YOLO("yolo_models/best_train6.pt")
BLUE_THRESHOLD = 87
try:
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        resized_frame = cv2.resize(frame, (640, 640))
        results = model(resized_frame)[0].boxes.xyxyn


        blue_channel = frame[:, :, 0]
        green_channel = frame[:, :, 1]
        red_channel = frame[:, :, 2]

        # print channel means
        # print(f"Blue channel mean: {np.mean(blue_channel)}")
        # print(f"Green channel mean: {np.mean(green_channel)}")
        # print(f"Red channel mean: {np.mean(red_channel)}")

        # blue relative to green
        blue_relative_divided = cv2.subtract(blue_channel, green_channel)

        # for each box, take the average blue relative value and print it
        for box in results:
            x1 = int(box[0] * frame.shape[1])
            y1 = int(box[1] * frame.shape[0])
            x2 = int(box[2] * frame.shape[1])
            y2 = int(box[3] * frame.shape[0])

            box_blue_relative = blue_relative_divided[y1:y2, x1:x2]
            # in each box, ignore high values above 200 to reduce noise
            box_blue_relative = box_blue_relative[box_blue_relative < BLUE_THRESHOLD]
    
            mean_blue_relative = np.mean(box_blue_relative)

            # take the lowest value in the box to avoid noise
            # min_blue_relative = np.min(box_blue_relative)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"Blue Rel: {mean_blue_relative:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # show the relative blue channel with a color gradient
        blue_relative_gradient = cv2.applyColorMap(blue_relative_divided, cv2.COLORMAP_JET)

        # inverse so that higher values are more blue
        # blue_relative_gradient = cv2.bitwise_not(blue_relative_gradient)
        # show the channel with a color gradient
        # blue_gradient = cv2.applyColorMap(blue_relative_gradient, cv2.COLORMAP_JET)
        # show the frame with a mask of the blue threshold
        # blue_mask = cv2.inRange(blue_relative_divided, BLUE_THRESHOLD, 255)
        # cv2.imshow("Frame", blue_mask)
        cv2.imshow("Blue Channel", frame)
 
        cv2.waitKey(1)
        input("Press Enter to continue...")

except KeyboardInterrupt:
    cap.release()
    cv2.destroyAllWindows()
    pass
finally:
    cap.release()
    cv2.destroyAllWindows()