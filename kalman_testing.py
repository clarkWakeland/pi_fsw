import torch
from yolox.data.data_augment import ValTransform
from yolox.exp import get_exp
from yolox.utils import postprocess
from yolox.tracker.byte_tracker import BYTETracker, STrack
import argparse
import cv2
import time
import numpy as np

# hailo = Hailo('detr_resnet50_v1_18_bn.hef')
# hailo = Hailo('hailo_models/best_train6.hef')
YOLOX_EXP_FILE = "/home/clark/repos/YOLOX/exps/example/custom/yolox_s.py"
YOLOX_CKPT_FILE = "/home/clark/repos/YOLOX/aquatic_alphaV2.pth"
YOLOX_TEST_SIZE = (640, 640)
YOLOX_CONF = 0.3
YOLOX_NMS = 0.6
YOLOX_FP16 = False
YOLOX_LEGACY = False
YOLOX_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def load_yolox_model():
    exp = get_exp(YOLOX_EXP_FILE, None)
    exp.test_conf = YOLOX_CONF
    exp.nmsthre = YOLOX_NMS
    exp.test_size = YOLOX_TEST_SIZE
    model = exp.get_model()
    ckpt = torch.load(YOLOX_CKPT_FILE, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model"])
    model.eval()
    if YOLOX_DEVICE == "cuda":
        model.cuda()
        if YOLOX_FP16:
            model.half()
    return model, exp, ValTransform()


yolox_model, yolox_exp, yolox_preproc = load_yolox_model()

# picam = Picamera2()
# # Just make the format RGB
# picam_config = picam.create_preview_configuration(main={"format": 'BGR888', "size": (1920, 1080)}, transform=Transform(hflip=1, vflip=1))
# picam.configure(picam_config)
# picam.start()
cap = cv2.VideoCapture('/home/clark/Desktop/model_training/validation_vids/masters_2.mp4')
# Initialize the BYTETracker
parser = argparse.ArgumentParser("basic args")
parser.add_argument("--track_thresh", type=float, default=0.35, help="tracking confidence threshold")
parser.add_argument("--track_buffer", type=int, default=60, help="the frames for keep lost tracks")
parser.add_argument("--match_thresh", type=float, default=0.80, help="matching threshold for tracking")
parser.add_argument('--min-box-area', type=float, default=10, help='filter out tiny boxes')
parser.add_argument("--mot20", dest="mot20", default=True, action="store_true", help="test mot20.")

btrack = BYTETracker(args=parser.parse_args())
frame_count = 0
tracks = []
useByteTrack = True
new_results = []
WIDTH_SCALE = 1080/640
HEIGHT_SCALE = 1920/640
BLUE_THRESHOLD = 87 # Adjust this threshold based on your requirements. 87 experimentally works well
VELOCITY_SCALE = 10
SHOW_BACKGROUND = True

def process_image():
    global tracks, new_results, resized_frame

    img, _ = yolox_preproc(resized_frame, None, yolox_exp.test_size)
    img = torch.from_numpy(img).unsqueeze(0).float()
    if YOLOX_DEVICE == "cuda":
        img = img.cuda()
        if YOLOX_FP16:
            img = img.half()

    with torch.no_grad():
        outputs = yolox_model(img)
        print(outputs)
        outputs = postprocess(
            outputs,
            yolox_exp.num_classes,
            yolox_exp.test_conf,
            yolox_exp.nmsthre
        )

    new_results = []
    if outputs is not None and outputs[0] is not None:
        dets = outputs[0].cpu().numpy()
        for det in dets:
            x1, y1, x2, y2, obj_conf, cls_conf, _cls = det
            score = float(obj_conf * cls_conf)
            new_results.append([x1, y1, x2, y2, score])
    print(new_results)

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
        process_image()
        
        blue_channel = resized_frame[:, :, 0]
        green_channel = resized_frame[:, :, 1]

        blue_relative_channel = cv2.subtract(blue_channel, green_channel)

        if not SHOW_BACKGROUND:
            frame = np.zeros_like(frame)

        if useByteTrack:

            for track in tracks:
                
                x1, y1, x2, y2 = track.tlbr 
                x1 *= HEIGHT_SCALE
                y1 *= WIDTH_SCALE
                x2 *= HEIGHT_SCALE
                y2 *= WIDTH_SCALE
        
                box_blue_relative = blue_relative_channel[int(y1):int(y2), int(x1):int(x2)]
                box_blue_relative = box_blue_relative[box_blue_relative < BLUE_THRESHOLD]
                mean_blue_relative = np.mean(box_blue_relative) if box_blue_relative.size > 0 else 0

                conf = track.score
                id = track.track_id
                x, y = track.to_xy()
                # draw predicted center + velocity vector in red
                if track.predicted is not None:
                    x_pred, y_pred, vx_pred, vy_pred = track.predicted
                    center = (int(x_pred * HEIGHT_SCALE), int(y_pred * WIDTH_SCALE))
                    velocity_end = (
                        int((x_pred + vx_pred * VELOCITY_SCALE) * HEIGHT_SCALE),
                        int((y_pred + vy_pred * VELOCITY_SCALE) * WIDTH_SCALE),
                    )
                    cv2.circle(frame, center, 4, (0, 0, 255), -1)
                    cv2.arrowedLine(frame, center, velocity_end, (0, 0, 255), 2, tipLength=0.2)

                # draw actual detection box in green
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                # put track ID and confidence
                cv2.putText(frame, f"ID: {id}", (int(x1), int(y1)-25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                cv2.putText(frame, f"{conf:.2f}", (int(x1), int(y1)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                # put x, y, a, and h on bounding box
                # cv2.putText(frame, f"x:{x:.3f} y:{y:.3f} a:{a:.3f} h:{h:.3f}", (int(x1), int(y2)+15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                # cv2.putText(frame, f"BlueRel: {mean_blue_relative:.2f}", (int(x1), int(y2)+15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        else:
            for result in new_results:
                x1, y1, x2, y2, conf = result
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                cv2.putText(frame, f"{conf:.2f}", (int(x1), int(y1)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        cv2.imshow("Detections", frame)
        cv2.waitKey(0)
        # Save image to folder
        # cv2.imwrite(f"output/frame_{frame_count:04d}.png", frame)

        # input("Press enter to continue...")
        frame_count += 1


except KeyboardInterrupt:
    print("exiting")

        
