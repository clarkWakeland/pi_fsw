from ultralytics import RTDETR
from picamera2 import Picamera2
from picamera2.devices import Hailo
import cv2
import time

# hailo = Hailo('detr_resnet50_v1_18_bn.hef')
hailo = Hailo('/usr/share/hailo-models/yolov8s_h8l.hef')

picam = Picamera2()
# Just make the format RGB
picam_config = picam.create_preview_configuration(main={"format": 'RGB', "size": (1920, 1080)})
picam.configure(picam_config)
picam.start()

try:
    while True:
        frame = picam.capture_array()
        resized_frame = cv2.resize(frame, (640, 640))
        start = time.time()
        results = hailo.run(resized_frame)[0]
        end = time.time()
        print(results)
        print(f"Inference time: {end - start:.3f} seconds")
        time.sleep(5)

except KeyboardInterrupt:
    print("exiting")

        