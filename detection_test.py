from ultralytics import YOLO
from PIL import Image
import cv2
import numpy as np
from picamera2 import Picamera2
import time
import threading
from interpreter_test import detect_objects
from adafruit_servo_test import MotorControl

class PersonTracking:
    def __init__(self):
        # self.picam2 = camera
        # camera_config = self.picam2.create_video_configuration(main={"size": (640, 640)},)
        # self.picam2.configure(camera_config)
        # self.picam2.start()
        self.frame = None
        self.running = True
        
        self.interpreter = Interpreter(
        model_path = "ssd_mobilenet_v2_coco_quant_postprocess_edgetpu.tflite",
        experimental_delegates=[load_delegate("libedgetpu.so.1", options={"device": ":0"})]
        )
        self.interpreter.allocate_tensors()
        
        # Get model details
        input_details = self.interpreter.get_input_details()
        output_details = self.interpreter.get_output_details()
        input_shape = input_details[0]['shape']
        self.iheight, self.iwidth = input_shape[1], input_shape[2]
        
        # self.mc = MotorControl()
        # Print model info
        # print(f"Input shape: {input_details[0]['shape']}")
        # print(f"Input type: {input_details[0]['dtype']}")
        
        self.x_delta = 0
        self.y_delta = 0
        # self.ty = threading.Thread(target = self.move_y_servo, daemon = True)
        # self.tx = threading.Thread(target = self.move_x_servo, daemon = True)
        # self.ty.start()
        # self.tx.start()
        
        
        # Load labels
        # self.labels = self.load_labels()
        # print(f"Loaded {len(labels)} labels")
    
    def load_labels(path="coco_labels.txt"):
        """Load the labels file associated with the model."""
        with open(path, 'r') as f:
            labels = [line.strip() for line in f.readlines()]
        # Remove the first label if it's '???'
        if labels[0] == '???':
            del(labels[0])
        return labels
            

    def stop(self):
        print("Stopping person tracking...")
        self.running = False
        self.picam2.stop()
        cv2.destroyAllWindows()

    def basic_video(self, frame):
        if frame is not None:
            results = self.process_image(frame)
            print(results)
            for result in results:
                if result["score"] > 0.5 and result['class_id'] == 0:  # Assuming class_id 0 is 'person'
                    box = result['bounding_box'] * 640
                    y1, x1, y2, x2 = map(int, box)
                    # cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    self.move_servo([x1, y1, x2, y2])
    
    def process_image(self, image):
        # Resize image to model input size
        
        original_image = Image.fromarray(image)#.convert("RGB")
        resized_image = original_image.resize((self.iwidth, self.iheight))
        
        # Run object detection
        results = detect_objects(self.interpreter, resized_image, 0.5)
        
        return results
        
        
    def move_servo(self, coords):
        # input list of [x1, y1, x2, y2]
        SCREEN_CENTER = (320, 320)
        x_cent = (coords[2] + coords[0])/2 
        y_cent = (coords[3] + coords[1])/2
        
        self.x_delta = x_cent - SCREEN_CENTER[0]
        self.y_delta = y_cent - SCREEN_CENTER[1]
        
        
        
    # def move_y_servo(self):
    #     while True:
    #         print(f"y_delta: {self.y_delta}")
    #         if self.y_delta > 20:
    #             self.mc.set_angle(self.y_delta)
    #         elif self.y_delta < -20:
    #             self.mc.set_angle(self.y_delta)
    #         time.sleep(0.05)
    
    # def move_x_servo(self):
    #     while True:
    #         # print(f"x_delta: {self.x_delta}")
    #         if self.x_delta > 20:
    #             self.mc.set_angle(self.x_delta * -1) # actually going right
    #         elif self.x_delta < -20:
    #             self.mc.set_angle(self.x_delta * -1) # actually going left
    #         time.sleep(0.05)
        
def main():
    tracker = PersonTracking()
    try:
        tracker.basic_video()
    except KeyboardInterrupt:
        tracker.stop()
    
if __name__ == "__main__":
    main()



