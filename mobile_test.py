#!/usr/bin/env python3
"""
Concise MobileNetV2 image classifier for Raspberry Pi
Uses TensorFlow Lite for optimal performance
"""

import numpy as np
import cv2
from picamera2 import Picamera2
import tflite_runtime.interpreter as tflite
import time

class MobileNetClassifier:
    def __init__(self, model_path="MobileNet-v2.tflite"):
        """Initialize MobileNetV2 classifier"""
        # Load ONNX model
        self.interpreter = tflite.Interpreter(model_path=model_path)
        self.interpreter.allocate_tensors()
        
        # Get input/output details
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        
        # Model expects 224x224 RGB images
        self.input_shape = self.input_details[0]['shape']
        print(f"Model input shape: {self.input_shape}")
        
        # Load ImageNet labels
        self.labels = self.load_labels()
        
    def load_labels(self):
        """Load ImageNet class labels"""
        # Download labels if not present
        try:
            with open('imagenet_labels.txt', 'r') as f:
                labels = [line.strip() for line in f.readlines()]
        except FileNotFoundError:
            print("Creating basic labels (download imagenet_labels.txt for full labels)")
            labels = [f"class_{i}" for i in range(1001)]
        
        return labels
    
    def preprocess_image(self, image):
        """Preprocess image for MobileNetV2"""
        # Resize to 224x224
        resized = cv2.resize(image, (224, 224))
        
        # Convert BGR to RGB
        rgb_image = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        
        # Normalize to [0, 1] and expand dimensions
        normalized = rgb_image.astype(np.float32) / 255.0
        input_data = np.expand_dims(normalized, axis=0)
        
        return input_data
    
    def classify(self, image):
        """Classify single image"""
        # Preprocess
        input_data = self.preprocess_image(image)
        
        # Run inference
        self.interpreter.set_tensor(self.input_details[0]['index'], input_data)
        self.interpreter.invoke()
        
        # Get predictions
        output_data = self.interpreter.get_tensor(self.output_details[0]['index'])
        predictions = output_data[0]
        
        # Get top 3 predictions
        top_indices = np.argsort(predictions)[-3:][::-1]
        
        results = []
        for idx in top_indices:
            confidence = predictions[idx]
            label = self.labels[idx] if idx < len(self.labels) else f"class_{idx}"
            results.append((label, confidence))
        
        return results

def download_model():
    """Download MobileNetV2 TFLite model if not present"""
    import os
    import urllib.request
    
    model_file = "mobilenet_v2_1.0_224_quant.tflite"
    labels_file = "imagenet_labels.txt"
    
    if not os.path.exists(model_file):
        print("Downloading MobileNetV2 model...")
        model_url = "https://storage.googleapis.com/download.tensorflow.org/models/mobilenet_v2_1.0_224_quant_20181010.tflite"
        urllib.request.urlretrieve(model_url, model_file)
        print(f"Downloaded {model_file}")
    
    if not os.path.exists(labels_file):
        print("Downloading ImageNet labels...")
        labels_url = "https://storage.googleapis.com/download.tensorflow.org/data/ImageNetLabels.txt"
        urllib.request.urlretrieve(labels_url, labels_file)
        print(f"Downloaded {labels_file}")

def test_with_camera():
    """Test classifier with Pi Camera"""

    
    # Initialize classifier
    classifier = MobileNetClassifier()
    
    # Setup camera
    picam2 = Picamera2()
    picam2.configure(picam2.create_preview_configuration(
        main={"format": 'XRGB8888', "size": (640, 480)}
    ))
    picam2.start()
    
    print("Press 'c' to classify, 'q' to quit")
    
    try:
        while True:
            # Capture frame
            frame = picam2.capture_array()
            
            # Display frame
            cv2.imshow('Camera', frame)
            
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('c'):
                print("Classifying...")
                start_time = time.time()
                
                # Classify current frame
                results = classifier.classify(frame)
                
                inference_time = time.time() - start_time
                print(f"Inference time: {inference_time:.2f}s")
                
                # Display results
                print("Top 3 predictions:")
                for i, (label, confidence) in enumerate(results):
                    print(f"{i+1}. {label}: {confidence:.3f}")
                print("-" * 40)
                
            elif key == ord('q'):
                break
    
    finally:
        picam2.stop()
        cv2.destroyAllWindows()

def test_with_image(image_path):
    """Test classifier with single image"""
    
    
    # Initialize classifier
    classifier = MobileNetClassifier()
    
    # Load and classify image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Could not load image: {image_path}")
        return
    
    print(f"Classifying {image_path}...")
    start_time = time.time()
    
    results = classifier.classify(image)
    
    inference_time = time.time() - start_time
    print(f"Inference time: {inference_time:.2f}s")
    
    # Display results
    print("Top 3 predictions:")
    for i, (label, confidence) in enumerate(results):
        print(f"{i+1}. {label}: {confidence:.3f}")
    
    # Show image with results
    cv2.putText(image, f"{results[0][0]}: {results[0][1]:.2f}", 
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.imshow('Classification Result', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def continuous_classification():
    """Continuous classification with camera"""
    # Download model if needed
    download_model()
    
    # Initialize classifier
    classifier = MobileNetClassifier()
    
    # Setup camera
    picam2 = Picamera2()
    picam2.configure(picam2.create_preview_configuration(
        main={"format": 'XRGB8888', "size": (320, 240)}  # Lower res for speed
    ))
    picam2.start()
    
    print("Continuous classification started. Press 'q' to quit")
    
    try:
        while True:
            # Capture frame
            frame = picam2.capture_array()
            
            # Classify every 30 frames (roughly 1 second at 30fps)
            if cv2.getTickCount() % 30 == 0:
                results = classifier.classify(frame)
                
                # Draw top prediction on frame
                if results:
                    label, confidence = results[0]
                    text = f"{label}: {confidence:.2f}"
                    cv2.putText(frame, text, (10, 30), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            # Display frame
            cv2.imshow('Live Classification', frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    
    finally:
        picam2.stop()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        # Test with provided image
        test_with_image(sys.argv[1])
    else:
        # Show menu
        print("MobileNetV2 Classifier Options:")
        print("1. Test with camera (press 'c' to classify)")
        print("2. Continuous classification")
        print("3. Quit")
        
        choice = input("Choose option (1-3): ").strip()
        
        if choice == '1':
            test_with_camera()
        elif choice == '2':
            continuous_classification()
        else:
            print("Goodbye!")