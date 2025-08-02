#!/usr/bin/env python3

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from tflite_runtime.interpreter import Interpreter, load_delegate
import argparse
from time import time

def load_labels(path):
    """Load the labels file associated with the model."""
    with open(path, 'r') as f:
        labels = [line.strip() for line in f.readlines()]
    # Remove the first label if it's '???'
    if labels[0] == '???':
        del(labels[0])
    return labels

def set_input_tensor(interpreter, image):
    """Sets the input tensor."""
    tensor_index = interpreter.get_input_details()[0]['index']
    input_tensor = interpreter.tensor(tensor_index)()[0]
    input_tensor[:, :] = image

def get_output_tensor(interpreter, index):
    """Returns the output tensor at the given index."""
    output_details = interpreter.get_output_details()[index]
    tensor = np.squeeze(interpreter.get_tensor(output_details['index']))
    return tensor

def detect_objects(interpreter, image, threshold):
    """Returns a list of detection results, each a dictionary of object info."""
    set_input_tensor(interpreter, image)
    interpreter.invoke()
    
    # Get all output details
    boxes = get_output_tensor(interpreter, 0)
    classes = get_output_tensor(interpreter, 1)
    scores = get_output_tensor(interpreter, 2)
    count = int(get_output_tensor(interpreter, 3))
    
    results = []
    for i in range(count):
        if scores[i] >= threshold:
            result = {
                'bounding_box': boxes[i],
                'class_id': int(classes[i]),
                'score': scores[i]
            }
            results.append(result)
    return results

def draw_boxes(image, results, labels):
    """Draw bounding boxes on the image."""
    draw = ImageDraw.Draw(image)
    
    # Try to use a better font, fall back to default if not available
    try:
        font = ImageFont.truetype("arial.ttf", 20)
    except:
        font = ImageFont.load_default()
    
    for result in results:
        ymin, xmin, ymax, xmax = result['bounding_box']
        xmin = int(xmin * image.width)
        xmax = int(xmax * image.width)
        ymin = int(ymin * image.height)
        ymax = int(ymax * image.height)
        
        # Draw bounding box
        draw.rectangle([(xmin, ymin), (xmax, ymax)], outline='red', width=3)
        
        # Draw label and confidence
        class_id = result['class_id']
        if class_id < len(labels):
            label = labels[class_id]
        else:
            label = f'Class {class_id}'
        
        confidence = result['score']
        text = f'{label}: {confidence:.2f}'
        
        # Calculate text position (above the box)
        text_y = max(0, ymin - 25)
        draw.text((xmin, text_y), text, fill='red', font=font)

def init():
    parser = argparse.ArgumentParser(description='TensorFlow Lite Object Detection')
    parser.add_argument('--model', default="ssd_mobilenet_v2_coco_quant_postprocess_edgetpu.tflite", help='Path to the .tflite model file')
    parser.add_argument('--labels', default="coco_labels.txt", help='Path to the labels file')
    parser.add_argument('--image', default="kite_and_cold.jpg", help='Path to the input image')
    parser.add_argument('--output', default='output.jpg', help='Path to save output image')
    parser.add_argument('--threshold', type=float, default=0.2, help='Detection threshold (0-1)')
    
    args = parser.parse_args()
    
    # Load the TFLite model
    print(f"Loading model from {args.model}...")
    interpreter = Interpreter(
    model_path = args.model,
    experimental_delegates=[load_delegate("libedgetpu.so.1", options={"device": ":0"})]
    )
    interpreter.allocate_tensors()
    
    # Get model details
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    # Print model info
    print(f"Input shape: {input_details[0]['shape']}")
    print(f"Input type: {input_details[0]['dtype']}")
    
    # Load labels
    labels = load_labels(args.labels)
    print(f"Loaded {len(labels)} labels")

def perform_inference(image, labels):
    # Load and preprocess image
    # get time it takes to preprocess image
    start_time = time()
    print(f"Loading image from {args.image}...")
    original_image = Image.open(args.image).convert('RGB')

    # Get input size from model
    input_shape = input_details[0]['shape']
    height, width = input_shape[1], input_shape[2]
    
    # Resize image to model input size
    resized_image = original_image.resize((width, height))
    input_data = np.expand_dims(resized_image, axis=0)
    
    preprocessing_time = time() - start_time
    print(f"Image preprocessing time: {preprocessing_time:.2f} seconds")
    
    print("Running inference...")
    # Run object detection
    start_time = time()
    results = detect_objects(interpreter, input_data, args.threshold)

    inference_time = time() - start_time
    print(f"Inference time: {inference_time:.2f} seconds")
    print(f"Found {len(results)} objects above threshold {args.threshold}")
    
    # Print results
    for i, result in enumerate(results):
        class_id = result['class_id']
        label = labels[class_id] if class_id < len(labels) else f'Class {class_id}'
        confidence = result['score']
        bbox = result['bounding_box']
        print(f"Object {i+1}: {label} ({confidence:.2f}) at {bbox}")
    
    # Draw bounding boxes on original image
    output_image = original_image.copy()
    draw_boxes(output_image, results, labels)
    
    # Save result
    output_image.save(args.output)
    print(f"Output saved to {args.output}")

if __name__ == '__main__':
    main()