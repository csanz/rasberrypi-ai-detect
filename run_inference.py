import tflite_runtime.interpreter as tflite
import numpy as np
from PIL import Image
import sys
import os
from pathlib import Path

# Check if image path is provided
if len(sys.argv) != 2:
    print("Usage: python3 run_inference.py <image_path>")
    sys.exit(1)

# Define paths relative to current directory
MODEL_DIR = "./models"
model_path = os.path.join(MODEL_DIR, "mobilenet_v2.tflite")
labels_path = os.path.join(MODEL_DIR, "mobilenet_v2.txt")

# Load the image path from command-line argument
image_path = sys.argv[1]

# Verify files exist
if not os.path.exists(model_path):
    print(f"Error: Model file not found at {model_path}")
    sys.exit(1)
if not os.path.exists(labels_path):
    print(f"Error: Labels file not found at {labels_path}")
    sys.exit(1)

# Load the TFLite model
interpreter = tflite.Interpreter(model_path=model_path)
interpreter.allocate_tensors()

# Get input and output details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Load labels
with open(labels_path, "r") as f:
    labels = [line.strip() for line in f.readlines()]

# Preprocess the image
try:
    image = Image.open(image_path).resize((224, 224))
    input_data = np.expand_dims(np.array(image, dtype=np.float32) / 255.0, axis=0)
except Exception as e:
    print(f"Error processing the image: {e}")
    sys.exit(1)

# Perform inference
interpreter.set_tensor(input_details[0]['index'], input_data)
interpreter.invoke()

# Get results
output_data = interpreter.get_tensor(output_details[0]['index'])

# Get the index of the highest probability
top_prediction = np.argmax(output_data)
confidence = output_data[0][top_prediction]

# Map to the corresponding label
predicted_label = labels[top_prediction]
print(f"Predicted Label: {predicted_label} (Confidence: {confidence:.2f})")

