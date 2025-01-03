import tflite_runtime.interpreter as tflite
import numpy as np
from PIL import Image
import sys

# Check if image path is provided
if len(sys.argv) != 2:
    print("Usage: python3 run_inference.py <image_path>")
    sys.exit(1)

# Load the image path from command-line argument
image_path = sys.argv[1]

# Load the TFLite model
model_path = "/home/csanz/models/mobilenet_v2.tflite"  # Adjust path if needed
interpreter = tflite.Interpreter(model_path=model_path)
interpreter.allocate_tensors()

# Get input and output details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Load labels
labels_path = "/home/csanz/models/imagenet_labels.txt"  # Adjust path if needed
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

