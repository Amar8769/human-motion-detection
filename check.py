import tensorflow as tf
import numpy as np
from PIL import Image
import os

# ===== Load TFLite model =====
MODEL_PATH = r"human_detection_fixed.tflite"
interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

print("✅ Model loaded:", MODEL_PATH)

# ===== Prediction function =====
def predict_image(image_path):
    img = Image.open(image_path).convert("RGB")
    img = img.resize((128,128))  # adjust size to your model
    img = np.array(img, dtype=np.float32) / 255.0
    img = np.expand_dims(img, axis=0)

    interpreter.set_tensor(input_details[0]['index'], img)
    interpreter.invoke()
    result = interpreter.get_tensor(output_details[0]['index'])[0][0]

    return result

# ===== Test folder of images =====
test_folder = r"C:\Users\thvai\OneDrive\Desktop\Iotproject\test_images"

for fname in os.listdir(test_folder):
    if fname.lower().endswith((".jpg", ".jpeg", ".png")):
        path = os.path.join(test_folder, fname)
        pred = predict_image(path)
        label = "HUMAN" if pred < 0.3 else "NO HUMAN"  # adjust threshold if needed
        print(f"{fname}: {pred:.3f} → {label}")
