import cv2
import numpy as np
import tensorflow as tf
from send_signal import send_signal

# ========== CONFIG ==========
MODEL_PATH = "human_detection_fixed.tflite"
ESP_STREAM_URL = "http://10.184.121.106:81/stream"   # typical ESP32-CAM stream URL
CONFIDENCE_THRESHOLD = 0.7
FRAME_INTERVAL = 10   # run detection every 10 frames to save CPU
# ============================

# Load TFLite model
interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

input_shape = input_details[0]['shape'][1:3]
print(f"📸 Model input shape: {input_shape}")

# Open video stream
cap = cv2.VideoCapture(ESP_STREAM_URL)
if not cap.isOpened():
    raise RuntimeError("❌ Failed to open ESP32-CAM stream. Check IP and stream URL.")

frame_count = 0
print("🚀 Live detection started... Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("⚠️ Stream read failed.")
        continue

    frame_count += 1
    # Show stream in window
    cv2.imshow("ESP32-CAM Live", frame)

    # Process every few frames (for speed)
    if frame_count % FRAME_INTERVAL == 0:
        # Resize and preprocess
        resized = cv2.resize(frame, tuple(input_shape))
        input_data = np.expand_dims(resized.astype(np.float32) / 255.0, axis=0)

        # Run inference
        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()
        output_data = interpreter.get_tensor(output_details[0]['index'])
        confidence = float(output_data[0][0])

        print(f"🔍 Confidence: {confidence:.3f}")

        # Send signal
        if confidence > CONFIDENCE_THRESHOLD:
            print("🚨 Human detected!")
            send_signal(True)
        else:
            send_signal(False)

    # Exit key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
