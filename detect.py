from flask import Flask, request
import requests

# ===== Roboflow API Configuration =====
ROBOFLOW_API_KEY = "keyPlaceholder"
MODEL_NAME = "person-qrleq"  # short model name
VERSION = "1"
ROBOFLOW_URL = f"https://detect.roboflow.com/{MODEL_NAME}/{VERSION}?api_key={ROBOFLOW_API_KEY}"

# ===== Telegram Setup =====
TELEGRAM_TOKEN = "tokenPlaceholder"
TELEGRAM_CHAT_ID = "idPlaceholder"

# Optional confidence threshold
CONFIDENCE_THRESHOLD = 0.3

def send_telegram_image(image_bytes, caption="Human detected!"):
    """Send an image + caption to Telegram"""
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendPhoto"
    files = {"photo": ("image.jpg", image_bytes)}
    data = {"chat_id": TELEGRAM_CHAT_ID, "caption": caption}
    try:
        r = requests.post(url, files=files, data=data)
        if r.status_code != 200:
            print("⚠️ Telegram response:", r.text)
        else:
            print("✅ Telegram image sent successfully.")
    except Exception as e:
        print("❌ Telegram send failed:", e)

def predict_image(image_bytes):
    """Send image to Roboflow and detect human (class_id == 0 only)"""
    try:
        response = requests.post(
            ROBOFLOW_URL,
            files={"file": ("image.jpg", image_bytes, "image/jpeg")},
            timeout=10
        )
        if response.status_code != 200:
            print(f"❌ Roboflow HTTP {response.status_code}")
            print("Response body:", response.text)
            return False

        data = response.json()
        predictions = data.get("predictions", [])

        if not predictions:
            return False  # no objects detected → no human

        # Only consider class_id 0 as human
        for pred in predictions:
            if pred.get("class_id") == 0 and pred.get("confidence", 0) >= CONFIDENCE_THRESHOLD:
                print(f"✅ Human detected with confidence {pred.get('confidence')}")
                return True

        # No human found
        return False

    except requests.exceptions.RequestException as e:
        print("❌ Roboflow request failed:", e)
        return False

# ===== Flask App =====
app = Flask(__name__)

@app.route('/')
def home():
    return "🚀 Flask server running!"

@app.route('/upload', methods=['POST'])
def upload_image():
    print("📸 Image received from ESP32")

    image_bytes = request.data
    if not image_bytes:
        return "❌ No image data", 400

    # Check for human
    human_detected = predict_image(image_bytes)

    if human_detected:
        print("👤 Human detected! Sending image and triggering buzzer...")
        send_telegram_image(image_bytes, caption="Human detected!")
        # Here you would also trigger your ESP32 buzzer
        return "HUMAN", 200
    else:
        print("✅ No human detected. No alert triggered.")
        return "NO HUMAN", 200

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)
