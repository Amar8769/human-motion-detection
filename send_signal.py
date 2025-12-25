import requests

# Replace with your ESP32-CAM's actual IP
ESP_IP = "http://10.184.121.106"

def send_signal(human_detected: bool):
    """
    Sends a simple HTTP GET request to the ESP32-CAM.
    Example: http://10.184.121.106/?human=1
    """
    url = f"{ESP_IP}/?human={'1' if human_detected else '0'}"
    try:
        response = requests.get(url, timeout=5)
        if response.status_code == 200:
            print(f"✅ Signal sent successfully → {url}")
        else:
            print(f"⚠️ ESP responded with: {response.status_code}")
    except Exception as e:
        print("❌ Failed to reach ESP32:", e)
