import time
import board
import adafruit_scd4x
import requests
import json
from datetime import datetime
import pytz  # Ensure pytz is installed: pip install pytz

# API Endpoint
API_URL = "http://pi.local:8000/data"

# Presence file path
presence_file = '/home/moadt/indoor_monitor/presence.txt'

# Set Belgium timezone
belgium_tz = pytz.timezone("Europe/Brussels")

# Initialize I2C for SCD4x
i2c = board.I2C()
scd4x = adafruit_scd4x.SCD4X(i2c)

print("Starting SCD41 CO₂ measurement...")
scd4x.start_periodic_measurement()
time.sleep(5)  # Allow sensor to collect data

# Default presence
presence_detected = False

# Read presence from file
try:
    with open(presence_file, 'r') as f:
        presence_value = f.read().strip()
        if presence_value == "ONN":
            presence_detected = True
        else:
            presence_detected = False
except Exception as e:
    print(f"🚨 Failed to read presence file: {e}")
    presence_detected = False  # fallback if error

# If data ready, send to API
if scd4x.data_ready:
    # Get current Belgium time
    timestamp = datetime.now(belgium_tz).isoformat()

    data = {
        "timestamp": timestamp,
        "co2": scd4x.CO2,
        "temperature": round(scd4x.temperature, 2),
        "humidity": round(scd4x.relative_humidity, 2),
        "presence": presence_detected
    }

    print(f"Sending data: {data}")

    try:
        response = requests.post(API_URL, json=data, headers={"Content-Type": "application/json"})
        if response.status_code == 200:
            print("✅ Data successfully sent to API.")
        else:
            print(f"❌ Error: Received status code {response.status_code} - {response.text}")
    except requests.exceptions.RequestException as e:
        print(f"🚨 Failed to send data: {e}")
else:
    print("⚠️ Sensor data not ready. Try again in a few seconds.")

