import subprocess
import requests
import time
import speech_recognition as sr
from elevenlabs import play
from elevenlabs.client import ElevenLabs
from speech_recognition import Microphone
from openai import OpenAI
from dateutil import parser
from flask import Flask, request
import threading
import os
import sys
import socket
import signal
import json
from datetime import datetime
import logging
from fastapi import FastAPI
from fastapi.responses import JSONResponse
import uvicorn
import json


is_speaking = threading.Event()



from pixels import Pixels

# Initialize the Pixels object
pixels = Pixels()

# Optional: Test the LEDs on startup
pixels.wakeup()
time.sleep(3)



# API Keys
# Dynamisch pad naar api_keys.json
base_dir = os.path.dirname(__file__)
key_file = os.path.join(base_dir, "api_keys.json")

with open(key_file, "r") as f:
    keys = json.load(f)

openai_api_key = keys["openai_api_key"]
elevenlabs_api_key = keys["elevenlabs_api_key"]

openai_client = OpenAI(api_key=openai_api_key)
elevenlabs_client = ElevenLabs(api_key=elevenlabs_api_key)

# Config
API_PORT = 8000
LOCAL_API_URL = f"http://pi.local:{API_PORT}/data"

# Speech Recognizer
recognizer = sr.Recognizer()
mic = Microphone()

# Conversation Memory
conversation_history = [{"role": "system", "content": "You are Luna, a friendly, creative, and emotional human assistant. Never mention you're an AI. Talk like a real human with genuine feelings. When the user mentions indoor air quality, temperature, humidity, CO₂, or air comfort, use provided climate data creatively in your response."}]

# Check if a port is in use
def is_port_in_use(port):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(('localhost', port)) == 0

# Find available port starting from base_port
def find_available_port(base_port):
    port = base_port
    while is_port_in_use(port):
        print(f"Port {port} is already in use, trying next port...")
        port += 1
    return port

# Kill process using a specific port
def kill_process_on_port(port):
    try:
        # Find process using the port
        result = subprocess.run(['lsof', '-i', f':{port}'], 
                              stdout=subprocess.PIPE, 
                              text=True)
        
        lines = result.stdout.strip().split('\n')
        if len(lines) > 1:  # Header + at least one process
            # Extract PID from the second line (first process)
            # Format is: COMMAND  PID  USER  ... 
            process_info = lines[1].split()
            if len(process_info) > 1:
                pid = process_info[1]
                print(f"Killing process {pid} using port {port}")
                os.kill(int(pid), signal.SIGTERM)
                time.sleep(1)  # Give process time to terminate
                return True
    except Exception as e:
        print(f"Error killing process on port {port}: {e}")
    
    return False


# Fetch latest air quality data
def get_latest_air_quality():
    try:
        print(f"Fetching data from {LOCAL_API_URL}")
        response = requests.get(LOCAL_API_URL, timeout=5)
        print(f"Response status: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            
            # If data is a list, get the first item
            if isinstance(data, list) and len(data) > 0:
                latest = data[0]
                print(f"Found array data, using first item: ID={latest.get('id')}, CO2={latest.get('co2')}")
                return latest
            else:
                print(f"Data is not an array, using as is")
                return data
        else:
            print(f"Error fetching data: HTTP {response.status_code}")
            return None
    except Exception as e:
        print(f"Error fetching air quality data: {e}")
        return None



# Helper for detecting climate-related questions
def is_climate_related(prompt):
    keywords = ["indoor air quality", "temperature", "humidity", "co2", "air", "climate", "room", "hot", "cold", "warm", "cool", "fresh"]
    return any(kw in prompt.lower() for kw in keywords)


def is_forecast_related(prompt):
    keywords = ["forecast", "voorspelling", "verwachting"]
    return any(kw in prompt.lower() for kw in keywords)


def is_weather_related(prompt: str) -> bool:
    prompt_lower = prompt.lower()
    return any(keyword in prompt_lower for keyword in [
        "weather"
    ])


def get_weather_forecast_for(prompt: str) -> str:
    import re
    import requests
    import logging
    from datetime import datetime, timedelta

    logging.info("🌤️ [WEER] Start weather forecast handler...")
    base_url = "https://api.open-meteo.com/v1/forecast"
    lat = 50.8503  # Brussels
    lon = 4.3517
    today = datetime.now().date()
    target_date = today

    # Detect relative date
    clean_prompt = prompt.lower()
    if "tomorrow" in clean_prompt:
        target_date = today + timedelta(days=1)
        logging.info(f"🌤️ [WEER] Detected 'tomorrow'. Target date: {target_date}")
    elif "yesterday" in clean_prompt:
        target_date = today - timedelta(days=1)
        logging.info(f"🌤️ [WEER] Detected 'yesterday'. Target date: {target_date}")
    elif "today" in clean_prompt:
        target_date = today
        logging.info(f"🌤️ [WEER] Detected 'today'. Target date: {target_date}")
    elif "day after tomorrow" in clean_prompt or "overmorrow" in clean_prompt:
        target_date = today + timedelta(days=2)
        logging.info(f"🌤️ [WEER] Detected 'day after tomorrow'. Target date: {target_date}")
    elif "day before yesterday" in clean_prompt:
        target_date = today - timedelta(days=2)
        logging.info(f"🌤️ [WEER] Detected 'day before yesterday'. Target date: {target_date}")
    elif "next week" in clean_prompt:
        target_date = today + timedelta(days=7)
        logging.info(f"🌤️ [WEER] Detected 'next week'. Target date: {target_date}")
    elif "last week" in clean_prompt:
        target_date = today - timedelta(days=7)
        logging.info(f"🌤️ [WEER] Detected 'last week'. Target date: {target_date}")
    else:
        # Clean prompt for better regex matching (remove punctuation)
        clean_prompt = re.sub(r'[?!.,;]', '', prompt.lower())
        
        # Detect specific date - supports multiple formats:
        # "may 25", "25 may", "may 25 2025", "25 may 2025", "25/05", "05/25/2025"
        date_patterns = [
            # Month day format: "may 25", "may 25 2025"
            r'(january|february|march|april|may|june|july|august|september|october|november|december|jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)\s+(\d{1,2})(?:\s+(\d{4}))?',
            # Day month format: "25 may", "25 may 2025"  
            r'(\d{1,2})\s+(january|february|march|april|may|june|july|august|september|october|november|december|jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)(?:\s+(\d{4}))?',
            # Numeric formats: "05/25", "25/05", "05/25/2025", "25/05/2025"
            r'(\d{1,2})[\/\-](\d{1,2})(?:[\/\-](\d{4}))?'
        ]
        
        day = month = year = None
        
        for pattern in date_patterns:
            match = re.search(pattern, clean_prompt)
            if match:
                logging.info(f"🌤️ [WEER] Regex match: {match.groups()}")
                groups = match.groups()
                
                if len(groups) == 3 and groups[0].isalpha():  # Month day format
                    month, day, year = groups
                elif len(groups) == 3 and groups[1].isalpha():  # Day month format  
                    day, month, year = groups
                else:  # Numeric format
                    # Assume MM/DD format for now, but could be enhanced
                    month, day, year = groups
                break
        
        if day and month:
            try:
                if not year:
                    year = str(today.year)
                
                # English month map only
                month_map = {
                    "january": 1, "february": 2, "march": 3, "april": 4, "may": 5, "june": 6,
                    "july": 7, "august": 8, "september": 9, "october": 10, "november": 11, "december": 12,
                    "jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5, "jun": 6,
                    "jul": 7, "aug": 8, "sep": 9, "oct": 10, "nov": 11, "dec": 12
                }
                
                month_num = int(month) if month.isdigit() else month_map.get(month.lower(), 0)
                
                if month_num == 0:
                    logging.warning(f"🌤️ [WEER] Unknown month: {month}")
                    return f"Sorry, I don't recognize the month '{month}'."
                
                target_date = datetime(int(year), month_num, int(day)).date()
                logging.info(f"🌤️ [WEER] Parsed target date: {target_date}")
                
            except ValueError as e:
                logging.warning(f"🌤️ [WEER] Invalid date values: {e}")
                return "Sorry, the date you provided is not valid."
            except Exception as e:
                logging.warning(f"🌤️ [WEER] Failed to parse date: {e}")
                # If parsing fails, use today's date
                target_date = today

    # Check API limit
    if target_date > today + timedelta(days=15):
        logging.warning(f"🌤️ [WEER] Target date {target_date} is too far in the future.")
        return "Sorry, ik kan geen weerdata ophalen voor zo ver in de toekomst. Vraag iets binnen de komende 15 dagen 💨."

    start_date = end_date = target_date.isoformat()

    # Fix: daily parameter should be a comma-separated string, not a list
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start_date,
        "end_date": end_date,
        "daily": "temperature_2m_max,temperature_2m_min,precipitation_sum,wind_speed_10m_max,cloud_cover_mean",
        "timezone": "Europe/Brussels"
    }

    logging.info(f"🌤️ [WEER] Requesting Open-Meteo with params: {params}")

    try:
        response = requests.get(base_url, params=params, timeout=5)
        logging.info(f"🌤️ [WEER] API status code: {response.status_code}")

        if response.status_code != 200:
            logging.error(f"🌤️ [WEER] API returned error status: {response.status_code}")
            logging.error(f"🌤️ [WEER] API response: {response.text}")
            return "Sorry, ik kon het weer niet ophalen."

        data = response.json()
        logging.debug(f"🌤️ [WEER] API raw JSON: {data}")

        # Check for errors in the response
        if "error" in data:
            logging.error(f"🌤️ [WEER] API returned error: {data['error']}")
            return "Sorry, ik kon het weer niet ophalen."

        if "daily" not in data:
            logging.error("🌤️ [WEER] 'daily' field missing in Open-Meteo response.")
            logging.error(f"🌤️ [WEER] Full response: {data}")
            return "Sorry, ik kon het weer niet ophalen."

        day_data = data["daily"]
        i = 0

        # Add safety checks for data availability
        if not day_data or len(day_data.get('temperature_2m_max', [])) == 0:
            logging.error("🌤️ [WEER] No weather data available for the requested date.")
            return "Sorry, er zijn geen weergegevens beschikbaar voor deze datum."

        return (
            f"📅 Weather forecast for {target_date.strftime('%B %d, %Y')} in Brussels:\n"
            f"- 🌡️ Max temp: {day_data['temperature_2m_max'][i]}°C\n"
            f"- 🌡️ Min temp: {day_data['temperature_2m_min'][i]}°C\n"
            f"- ☁️ Cloud cover: {day_data['cloud_cover_mean'][i]}%\n"
            f"- 💨 Wind: {day_data['wind_speed_10m_max'][i]} km/h\n"
            f"- 🌧️ Precipitation: {day_data['precipitation_sum'][i]} mm\n"
        )

    except requests.exceptions.Timeout:
        logging.error("🌤️ [WEER] Request timed out")
        return "Sorry, de weerservice reageert niet. Probeer het later opnieuw."
    except requests.exceptions.RequestException as e:
        logging.error(f"🌤️ [WEER] Request error: {e}")
        return "Sorry, ik kon het weer niet ophalen vanwege een netwerkfout."
    except Exception as e:
        logging.error(f"🌤️ [WEER] Error fetching weather data: {e}")
        return "Sorry, ik kon het weer niet ophalen."



logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

from datetime import datetime
import pytz
import requests
import logging

def get_chatgpt_response(prompt):
    """
    Fetch data from /forecast or /data depending on prompt content, and generate a personalized response.
    """
    try:
        pixels.think()
        logging.info("Analyzing prompt...")
        if is_weather_related(prompt):
            logging.info("Prompt is weather-related. Fetching Open-Meteo data for Brussels.")
            weather_response = get_weather_forecast_for(prompt)
            return weather_response

        # ----- Forecast prompt -----
        if is_forecast_related(prompt):
            logging.info("Prompt is forecast-related. Fetching forecast data.")
            try:
                forecast_resp = requests.get("http://pi.local:8001/forecast", timeout=5)
                if forecast_resp.status_code == 200:
                    forecast_data = forecast_resp.json()

                    forecast_table = "\n".join(
                        f"{entry['ds'][:10]} → Predicted CO2: {round(entry['pred_co2'])} ppm, "
                        f"Predicted Temp: {round(entry['pred_temp'], 1)}°C, "
                        f"Outside max: {entry['outside_temp_max']}°C, "
                        f"Solar: {entry['solar_radiation']}, "
                        f"Rain: {entry['precipitation']} mm, "
                        f"Wind: {entry['wind']} km/h, "
                        f"Clouds: {entry['cloudcover']}%, "
                        f"Advice: {entry.get('advies', '—')}"
                        for entry in forecast_data
                    )

                    extended_prompt = (
                        "The following is a 7-day forecast for indoor climate (CO₂ and temperature) and outdoor weather conditions "
                        "based on predicted sensor data and external weather input from Open-Meteo. The data includes:\n"
                        "- Predicted indoor CO₂ levels (`pred_co2`)\n"
                        "- Predicted indoor temperature (`pred_temp`)\n"
                        "- Outside weather data: `outside_temp_max`, `solar_radiation`, `precipitation`, `cloudcover`, `wind`\n\n"
                        "Your task is to provide a personalized daily forecast for each of the 7 days **in the future**.\n"
                        "Always use this format for each day:\n"
                        "- **Date**\n"
                        "- **Forecast**: Clearly state the predicted CO₂ and indoor temperature, followed by all relevant weather data (outside max temp, solar, wind, rain, etc.)\n"
                        "- **Advice**: Give specific ventilation and heating recommendations **as if the day is still coming**.\n"
                        "Be realistic and human: suggest when to open or close windows, reduce heating, or take no action. Always explain *why* based on the numbers.\n\n"
                        f"Forecast data:\n{forecast_table}\n\n"
                        f"User question: {prompt}\n"
                        "Respond with a full 7-day forecast breakdown using the format above. Write like a helpful assistant giving future-oriented advice."

                    )
                else:
                    logging.warning("Could not retrieve forecast data. Using fallback prompt.")
                    extended_prompt = f"The user asked: {prompt}\nRespond as Luna, a helpful and emotional human assistant."
            except Exception as e:
                logging.error(f"Error while fetching forecast data: {e}")
                extended_prompt = f"The user asked: {prompt}\nRespond as Luna, a helpful and emotional human assistant."

        # ----- Climate data prompt (/data) -----
        elif is_climate_related(prompt):
            logging.info("Prompt is climate-related. Fetching data from /data API.")
            try:
                response = requests.get("http://pi.local:8000/data", timeout=5)
                filtered_data = []

                if response.status_code == 200:
                    data = response.json()
                    local_tz = pytz.timezone("Europe/Brussels")
                    today = datetime.now(local_tz).strftime("%Y-%m-%d")
                    hourly_data = {}

                    for entry in data:
                        timestamp = datetime.fromisoformat(entry["timestamp"])
                        timestamp_local = timestamp.astimezone(local_tz)

                        if timestamp_local.strftime("%Y-%m-%d") != today:
                            continue

                        hour_key = timestamp_local.strftime("%Y-%m-%d %H:00")
                        if hour_key not in hourly_data or timestamp_local > datetime.fromisoformat(hourly_data[hour_key]["timestamp"]).astimezone(local_tz):
                            hourly_data[hour_key] = entry

                    filtered_data = list(hourly_data.values())
                    logging.info(f"Filtered to {len(filtered_data)} hourly entries for {today}")

                if filtered_data:
                    sorted_data = sorted(
                        filtered_data,
                        key=lambda x: datetime.fromisoformat(x["timestamp"]).astimezone(local_tz)
                    )

                    highest_entry = max(filtered_data, key=lambda x: x["co2"])
                    lowest_entry = min(filtered_data, key=lambda x: x["co2"])

                    climate_table = "\n".join(
                        f"Timestamp: {datetime.fromisoformat(entry['timestamp']).strftime('%Y-%m-%d %H:%M:%S')}, "
                        f"CO₂: {entry['co2']} ppm, "
                        f"Temperature: {entry['temperature']}°C, "
                        f"Humidity: {entry['humidity']}%, "
                        f"Presence: {entry['presence']}"
                        for entry in sorted_data
                    )

                    extended_prompt = (
                        "Here is today's climate data, sorted by time:\n"
                        f"{climate_table}\n"
                        "\nImportant data:\n"
                        f"Highest CO₂ level: {highest_entry['co2']} ppm at {highest_entry['timestamp']}\n"
                        f"Lowest CO₂ level: {lowest_entry['co2']} ppm at {lowest_entry['timestamp']}\n"
                        f"User question: {prompt}\n"
                        "Please use this data to answer the user's question in a concise and helpful way."
                    )
                else:
                    extended_prompt = f"The user asked: {prompt}\nRespond as Luna, a helpful and emotional human assistant."
            except Exception as e:
                logging.error(f"Error fetching /data: {e}")
                extended_prompt = f"The user asked: {prompt}\nRespond as Luna, a helpful and emotional human assistant."

        # ----- General prompt -----
        else:
            logging.info("Prompt is not forecast- or climate-related.")
            extended_prompt = f"The user asked: {prompt}\nRespond as Luna, a helpful and emotional human assistant."

        # Trim conversation history
        trimmed_history = trim_conversation_history(conversation_history)
        trimmed_history.append({"role": "user", "content": extended_prompt})
        logging.info(f"🧠 Final prompt to OpenAI:\n{extended_prompt}")

        # Call OpenAI
        ai_resp = openai_client.chat.completions.create(
            model="gpt-4-turbo",
            messages=trimmed_history,
            max_tokens=800,
            temperature=0.8 #adds creativity
        )
        message = ai_resp.choices[0].message.content.strip()
        logging.info(f"ChatGPT response: {message}")
        pixels.off()
        conversation_history.append({"role": "assistant", "content": message})
        return message

    except Exception as e:
        logging.error(f"Unexpected top-level error: {e}")
        return "Sorry, something went wrong."

MAX_TOKENS = 16385  # Global max tokens for GPT-4
RESERVED_TOKENS = 1000  # Leave space for current prompt

def trim_conversation_history(history, max_tokens=MAX_TOKENS, buffer=RESERVED_TOKENS):
    """
    Trim the conversation history to fit within the maximum token limit.
    """
    total_tokens = 0
    trimmed_history = []

    # Start from the most recent messages and work backwards
    for message in reversed(history):
        message_tokens = len(message["content"].split())
        if total_tokens + message_tokens + buffer > max_tokens:
            break
        trimmed_history.insert(0, message)
        total_tokens += message_tokens

    logging.info(f"Trimmed history to {len(trimmed_history)} messages with {total_tokens} tokens.")
    return trimmed_history


# Generate speech with ElevenLabs
def speak_text(text):
    is_speaking.set()
    try:
        audio = elevenlabs_client.generate(
            text=text,
            voice="EXAVITQu4vr4xnSDxMaL",
            model="eleven_turbo_v2",
            stream=False,
            optimize_streaming_latency=1
        )
        pixels.speak()
        play(audio)
    except Exception as e:
        logging.error(f"Audio error: {e}")
        print("Audio error:", e)
        # Fallback to print only if audio fails
        print(f"Luna (text only): {text}")
    finally:
        is_speaking.clear()
        pixels.off()

from flask import Flask, request, jsonify
import threading
# Flask App for receiving prompts from Windows Forms
app = Flask(__name__)

@app.route('/prompt', methods=['POST'])
def receive_prompt():
    data = request.json
    prompt = data.get('prompt')
    if not prompt:
        return jsonify(status="error", response="No prompt received."), 400

    # 1) Haal de GPT-response op
    response_text = get_chatgpt_response(prompt)

    # 2) Start de TTS in een daemon-thread: geen blokkade meer op de HTTP-request
    threading.Thread(
        target=lambda: speak_text(response_text),
        daemon=True
    ).start()

    # 3) Stuur de client meteen de textuele response
    return jsonify(status="success", response=response_text)

def run_flask():
    app.run(host='0.0.0.0', port=5000, debug=False)


def start_fastapi_server():
    try:
        print(f"Start de geïntegreerde FastAPI-server op poort {API_PORT}...")
        threading.Thread(
            target=uvicorn.run,
            args=(fastapi_app,),
            kwargs={"host": "0.0.0.0", "port": API_PORT, "log_level": "info"},
            daemon=True
        ).start()
        time.sleep(2)
    except Exception as e:
        print(f"Kon FastAPI-server niet starten: {e}")


'''

def voice_recognition_loop():
    # Voice loop
    while True:
        with mic as source:
            recognizer.adjust_for_ambient_noise(source, duration=1)
           print("I'm listening now.")
            print("Luna luistert...")
            try:
                audio = recognizer.listen(source, timeout=5, phrase_time_limit=10)
                command = recognizer.recognize_google(audio)
                print(f"Jij zei: {command}")
                response_text = get_chatgpt_response(command)
                print(f"Luna: {response_text}")
                speak_text(response_text)
            except sr.WaitTimeoutError:
                print("Ik hoor niets, probeer opnieuw.")
            except sr.UnknownValueError:
                print("Sorry, dat verstond ik niet.")
            except sr.RequestError as e:
                print("Speech recognition error, probeer het opnieuw.")


'''

import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


def voice_conversation_loop():
    """
    1) Listen for 'hey luna'
    2) On wake-word: speak 'ja' and enter conversation mode
    3) In conversation mode: listen repeatedly (3s timeout) for user speech
       - On each utterance: process & respond, then continue listening
       - On timeout/unknown: return to wake-word detection
    """
    while True:
        # 1) Wake-word detection
        while is_speaking.is_set():
            time.sleep(0.1)
        with mic as source:
            recognizer.adjust_for_ambient_noise(source, duration=0.5)
            logging.info("Waiting for wake-word 'hey luna'…")
            pixels.listen()
            try:
                audio = recognizer.listen(source, timeout=None)
                phrase = recognizer.recognize_google(audio).lower()
            except (sr.UnknownValueError, sr.WaitTimeoutError):
                continue
            except sr.RequestError as e:
                logging.error(f"Speech API error: {e}")
                time.sleep(1)
                continue

        if "hey luna" not in phrase:
            continue

        # 2) Wake-word heard: respond and enter conversation mode
        speak_text("I'm listening!")

        # 3) Conversation loop
        while True:
            # Don’t start listening until Luna’s done speaking
            while is_speaking.is_set():
                time.sleep(0.1)

            with mic as source_cmd:
                recognizer.adjust_for_ambient_noise(source_cmd, duration=0.5)
                logging.info("Listening for your command (3s)…")
                pixels.listen()
                try:
                    cmd_audio = recognizer.listen(
                        source_cmd,
                        timeout=3,
                        phrase_time_limit=10
                    )
                    command = recognizer.recognize_google(cmd_audio)
                    logging.info(f"User said: {command}")
                except (sr.WaitTimeoutError, sr.UnknownValueError):
                    logging.info("Silence → back to wake-word.")
                    break
                except sr.RequestError as e:
                    logging.error(f"Speech API error: {e}")
                    speak_text("Sorry, er ging iets mis.")
                    break


            # Let ChatGPT handle the query, including specific, historical, average, etc.
            response = get_chatgpt_response(command)
            speak_text(response)



def speak_message(text):
    speak_text(text)


def monitor_co2_threshold(interval_seconds=5, threshold=1000):
    """
    Elke `interval_seconds` seconden ophalen van http://pi.local:8000/data,
    vergelijken op ID en bij CO₂ ≥ threshold waarschuwen.
    """
    processed_ids = set()
    logging.info(f"🟢 Start CO₂-monitoring elke {interval_seconds}s (drempel ≥ {threshold} ppm)")

    while True:
        try:
            # Forceren van vaste URL naar poort 8000
            response = requests.get("http://pi.local:8000/data", timeout=2)
            response.raise_for_status()
            data = response.json()

            if isinstance(data, list) and data:
                latest = max(data, key=lambda x: x.get("id", 0))
            else:
                latest = data

            data_id = latest.get("id")
            co2     = latest.get("co2", 0)
            ts_raw  = latest.get("timestamp")
            timestamp = ts_raw if ts_raw else "onbekend"

            if data_id is None:
                logging.warning(f"⚠️ Ontvangen meting zonder ID: {latest}")
            else:
                if data_id not in processed_ids:
                    processed_ids.add(data_id)
                    logging.info(f"[{timestamp}] 📡 Nieuwe data ID={data_id} – CO₂={co2} ppm")

                    if co2 >= threshold:
                        alert = (f"Warning: the CO₂-level is now at {co2} ppm. "
                                 "Please ventilate the room!")
                        logging.warning(f"[{timestamp}] {alert}")
                        speak_message(alert)
                else:
                    logging.debug(f"[{timestamp}] ⏳ Geen nieuwe data (ID {data_id})")

        except Exception as e:
            logging.error(f"❌ Fout tijdens CO₂-monitoring: {e}")

        time.sleep(interval_seconds)

fastapi_app = FastAPI()

@fastapi_app.get("/forecast")
def get_forecast():
    try:
        base_dir = os.path.dirname(__file__)
        file_path = os.path.join(base_dir, "model", "forecast_result.json")
        with open(file_path, "r") as f:
            data = json.load(f)
        return data
    except Exception as e:
        logging.error(f"Error reading forecast_result.json: {e}")
        return JSONResponse(content={"error": "Could not read forecast data."}, status_code=500)

@fastapi_app.get("/forecast/history")
def get_forecast_history(limit_weeks: int = 4):
    try:
        base_dir = os.path.dirname(__file__)
        history_path = os.path.join(base_dir, "model", "forecast_history.json")
        if not os.path.exists(history_path):
        	return {"message": "Geen historiek beschikbaar."}
        with open(history_path, "r") as f:
            history = json.load(f)
        return {"history": history[-limit_weeks:]}
    except Exception as e:
        logging.error(f"Error reading forecast_history.json: {e}")
        return JSONResponse(content={"error": "Could not read forecast history."}, status_code=500)


def main():
    global API_PORT
    global LOCAL_API_URL

    # Check if port is in use and handle it
    if is_port_in_use(API_PORT):
        print(f"Port {API_PORT} is already in use.")
        if kill_process_on_port(API_PORT):
            print(f"Successfully freed port {API_PORT}")
        else:
            new_port = find_available_port(API_PORT + 1)
            print(f"Using alternative port: {new_port}")
            API_PORT = new_port
            LOCAL_API_URL = f"http://pi.local:{API_PORT}/data"

    try:
        start_fastapi_server()
    except Exception as e:
        print(f"Failed to start FastAPI server: {e}")
        return

    # Start de Flask-server (voor prompts via HTTP)
    flask_thread = threading.Thread(target=run_flask, daemon=True)
    flask_thread.start()

    # Start CO₂ monitoring thread
    co2_monitor_thread = threading.Thread(target=monitor_co2_threshold, daemon=True)
    co2_monitor_thread.start()

    # Welkomstbericht en logging
    welcome_message = "Hey there! Luna here, ready to help."
    speak_text(welcome_message)
    print(f"""
    --------------------------------------
    Luna is klaar om te luisteren...
    FastAPI server running on port {API_PORT}
    Flask server running on port 5000
    Voice recognition active
    CO₂ monitoring active
    System is ready 🚀
    --------------------------------------
    """)

    # Start voice recognition loop (blokkeert verder)
    voice_conversation_loop()


if __name__ == "__main__":
    main()

