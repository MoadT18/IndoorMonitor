from fastapi import FastAPI
from pydantic import BaseModel
import sqlite3
from datetime import datetime

# Initialize FastAPI
app = FastAPI()

# Database file
DB_FILE = "sensor_data.db"

# Ensure the database table exists and has all needed columns
def init_db():
    with sqlite3.connect(DB_FILE) as conn:
        cursor = conn.cursor()
        # Create table if not exists
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS measurements (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT,
                co2 INTEGER,
                temperature REAL,
                humidity REAL,
                presence BOOLEAN DEFAULT 0
            )
        ''')
        # Check if 'presence' column exists (for upgrades)
        cursor.execute("PRAGMA table_info(measurements)")
        columns = [col[1] for col in cursor.fetchall()]
        if 'presence' not in columns:
            cursor.execute('ALTER TABLE measurements ADD COLUMN presence BOOLEAN DEFAULT 0')
        conn.commit()

# Initialize database on startup
init_db()

# Define the data model for incoming sensor data
class SensorData(BaseModel):
    timestamp: str
    co2: int
    temperature: float
    humidity: float
    presence: bool

# Define the data model for updating only presence
class PresenceUpdate(BaseModel):
    presence: bool

# Endpoint to receive and store sensor data
@app.post("/data")
def receive_data(data: SensorData):
    with sqlite3.connect(DB_FILE) as conn:
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO measurements (timestamp, co2, temperature, humidity, presence)
            VALUES (?, ?, ?, ?, ?)
        ''', (data.timestamp, data.co2, data.temperature, data.humidity, int(data.presence)))
        conn.commit()
    return {"message": "✅ Data received successfully"}

# Endpoint to retrieve the last sensor readings
@app.get("/data")
def get_data():
    with sqlite3.connect(DB_FILE) as conn:
        cursor = conn.cursor()
        cursor.execute('''
            SELECT * FROM measurements ORDER BY timestamp DESC
        ''')
        rows = cursor.fetchall()

    return [
        {
            "id": row[0],
            "timestamp": row[1],
            "co2": row[2],
            "temperature": row[3],
            "humidity": row[4],
            "presence": bool(row[5])
        }
        for row in rows
    ]

# Endpoint to delete a measurement by ID
@app.delete("/data/{item_id}")
def delete_data(item_id: int):
    with sqlite3.connect(DB_FILE) as conn:
        cursor = conn.cursor()
        cursor.execute('''
            DELETE FROM measurements WHERE id=?
        ''', (item_id,))
        conn.commit()
    return {"message": f"✅ Measurement with ID {item_id} deleted successfully"}

# ✨ NEW: Endpoint to update presence by ID
@app.put("/data/{item_id}")
def update_presence(item_id: int, update: PresenceUpdate):
    with sqlite3.connect(DB_FILE) as conn:
        cursor = conn.cursor()
        cursor.execute('''
            UPDATE measurements
            SET presence = ?
            WHERE id = ?
        ''', (int(update.presence), item_id))
        conn.commit()
    return {"message": f"✅ Presence for ID {item_id} updated to {update.presence}"}

