import os
import sqlite3
from io import BytesIO
from typing import Optional
import pandas as pd
import plotly.express as px
import streamlit as st
from google import genai
from google.genai import types
from PIL import ExifTags, Image
import pillow_heif
from pydantic import BaseModel, Field

# Enable support for phone camera formats (including Apple HEIC)
pillow_heif.register_heif_opener()

# 1. Page Configuration
st.set_page_config(page_title="Spin Lab", page_icon="🚴", layout="centered")
st.title("🚴 Spin Performance Lab")

# 2. API Key Management
# Checks Streamlit Secrets first (for cloud deployment), then local environment, then fallback to sidebar
api_key = None
if hasattr(st, "secrets") and "GEMINI_API_KEY" in st.secrets:
    api_key = st.secrets["GEMINI_API_KEY"]
elif os.getenv("GEMINI_API_KEY"):
    api_key = os.getenv("GEMINI_API_KEY")
else:
    api_key = st.sidebar.text_input("Enter Gemini API Key", type="password")

client = genai.Client(api_key=api_key) if api_key else None

# 3. Database Setup (SQLite)
conn = sqlite3.connect("workouts.db", check_same_thread=False)
cursor = conn.cursor()
cursor.execute("""
CREATE TABLE IF NOT EXISTS workouts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    workout_date TEXT,
    duration_min REAL,
    avg_watts REAL,
    avg_cadence REAL,
    total_kj REAL,
    calories REAL,
    distance_miles REAL
)
""")
conn.commit()

# 4. Schema Definition for Structured Vision Extraction
class SpinMetrics(BaseModel):
    duration_min: float = Field(description="Workout duration in minutes")
    avg_watts: Optional[float] = Field(default=None, description="Average power output in watts")
    avg_cadence: Optional[float] = Field(default=None, description="Average cadence in RPM")
    total_kj: Optional[float] = Field(default=None, description="Total work output in kiloJoules (kJ)")
    calories: Optional[float] = Field(default=None, description="Total estimated calories burned")
    distance_miles: Optional[float] = Field(default=None, description="Total distance in miles")

def extract_photo_timestamp(raw_bytes: bytes) -> str:
    """Reads capture timestamp from image EXIF metadata; falls back to current time."""
    try:
        image = Image.open(BytesIO(raw_bytes))
        exif = image._getexif()
        if exif:
            for tag_id, val in exif.items():
                if ExifTags.TAGS.get(tag_id) == "DateTimeOriginal":
                    parts = str(val).split(" ")
                    date_part = parts[0].replace(":", "-")
                    return f"{date_part} {parts[1]}"
    except Exception:
        pass
    return pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")

# 5. Mobile Photo Upload Section
st.subheader("📤 Upload Console Photos")
uploaded_files = st.file_uploader(
    "Select 1 or more photos directly from your camera roll",
    type=["jpg", "jpeg", "png", "heic"],
    accept_multiple_files=True
)

if uploaded_files and st.button("Process & Log Rides", use_container_width=True):
    if not client:
        st.error("Please provide a Gemini API Key to continue.")
    else:
        with st.spinner(f"Analyzing {len(uploaded_files)} ride photo(s)..."):
            for file in uploaded_files:
                raw_bytes = file.read()
                captured_date = extract_photo_timestamp(raw_bytes)

                mime_type = file.type if file.type else "image/jpeg"
                prompt = (
                    "Extract the final spin workout metrics visible on this screen. "
                    "Extract duration, average watts, average cadence, total kJ, calories, and distance. "
                    "If a value is not visible or obscured, leave it as null."
                )

                try:
                    response = client.models.generate_content(
                        model="gemini-2.5-flash",
                        contents=[
                            types.Part.from_bytes(data=raw_bytes, mime_type=mime_type),
                            prompt
                        ],
                        config=types.GenerateContentConfig(
                            response_mime_type="application/json",
                            response_schema=SpinMetrics,
                        )
