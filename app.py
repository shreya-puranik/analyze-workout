import os
import sqlite3
from io import BytesIO
from typing import Optional, List
import pandas as pd
import plotly.express as px
import streamlit as st
from google import genai
from google.genai import types
from PIL import ExifTags, Image
import pillow_heif
from pydantic import BaseModel, Field

# Enable camera and phone image formats (including HEIC)
pillow_heif.register_heif_opener()

# 1. Page Configuration
st.set_page_config(
    page_title="Spin Lab", 
    page_icon="🚴", 
    layout="centered",
    initial_sidebar_state="collapsed"
)

# 2. Database Initialization
conn = sqlite3.connect("workouts.db", check_same_thread=False)
cursor = conn.cursor()
cursor.execute("""
CREATE TABLE IF NOT EXISTS workouts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    workout_date TEXT UNIQUE,
    duration_min REAL,
    avg_watts REAL,
    avg_cadence REAL,
    total_kj REAL,
    calories REAL,
    distance_miles REAL
)
""")
conn.commit()

# 3. Gemini API Client Initialization
api_key = None
if hasattr(st, "secrets") and "GEMINI_API_KEY" in st.secrets:
    api_key = st.secrets["GEMINI_API_KEY"]
elif os.getenv("GEMINI_API_KEY"):
    api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    api_key = st.sidebar.text_input("Gemini API Key", type="password")

client = genai.Client(api_key=api_key) if api_key else None

# 4. Schema Definition
class SpinMetrics(BaseModel):
    is_valid_spin_screen: bool = Field(description="True if the photo depicts a spin bike or exercise console summary, False otherwise")
    duration_min: Optional[float] = Field(default=None, description="Workout duration in minutes")
    avg_watts: Optional[float] = Field(default=None, description="Average power output in watts")
    avg_cadence: Optional[float] = Field(default=None, description="Average cadence in RPM")
    total_kj: Optional[float] = Field(default=None, description="Total work in kiloJoules (kJ)")
    calories: Optional[float] = Field(default=None, description="Total estimated calories burned")
    distance_miles: Optional[float] = Field(default=None, description="Total distance in miles")
    rejection_reason: Optional[str] = Field(default=None, description="Reason if not a valid workout summary screen")

def extract_photo_timestamp(raw_bytes: bytes) -> str:
    """Extracts original camera timestamp from EXIF metadata; falls back to current time."""
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

# 5. Application UI Header
st.title("🚴 Spin Performance Lab")

# Section: Ingestion (Tabs for Camera vs Upload)
st.subheader("📥 Add Workout")
tab_upload, tab_camera = st.tabs(["📁 Upload from Gallery", "📸 Snap Photo"])

files_to_process = []

with tab_upload:
    uploaded = st.file_uploader(
        "Select console photos",
        type=["jpg", "jpeg", "png", "heic", "webp"],
        accept_multiple_files=True,
        key="file_uploader"
    )
    if uploaded:
        files_to_process.extend(uploaded)

with tab_camera:
    camera_pic = st.camera_input("Take a photo of your bike console", key="camera_input")
    if camera_pic:
        files_to_process.append(camera_pic)

# Processing Workflow
if files_to_process:
    st.info(f"Selected {len(files_to_process)} image(s). Ready to analyze.")
    
    if st.button("🚀 Process & Save Workouts", use_container_width=True, type="primary"):
        if not client:
            st.error("Missing Gemini API Key. Please provide one to proceed.")
        else:
            success_count = 0
            failed_count = 0
            
            with st.status("Analyzing images with Gemini...", expanded=True) as status:
                for idx, item in enumerate(files_to_process):
                    st.write(f"Processing photo {idx + 1} of {len(files_to_process)}...")
                    raw_bytes = item.getvalue()
                    captured_date = extract_photo_timestamp(raw_bytes)
                    
                    prompt = (
                        "Inspect this image. Determine if it is a spin/stationary bike summary screen. "
                        "If yes, extract duration_min, avg_watts, avg_cadence, total_kj, calories, and distance_miles. "
                        "If it is not a console, or metrics are completely unreadable, set is_valid_spin_screen=False "
                        "and explain why in rejection_reason."
                    )
                    
                    try:
                        response = client.models.generate_content(
                            model="gemini-2.5-flash",
                            contents=[
                                types.Part.from_bytes(data=raw_bytes, mime_type="image/jpeg"),
                                prompt
                            ],
                            config=types.GenerateContentConfig(
                                response_mime_type="application/json",
                                response_schema=SpinMetrics,
                            )
                        )
                        metrics = SpinMetrics.model_validate_json(response.text)
                        
                        if not metrics.is_valid_spin_screen:
                            st.warning(f"⚠️ Photo {idx + 1} rejected: {metrics.rejection_reason or 'Not a recognized workout screen'}")
                            failed_count += 1
                            continue
                        
                        # Insert into database with duplicate prevention
                        cursor.execute("""
                        INSERT OR IGNORE INTO workouts (workout_date, duration_min, avg_watts, avg_cadence, total_kj, calories, distance_miles)
                        VALUES (?, ?, ?, ?, ?, ?, ?)
                        """, (
                            captured_date,
                            metrics.duration_min,
                            metrics.avg_watts,
                            metrics.avg_cadence,
                            metrics.total_kj,
                            metrics.calories,
                            metrics.distance_miles
                        ))
                        conn.commit()
                        
                        if cursor.rowcount == 0:
                            st.info(f"ℹ️ Ride from {captured_date} was already logged. Skipped duplicate.")
                        else:
                            st.write(f"✅ Successfully logged ride from **{captured_date}** ({metrics.avg_watts or 0}W, {metrics.avg_cadence or 0} RPM)")
                            success_count += 1
                            
                    except Exception as err:
                        st.error(f"❌ Error processing photo {idx + 1}: {err}")
                        failed_count += 1
                
                status.update(
                    label=f"Done! {success_count} ride(s) saved, {failed_count} failed.", 
                    state="complete", 
                    expanded=True
                )
            st.rerun()

# 6. Data Visualization & Analytics
st.divider()
df = pd.read_sql_query("SELECT * FROM workouts ORDER BY workout_date ASC", conn)

if not df.empty:
    df["workout_date"] = pd.to_datetime(df["workout_date"])
    
    st.subheader("📊 Performance Analytics")
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Rides", len(df))
    col2.metric("Max Watts", f"{df['avg_watts'].max():.0f} W" if df['avg_watts'].notnull().any() else "--")
    col3.metric("Avg Cadence", f"{df['avg_cadence'].mean():.0f} RPM" if df['avg_cadence'].notnull().any() else "--")

    metric_to_plot = st.selectbox(
        "Metric to Trend", 
        ["avg_watts", "avg_cadence", "total_kj", "calories", "distance_miles", "duration_min"]
    )
    fig = px.line(
        df, 
        x="workout_date", 
        y=metric_to_plot, 
        markers=True,
        title=f"{metric_to_plot.replace('_', ' ').title()} Over Time"
    )
    st.plotly_chart(fig, use_container_width=True)

    with st.expander("📋 View Raw Workout Log"):
        st.dataframe(df.sort_values(by="workout_date", ascending=False), use_container_width=True)

    # 7. Conversational Chat Interface
    st.divider()
    st.subheader("💬 Coach Chat")
    st.caption("Ask specific questions about your trends, power output, or progression.")

    # Initialize chat history in session state
    if "messages" not in st.session_state:
        st.session_state["messages"] = [
            {"role": "assistant", "content": "I've reviewed your workout log. What would you like to analyze today?"}
        ]

    # Render conversation history
    for msg in st.session_state["messages"]:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Chat user input
    if prompt_text := st.chat_input("e.g., Compare my last 3 rides or check if my cadence is dropping."):
        # Display user message
        st.session_state["messages"].append({"role": "user", "content": prompt_text})
        with st.chat_message("user"):
            st.markdown(prompt_text)

        if not client:
            st.error("Please configure the Gemini API Key.")
        else:
            with st.chat_message("assistant"):
                with st.spinner("Analyzing workouts..."):
                    chat_context = f"""
                    You are an analytical cycling performance coach.
                    Below is the user's complete historical workout log in JSON:
                    {df.to_json(orient='records', date_format='iso')}

                    User question: {prompt_text}

                    Instructions:
                    - Base all statements on the numerical data provided.
                    - State specific dates, power deltas, and cadence trends.
                    - Offer actionable coaching recommendations.
                    - Keep the tone encouraging, concise, and direct.
                    """
                    response = client.models.generate_content(
                        model="gemini-2.5-flash",
                        contents=chat_context
                    )
                    reply = response.text
                    st.markdown(reply)
                    st.session_state["messages"].append({"role": "assistant", "content": reply})

else:
    st.info("No rides logged yet. Snap or upload your first bike console photo above to populate your dashboard.")
