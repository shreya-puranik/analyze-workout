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

# Enable mobile formats (HEIC, WebP, etc.)
pillow_heif.register_heif_opener()

st.set_page_config(page_title="Spin Lab", page_icon="🚴", layout="centered")
st.title("🚴 Spin Performance Lab")

# 1. Persistent Session State Initialization
if "processing_logs" not in st.session_state:
    st.session_state["processing_logs"] = []

if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {"role": "assistant", "content": "I'm ready. Upload your ride photos or ask questions about logged workouts."}
    ]

# 2. Database Connection
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

# 3. API Key & Client Setup
api_key = None
if hasattr(st, "secrets") and "GEMINI_API_KEY" in st.secrets:
    api_key = st.secrets["GEMINI_API_KEY"]
elif os.getenv("GEMINI_API_KEY"):
    api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    api_key = st.sidebar.text_input("Enter Gemini API Key", type="password")

client = genai.Client(api_key=api_key) if api_key else None

# 4. Structured Output Schema
class SpinMetrics(BaseModel):
    is_valid_spin_screen: bool = Field(description="True if photo is a stationary/spin bike summary screen, False otherwise")
    duration_min: Optional[float] = Field(default=None, description="Workout duration in minutes")
    avg_watts: Optional[float] = Field(default=None, description="Average power in watts")
    avg_cadence: Optional[float] = Field(default=None, description="Average cadence in RPM")
    total_kj: Optional[float] = Field(default=None, description="Total work in kiloJoules (kJ)")
    calories: Optional[float] = Field(default=None, description="Total estimated calories burned")
    distance_miles: Optional[float] = Field(default=None, description="Total distance in miles")
    rejection_reason: Optional[str] = Field(default=None, description="Explanation if not a valid workout summary screen")

def extract_photo_timestamp(raw_bytes: bytes) -> str:
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

# 5. Ingestion UI
st.subheader("📥 Add Workout Photos")
tab_upload, tab_camera = st.tabs(["📁 Choose from Photos", "📸 Take Photo"])

selected_files = []
with tab_upload:
    uploaded = st.file_uploader(
        "Select console photos",
        type=["jpg", "jpeg", "png", "heic", "webp"],
        accept_multiple_files=True,
        key="gallery_uploader"
    )
    if uploaded:
        selected_files.extend(uploaded)

with tab_camera:
    camera_pic = st.camera_input("Snap console screen", key="camera_snap")
    if camera_pic:
        selected_files.append(camera_pic)

if selected_files:
    st.info(f"📂 {len(selected_files)} photo(s) selected and ready for analysis.")
    
    if st.button("🚀 Process & Save Workouts", type="primary", use_container_width=True):
        if not client:
            st.error("Missing Gemini API Key. Add it to Streamlit Secrets or sidebar.")
        else:
            st.session_state["processing_logs"] = []
            progress_bar = st.progress(0)
            status_text = st.empty()

            for i, file_obj in enumerate(selected_files):
                status_text.text(f"Processing photo {i + 1} of {len(selected_files)} ({file_obj.name})...")
                progress_bar.progress((i + 1) / len(selected_files))
                
                raw_bytes = file_obj.getvalue()
                captured_date = extract_photo_timestamp(raw_bytes)

                prompt = (
                    "Inspect this image. Determine if it is a stationary spin bike summary screen. "
                    "Extract duration_min, avg_watts, avg_cadence, total_kj, calories, and distance_miles. "
                    "If obscured, unclear, or not a workout console, set is_valid_spin_screen=False."
                )

                try:
                    # Uses production gemini-2.0-flash with fallback
                    try:
                        model_name = "gemini-2.0-flash"
                        response = client.models.generate_content(
                            model=model_name,
                            contents=[
                                types.Part.from_bytes(data=raw_bytes, mime_type="image/jpeg"),
                                prompt
                            ],
                            config=types.GenerateContentConfig(
                                response_mime_type="application/json",
                                response_schema=SpinMetrics,
                            )
                        )
                    except Exception:
                        model_name = "gemini-1.5-flash"
                        response = client.models.generate_content(
                            model=model_name,
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
                        st.session_state["processing_logs"].append({
                            "status": "warning",
                            "msg": f"⚠️ {file_obj.name}: Rejected ({metrics.rejection_reason or 'Not a recognized workout screen'})."
                        })
                        continue

                    cursor.execute("""
                    INSERT OR IGNORE INTO workouts (workout_date, duration_min, avg_watts, avg_cadence, total_kj, calories, distance_miles)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    """, (
                        captured_date, metrics.duration_min, metrics.avg_watts,
                        metrics.avg_cadence, metrics.total_kj, metrics.calories, metrics.distance_miles
                    ))
                    conn.commit()

                    if cursor.rowcount == 0:
                        st.session_state["processing_logs"].append({
                            "status": "info",
                            "msg": f"ℹ️ {file_obj.name}: Ride from {captured_date} was already logged. Skipped duplicate."
                        })
                    else:
                        st.session_state["processing_logs"].append({
                            "status": "success",
                            "msg": f"✅ {file_obj.name}: Saved ride ({captured_date}) &rarr; {metrics.avg_watts or 0}W, {metrics.avg_cadence or 0} RPM, {metrics.duration_min} min."
                        })

                except Exception as ex:
                    st.session_state["processing_logs"].append({
                        "status": "error",
                        "msg": f"❌ {file_obj.name}: API/Processing Error: {str(ex)}"
                    })

            status_text.text("Processing complete!")

# 6. Persistent Batch Feedback Container
if st.session_state["processing_logs"]:
    st.markdown("### 📋 Upload Results")
    for log in st.session_state["processing_logs"]:
        if log["status"] == "success":
            st.success(log["msg"])
        elif log["status"] == "warning":
            st.warning(log["msg"])
        elif log["status"] == "info":
            st.info(log["msg"])
        elif log["status"] == "error":
            st.error(log["msg"])
    
    if st.button("Clear Log Results"):
        st.session_state["processing_logs"] = []
        st.rerun()

# 7. Dashboard Analytics
st.divider()
df = pd.read_sql_query("SELECT * FROM workouts ORDER BY workout_date ASC", conn)

if not df.empty:
    df["workout_date"] = pd.to_datetime(df["workout_date"])

    st.subheader("📊 Performance Analytics")
    c1, c2, c3 = st.columns(3)
    c1.metric("Total Rides", len(df))
    c2.metric("Max Watts", f"{df['avg_watts'].max():.0f} W" if df['avg_watts'].notnull().any() else "--")
    c3.metric("Avg Cadence", f"{df['avg_cadence'].mean():.0f} RPM" if df['avg_cadence'].notnull().any() else "--")

    metric_choice = st.selectbox(
        "Metric to Trend",
        ["avg_watts", "avg_cadence", "total_kj", "calories", "distance_miles", "duration_min"]
    )
    fig = px.line(df, x="workout_date", y=metric_choice, markers=True, title=f"{metric_choice.replace('_', ' ').title()} Over Time")
    st.plotly_chart(fig, use_container_width=True)

    with st.expander("📋 View Stored Database"):
        st.dataframe(df.sort_values(by="workout_date", ascending=False), use_container_width=True)

    # 8. Interactive Chat Interface
    st.divider()
    st.subheader("💬 Coach Chat")

    for msg in st.session_state["messages"]:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    if prompt_text := st.chat_input("Ask a question about your workouts..."):
        st.session_state["messages"].append({"role": "user", "content": prompt_text})
        with st.chat_message("user"):
            st.markdown(prompt_text)

        if client:
            with st.chat_message("assistant"):
                with st.spinner("Analyzing workouts..."):
                    system_ctx = f"""
                    You are an analytical cycling performance coach.
                    User's workout logs (JSON):
                    {df.to_json(orient='records', date_format='iso')}

                    Question: {prompt_text}

                    Provide clear, data-driven coaching insights with specific dates and comparisons.
                    """
                    chat_resp = client.models.generate_content(
                        model="gemini-2.0-flash",
                        contents=system_ctx
                    )
                    reply = chat_resp.text
                    st.markdown(reply)
                    st.session_state["messages"].append({"role": "assistant", "content": reply})
else:
    st.info("No rides logged yet. Upload or snap a console photo above to populate your dashboard.")
