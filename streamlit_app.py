import streamlit as st
import cv2
import yt_dlp
from ultralytics import YOLO
import os

# Safety switch for cloud servers
os.environ["QT_QPA_PLATFORM"] = "offscreen"

st.title("🤖 FRC 2026 AI Scout")

# Loads a tiny starter AI (downloads automatically)
@st.cache_resource
def load_model():
    return YOLO("yolov8n.pt")

model = load_model()

url = st.text_input("Paste YouTube Link:", "")

if url and st.button("Start Scouting"):
    st.write("⏳ Connecting...")
    try:
        ydl_opts = {"format": "best[ext=mp4]/best", "quiet": True}
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=False)
            video_url = info["url"]

        cap = cv2.VideoCapture(video_url)
        frame_placeholder = st.empty()

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break

            results = model(frame, conf=0.3)
            annotated_frame = results[0].plot()
            frame_placeholder.image(annotated_frame, channels="BGR")

        cap.release()
    except Exception as e:
        st.error(f"Error: {e}")
