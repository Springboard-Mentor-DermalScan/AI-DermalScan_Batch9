import os
import cv2
import time
import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf
from PIL import Image
from datetime import datetime
import random

# =============================
# Create Export Directories
# =============================
os.makedirs("exports/images", exist_ok=True)
os.makedirs("exports/logs", exist_ok=True)

# =============================
# PAGE CONFIG
# =============================
st.set_page_config(
    page_title="AI DermalScan",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# =============================
# SESSION HISTORY
# =============================
if "history" not in st.session_state:
    st.session_state.history = []

# =============================
# FIXED DARK THEME
# =============================
st.session_state.theme = "Dark"

st.markdown("""
<style>
.stApp { background: linear-gradient(135deg,#020617,#0f172a); }
h1,h2,h3,p,label,span { color:#e5e7eb !important; }
button { background:#1e293b !important; color:#e5e7eb !important; }
</style>
""", unsafe_allow_html=True)

# =============================
# TITLE
# =============================
st.markdown(
    "<h1 style='text-align:center'>AI DermalScan – Skin Condition Detection</h1>",
    unsafe_allow_html=True
)

# =============================
# LOAD MODEL
# =============================
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("model/mobilenetv2_module3.h5", compile=False)

model = load_model()

CLASS_NAMES = ["clear skin", "dark spots", "puffy eyes", "wrinkles"]

AGE_RANGES = {
    "clear skin": (20, 30),
    "dark spots": (25, 40),
    "puffy eyes": (40, 55),
    "wrinkles": (55, 70)
}

# =============================
# FACE DETECTOR
# =============================
face_detector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

# =============================
# IMAGE UPLOAD
# =============================
uploaded_file = st.file_uploader(
    "Upload skin image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file:
    start_time = time.time()

    image = Image.open(uploaded_file).convert("RGB")
    img_np = np.array(image)
    gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)

    faces = face_detector.detectMultiScale(gray, 1.3, 5)

    annotated = img_np.copy()
    results = []

    for i, (x, y, w, h) in enumerate(faces):
        face = img_np[y:y+h, x:x+w]
        face_resized = cv2.resize(face, (224, 224)) / 255.0
        face_input = np.expand_dims(face_resized, axis=0)

        preds = model.predict(face_input, verbose=0)[0]
        idx = np.argmax(preds)

        label = CLASS_NAMES[idx]
        conf = preds[idx] * 100
        age = random.randint(*AGE_RANGES[label])

        thickness = max(3, w // 150)
        font_scale = max(0.6, w / 300)

        cv2.rectangle(annotated, (x, y), (x+w, y+h), (0,255,0), thickness)

        text = f"{label} | Age {age}"
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)

        cv2.rectangle(
            annotated,
            (x, y - th - 12),
            (x + tw + 6, y),
            (255,105,180),
            -1
        )

        cv2.putText(
            annotated,
            text,
            (x+3, y-6),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (0,0,0),
            thickness
        )

        record = {
            "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "Face": i + 1,
            "Class": label,
            "Estimated Age": age,
            "Confidence (%)": round(conf, 2),
            "X": x,
            "Y": y,
            "Width": w,
            "Height": h
        }

        results.append(record)
        st.session_state.history.append(record)

    processing_time = round(time.time() - start_time, 2)

    # =============================
    # SAVE ANNOTATED IMAGE
    # =============================
    image_filename = f"annotated_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
    image_path = os.path.join("exports/images", image_filename)
    cv2.imwrite(image_path, cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR))

    # =============================
    # DISPLAY
    # =============================
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Default input image")
        st.image(image, width=320)

    with col2:
        st.subheader("Annotated image")
        st.image(annotated, width=320)

        with open(image_path, "rb") as f:
            st.download_button(
                "Download annotated image",
                f,
                image_filename,
                "image/jpeg"
            )

    st.success(f"Estimated processing time: {processing_time} seconds")

    df = pd.DataFrame(results)
    st.subheader("Prediction summary")
    st.dataframe(df, use_container_width=True)

    hist_df = pd.DataFrame(st.session_state.history)
    csv = hist_df.to_csv(index=False).encode("utf-8")

    st.download_button(
        "Export history as CSV",
        csv,
        "prediction_history.csv",
        "text/csv"
    )
