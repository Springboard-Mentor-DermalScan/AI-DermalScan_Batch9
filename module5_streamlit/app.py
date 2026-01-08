import streamlit as st
import numpy as np
import pandas as pd
import cv2
import time
import tensorflow as tf
from PIL import Image
from datetime import datetime
import random

# =============================
# PAGE CONFIG
# =============================
st.set_page_config(
    page_title="AI DermaScan",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# =============================
# THEME (FIXED – NO TOGGLE SHOWN)
# =============================
st.session_state.theme = "Dark"


# =============================
# STYLING + ANIMATIONS
# =============================
if st.session_state.theme == "Dark":
    st.markdown("""
    <style>
    /* ===== GLOBAL ===== */
    .stApp {
        background: linear-gradient(135deg,#020617,#0f172a);
        animation: fadeIn 0.8s ease-in-out;
    }

    h1,h2,h3,p,label,span {
        color:#e5e7eb !important;
        animation: slideUp 0.6s ease-in-out;
    }

    /* ===== BUTTONS ===== */
    button {
        background:#1e293b !important;
        color:#e5e7eb !important;
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 20px rgba(0,0,0,0.4);
    }

    /* ===== IMAGES ===== */
    img {
        animation: zoomIn 0.6s ease-in-out;
        border-radius: 12px;
    }

    /* ===== TABLE ===== */
    .stDataFrame {
        animation: fadeIn 0.7s ease-in-out;
    }
    .stDataFrame tbody tr:hover {
        background-color: rgba(255,255,255,0.05) !important;
        transition: background-color 0.2s ease;
    }

    /* ===== METRICS ===== */
    [data-testid="metric-container"] {
        background: linear-gradient(135deg,#022c22,#064e3b);
        border-radius: 12px;
        padding: 12px;
        animation: slideUp 0.6s ease-in-out;
    }

    /* ===== ANIMATIONS ===== */
    @keyframes fadeIn {
        from { opacity: 0; }
        to { opacity: 1; }
    }

    @keyframes slideUp {
        from { transform: translateY(20px); opacity: 0; }
        to { transform: translateY(0); opacity: 1; }
    }

    @keyframes zoomIn {
        from { transform: scale(0.95); opacity: 0; }
        to { transform: scale(1); opacity: 1; }
    }
    </style>
    """, unsafe_allow_html=True)
else:
    st.markdown("""
    <style>
    .stApp {
        background: linear-gradient(135deg,#f9fafb,#eef2f7);
        animation: fadeIn 0.8s ease-in-out;
    }

    h1,h2,h3,p,label,span {
        color:#111827 !important;
        animation: slideUp 0.6s ease-in-out;
    }

    button {
        background:#2563eb !important;
        color:white !important;
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 20px rgba(0,0,0,0.2);
    }

    img {
        animation: zoomIn 0.6s ease-in-out;
        border-radius: 12px;
    }

    .stDataFrame {
        animation: fadeIn 0.7s ease-in-out;
    }

    @keyframes fadeIn {
        from { opacity: 0; }
        to { opacity: 1; }
    }

    @keyframes slideUp {
        from { transform: translateY(20px); opacity: 0; }
        to { transform: translateY(0); opacity: 1; }
    }

    @keyframes zoomIn {
        from { transform: scale(0.95); opacity: 0; }
        to { transform: scale(1); opacity: 1; }
    }
    </style>
    """, unsafe_allow_html=True)

# =============================
# TITLE
# =============================
st.markdown(
    "<h1 style='text-align:center'>AI DermaScan – Skin Condition Detection</h1>",
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
    "clear skin": (18, 25),
    "dark spots": (25, 40),
    "puffy eyes": (40, 55),
    "wrinkles": (55, 70)
}

# =============================
# FACE DETECTOR
# =============================
face_detector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

# =============================
# SESSION HISTORY
# =============================
if "history" not in st.session_state:
    st.session_state.history = []

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

        cv2.rectangle(annotated, (x,y), (x+w,y+h), (0,255,0), thickness)

        text = f"{label} | Age {age}"
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)

        cv2.rectangle(
            annotated,
            (x, y - th - 12),
            (x + tw + 6, y),
            (255, 105, 180),
            -1
        )

        cv2.putText(
            annotated,
            text,
            (x + 3, y - 6),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (0,0,0),
            thickness
        )

        results.append({
            "Face": i+1,
            "Class": label,
            "Estimated Age": age,
            "Confidence (%)": round(conf,2),
            "X": x,
            "Y": y,
            "Width": w,
            "Height": h
        })

    processing_time = round(time.time() - start_time, 2)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Default input image")
        st.image(image, width=320)

    with col2:
        st.subheader("Annotated image")
        st.image(annotated, width=320)

    st.success(f"Estimated processing time: {processing_time} seconds")

    df = pd.DataFrame(results)
    st.subheader("Prediction summary")
    st.dataframe(df, use_container_width=True)

    for r in results:
        r["Timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        st.session_state.history.append(r)

    hist_df = pd.DataFrame(st.session_state.history)
    st.subheader("Previous predictions")
    st.dataframe(hist_df, use_container_width=True)

    csv = hist_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Export history as CSV",
        csv,
        "prediction_history.csv",
        "text/csv"
    )
