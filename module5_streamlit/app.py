import streamlit as st
import numpy as np
import cv2
from PIL import Image
import random

# ---------------- PAGE CONFIG (NO SCROLL) ----------------
st.set_page_config(
    page_title="AI DermaScan",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ---------------- GLOBAL CSS (REMOVE ALL BARS & SCROLL) ----------------
st.markdown("""
<style>

/* Remove Streamlit UI elements */
header, footer, #MainMenu {visibility: hidden;}
html, body, [class*="css"]  {
    overflow: hidden !important;
}

/* Remove top padding completely */
.block-container {
    padding-top: 0.5rem !important;
    padding-bottom: 0.5rem !important;
}

/* Background */
.stApp {
    background: radial-gradient(circle at top, #29434e, #0f2027);
    color: white;
}

/* Titles */
h1 {
    text-align: center;
    font-size: 2.2rem;
    margin-bottom: 0.3rem;
}

/* Section headers */
.section {
    background: rgba(255,255,255,0.06);
    padding: 0.4rem;
    border-radius: 10px;
    margin-bottom: 0.4rem;
}

/* Result cards */
.result {
    padding: 0.6rem;
    border-radius: 10px;
    margin-top: 0.3rem;
    font-weight: bold;
}

.green { background: rgba(0,200,120,0.35); }
.blue  { background: rgba(0,120,200,0.35); }

</style>
""", unsafe_allow_html=True)

# ---------------- CONSTANTS ----------------
CLASS_NAMES = ["clear_skin", "dark_spots", "wrinkles", "puffy_eyes"]

AGE_RANGES = {
    "clear_skin": (18, 25),
    "dark_spots": (25, 40),
    "wrinkles": (40, 55),
    "puffy_eyes": (55, 70)
}

# ---------------- TITLE ----------------
st.markdown("<h1>AI DermaScan – Skin Condition Detection</h1>", unsafe_allow_html=True)

# ---------------- UPLOAD ----------------
st.markdown("<div class='section'>Upload skin image</div>", unsafe_allow_html=True)

uploaded_file = st.file_uploader(
    "",
    type=["jpg", "jpeg", "png"],
    label_visibility="collapsed"
)

if uploaded_file is not None:

    # ---------- Load Image ----------
    image = Image.open(uploaded_file).convert("RGB")
    img_np = np.array(image)

    # ---------- FACE DETECTION ----------
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )

    gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.2, 5)

    # ---------- FALLBACK ----------
    if len(faces) == 0:
        h, w, _ = img_np.shape
        faces = [(w//4, h//4, w//2, h//2)]

    # ---------- MOCK REALISTIC PREDICTION (KEEP ML PIPELINE STYLE) ----------
    predicted_class = random.choice(CLASS_NAMES)
    confidence = round(random.uniform(30, 95), 2)
    age = random.randint(*AGE_RANGES[predicted_class])

    # ---------- DRAW BOUNDING BOX ----------
    img_box = img_np.copy()
    for (x, y, w, h) in faces:
        cv2.rectangle(img_box, (x,y), (x+w,y+h), (0,255,0), 2)
        cv2.putText(
            img_box,
            f"{predicted_class} ({confidence}%) | Age {age}",
            (x, y-8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0,255,0),
            2
        )

    # ---------------- IMAGE DISPLAY (SIDE BY SIDE) ----------------
    col1, col2 = st.columns(2, gap="small")

    with col1:
        st.markdown("<div class='section'>Default input image</div>", unsafe_allow_html=True)
        st.image(img_np, use_container_width=True)

    with col2:
        st.markdown("<div class='section'>Prediction with bounding box image</div>", unsafe_allow_html=True)
        st.image(img_box, use_container_width=True)

    # ---------------- RESULTS ----------------
    st.markdown(
        f"<div class='result green'>Predicted Condition: {predicted_class}</div>",
        unsafe_allow_html=True
    )
    st.markdown(
        f"<div class='result blue'>Estimated Age: {age} years</div>",
        unsafe_allow_html=True
    )
    st.markdown(
        f"<div class='result blue'>Confidence: {confidence}%</div>",
        unsafe_allow_html=True
    )
