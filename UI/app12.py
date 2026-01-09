import streamlit as st
import numpy as np
import pandas as pd
import cv2
import time
import io
from PIL import Image

from inference import detect_single_face, predict_skin, estimate_age

# ================= PAGE CONFIG =================
st.set_page_config(
    page_title="Dermal AI",
    page_icon="🧠",
    layout="wide"
)

# ================= STYLES =================
st.markdown("""
<style>
.main-title {
    font-size: 46px;
    font-weight: 700;
    text-align: center;
    color: #6EE7F9;
}
.subtitle {
    text-align: center;
    font-size: 18px;
    color: #9CA3AF;
    margin-bottom: 25px;
}
.card {
    background: rgba(255,255,255,0.04);
    backdrop-filter: blur(14px);
    border-radius: 16px;
    padding: 22px;
    border: 1px solid rgba(255,255,255,0.12);
    margin-top: 20px;
}
.glass {
    background: rgba(255,255,255,0.05);
    backdrop-filter: blur(14px);
    border-radius: 14px;
    padding: 16px;
    border: 1px solid rgba(255,255,255,0.12);
}
</style>
""", unsafe_allow_html=True)

# ================= TITLE =================
st.markdown('<div class="main-title">🧠 Dermal AI</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="subtitle">AI-powered skin analysis</div>',
    unsafe_allow_html=True
)

# st.warning("⚠️ Educational purpose only. Not a medical diagnosis.")

# ================= UPLOAD =================
uploaded_file = st.file_uploader(
    "📤 Upload a clear face image",
    type=["jpg", "jpeg", "png"]
)

# ================= MAIN =================
if uploaded_file:
    start = time.time()

    pil_img = Image.open(uploaded_file).convert("RGB")
    img_np = np.array(pil_img)

    face_data = detect_single_face(img_np)

    if face_data is None:
        st.error("❌ No face detected.")
        st.stop()

    x1, y1, x2, y2 = face_data["box"]
    w, h = face_data["width"], face_data["height"]

    face_crop = img_np[y1:y2, x1:x2]

    result = predict_skin(face_crop)
    predicted_class = result["class"]
    confidence = result["confidence"]
    class_probs = result["all_confidences"]

    age = estimate_age(w, h)

    label = f"{predicted_class} ({confidence*100:.1f}%) | Age: {age}"

    annotated = img_np.copy()
    cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 255), 2)
    cv2.putText(
        annotated,
        label,
        (x1, y1 - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 255, 255),
        2
    )

    # ================= IMAGES =================
    st.markdown("## 🖼 Image Analysis")
    col1, col2 = st.columns(2)

    with col1:
        st.image(img_np, caption="Original Image", use_container_width=True)

    with col2:
        st.image(annotated, caption="Annotated Image", use_container_width=True)

    # ================= DOWNLOAD =================
    buf = io.BytesIO()
    Image.fromarray(annotated).save(buf, format="PNG")

    st.download_button(
        "⬇️ Download Annotated Image",
        data=buf.getvalue(),
        file_name="dermal_ai_annotated.png",
        mime="image/png"
    )

    # ================= SUMMARY =================
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("🔍 Analysis Summary")
    st.write(f"**Primary Condition:** {predicted_class}")
    st.write(f"**Confidence:** {confidence*100:.2f}%")
    st.write(f"**Estimated Age:** {age}")
    st.write(f"**Processing Time:** {time.time() - start:.2f} sec")
    st.markdown('</div>', unsafe_allow_html=True)

    # ================= DETAILED CONFIDENCE =================
    with st.expander("📊 Detailed Skin Confidence"):
        df_conf = pd.DataFrame({
            "Condition": class_probs.keys(),
            "Confidence (%)": [round(v * 100, 2) for v in class_probs.values()]
        }).sort_values("Confidence (%)", ascending=False)

        st.bar_chart(df_conf.set_index("Condition"), use_container_width=True)

    # ================= PREDICTION TABLE =================
    st.markdown("## 📋 Predictions")

    df_pred = pd.DataFrame([{
        "Filename": uploaded_file.name,
        "box_x1": x1,
        "box_y1": y1,
        "box_x2": x2,
        "box_y2": y2,
        "Class": predicted_class,
        "Confidence (%)": round(confidence * 100, 2),
        "Age": age,
        "Detector Conf (%)": 100.0
    }])

    st.markdown('<div class="glass">', unsafe_allow_html=True)
    st.dataframe(df_pred, use_container_width=True, hide_index=True)
    st.markdown('</div>', unsafe_allow_html=True)
