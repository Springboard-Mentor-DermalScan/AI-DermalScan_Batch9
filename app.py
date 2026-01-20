import streamlit as st
import cv2
import time
import pandas as pd
import uuid
import os
import sys

sys.path.append(os.path.abspath("../backend"))
from inference import predict_image

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="AI DermalScan",
    page_icon="🧬",
    layout="wide"
)

# ---------------- BACKGROUND STYLE ----------------
st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #0f2027, #203a43, #2c5364);
}
h1, h2, h3, h4, h5, h6, p, label {
    color: white;
}
</style>
""", unsafe_allow_html=True)

st.title("🧬 AI DermalScan")
st.subheader("Skin Disease & Age Prediction System")

uploaded_files = st.file_uploader(
    "📤 Upload Skin Images",
    type=["jpg", "jpeg", "png"],
    accept_multiple_files=True
)

os.makedirs("uploads", exist_ok=True)
os.makedirs("outputs", exist_ok=True)

results = []

if uploaded_files:
    for file in uploaded_files:
        uid = str(uuid.uuid4())
        img_path = f"uploads/{uid}.jpg"
        out_path = f"outputs/annotated_{uid}.jpg"

        with open(img_path, "wb") as f:
            f.write(file.read())

        start = time.perf_counter()
        annotated, disease, confidence, age_bucket, status, age = predict_image(img_path)
        end = time.perf_counter()

        time_taken = round(end - start, 3)

        cv2.imwrite(out_path, annotated)

        st.image(
            annotated,
            caption=f"{file.name} | {disease} | {confidence}% | Age {age} | ⏱ {time_taken}s",
            channels="BGR"
        )

        with open(out_path, "rb") as img_file:
            st.download_button(
                label="⬇️ Download This Annotated Image",
                data=img_file,
                file_name=f"annotated_{file.name}",
                mime="image/jpeg",
                key=uid
            )

        results.append({
            "File Name": file.name,
            "Disease": disease,
            "Prediction Confidence (%)": confidence,
            "Age Bucket": age_bucket,
            "Predicted Age": age,
            "Status": status,
            "Time Taken (sec)": time_taken
        })

if results:
    st.subheader("📊 Prediction Summary")
    df = pd.DataFrame(results)
    st.dataframe(df, use_container_width=True)

    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button(
        "⬇️ Download Prediction Report (CSV)",
        csv,
        "AI_DermalScan_Report.csv",
        "text/csv"
    )
