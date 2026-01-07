import streamlit as st
import pandas as pd
import cv2
import tempfile
import os
import sys

# ---------------- PATH FIX ----------------
sys.path.append(os.path.abspath(".."))
from backend.inference import predict_image

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="AI DermalScan",
    page_icon="🧬",
    layout="wide"
)

st.title("🧬 AI DermalScan")
st.subheader("Skin Disease & Age Prediction System")

# ---------------- SESSION STATE ----------------
if "results" not in st.session_state:
    st.session_state.results = []

# ---------------- FILE UPLOADER ----------------
uploaded_files = st.file_uploader(
    "📤 Upload Skin Images",
    type=["jpg", "jpeg", "png"],
    accept_multiple_files=True
)

# ---------------- PROCESS IMAGES ----------------
if uploaded_files:
    for file in uploaded_files:
        existing_files = [r["File Name"] for r in st.session_state.results]
        if file.name in existing_files:
            continue


        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
            tmp.write(file.read())
            img_path = tmp.name

        # 🔮 Prediction
        annotated_img, disease, confidence, age_bucket, status, predicted_age = predict_image(img_path)

        # 📊 Store results
        st.session_state.results.append({
            "File Name": file.name,
            "Disease": disease,
            "Prediction Confidence (%)": confidence,
            "Age Bucket": age_bucket,
            "Predicted Age": predicted_age,
            "Status": status
        })

        # 🖼️ Show Image
        st.image(
            cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB),
            caption=f"Annotated Image - {file.name}",
            width=450
        )

        # ⬇️ Image Download Button
        _, buffer = cv2.imencode(".jpg", annotated_img)

        st.download_button(
            label=f"⬇️ Download Annotated Image ({file.name})",
            data=buffer.tobytes(),
            file_name=f"annotated_{file.name}",
            mime="image/jpeg"
        )

# ---------------- RESULTS TABLE ----------------
if st.session_state.results:
    st.subheader("🔍 Predictions")
    df = pd.DataFrame(st.session_state.results)
    st.dataframe(df, use_container_width=True)

    # ⬇️ CSV Download
    st.download_button(
        label="⬇️ Download Prediction Report (CSV)",
        data=df.to_csv(index=False).encode("utf-8"),
        file_name="AI_DermalScan_Predictions.csv",
        mime="text/csv"
    )
