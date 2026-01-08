import streamlit as st
import tempfile
import time
import hashlib
import cv2
import pandas as pd

from inference import run_inference

st.set_page_config(page_title="Dermal Scan", layout="wide")

st.markdown(
    """
    <style>

    /* Global background */
    html, body, .stApp {
        background-color: #020617;
        color: #f8fafc;
    }

    /* Reduce default spacing */
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }

    /* Header */
    .app-header {
        background: linear-gradient(135deg, #020617, #0f172a);
        padding: 28px;
        border-radius: 18px;
        text-align: center;
        margin-bottom: 32px;
        box-shadow: 0 12px 30px rgba(0,0,0,0.7);
    }

    .app-header h1 {
        font-size: 44px;
        font-weight: 900;
        color: #e0f2fe;
        margin-bottom: 6px;
    }

    .app-header p {
        font-size: 16px;
        color: #cbd5f5;
    }

    /* File uploader */
    section[data-testid="stFileUploader"] {
        background-color: #020617;
        border: 2px dashed #38bdf8;
        border-radius: 14px;
        padding: 20px;
    }

    section[data-testid="stFileUploader"] * {
        color: #e0f2fe !important;
        font-weight: 600;
    }

    /* Buttons */
    button {
        background: linear-gradient(135deg, #38bdf8, #2563eb) !important;
        color: #020617 !important;
        font-weight: 800 !important;
        border-radius: 12px !important;
        border: none !important;
        padding: 10px 22px !important;
    }

    button span {
        color: #020617 !important;
        font-weight: 900;
    }

    /* Text */
    label,
    .stMarkdown,
    .stText,
    .stCaption,
    .stSubheader,
    .stHeader,
    .stTitle {
        color: #f8fafc !important;
    }

    /* Tables */
    .stDataFrame {
        background-color: #020617;
        border: 1px solid #38bdf8;
        border-radius: 14px;
    }

    .stDataFrame thead tr th {
        background-color: #020617 !important;
        color: #38bdf8 !important;
        font-weight: 800;
    }

    .stDataFrame tbody tr td {
        color: #e5e7eb !important;
    }

    /* Success alert */
    .stAlert {
        background-color: #020617 !important;
        border-left: 5px solid #22c55e !important;
    }

    .stAlert * {
        color: #dcfce7 !important;
        font-weight: 600;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# SESSION STATE

st.session_state.setdefault("history_rows", [])
st.session_state.setdefault("last_file_hash", None)

# HEADER

st.markdown(
    """
    <div class="app-header">
        <h1>DermalScan – AI Skin Analysis</h1>
        <p>Upload a face image to detect skin conditions and estimated age</p>
    </div>
    """,
    unsafe_allow_html=True
)


# IMAGE UPLOAD

uploaded_file = st.file_uploader(
    "Upload Face Image",
    type=["jpg", "jpeg", "png"]
)


# RUN INFERENCE

if uploaded_file:
    file_bytes = uploaded_file.getvalue()
    file_hash = hashlib.md5(file_bytes).hexdigest()

    if file_hash != st.session_state.last_file_hash:
        st.session_state.last_file_hash = file_hash
        start_time = time.time()

        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
            tmp.write(file_bytes)
            image_path = tmp.name

        annotated_img, result_table = run_inference(image_path)

        st.session_state.image_path = image_path
        st.session_state.annotated_img = annotated_img
        st.session_state.elapsed = time.time() - start_time

        if not result_table.empty:
            st.session_state.history_rows.append(
                result_table.iloc[0].to_dict()
            )


# RESULTS

if "annotated_img" in st.session_state:

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Original Image")
        st.image(st.session_state.image_path, use_container_width=True)

    with col2:
        st.subheader("Annotated Image")
        st.image(st.session_state.annotated_img, use_container_width=True)

    st.success(f"Processing Time: {st.session_state.elapsed:.2f} seconds")

    _, buffer = cv2.imencode(
        ".png",
        cv2.cvtColor(st.session_state.annotated_img, cv2.COLOR_RGB2BGR)
    )

    st.download_button(
        "Download Annotated Image",
        buffer.tobytes(),
        "annotated.png",
        "image/png"
    )

# HISTORY

if st.session_state.history_rows:

    st.subheader("Analysis History")

    df = pd.DataFrame(st.session_state.history_rows)

    if not df.empty:
        st.dataframe(df, use_container_width=True)

        st.download_button(
            "Download History (CSV)",
            df.to_csv(index=False).encode(),
            "history.csv",
            "text/csv"
        )
