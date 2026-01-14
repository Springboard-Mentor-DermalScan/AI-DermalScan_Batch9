import streamlit as st
import numpy as np
import cv2
import pandas as pd
import tensorflow as tf 
from PIL import Image
from scripts.facedetection import predict
from tensorflow.keras.applications.resnet50 import preprocess_input
import os
import time
from datetime import datetime


st.set_page_config(
    page_title="Facial Age Detection",
    layout="wide"
)

st.markdown(
    """
    <style>
    .stApp {
        background: linear-gradient(
        135deg,
        #000000,
        #010101
);

    }
    """,
    unsafe_allow_html=True
)





st.title("Facial Skin Age Detection")

st.markdown("""
Upload a facial image to detect:
- Skin condition
- Estimated age group
- Confidence score
""")

@st.cache_resource
def load_model():
    return tf.keras.models.load_model(os.path.join("models", "resnet_best_model.h5"))

model = load_model()
#warm up to reduce first prediction delay
try:
    _ = model.predict(np.zeros((1, 224, 224, 3)), verbose=0)
except Exception:
    pass

uploaded = st.file_uploader("Upload image",["jpg","png","jpeg"])
filename = uploaded.name if uploaded else None

if "results_df" not in st.session_state:
    st.session_state.results_df = pd.DataFrame(
        columns=[
            "File Name", "X", "Y", "Width", "Height",
            "Class Name", "Age", "Confidence (%)","Timestamp"
        ]
    )


if uploaded:
    upload_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")


    img = Image.open(uploaded).convert("RGB")
    img = np.array(img)
    preview = st.empty()
    preview.image(img, caption="Uploaded image", width=900)

    start_time = time.time()
    progress = st.progress(0)

    

    with st.spinner("Processing image..."):
        
        facecascade = cv2.CascadeClassifier(os.path.join("scripts", "haarcascade_frontalface_default.xml"))

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        faces = facecascade.detectMultiScale(
            gray,
            scaleFactor=1.3,
            minNeighbors=5
        )

        if len(faces) == 0:
            st.warning("No face detected in the image. Please upload a clear, frontal face image.")
            st.stop()   
        print(faces)
        
        progress.progress(30)

        results_list = []
        start_time = time.time()

        for i, (x, y, w, h) in enumerate(faces):
            faceimg = img[y:y+h, x:x+w]

            faceimg = cv2.resize(faceimg, (224, 224))
            faceimg = preprocess_input(faceimg)
            faceimg = np.expand_dims(faceimg, axis=0)

            results = predict(faceimg, model, faces)
            results_list.append(results)

            st.session_state.results_df = pd.concat(
            [
            st.session_state.results_df,
            pd.DataFrame([{
                "File Name": filename,
                "X": x,
                "Y": y,
                "Width": w,
                "Height": h,
                "Class Name": results["label"],
                "Age": results["age"],
                "Confidence (%)": round(results["confidence"], 2),
                "Timestamp": upload_time
                    }])
                ],
                ignore_index=True
            )

            progress.progress(70)

            cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)

            cv2.putText(
                img,
                f"{results['label']} ({results['confidence']:.2f}%) Age:{results['age']}",
                (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2
            )
        progress.progress(100)
        annotated_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        success, buffer = cv2.imencode(".png", annotated_img)

        if success:
            annotated_bytes = buffer.tobytes()
        
        st.subheader("Export Results")

        col_a, col_b = st.columns(2)

        with col_a:
            st.download_button(
                "⬇ Download Annotated Image",
                annotated_bytes,
                file_name=f"annotated_{filename}",
                mime="image/png"
            )

        with col_b:
            csv = st.session_state.results_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                "⬇ Download CSV Log",
                csv,
                "face_detection_results.csv",
                "text/csv"
            )


        
        end_time = time.time()
        processing_time = end_time - start_time
        col1, col2 = st.columns([2, 1])

        with col1:
            preview = st.empty()
            preview.image(img, caption="Annotated Image", use_container_width=True)

        with col2:
            st.metric("Faces Detected", len(faces))
            st.metric("Processing Time (s)", f"{processing_time:.2f}")


        for idx, res in enumerate(results_list, start=1):
            with st.expander(f"Face {idx} Details"):
                st.write(f"**Skin Type:** {res['label']}")
                st.write(f"**Estimated Age:** {res['age']}")
                st.write(f"**Confidence:** {res['confidence']:.2f}%")

        

    st.subheader("Detection Logs")

    st.dataframe(
        st.session_state.results_df,
        use_container_width=True,
        height=300
    )