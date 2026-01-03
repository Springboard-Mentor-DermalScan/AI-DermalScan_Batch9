import streamlit as st
import numpy as np
import cv2
import tensorflow as tf 
from PIL import Image
from scripts.facedetection import predict
from tensorflow.keras.applications.resnet50 import preprocess_input
import os
import time


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



if uploaded:
    img = Image.open(uploaded).convert("RGB")
    img = np.array(img)
    preview = st.empty()
    preview.image(img, caption="Uploaded image", use_container_width=True)

    start_time = time.time()

    

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
        
        x, y, w, h = faces[0]
        faceimg = img[y:y+h, x:x+w]

        faceimg = cv2.resize(faceimg,(224,224))
        faceimg = preprocess_input(faceimg)
        faceimg = np.expand_dims(faceimg,axis=0)

        results = predict(faceimg,model,faces)
        end_time = time.time()
        processing_time = end_time - start_time

        cv2.rectangle(img, (x,y), (x+w,y+h), (0,255,0), 2)
        cv2.putText(
            img,
            f"{results['label']} ({results['confidence']:.2f}%) Age:{results['age']}",
            (x, y-10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0,255,0),
            2
        )
        st.subheader("Prediction Result")
        preview.image(img, use_container_width=True)
        st.success(f"Skin Type: {results['label']}")
        st.info(f"Confidence: {results['confidence']:.2f}%")
        st.write(f"Estimated Age:  {results['age']}")
        st.success(f"Processing time: {processing_time:.3f} seconds")