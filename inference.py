import os
import cv2
import random
import numpy as np
import pandas as pd
import tensorflow as tf
import mediapipe as mp


#  trained model

MODEL_PATH = r"C:\Users\ambat\OneDrive\Desktop\dermalscan\.venv\mobilenetv2_best.h5"
model = tf.keras.models.load_model(MODEL_PATH)

# Class labels 

LABELS = [
    "clear skin",
    "dark spots",
    "puffy eyes",
    "wrinkles"
    
]

# MediaPipe face detector

mp_face = mp.solutions.face_detection
face_detector = mp_face.FaceDetection(
    model_selection=1,
    min_detection_confidence=0.5
)


# Preprocess face image for model

def preprocess_face(face_img):
    face_img = cv2.resize(face_img, (224, 224))
    face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
    face_img = face_img.astype(np.float32) / 255.0
    return np.expand_dims(face_img, axis=0)


# Age Estimate 

def estimate_age_from_class(label):
    label = label.lower()

    if label == "clear skin":
        return random.randint(20, 28)
    if label == "puffy eyes":
        return random.randint(30, 38)
    if label == "dark spots":
        return random.randint(35, 42)
    if label == "wrinkles":
        return random.randint(45, 55)

    return random.randint(22, 30)


#  inference function

def run_inference(image_path):
    image_bgr = cv2.imread(image_path)
    if image_bgr is None:
        raise ValueError("Invalid image path")

    img_height, img_width, _ = image_bgr.shape

    # Convert to RGB for face detection
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    detection_results = face_detector.process(image_rgb)

    
    x, y, w, h = 0, 0, img_width, img_height
    detector_confidence = 100.0

    if detection_results.detections:
        detection = detection_results.detections[0]
        detector_confidence = float(detection.score[0]) * 100

        box = detection.location_data.relative_bounding_box
        x = int(box.xmin * img_width)
        y = int(box.ymin * img_height)
        w = int(box.width * img_width)
        h = int(box.height * img_height)

        x = max(0, x)
        y = max(0, y)

    # Crop face region
    face_img = image_bgr[y:y + h, x:x + w]

    # Run model prediction
    predictions = model.predict(preprocess_face(face_img), verbose=0)[0]
    class_index = int(np.argmax(predictions))

    label = LABELS[class_index].lower()
    confidence = float(predictions[class_index]) * 100
    estimated_age = estimate_age_from_class(label)

    # Draw results on image
  
    annotated_bgr = image_bgr.copy()

    cv2.rectangle(
        annotated_bgr,
        (x, y),
        (x + w, y + h),
        (0, 255, 0),
        2
    )

    cv2.putText(
        annotated_bgr,
        f"{label} | {confidence:.1f}% | Age: {estimated_age}",
        (x, max(y - 10, 20)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0),
        2
    )

    annotated_rgb = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)

    # Build results table
   
    results_df = pd.DataFrame([{
        "Filename": os.path.basename(image_path),
        "Condition": label,
        "Confidence (%)": round(confidence, 2),
        "Estimated Age": estimated_age,
        "Face Detector Confidence (%)": round(detector_confidence, 2),
        "X1": x,
        "Y1": y,
        "X2": x + w,
        "Y2": y + h
    }])

    return annotated_rgb, results_df