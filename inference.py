import os
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.nn import softmax

# ---------------- BASE PATH ----------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__)) 
MODEL_PATH = os.path.join( 
    BASE_DIR, 
    "models", 
    "MobileNetV2_Module3_finetuned.h5" 
    ) 
# ---------------- LOAD MODEL ----------------
model = load_model(MODEL_PATH)

# ---------------- CONSTANTS ----------------
CLASSES = ["Clear skin", "Dark spots", "Puffy eyes", "Wrinkles"]

AGE_BASE = {
    "Clear skin": 22,
    "Dark spots": 30,
    "Puffy eyes": 35,
    "Wrinkles": 45
}

AGE_BUCKETS = {
    "0-18": (0, 18),
    "19-25": (19, 25),
    "26-35": (26, 35),
    "36-50": (36, 50),
    "50+": (51, 100)
}

# ---------------- RISK STATUS ----------------
def risk_status(disease):
    if disease == "Clear skin":
        return "Normal"
    elif disease == "Wrinkles":
        return "Risk"
    else:
        return "Moderate"

# ---------------- MAIN PREDICTION ----------------
def predict_image(img_path):
    # Load & preprocess image
    img = image.load_img(img_path, target_size=(224, 224))
    arr = image.img_to_array(img) / 255.0
    arr = np.expand_dims(arr, axis=0)

    # Model prediction
    preds = softmax(model.predict(arr, verbose=0)[0]).numpy()
    idx = np.argmax(preds)

    disease = CLASSES[idx]
    confidence = round(preds[idx] * 100, 2)

    # Age estimation
    age = AGE_BASE[disease] + int(confidence // 20)
    age = max(18, min(age, 60))

    # Age bucket
    age_bucket = "Unknown"
    for bucket, (low, high) in AGE_BUCKETS.items():
        if low <= age <= high:
            age_bucket = bucket
            break

    # Risk level
    status = risk_status(disease)

    # Annotated image
    img_cv = cv2.imread(img_path)
    cv2.putText(
        img_cv,
        f"{disease} | {confidence}% | Age {age}",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2
    )

    return img_cv, disease, confidence, age_bucket, status, age
