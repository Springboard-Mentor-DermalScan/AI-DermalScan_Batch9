import os
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.nn import softmax

# ---------------- BASE SETUP ----------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(
    BASE_DIR,
    "models",
    "MobileNetV2_Module3.h5"
)

model = load_model(MODEL_PATH)

CLASS_NAMES = ["Clear skin", "Dark spots", "Puffy eyes", "Wrinkles"]

AGE_BUCKETS = {
    "0-18": (0, 18),
    "19-25": (19, 25),
    "26-35": (26, 35),
    "36-50": (36, 50),
    "50+": (51, 100)
}

BASE_AGE = {
    "Clear skin": 22,
    "Dark spots": 30,
    "Puffy eyes": 35,
    "Wrinkles": 45
}

def risk_status(disease):
    if disease == "Clear skin":
        return "Normal"
    elif disease == "Wrinkles":
        return "Risk"
    else:
        return "Moderate"

# ---------------- PREDICTION FUNCTION ----------------
def predict_image(img_path):

    # Load image
    img = image.load_img(img_path, target_size=(224, 224))
    img_arr = image.img_to_array(img) / 255.0
    img_arr = np.expand_dims(img_arr, axis=0)

    # Disease prediction
    raw_preds = model.predict(img_arr)[0]
    probs = softmax(raw_preds).numpy()

    idx = np.argmax(probs)
    disease = CLASS_NAMES[idx]
    confidence = probs[idx] * 100

    # -------- STABLE AGE LOGIC --------
    predicted_age = BASE_AGE[disease] + int(confidence // 20)
    predicted_age = max(18, min(predicted_age, 60))

    # Age bucket
    age_bucket = "Unknown"
    for bucket, (low, high) in AGE_BUCKETS.items():
        if low <= predicted_age <= high:
            age_bucket = bucket
            break

    # Risk status
    status = risk_status(disease)

    # -------- Annotated Image --------
    img_cv = cv2.imread(img_path)
    h, w, _ = img_cv.shape

    x1, y1 = int(w * 0.25), int(h * 0.25)
    x2, y2 = int(w * 0.75), int(h * 0.75)

    cv2.rectangle(img_cv, (x1, y1), (x2, y2), (0, 255, 0), 2)

    label = f"{disease} | {confidence:.2f}% | Age: {predicted_age}"
    cv2.putText(
        img_cv,
        label,
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 0, 0),
        2
    )

    return img_cv, disease, round(confidence, 2), age_bucket, status, predicted_age
