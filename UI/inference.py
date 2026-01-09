import cv2
import numpy as np
import tensorflow as tf

# ---------------- LOAD MODEL ----------------
MODEL_PATH = "models/final_model.keras"

skin_model = tf.keras.models.load_model(MODEL_PATH, compile=False)

CLASS_NAMES = ["Clear Skin", "Dark Spots", "Puffy Eyes", "Wrinkles"]

# ---------------- FACE DETECTOR ----------------
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# ---------------- HELPERS ----------------
def detect_single_face(image_np):
    gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(80, 80)
    )

    if len(faces) == 0:
        return None

    # largest face only
    faces = sorted(faces, key=lambda x: x[2] * x[3], reverse=True)
    x, y, w, h = faces[0]

    return {
        "box": (x, y, x + w, y + h),
        "width": w,
        "height": h
    }

def preprocess_face(face_img):
    face = cv2.resize(face_img, (224, 224))
    face = face / 255.0
    return np.expand_dims(face, axis=0)

def predict_skin(face_img):
    inp = preprocess_face(face_img)
    preds = skin_model.predict(inp, verbose=0)[0]

    top_idx = int(np.argmax(preds))

    return {
        "class": CLASS_NAMES[top_idx],
        "confidence": float(preds[top_idx]),
        "all_confidences": dict(zip(CLASS_NAMES, preds))
    }

def estimate_age(face_w, face_h):
    area = face_w * face_h
    if area > 90000:
        return np.random.randint(18, 26)
    elif area > 70000:
        return np.random.randint(26, 35)
    elif area > 50000:
        return np.random.randint(35, 45)
    else:
        return np.random.randint(45, 60)
