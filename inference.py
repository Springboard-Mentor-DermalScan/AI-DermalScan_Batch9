import cv2
import numpy as np
from tensorflow.keras.models import load_model
import random
import pandas as pd
import os
import time
from datetime import datetime

# PATHS
MODEL_PATH = "models/AIDermalScan_MobileNetV2_Final.h5"
PROTO = "face_detector/deploy.prototxt"
WEIGHTS = "face_detector/res10_300x300_ssd_iter_140000.caffemodel"
CSV_PATH = "output/predictions.csv"

os.makedirs("output", exist_ok=True)

CLASS_NAMES = ["clear skin", "dark spots", "puffy eyes", "wrinkles"]

AGE_BUCKETS = {
    "clear skin": (18, 25),
    "dark spots": (26, 40),
    "puffy eyes": (30, 45),
    "wrinkles": (40, 60)
}

IMG_SIZE = 224
CONF_THRESHOLD = 0.3

# LOAD MODEL & FACE DETECTOR
model = load_model(MODEL_PATH)
face_net = cv2.dnn.readNetFromCaffe(PROTO, WEIGHTS)

# SAVE TO CSV
def save_to_csv(row):
    df = pd.DataFrame([row])
    if os.path.exists(CSV_PATH):
        df.to_csv(CSV_PATH, mode="a", header=False, index=False)
    else:
        df.to_csv(CSV_PATH, index=False)

def process_image(image, filename):
    start_time = time.time()   # START TIMER

    h, w = image.shape[:2]

    # -------- FACE DETECTION --------
    blob = cv2.dnn.blobFromImage(
        image,
        scalefactor=1.0,
        size=(300, 300),
        mean=(104.0, 177.0, 123.0)
    )

    face_net.setInput(blob)
    detections = face_net.forward()

    if detections.shape[2] == 0:
        return image, None

    i = np.argmax(detections[0, 0, :, 2])
    detector_conf = float(detections[0, 0, i, 2])

    if detector_conf < 0.5:
        return image, None

    box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
    box_x1, box_y1, box_x2, box_y2 = box.astype(int)

    box_x1 = max(0, box_x1)
    box_y1 = max(0, box_y1)
    box_x2 = min(w, box_x2)
    box_y2 = min(h, box_y2)

    # -------- FACE CROP --------
    face = image[box_y1:box_y2, box_x1:box_x2]
    face = cv2.resize(face, (IMG_SIZE, IMG_SIZE))
    face = face.astype("float32") / 255.0
    face = np.expand_dims(face, axis=0)

    # -------- MODEL PREDICTION --------
    preds = model.predict(face, verbose=0)[0]
    class_index = int(np.argmax(preds))
    class_prob = float(preds[class_index])
    predicted_class = CLASS_NAMES[class_index]

    low_confidence = class_prob < CONF_THRESHOLD

    # -------- AGE ESTIMATION --------
    low, high = AGE_BUCKETS[predicted_class]
    age = random.randint(low, high)
    age_bucket = f"{low}-{high}"

    # -------- TIME TAKEN --------
    time_taken = round(time.time() - start_time, 2)
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # -------- ANNOTATION --------
    annotated = image.copy()
    label = f"{predicted_class} | {class_prob:.2f} | Age: {age}"

    if low_confidence:
        label += " (low confidence)"

    cv2.rectangle(
        annotated,
        (box_x1, box_y1),
        (box_x2, box_y2),
        (0, 255, 0),
        2
    )

    cv2.putText(
        annotated,
        label,
        (box_x1, box_y1 - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (0, 255, 0),
        2
    )


    # -------- CSV ROW --------
    row = {
        "Filename": filename,
        "Predicted_Class": predicted_class,
        "Confidence": round(class_prob* 100, 3),
        "Age": age,
        "Age_Bucket": age_bucket,
        "Detector_Conf": round(detector_conf, 2),
        "Time_Taken_secs": time_taken,
        
    }

    save_to_csv(row)
    return annotated, row
