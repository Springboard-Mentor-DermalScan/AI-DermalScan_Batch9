import cv2
import numpy as np
import tensorflow as tf
import os
import pandas as pd
import random

model = tf.keras.models.load_model("skin_classifier_mobilenetv2.h5")

CLASSES = ["Clear Skin", "Dark Spot", "Puffy Eyes", "Wrinkles"]

def detect_skin_issue(image_path):
    img = cv2.imread(image_path)
    h, w, _ = img.shape

    resized = cv2.resize(img, (224, 224)) / 255.0
    pred = model.predict(np.expand_dims(resized, axis=0), verbose=0)[0]

    class_id = int(np.argmax(pred))
    confidence = float(pred[class_id])
    label = CLASSES[class_id]

    if label == "Clear Skin":
        age = random.randint(22, 28)
        age_bucket = "22-28"
    elif label == "Dark Spot":
        age = random.randint(30, 38)
        age_bucket = "30-38"
    elif label == "Puffy Eyes":
        age = random.randint(40, 48)
        age_bucket = "40-48"
    else:
        age = random.randint(70, 85)
        age_bucket = "70-85"

    x1, y1 = int(w * 0.15), int(h * 0.15)
    x2, y2 = int(w * 0.85), int(h * 0.85)

    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 3)
    text = f"{label} ({confidence*100:.2f}%) | Age: {age}"
    cv2.putText(img, text, (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)

    os.makedirs("static/outputs", exist_ok=True)
    output_image = "static/outputs/result.jpg"
    cv2.imwrite(output_image, img)

    csv_path = "static/outputs/result.csv"
    df = pd.DataFrame([{
        "filename": os.path.basename(image_path),
	"box_x1":x1,
	"box_y1":y1,
	"box_x2":x2,
	"box_y2":y2,
        "class": label,
        "confidence": round(confidence, 4),
        "age": age,
        "age_bucket": age_bucket
    }])
    df.to_csv(csv_path, index=False)

    return label, confidence, age, image_path, output_image, csv_path, x1, y1, x2, y2
