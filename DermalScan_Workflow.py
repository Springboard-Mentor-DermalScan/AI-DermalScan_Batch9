"""
DermalScan - Module 4
Skin Problem Detection + Age Estimation + Face Highlightingx`
"""

import cv2
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model


IMG_SIZE = 224
CLASS_NAMES = ["clear skin", "dark spots", "puffy eyes", "wrinkles"]


MODEL_PATH = "DermalScan_Final_Model.h5"


DATASET_PATH = r"C:\Users\akhta\Downloads\Downloads 2025\Self Development\Projects\DermalScan\Dermal Scan Main\dataset_clean"


print("\n-------------------------------")
print(" Loading DermalScan Model...")
print("-------------------------------")

model = load_model(MODEL_PATH)
print(" Model Loaded Successfully!")


face_detector = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)



def estimate_age(label):
    if label == "clear skin":
        return "18 to 25"
    elif label == "puffy eyes":
        return "25 to 35"
    elif label == "dark spots":
        return "30 to 45"
    else:
        return "40 to 60"


def preprocess_face(face):
    face = cv2.resize(face, (IMG_SIZE, IMG_SIZE))
    face = face.astype("float32") / 255.0
    face = np.expand_dims(face, axis=0)
    return face



def tta_prediction(face):
    preds = []

    # Original
    preds.append(model.predict(face, verbose=0)[0])

    # Flipped
    flipped = np.flip(face, axis=2)
    preds.append(model.predict(flipped, verbose=0)[0])

    # Brightness Adjustment
    bright = np.clip(face + 0.1, 0, 1)
    preds.append(model.predict(bright, verbose=0)[0])

    return np.mean(preds, axis=0)



def predict_skin(image_path):
    img = cv2.imread(image_path)

    if img is None:
        print("❌ Image not found:", image_path)
        return

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_detector.detectMultiScale(gray, 1.3, 5)

    if len(faces) == 0:
        print("⚠ No Face Detected — Using Whole Image")
        faces = [(0, 0, img.shape[1], img.shape[0])]

    for (x, y, w, h) in faces:
        face = img[y:y+h, x:x+w]

        processed = preprocess_face(face)

       
        preds = tta_prediction(processed)

        index = np.argmax(preds)
        confidence = preds[index] * 100
        condition = CLASS_NAMES[index]
        age = estimate_age(condition)

      
     
        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 3)

       
        header_y1 = max(y - 40, 0)
        header_y2 = y
        cv2.rectangle(img, (x, header_y1), (x+w, header_y2), (0, 0, 0), -1)

        display_text = f"{condition} ({confidence:.2f}%) | Age: {age}"

        cv2.putText(img,
            display_text,
            (x + 5, header_y1 + 27),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2)


        print("\nPrediction Result")
        print("---------------------------")
        print("Skin Problem:", condition)
        print("Confidence:", f"{confidence:.2f}%")
        print("Estimated Age:", age)

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    cv2.imshow("DermalScan Result", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()



def pick_random_image():
    import random

    cls = random.choice(CLASS_NAMES)
    folder = os.path.join(DATASET_PATH, cls)

    img_name = random.choice(os.listdir(folder))
    return os.path.join(folder, img_name)


if __name__ == "__main__":
    test_image = pick_random_image()
    print("\nTesting Image:", test_image)

    predict_skin(test_image)
