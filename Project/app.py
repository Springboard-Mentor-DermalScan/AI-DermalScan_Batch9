import os
import cv2
import csv
import random
import numpy as np
from flask import Flask, render_template, request, send_file
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.efficientnet import preprocess_input

# ------------------ Flask Setup ------------------
app = Flask(__name__)

UPLOAD_FOLDER = "static/uploads"
RESULT_FOLDER = "static/results"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)

# ------------------ Load Model ------------------
model = load_model("best_dermalscan_model.h5")
CLASS_LABELS = ["clear skin", "dark spots", "puffy eyes", "wrinkles"]

# ------------------ Load DNN Face Detector ------------------
face_net = cv2.dnn.readNetFromCaffe(
    "deploy.prototxt",
    "res10_300x300_ssd_iter_140000.caffemodel"
)

# ------------------ Age Generator ------------------
def generate_age(label):
    age_map = {
        "clear skin": (18, 25),
        "dark spots": (26, 40),
        "puffy eyes": (25, 45),
        "wrinkles": (35, 60)
    }
    return random.randint(*age_map.get(label, (20, 40)))

# ------------------ Global Results ------------------
LAST_RESULTS = []

# ------------------ Face Detection ------------------
def detect_faces(image, threshold=0.5):
    h, w = image.shape[:2]
    blob = cv2.dnn.blobFromImage(
        cv2.resize(image, (300, 300)),
        1.0,
        (300, 300),
        (104.0, 177.0, 123.0)
    )
    face_net.setInput(blob)
    detections = face_net.forward()

    faces = []
    for i in range(detections.shape[2]):
        conf = detections[0, 0, i, 2]
        if conf > threshold:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            x1, y1, x2, y2 = box.astype(int)
            faces.append((x1, y1, x2 - x1, y2 - y1))
    return faces

# ------------------ Main Route ------------------
@app.route("/", methods=["GET", "POST"])
def index():
    global LAST_RESULTS
    LAST_RESULTS = []
    image_path = None

    if request.method == "POST":
        file = request.files.get("image")
        if file:
            input_path = os.path.join(UPLOAD_FOLDER, file.filename)
            file.save(input_path)

            image = cv2.imread(input_path)
            faces = detect_faces(image)

            if len(faces) == 0:
                LAST_RESULTS.append(["—", "No face detected", "0%", "—"])
                cv2.putText(
                    image, "No face detected",
                    (30, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9, (0, 0, 255), 2
                )
            else:
                for idx, (x, y, w, h) in enumerate(faces, start=1):
                    face = image[y:y+h, x:x+w]
                    face = cv2.resize(face, (224, 224))
                    face = img_to_array(face)
                    face = np.expand_dims(face, axis=0)
                    face = preprocess_input(face)

                    preds = model.predict(face, verbose=0)[0]
                    top_idx = np.argmax(preds)
                    confidence = preds[top_idx] * 100
                    label = CLASS_LABELS[top_idx]
                    age = generate_age(label)

                    display_label = label if confidence >= 30 else f"Uncertain ({label})"

                    LAST_RESULTS.append([
                        f"Face {idx}",
                        display_label,
                        f"{confidence:.2f}%",
                        age
                    ])

                    # ---------- Draw Face Box ----------
                    cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)

                    # ---------- Face ID (INSIDE BOX) ----------
                    cv2.putText(
                        image,
                        f"Face {idx}",
                        (x + 6, y + 18),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.55,
                        (0, 255, 0),
                        2,
                        cv2.LINE_AA
                    )

                    # ---------- Text Block ----------
                    line1 = f"{display_label} | {confidence:.1f}%"
                    line2 = f"Age: {age}"

                    font = cv2.FONT_HERSHEY_SIMPLEX
                    scale1, thick1 = 0.55, 2
                    scale2, thick2 = 0.5, 2

                    (w1, h1), _ = cv2.getTextSize(line1, font, scale1, thick1)
                    (w2, h2), _ = cv2.getTextSize(line2, font, scale2, thick2)

                    block_h = h1 + h2 + 12
                    block_w = max(w1, w2)

                    tx = x
                    ty = y - 12

                    if ty - block_h < 0:
                        ty = y + h + block_h + 10

                    if tx + block_w > image.shape[1]:
                        tx = image.shape[1] - block_w - 10

                    cv2.rectangle(
                        image,
                        (tx - 4, ty - block_h - 6),
                        (tx + block_w + 6, ty + 6),
                        (0, 0, 0),
                        -1
                    )

                    cv2.putText(
                        image, line1,
                        (tx, ty - h2 - 6),
                        font, scale1, (0, 0, 255),
                        thick1, cv2.LINE_AA
                    )

                    cv2.putText(
                        image, line2,
                        (tx, ty),
                        font, scale2, (0, 180, 0),
                        thick2, cv2.LINE_AA
                    )

            output_path = os.path.join(RESULT_FOLDER, "result.jpg")
            cv2.imwrite(output_path, image)
            image_path = output_path

    return render_template("index.html", image=image_path, results=LAST_RESULTS)

# ------------------ Downloads ------------------
@app.route("/download_image")
def download_image():
    return send_file(os.path.join(RESULT_FOLDER, "result.jpg"), as_attachment=True)

@app.route("/download_csv")
def download_csv():
    csv_path = os.path.join(RESULT_FOLDER, "results.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Face", "Class", "Confidence", "Age"])
        writer.writerows(LAST_RESULTS)
    return send_file(csv_path, as_attachment=True)

# ------------------ Run ------------------
if __name__ == "__main__":
    app.run(debug=True)
