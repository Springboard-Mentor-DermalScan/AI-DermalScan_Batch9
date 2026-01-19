#Importing necessary libraries
import os
import cv2
import csv
import time
import io
import uuid
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from flask import Flask, render_template, request, jsonify, send_file, session
from tensorflow.keras.models import load_model


#Flask app
app = Flask(__name__)
app.secret_key = "dermalscan_session_secret"

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, "static", "uploads")
OUTPUT_FOLDER = os.path.join(BASE_DIR, "static", "outputs")

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)


#Loading the pre-trained model and defining constants
MODEL_PATH = "MobileNetV2_Model2_Final.h5"
IMG_SIZE = 224

model = load_model(MODEL_PATH)

class_labels = ["Clear Skin", "Dark Spots", "Puffy Eyes", "Wrinkles"]

age_ranges = {
    "Clear Skin": (18, 30),
    "Dark Spots": (25, 45),
    "Puffy Eyes": (20, 40),
    "Wrinkles": (40, 60)
}

#CSV columns for download
CSV_FIELDS = [
    "face_id",
    "filename",
    "box_x1",
    "box_y1",
    "box_x2",
    "box_y2",
    "class",
    "class_prob",
    "age_estimation",
    "detector_conf",
    "evaluation_time_sec"
]


#DNN Face Detector
PROTO = "deploy.prototxt"
WEIGHTS = "res10_300x300_ssd_iter_140000.caffemodel"
face_net = cv2.dnn.readNetFromCaffe(PROTO, WEIGHTS)


def preprocess_face(face):
    face = cv2.resize(face, (IMG_SIZE, IMG_SIZE))
    face = face / 255.0
    return np.expand_dims(face, axis=0)


# Routes
@app.route("/")
def index():
    session.clear()
    session["predictions"] = []
    session["next_face_id"] = 1
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    start_time = time.time()

    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    file = request.files["image"]
    filename = f"{uuid.uuid4().hex}.jpg"
    upload_path = os.path.join(UPLOAD_FOLDER, filename)
    file.save(upload_path)

    image = cv2.imread(upload_path)
    if image is None:
        return jsonify({"error": "Invalid image"}), 400

    (h, w) = image.shape[:2]

    # Face Detection
    blob = cv2.dnn.blobFromImage(
        cv2.resize(image, (300, 300)),
        1.0,
        (300, 300),
        (104.0, 177.0, 123.0)
    )

    face_net.setInput(blob)
    detections = face_net.forward()

    boxes, confidences = [], []

    for i in range(detections.shape[2]):
        conf = detections[0, 0, i, 2]
        if conf > 0.5:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            x1, y1, x2, y2 = box.astype("int")
            boxes.append([x1, y1, x2 - x1, y2 - y1])
            confidences.append(float(conf))

    if not boxes:
        return jsonify({"error": "No face detected"}), 400
    
    #Non-Maximum Suppression (NMS)
    indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(img_rgb)
    ax.axis("off")

    results = []
    face_id = session.get("next_face_id", 1)
    MAX_FACES = 3
    is_congested = len(indices) > MAX_FACES

    for idx in indices.flatten():
        x, y, bw, bh = boxes[idx]
        x1, y1 = max(0, x), max(0, y)
        x2, y2 = min(w, x + bw), min(h, y + bh)

        face = image[y1:y2, x1:x2]
        if face.size == 0:
            continue

        preds = model.predict(preprocess_face(face), verbose=0)[0]
        cls_idx = int(np.argmax(preds))
        label = class_labels[cls_idx]
        confidence = round(float(preds[cls_idx] * 100), 2)

        min_age, max_age = age_ranges[label]
        age = int(min_age + (confidence / 100) * (max_age - min_age))
        
        # Bounding box
        rect = plt.Rectangle(
            (x1, y1), x2 - x1, y2 - y1,
            linewidth=3, edgecolor="lime", facecolor="none"
        )
        ax.add_patch(rect)
        
        # Label text based on congestion
        label_text = f"#{face_id} {label}" if is_congested else f"{label}\nAge: {age} | {confidence}%"

        ax.text(
            x1, max(0, y1 - 10),
            label_text,
            fontsize=11 if is_congested else 12,
            color="black",
            bbox=dict(facecolor="lime", edgecolor="lime", boxstyle="round,pad=0.35")
        )

        results.append({
            "face_id": face_id,
            "filename": filename,
            "box_x1": x1,
            "box_y1": y1,
            "box_x2": x2,
            "box_y2": y2,
            "class": label,
            "class_prob": confidence,
            "age_estimation": age,
            "detector_conf": round(confidences[idx] * 100, 2),
            "evaluation_time_sec": None  # to be filled later
        })

        face_id += 1

    # Evaluation Time
    evaluation_time = float(round(time.time() - start_time, 3))
    for r in results:
        r["evaluation_time_sec"] = evaluation_time

    session["next_face_id"] = face_id
    session["predictions"].extend(results)
    session.modified = True

  
    output_path = os.path.join(OUTPUT_FOLDER, filename)
    plt.savefig(output_path, bbox_inches="tight", pad_inches=0.2)
    plt.close(fig)

    # Summary Response
    if len(results) == 1:
        r = results[0]
        response_text = (
            f"The predicted skin condition is <b>{r['class']}</b> "
            f"with a confidence score of <b>{r['class_prob']}%</b>. "
            f"The age is estimated to be <b>{r['age_estimation']}</b>."
        )
    else:
        summary = defaultdict(list)
        for r in results:
            summary[r["class"]].append((r["class_prob"], r["age_estimation"]))

        parts = []
        for cls, vals in summary.items():
            probs = [v[0] for v in vals]
            ages = [v[1] for v in vals]
            parts.append(
                f"<b>{len(vals)}</b> <b>{cls}</b> "
                f"(average confidence <b>{sum(probs)/len(probs):.1f}%</b>) "
                f"with an average age <b>{sum(ages)//len(ages)}</b>"
            )

        response_text = f"<b>Analysis Summary:</b><br>Detected <b>{len(results)}</b> faces: " + ", ".join(parts)
    
    for r in session["predictions"]:
        for k, v in r.items():
            if isinstance(v, np.generic):
                r[k] = v.item()


    return jsonify({
        "multi_face": len(results) > 1,
        "result_text": response_text,
        "evaluation_time": evaluation_time,
        "annotated_image": f"/static/outputs/{filename}",
        "download_url": f"/download_image/{filename}",
        "table": session["predictions"]
    })


# Download Routes
@app.route("/download_image/<filename>")
def download_image(filename):
    return send_file(os.path.join(OUTPUT_FOLDER, filename), as_attachment=True)


@app.route("/download_csv")
def download_csv():
    if not session.get("predictions"):
        return "No predictions available", 400

    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=CSV_FIELDS)
    writer.writeheader()
    writer.writerows(session["predictions"])
    output.seek(0)

    return send_file(
        io.BytesIO(output.getvalue().encode()),
        mimetype="text/csv",
        as_attachment=True,
        download_name="dermalscan_predictions.csv"
    )

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
