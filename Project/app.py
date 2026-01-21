import os
import cv2
import csv
import random
import time
import numpy as np
import tensorflow as tf
from flask import Flask, render_template, request, send_file
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.efficientnet import preprocess_input

# -------- TensorFlow Optimization --------
tf.config.optimizer.set_jit(True)

# -------- Flask Setup --------
app = Flask(__name__)

UPLOAD_FOLDER = "static/uploads"
RESULT_FOLDER = "static/results"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)

# -------- Load Model --------
model = load_model("model.h5")
CLASS_LABELS = ["clear skin", "dark spots", "puffy eyes", "wrinkles"]

# -------- Load Face Detector --------
face_net = cv2.dnn.readNetFromCaffe(
    "deploy.prototxt",
    "res10_300x300_ssd_iter_140000.caffemodel"
)

# -------- Age Generator --------
def generate_age(label):
    age_map = {
        "clear skin": (18, 30),
        "dark spots": (26, 40),
        "puffy eyes": (25, 45),
        "wrinkles": (60, 80)
    }
    return random.randint(*age_map.get(label, (20, 40)))

# -------- Resize Image (Speed + Consistency) --------
def resize_image(image, target_width=900):
    h, w = image.shape[:2]
    if w > target_width:
        scale = target_width / w
        image = cv2.resize(image, (target_width, int(h * scale)))
    return image

def letterbox_image(image, size=(900, 900), color=(245, 245, 245)):
    h, w = image.shape[:2]
    target_w, target_h = size

    scale = min(target_w / w, target_h / h)
    new_w = int(w * scale)
    new_h = int(h * scale)

    resized = cv2.resize(image, (new_w, new_h))

    canvas = np.full((target_h, target_w, 3), color, dtype=np.uint8)

    x_offset = (target_w - new_w) // 2
    y_offset = (target_h - new_h) // 2

    canvas[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized
    return canvas

# -------- Face Detection --------
def detect_faces(image, threshold=0.5):
    h, w = image.shape[:2]
    blob = cv2.dnn.blobFromImage(
        image, 1.0, (300, 300),
        (104.0, 177.0, 123.0),
        swapRB=False, crop=False
    )
    face_net.setInput(blob)
    detections = face_net.forward()

    faces = []
    for i in range(detections.shape[2]):
        conf = detections[0, 0, i, 2]
        if conf > threshold:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            x1, y1, x2, y2 = box.astype(int)
            if (x2 - x1) > 60 and (y2 - y1) > 60:
                faces.append((x1, y1, x2, y2))
    return faces

# -------- Globals --------
LAST_RESULTS = []
SUMMARY = {}

# -------- Main Route --------
@app.route("/", methods=["GET", "POST"])
def index():
    global LAST_RESULTS, SUMMARY
    LAST_RESULTS = []
    SUMMARY = {}
    image_path = None

    if request.method == "POST":
        start_time = time.time()
        file = request.files.get("image")

        if file:
            input_path = os.path.join(UPLOAD_FOLDER, file.filename)
            file.save(input_path)

            image = cv2.imread(input_path)
            image = resize_image(image)

            faces = detect_faces(image)

            if len(faces) == 0:
                LAST_RESULTS.append(
                    ["—", "No face detected", "0%", "—", "-", "-", "-", "-"]
                )
                cv2.putText(
                    image, "No face detected",
                    (30, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9, (0, 0, 255), 2
                )

            else:
                for idx, (x1, y1, x2, y2) in enumerate(faces, start=1):
                    face_crop = image[y1:y2, x1:x2]
                    face_crop = cv2.resize(face_crop, (224, 224))
                    face_arr = img_to_array(face_crop)
                    face_arr = np.expand_dims(face_arr, axis=0)
                    face_arr = preprocess_input(face_arr)

                    preds = model.predict(face_arr, verbose=0)[0]
                    top_idx = np.argmax(preds)
                    confidence = preds[top_idx] * 100
                    label = CLASS_LABELS[top_idx]
                    age = generate_age(label)

                    LAST_RESULTS.append([
                        f"Face {idx}",
                        label,
                        f"{confidence:.2f}%",
                        age,
                        x1, y1, x2, y2
                    ])

                    # ---- Draw Box ----
                    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

                    # ---- Face ID ----
                    cv2.putText(
                        image, f"Face {idx}",
                        (x1 + 6, y1 + 18),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (0, 255, 0), 2
                    )

                    # ---- Info Block ----
                    text1 = f"{label} | {confidence:.1f}%"
                    text2 = f"Age: {age}"

                    font = cv2.FONT_HERSHEY_SIMPLEX
                    s1, t1 = 0.6, 2
                    s2, t2 = 0.55, 2

                    (tw1, th1), _ = cv2.getTextSize(text1, font, s1, t1)
                    (tw2, th2), _ = cv2.getTextSize(text2, font, s2, t2)

                    block_w = max(tw1, tw2)
                    block_h = th1 + th2 + 14

                    tx = x1
                    ty = y1 - 10
                    if ty - block_h < 0:
                        ty = y2 + block_h + 10

                    if tx + block_w > image.shape[1]:
                        tx = image.shape[1] - block_w - 10

                    cv2.rectangle(
                        image,
                        (tx - 6, ty - block_h),
                        (tx + block_w + 6, ty + 6),
                        (0, 0, 0),
                        -1
                    )

                    cv2.putText(image, text1, (tx, ty - th2 - 6),
                                font, s1, (0, 0, 255), t1)
                    cv2.putText(image, text2, (tx, ty),
                                font, s2, (0, 200, 0), t2)

                # ---- Multi-Face Summary ----
                SUMMARY = {
                    "total_faces": len(LAST_RESULTS),
                    "faces": [],
                    "time": f"{time.time() - start_time:.2f} sec"
                }

                for r in LAST_RESULTS:
                    SUMMARY["faces"].append({
                        "id": r[0],
                        "class": r[1],
                        "confidence": r[2],
                        "age": r[3]
                    })

            image = letterbox_image(image, size=(900, 900))
            output_path = os.path.join(RESULT_FOLDER, "result.jpg")
            cv2.imwrite(output_path, image)
            image_path = output_path

    return render_template(
        "index.html",
        image=image_path,
        results=LAST_RESULTS,
        summary=SUMMARY
    )

# -------- Downloads --------
@app.route("/download_image")
def download_image():
    return send_file(os.path.join(RESULT_FOLDER, "result.jpg"), as_attachment=True)

@app.route("/download_csv")
def download_csv():
    csv_path = os.path.join(RESULT_FOLDER, "results.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Face", "Class", "Confidence", "Age", "x1", "y1", "x2", "y2"])
        writer.writerows(LAST_RESULTS)
    return send_file(csv_path, as_attachment=True)

# -------- Run --------
if __name__ == "__main__":
    app.run(debug=True)
