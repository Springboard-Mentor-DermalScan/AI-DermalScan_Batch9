import os
import cv2
import csv
import io
import uuid
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from flask import Flask, render_template, request, jsonify, send_file, session
from tensorflow.keras.models import load_model

#flask app 
app = Flask(__name__)
app.secret_key = "dermalscan_session_secret"

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, "static", "uploads")
OUTPUT_FOLDER = os.path.join(BASE_DIR, "static", "outputs")

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

#Loading model and face detector 
MODEL_PATH = "MobileNetV2_Model2_Final.h5"
CASCADE_PATH = "haarcascade_frontalface_default.xml"
IMG_SIZE = 224

model = load_model(MODEL_PATH)
face_cascade = cv2.CascadeClassifier(CASCADE_PATH)

class_labels = ["Clear Skin", "Dark Spots", "Puffy Eyes", "Wrinkles"]

age_ranges = {
	"Clear Skin": (18, 30),
	"Dark Spots": (25, 45),
	"Puffy Eyes": (20, 40),
	"Wrinkles": (40, 60)
}

#Preprocess face for model prediction
def preprocess_face(face):
	face = cv2.resize(face, (IMG_SIZE, IMG_SIZE))
	face = face / 255.0
	return np.expand_dims(face, axis=0)

#app routes
@app.route("/")
def index():
	# 🔥 Start fresh on new page load
	session["predictions"] = []
	return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():

	if "image" not in request.files:
		return jsonify({"error": "No image uploaded"}), 400

	file = request.files["image"]
	filename = f"{uuid.uuid4().hex}.jpg"

	upload_path = os.path.join(UPLOAD_FOLDER, filename)
	file.save(upload_path)

	img = cv2.imread(upload_path)
	if img is None:
		return jsonify({"error": "Invalid image"}), 400

	img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

	faces = face_cascade.detectMultiScale(
		gray, 1.1, 4, minSize=(100, 100)
	)

	if len(faces) == 0:
		return jsonify({"error": "No face detected"}), 400

	x, y, w, h = faces[0]

	
	pad = 20
	x = max(0, x - pad)
	y = max(0, y - pad)
	w = min(img.shape[1] - x, w + 2 * pad)
	h = min(img.shape[0] - y, h + 2 * pad)

	face = img[y:y+h, x:x+w]
	preds = model.predict(preprocess_face(face), verbose=0)[0]

	idx = int(np.argmax(preds))
	label = class_labels[idx]
	confidence = float(preds[idx] * 100)
	is_confident = confidence >= 50.0
	confidence_status = "CORRECT" if is_confident else "WRONG"
	confidence_note = (
	"Prediction confidence is low. Please upload a clearer image." if not is_confident else "Prediction confidence is sufficient."
)


	min_age, max_age = age_ranges[label]
	predicted_age = int(min_age + (confidence / 100) * (max_age - min_age))

	#Using matplotlib to draw bounding box and label
	fig, ax = plt.subplots(figsize=(6, 6))
	ax.imshow(img_rgb)
	ax.axis("off")

	rect = plt.Rectangle(
		(x, y),
		w,
		h,
		linewidth=3,
		edgecolor="lime",
		facecolor="none"
	)
	ax.add_patch(rect)

	label_text = f"{label}\nAge: {predicted_age} | {confidence:.2f}%"

	ax.text(
		x,
		max(0, y - 10),
		label_text,
		fontsize=12,
		color="black",
		verticalalignment="top",
		bbox=dict(
			facecolor="lime",
			edgecolor="lime",
			boxstyle="round,pad=0.35"
		)
	)

	output_path = os.path.join(OUTPUT_FOLDER, filename)
	plt.savefig(output_path, bbox_inches="tight", pad_inches=0.1)
	plt.close(fig)

	#Storing prediction in a session
	if "predictions" not in session:
		session["predictions"] = []

	session["predictions"].append({
	"filename": filename,
	"box_x1": int(x),
	"box_y1": int(y),
	"box_x2": int(x + w),
	"box_y2": int(y + h),
	"class": label,
	"class_prob": round(confidence, 2),
	"age_estimation": predicted_age,
	"detector_conf": round(confidence, 2),
	"status": confidence_status   # ✅ NEW
  })


	session.modified = True

	return jsonify({
	"label": label,
	"confidence": round(confidence, 2),
	"age": predicted_age,
	"status": confidence_status,        # ✅ NEW
	"note": confidence_note,             # ✅ NEW
	"annotated_image": f"/static/outputs/{filename}",
	"download_url": f"/download_image/{filename}",
	"table": session["predictions"]
})


@app.route("/download_image/<filename>")
def download_image(filename):
	return send_file(
		os.path.join(OUTPUT_FOLDER, filename),
		as_attachment=True
	)

@app.route("/download_csv")
def download_csv():

	if "predictions" not in session or not session["predictions"]:
		return "No predictions available", 400

	output = io.StringIO()
	writer = csv.DictWriter(
		output,
		fieldnames=session["predictions"][0].keys()
	)
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
	app.run(debug=True)
