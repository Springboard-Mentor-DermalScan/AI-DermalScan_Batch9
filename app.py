from flask import Flask, render_template, request, send_file
from inference import detect_skin_issue
import os
import time

app = Flask(__name__)

UPLOAD_FOLDER = "static/uploads"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER


@app.route("/", methods=["GET", "POST"])
def index():
    result = age = None
    original_image = predicted_image = None
    x1 = y1 = x2 = y2 = None
    confidence = None
    predicted_class = None
    execution_time = None

    if request.method == "POST":
        file = request.files["image"]
        if file:
            os.makedirs(UPLOAD_FOLDER, exist_ok=True)
            path = os.path.join(UPLOAD_FOLDER, file.filename)
            file.save(path)
            start_time = time.time()
            label,conf, age, orig_img, pred_img, csv_path, box_x1, box_y1, box_x2, box_y2 = detect_skin_issue(path)
            predicted_class = label
            end_time = time.time()
            execution_time = round(end_time-start_time, 2)

            result = f"{label} ({conf*100:.2f}%)"
            confidence = f"{conf*100:.2f}%"

            original_image = "/" + orig_img
            predicted_image = "/" + pred_img

            x1, y1, x2, y2 = box_x1, box_y1, box_x2, box_y2

    return render_template(
        "index.html",
        result=result,
        age=age,
        original_image=original_image,
        predicted_image=predicted_image,
        confidence=confidence,
	label=predicted_class,
        x1=x1, y1=y1, x2=x2, y2=y2,
        execution_time=execution_time
    )


@app.route("/download_image")
def download_image():
    return send_file("static/outputs/result.jpg", as_attachment=True)


@app.route("/download_csv")
def download_csv():
    return send_file("static/outputs/result.csv", as_attachment=True)


if __name__ == "__main__":
    app.run(debug=True)
