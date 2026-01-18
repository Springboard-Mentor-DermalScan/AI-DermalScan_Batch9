from flask import Flask, render_template, request, send_file
import cv2
import os
from backend import process_image, save_to_csv
from csv_report import csv_to_pdf

app = Flask(__name__)

UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

history = []

@app.route("/", methods=["GET", "POST"])
def index():
    global history

    current = None
    original_image = None
    annotated_image = None

    if request.method == "POST":
        file = request.files["image"]
        if file:
            original_path = os.path.join(UPLOAD_FOLDER, "original_" + file.filename)
            annotated_path = os.path.join(UPLOAD_FOLDER, "annotated_" + file.filename)

            file.save(original_path)
            image = cv2.imread(original_path)

            annotated, row = process_image(image, file.filename)

            if row is not None:
                cv2.imwrite(annotated_path, annotated)
                

                history.insert(0, row)
                history = history[:5]

                current = row
                original_image = original_path
                annotated_image = annotated_path

    return render_template(
        "index.html",
        current=current,
        previous=history[1:],
        original_image=original_image,
        annotated_image=annotated_image
    )

@app.route("/download_csv")
def download_csv():
    pdf_path = csv_to_pdf("output/predictions.csv")
    return send_file(pdf_path, as_attachment=True)

@app.route("/download_image")
def download_image():
    return send_file("static/uploads", as_attachment=True)


if __name__ == "__main__":
    app.run(debug=True)

