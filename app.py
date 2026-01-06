from flask import Flask, render_template, request, send_file
from inference import detect_skin_issue
import os

app = Flask(__name__)

UPLOAD_FOLDER = "static/uploads"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

@app.route("/", methods=["GET", "POST"])
def index():
    result = age = None
    original_image = predicted_image = None

    if request.method == "POST":
        file = request.files["image"]
        if file:
            os.makedirs(UPLOAD_FOLDER, exist_ok=True)
            path = os.path.join(UPLOAD_FOLDER, file.filename)
            file.save(path)

            label, conf, age, orig_img, pred_img, csv_path = detect_skin_issue(path)

            result = f"{label} ({conf*100:.2f}%)"
            original_image = "/" + orig_img
            predicted_image = "/" + pred_img

    return render_template(
        "index.html",
        result=result,
        age=age,
        original_image=original_image,
        predicted_image=predicted_image
    )

@app.route("/download_image")
def download_image():
    return send_file("static/outputs/result.jpg", as_attachment=True)

@app.route("/download_csv")
def download_csv():
    return send_file("static/outputs/result.csv", as_attachment=True)

if __name__ == "__main__":
    app.run(debug=True)
