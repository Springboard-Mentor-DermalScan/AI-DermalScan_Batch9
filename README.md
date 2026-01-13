🧠 DermalScan: AI Facial Skin Aging Detection App

DermalScan is an AI-powered web application that detects facial skin aging conditions from images. The system automatically identifies faces, predicts skin conditions such as clear skin, dark spots, puffy eyes, and wrinkles, estimates an approximate age, and visualizes results with confidence scores using deep learning and computer vision techniques.

This project integrates CNN-based deep learning models, face detection, and a responsive web interface to deliver fast and accurate predictions suitable for real-world usage.

🚀 Key Features

✅ Multi-face detection from a single image

✅ AI-based skin condition classification

✅ Confidence percentage for each prediction

✅ Approximate age estimation based on skin condition

✅ Annotated image with bounding boxes and labels

✅ Download annotated image

✅ Export predictions as CSV file

✅ Responsive web UI (HTML + CSS + Flask)

✅ End-to-end inference pipeline

✅ Processing time ≤ 5 seconds per image

🧠 Skin Conditions Detected

Clear Skin

Dark Spots

Puffy Eyes

Wrinkles

🏗️ System Architecture (High Level)

User uploads an image through the web interface

Backend detects faces using OpenCV DNN face detector

Each detected face is cropped and preprocessed

Trained CNN model (EfficientNetB0) predicts skin condition

Confidence score and age are generated

Results are visualized and made downloadable

🧪 Model Details

The skin condition classifier was trained using a public facial skin dataset with four classes.

📊 Model Performance Comparison
Model	Training Accuracy	Validation Accuracy
EfficientNetB0	90.36%	95.63%
MobileNetV2	94.69%	87.92%

✔ Final model used: EfficientNetB0
✔ Selected for better generalization and validation performance

📌 The trained model file is not uploaded to GitHub.
Instead, a Jupyter Notebook demonstrating the full training process is included.

🧰 Tools & Technologies
Languages

Python

Libraries & Frameworks

TensorFlow / Keras

OpenCV

NumPy

Pandas

Matplotlib

Pillow

SciPy

OS

Web Technologies

Flask

HTML

CSS

Platforms

Jupyter Notebook

VS Code

GitHub

📁 Project Structure
DermalScan/
│
├── app.py                    # Flask backend
├── templates/
│   └── index.html            # Frontend UI
│
├── static/
│   ├── uploads/              # Uploaded images
│   └── results/              # Annotated outputs
│
├── notebooks/
│   └── model_training.ipynb  # Model training notebook
│
├── requirements.txt
└── README.md

🖥️ How to Run This Project Locally

Follow these steps carefully 👇

1️⃣ Download the Project

Option A: Download ZIP

Go to the GitHub repository

Click Code → Download ZIP

Extract the ZIP file

Option B: Clone Repository

git clone <repository-url>
cd DermalScan

2️⃣ Create Virtual Environment (Recommended)
python -m venv venv


Activate it:

Windows

venv\Scripts\activate


Mac/Linux

source venv/bin/activate

3️⃣ Install Dependencies
pip install -r requirements.txt

4️⃣ Ensure Required Files

Make sure the following files are present:

deploy.prototxt

res10_300x300_ssd_iter_140000.caffemodel

Trained model file (place inside project root)

⚠️ The trained model (best_dermalscan_model.h5) should be added manually if not included.

5️⃣ Run the Application
python app.py

6️⃣ Open in Browser

Visit:

http://127.0.0.1:5000


Upload an image and view the results 🎉

📸 Output Description

Annotated image with:

Face ID (Face 1, Face 2, …)

Skin condition

Confidence percentage

Age estimate

Detailed analysis table:

Bounding box coordinates (x1, y1, x2, y2)

Download options:

Annotated image

CSV prediction report

📊 Final Outcome

DermalScan successfully demonstrates an end-to-end AI system for facial skin aging detection with:

High accuracy

Fast inference

Clean UI

Exportable results

The system is demo-ready, scalable, and suitable for academic and real-world applications.

📌 Future Enhancements

Real age regression model

Skin segmentation for finer analysis

Mobile-friendly deployment

Cloud hosting (AWS / Azure)
