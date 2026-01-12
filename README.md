## ✨ DermalScan – AI Facial Skin & Age Detection
Deep Learning–Powered Skin Condition Analysis & Age Estimations



⭐ Project Overview

DermalScan is an AI-driven facial skin analysis system that detects skin conditions such as:

Clear Skin

Dark Spots

Puffy Eyes

Wrinkles

The system uses a MobileNetV2 CNN classifier, a simple bounding-box-based facial cropping approach, and a fully integrated Flask web UI that allows users to:

✔ Upload an image

✔ View annotated prediction with bounding box

✔ Get estimated age

✔ Download annotated image

✔ Download structured CSV logs

This project is structured according to an 8-week milestone plan, covering dataset preparation, training, UI development, backend integration, export options, and final documentation.





brijesh/data

│── app.py                     # Flask backend

│── inference.py               # Model loading + preprocessing + prediction

│── templates/

│     └── index.html           # UI Template

│── static/

│     ├── uploads/             # User-uploaded images

│     └── outputs/             # Annotated results + CSV logs

│── skin_classifier_mobilenetv2.h5  # Trained model

│── requirements.txt

│── README.md





🧠 Core Features
🔍 1. Skin Condition Classification

MobileNetV2 CNN model

Predicts one of 4 classes

Outputs confidence score (%)






📦 2. Age Estimation

Class-based artificial age buckets:

Class	Age Range
Clear Skin	22–28
Dark Spot	30–38
Puffy Eyes	35–45
Wrinkles	70–85




85
🎯 3. Bounding Box Annotation

Fixed box covering central face region

Prediction text: Class (Confidence%) | Age: ##

Saved as static/outputs/result.jpg

📊 4. CSV Export

Saved as result.csv with:

filename

box coordinates

predicted class

confidence

age bucket

🌐 5. Full Web Interface (Flask)

Upload facial image

Preview original & annotated results

Display table summary (class, confidence, box coords, age)

Download button for image & CSV




🚀 Installation

1. Clone Repository

git clone https://github.com/your-username/Brijesh-Rath.git
cd Brijesh-Rath

3. Install Dependencies

pip install -r requirements.txt















