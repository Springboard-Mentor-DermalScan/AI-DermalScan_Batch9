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








