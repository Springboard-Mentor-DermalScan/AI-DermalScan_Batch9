<p align="center">
  <img src="https://img.shields.io/badge/DL-Classification%20|%20CNN-blue" />
  <img src="https://img.shields.io/badge/Python-3.10+-yellow" />
  <img src="https://img.shields.io/badge/Jupyter-Notebook-orange" />
  <img src="https://img.shields.io/badge/Status-Completed-success" />
</p>



## ✨ DermalScan – AI Facial Skin & Age Detection
Deep Learning–Powered Skin Condition Analysis & Age Estimations



## ⭐ Project Overview

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



## 📑 Schema

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





## 🧠 Core Features

Skin Condition Classification

MobileNetV2 CNN model

Predicts one of 4 classes

Outputs confidence score (%)






## 📦 Age Estimation

Class-based artificial age buckets:

Class	Age Range

Clear Skin	22–28

Dark Spot	30–38

Puffy Eyes	35–45

Wrinkles	70–85




## 🎯 Bounding Box Annotation

Fixed box covering central face region

Prediction text: Class (Confidence%) | Age: ##

Saved as static/outputs/result.jpg





## 📊 CSV Export

Saved as result.csv with:

filename

box coordinates

predicted class

confidence

age bucket






## 🌐 Full Web Interface (Flask)

Upload facial image

Preview original & annotated results

Display table summary (class, confidence, box coords, age)

Download button for image & CSV





### 🚀 Installation

### 1️⃣ Clone Repository
```bash
git clone -b Brijesh-Rath https://github.com/Springboard-Mentor-DermalScan/AI-DermalScan_Batch9.git
```

### 2️⃣ Create Virtual Environment
```bash
python -m venv venv
venv\Scripts\activate
```

### 3️⃣ Install Dependencies
```bash
pip install -r requirements.txt
```

### 4️⃣ Run Application
```bash
python app.py
```






## 🏗️ Project Workflow (Backend + UI Flow)

flowchart LR

    subgraph U[User Layer]
        U1([User])
        U2[Upload Image]
        U1 --> U2
    end

    subgraph F[Frontend Layer]
        F1[Streamlit UI]
        F2[Send Raw Image Bytes]
        F1 --> F2
    end

    subgraph B[Backend Processing Layer]
        B1[Read Image<br/>OpenCV]
        B2[Resize & Normalize]
        B3[Face Detection<br/>Haar Cascade]
        B4[Extract Face ROI]
        B1 --> B2 --> B3 --> B4
    end

    subgraph M[Model & Output Layer]
        M1[Model Prediction]
        M2[Assign Age Bucket]
        M3[Draw Bounding Box<br/>+ Label]
        M4[Save Annotated Image]
        M5[Generate CSV Log]
        M1 --> M2 --> M3
        M3 --> M4
        M3 --> M5
    end

    U2 --> F1
    F2 --> B1
    B4 --> M1








## 📊 Model Performance Summary

Model Used: MobileNetV2 (Transfer Learning)

Input Size: 224x224

Optimizer: Adam

Loss: Categorical Crossentropy

Accuracy Achieved: >90% on validation dataset






## 🧪 Technologies Used

## Category	Tools

Programming Language	Python

DL Framework	TensorFlow / Keras

Computer Vision	OpenCV

Web Framework	Flask

Frontend	HTML, CSS

Data Logging	Pandas (CSV Export)





## 🛠 Requirements (for running the project)

python==3.13

tensorflow==2.15.0

keras==2.15.0

numpy==1.26.4

opencv-python==4.9.0.80

matplotlib==3.8.2

scikit-learn==1.4.0

pillow==10.2.0

notebook==7.1.0





## 🎨 User Interface Preview

✔ Dual image frame (original + annotated)

✔ Results table (class, confidence, age, box coords)

✔ Stylish gradient theme

✔ Neon cyan borders

✔ Fully responsive





## 🔮 Future Scope

Integrate real facial landmark detection (dlib or mediapipe)

Add multi-class multi-region detection (cheeks, forehead, eye bags)

Deploy model as REST API on cloud (GCP / AWS / Azure)

Replace MobilenetV2 with EfficientNetB0 or ViT for higher accuracy

Add real-time camera streaming mode

Add multi-skin-type dataset for dermatology-grade predictions

Build a mobile app using Flutter or React Native





## 🏁 Final Conclusion

DermalScan successfully meets its goal of delivering a fast, lightweight, and accurate AI-based skin and age detection system. The project smoothly integrates deep learning, computer vision, and web development, providing:

Reliable predictions

Clean and modern UI

Easy export & logging

Fully documented workflow

The system is flexible enough to be extended into a full dermatology assistant, skincare analysis app, or age estimation tool.





## 🙌 **Acknowledgments**
This project is created for academic learning and practical understanding of deep learning in healthcare.

---

👨‍💻 Author


Brijesh Rath


📧 Email: rathbrijesh2006@gmail.com


💼 GitHub: (https://github.com/Brijeshrath67)

--















