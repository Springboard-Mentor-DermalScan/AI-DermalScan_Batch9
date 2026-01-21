# 🧴 DermalScan – AI-Based Facial Skin Analysis System

DermalScan is a deep learning–based application that analyzes facial images to detect skin aging conditions such as **wrinkles, dark spots, puffy eyes, and clear skin**.  
The system uses a **fine-tuned MobileNetV2 model** and provides results through an interactive **Streamlit web interface**.

---

## 📸 Project Preview

### 🔹 Web Interface
![AI DermalScan Home](assets/preview.png)

### 🔹 Prediction Output
![Prediction Result](assets/result.png)

### 🔹 CSV Report Download
![Prediction Report](assets/report.png)

---

## 📌 Project Objective

- Detect facial skin conditions from uploaded images
- Classify skin type using a CNN model (MobileNetV2)
- Predict confidence percentage and estimated age
- Display annotated output images
- Allow users to download results (image + CSV)

---

## 🧠 Technologies Used

- Python 3.10
- TensorFlow / Keras
- MobileNetV2 (Transfer Learning + Fine-Tuning)
- OpenCV
- NumPy, Pandas
- Streamlit (Frontend UI)

---

## 💻 Tech Stack

![Python](https://img.shields.io/badge/Python-3.10-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15-orange)
![Keras](https://img.shields.io/badge/Keras-Deep%20Learning-red)
![MobileNetV2](https://img.shields.io/badge/MobileNetV2-CNN-success)
![OpenCV](https://img.shields.io/badge/OpenCV-Computer%20Vision-green)
![NumPy](https://img.shields.io/badge/NumPy-Array-blue)
![Pandas](https://img.shields.io/badge/Pandas-Data%20Analysis-purple)
![Matplotlib](https://img.shields.io/badge/Matplotlib-Visualization-yellow)
![Streamlit](https://img.shields.io/badge/Streamlit-Web%20UI-ff4b4b)

---

## 🗂 Dataset Details

- Classes:
  - Clear Skin
  - Dark Spots
  - Puffy Eyes
  - Wrinkles
- Dataset split:
  - Train
  - Validation
  - Test
- Image size: **224 × 224**
- Labels encoded using **one-hot encoding**

---


## 📁 Project Structure

AI-DermalScan/
│
├── backend/
│ ├── inference.py
│ └── models/
│ └── MobileNetV2_Module3_finetuned.h5
│
├── frontend/
│ ├── app.py
│ ├── uploads/
│ └── outputs/
│
├── Dataset/
│ ├── train/
│ ├── val/
│ └── test/
│
├── notebooks/
│ └── AI_DermalScan_Project.ipynb
│
├── results/
├── requirements.txt
└── README.md

---


## ⚙️ Installation & Setup

### 1️⃣ Clone Repository
```bash
git clone https://github.com/Springboard-Mentor-DermalScan/AI-DermalScan_Batch9.git
cd AI-DermalScan_Batch9
```

### 2️⃣ Create Virtual Environment
```bash
python -m venv env
env\Scripts\activate
```

### 3️⃣  Install Dependencies
```bash
pip install -r requirements.txt
```

### 4️⃣ Run Streamlit App
```bash
streamlit run frontend/app.py
```

### 5️⃣ Open in Browser
```text
http://localhost:8501/
```
---

## 🔄 How DermalScan Works

- User uploads a facial skin image through Streamlit UI
- Image is preprocessed (resize, normalization)
- Trained MobileNetV2 (fine-tuned) model predicts skin condition
- Model outputs:
  - Skin type
  - Confidence percentage
  - Estimated age
  - Risk status
- OpenCV annotates the image with prediction details
- Results are displayed on UI
- User can download:
  - Annotated image
  - CSV prediction report

---

## 🏗 System Architecture

- Frontend
  - Streamlit web interface
  - Handles image upload & result display

- Backend
  - Image preprocessing module
  - MobileNetV2 inference engine
  - Age & risk estimation logic

- Model Layer
  - Fine-tuned MobileNetV2 (.h5)
  - Trained on facial skin dataset

- Output Layer
  - Annotated images (OpenCV)
  - CSV logs with predictions

## 🔁 Architecture Flow 

User Image
   ↓
Streamlit UI
   ↓
Image Preprocessing
   ↓
MobileNetV2 Model
   ↓
Prediction + Confidence
   ↓
Age & Risk Estimation
   ↓
Annotated Image + CSV Export

---


## 📊 Output Details

- Annotated image includes:
  - Skin condition
  - Confidence %
  - Predicted age

- CSV report includes:
  - File name
  - Prediction
  - Confidence
  - Age bucket
  - Time taken

---


## 📄 License

This project is developed for educational and academic purposes.

---


## 👨‍💻 Author

Meghana Sandya

### 📧Email: 22nn1a0480@gmail.com

### 💼GitHub: https://github.com/Meghanasandya28


