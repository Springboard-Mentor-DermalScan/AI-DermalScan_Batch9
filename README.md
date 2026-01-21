# 🧴 DermalScan – AI-Based Facial Skin Analysis System

DermalScan is a deep learning–based application that analyzes facial images to detect skin aging conditions such as **wrinkles, dark spots, puffy eyes, and clear skin**.  
The system uses a **fine-tuned MobileNetV2 model** and provides results through an interactive **Streamlit web interface**.

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

## 🛠 Tech Stack

![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)
![Keras](https://img.shields.io/badge/Keras-D00000?style=for-the-badge&logo=keras&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-27338e?style=for-the-badge&logo=opencv&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![Matplotlib](https://img.shields.io/badge/Matplotlib-11557C?style=for-the-badge)
![MobileNetV2](https://img.shields.io/badge/MobileNetV2-Deep%20Learning-blue?style=for-the-badge)


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
git clone https://github.com/Springboard-Mentor-DermalScan/AI-DermalScan_Batch9.git

### 2️⃣ Create Virtual Environment
python -m venv env
env\Scripts\activate

### 3️⃣  Install Dependencies
pip install -r requirements.txt

### 4️⃣ Run Streamlit App
streamlit run frontend/app.py

### 5️⃣ Open in Browser
http://localhost:8501/


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

