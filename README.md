# 🧴 AIDermalScan – AI Facial Skin Aging Detection

An **AI-powered web application** that analyzes facial images to detect **skin aging signs** such as wrinkles and fine lines using **Deep Learning and Computer Vision**.

---

## 📸 Website Preview

### 🖼️ Screenshots

<img width="1840" height="920" alt="Screenshot 1" src="https://github.com/user-attachments/assets/b7db2789-8132-4065-b3db-0c24823032ef" />

<img width="1844" height="889" alt="Screenshot 2" src="https://github.com/user-attachments/assets/bd81ea5d-b97e-48c8-ab09-33046cee2544" />

<img width="1847" height="899" alt="Screenshot 3" src="https://github.com/user-attachments/assets/db2db2e1-b5e3-4cab-b6af-0b4344e19267" />

<img width="1855" height="894" alt="Screenshot 4" src="https://github.com/user-attachments/assets/ebee753a-7534-483d-a147-3a2857fe2e9c" />

---

## 🚀 Features

- 📸 Upload facial images through a web interface  
- 🧠 AI-based skin aging prediction using **MobileNetV2**  
- 👤 Automatic face detection  
- 📊 Stores prediction history in **CSV** format  
- 🖥️ Simple and responsive frontend using **HTML & CSS**  
- ⚡ Lightweight and beginner-friendly implementation  

---

## 🏗️ Project Architecture

AIDermalScan/
│
├── app.py # Flask application
├── backend.py # Image processing & prediction logic
├── models/
│ └── AIDermalScan_MobileNetV2_Final.h5
│
├── face_detector/
│ ├── deploy.prototxt
│ └── res10_300x300_ssd_iter_140000.caffemodel
│
├── static/
│ ├── css/
│ ├── uploads/
│
├── templates/
│ └── index.html
│
├── dataset/
├── history.csv
├── requirements.txt
└── README.md


---

## 🛠️ Technologies Used

- 📝 **HTML & CSS** – Frontend  
- ⚙️ **Flask** – Backend Framework  
- 🧠 **TensorFlow / Keras** – Deep Learning  
- 📸 **OpenCV** – Face Detection  
- 📊 **Pandas & NumPy** – Data Handling  
- 📦 **MobileNetV2** – Transfer Learning Model  

---

## ⚙️ Installation & Setup

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/your-username/AIDermalScan.git
cd AIDermalScan

### 2️⃣ Create Virtual Environment (Optional)
python -m venv venv
venv\Scripts\activate

### 3️⃣ Install Dependencies
pip install -r requirements.txt

### 4️⃣ Run the Application
python app.py

### 5️⃣ Open in Browser
http://127.0.0.1:5000/

## 🧪 How It Works

User uploads a facial image

Face is detected using OpenCV’s DNN model

Image is preprocessed and resized

MobileNetV2 model predicts skin aging category

Result is displayed and stored in CSV

##📈 Dataset & Model

Dataset organized using folder-based structure

Images resized to 224 × 224

Data augmentation applied

Model trained using transfer learning (MobileNetV2)

#🎯 Use Cases

AI-based skincare analysis

Academic mini / major projects

Internship portfolio projects

Computer Vision learning projects

#👩‍💻 Author

Priya Ghosal
