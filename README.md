# 🧴 AIDermalScan – AI Facial Skin Aging Detection

An **AI-powered web application** that analyzes facial images to detect **skin aging signs** such as wrinkles and fine lines using **Deep Learning and Computer Vision**.

---


## 📸 Website Preview

### 🖼️ Screenshots

![Screenshot 1](https://github.com/user-attachments/assets/b7db2789-8132-4065-b3db-0c24823032ef)

![Screenshot 2](https://github.com/user-attachments/assets/bd81ea5d-b97e-48c8-ab09-33046cee2544)

![Screenshot 3](https://github.com/user-attachments/assets/db2db2e1-b5e3-4cab-b6af-0b4344e19267)

![Screenshot 4](https://github.com/user-attachments/assets/ebee753a-7534-483d-a147-3a2857fe2e9c)


---

## 🚀 Features

- 📸 Upload facial images through a web interface  
- 🧠 AI-based skin aging prediction using **MobileNetV2**  
- 👤 Automatic face detection  
- 📊 Stores prediction history in **CSV** format  
- 🖥️ Simple and responsive frontend using **HTML & CSS**  
- ⚡ Lightweight and beginner-friendly implementation  

---

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

---

## 🧪 How It Works

User uploads a facial image

Face is detected using OpenCV’s DNN model

Image is preprocessed and resized

MobileNetV2 model predicts skin aging category

Result is displayed and stored in CSV

---

##📈 Dataset & Model

Dataset organized using folder-based structure

Images resized to 224 × 224

Data augmentation applied

Model trained using transfer learning (MobileNetV2)
---

---

#👩‍💻 Author

Priya Ghosal
