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
```
### 2️⃣ Create Virtual Environment (Optional)
```bash
python -m venv venv
venv\Scripts\activate
```
### 3️⃣ Install Dependencies
```bash
pip install -r requirements.txt
```
### 4️⃣ Run the Application
```bash
python app.py
```

### 5️⃣ Open in Browser
```bash
http://127.0.0.1:5000/
```
---

## 🧪 How It Works

1. User uploads a facial image through the web interface  
2. The system detects the face using OpenCV’s DNN-based face detector  
3. The detected face is cropped and preprocessed  
4. Image is resized to **224 × 224** and normalized  
5. The **MobileNetV2 transfer learning model** predicts the skin aging category  
6. The prediction result is displayed on the UI  
7. Prediction details are saved in a CSV file for history tracking  

---

## 📈 Dataset & Model

- Dataset organized using a **folder-based structure**
- Images resized to **224 × 224**
- Data augmentation applied to improve generalization
- Model trained using **Transfer Learning with MobileNetV2**
- Trained model saved as `.h5` and loaded during inference

---

## 👩‍💻 Author

**Priya Ghosal**
