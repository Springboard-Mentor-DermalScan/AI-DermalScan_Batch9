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

## ⚙️ Project Modules Completed

### ✅ Module 1: Dataset Setup & Labeling
- Collected and organized facial skin images
- Labeled images into four skin categories
- Ensured balanced dataset distribution
- Generated class distribution plot

### ✅ Module 2: Image Preprocessing & Augmentation
- Resized images to 224×224
- Normalized pixel values
- Applied augmentation:
  - Rotation
  - Zoom
  - Horizontal flip

### ✅ Module 3: Model Training (MobileNetV2)
- Used pretrained MobileNetV2
- Applied transfer learning
- Fine-tuned top layers
- Used Adam optimizer and categorical cross-entropy
- Saved trained model as `.h5`

### ✅ Module 4: Prediction Pipeline
- Loaded trained model
- Preprocessed input images
- Generated class probabilities
- Predicted:
  - Skin condition
  - Confidence score
  - Estimated age
  - Risk status
- Annotated image using OpenCV

### ✅ Module 5: Streamlit Frontend
- Image upload support (single & multiple)
- Real-time result display
- Annotated image visualization
- Clean and responsive UI

### ✅ Module 6: Backend Integration
- Modular inference code (`inference.py`)
- Model loaded once for efficiency
- Smooth frontend–backend communication

### ✅ Module 7: Export & Logging
- Download annotated image
- Download prediction report as CSV
- Logged:
  - Disease
  - Confidence
  - Age
  - Time taken

### ✅ Module 8: Documentation
- README.md created
- Project structure documented
- GitHub repository prepared

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


## ▶️ How to Run the Project

### 1️⃣ Create Virtual Environment
```bash
python -m venv env
env\Scripts\activate
2️⃣ Install Dependencies
pip install -r requirements.txt
3️⃣ Run Streamlit App
streamlit run frontend/app.py


📊 ```md
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


📄 License

This project is developed for educational and academic purposes.