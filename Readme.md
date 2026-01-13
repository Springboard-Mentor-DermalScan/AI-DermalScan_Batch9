# DermalScan – AI-Based Facial Skin Aging Detection System

## 1. Project Overview
DermalScan is an end-to-end **AI-powered facial skin analysis system** that detects visible skin aging indicators from facial images. It is designed so that **even a beginner (fresher)** can understand how data flows from image upload to final prediction.

The system automatically:
- Detects a human face from an image
- Analyzes facial skin
- Classifies skin condition into:
  - Clear Skin
  - Dark Spots
  - Puffy Eyes
  - Wrinkles
- Displays confidence percentage
- Estimates age range
- Shows results visually through a web interface

The project combines **Deep Learning, Computer Vision, and Web UI** into one complete application.

---

## 2. Problem Statement
Skin aging signs such as wrinkles, dark spots, and puffy eyes are common but often require expert evaluation. Manual analysis is:
- Time-consuming
- Subjective
- Not easily accessible

DermalScan solves this by providing:
- Automated detection
- Consistent results
- Fast analysis using AI

---

## 3. Key Features
- AI-based skin condition classification
- Face detection using MediaPipe
- Deep learning model (MobileNetV2 / EfficientNetB0)
- Real-time web interface using Streamlit
- Downloadable annotated images
- Exportable prediction history (CSV)

---

## 4. Project Architecture (High-Level)
```
User Image Upload (UI)
        ↓
Streamlit Frontend (app.py)
        ↓
Backend Inference Engine (inference.py)
        ↓
Face Detection → Preprocessing → Model Prediction
        ↓
Annotated Image + Results Table
        ↓
Displayed & Downloaded in UI
```

---

## 5. Technology Stack

### Programming Language
- Python 3.x

### Libraries & Frameworks
| Category | Tools Used |
|-------|-----------|
| Deep Learning | TensorFlow, Keras |
| CNN Models | MobileNetV2, EfficientNetB0, ResNet50 |
| Image Processing | OpenCV, NumPy |
| Face Detection | MediaPipe |
| Visualization | Matplotlib |
| Web UI | Streamlit |
| Data Handling | Pandas |

---

## 6. Dataset Preparation (Milestone 1)

### 6.1 Dataset Structure
The dataset is organized into folders, **one folder per class**:
```
DATASET/
│── clear skin/
│── dark spots/
│── puffy eyes/
│── wrinkles/
```

Each folder contains facial images belonging to that class.

### 6.2 Dataset Validation
- Images are verified for correct format
- Corrupted images are removed
- Distribution of images per class is visualized using bar graphs

### 6.3 Why Dataset Balance Matters
If one class has more images than others, the model becomes biased. Therefore:
- Image counts are checked
- Augmentation is applied where needed

---

## 7. Image Preprocessing & Augmentation (Milestone 1 – Module 2)

### 7.1 Image Preprocessing
Each image is:
- Resized to **224 × 224** pixels
- Converted to NumPy array
- Normalized (pixel values between 0 and 1)

### 7.2 Image Augmentation
To increase dataset size and diversity, the following transformations are applied:
- Rotation
- Zoom
- Width & height shift
- Brightness adjustment
- Horizontal & vertical flip

This helps prevent **overfitting** and improves generalization.

---

## 8. Model Training (Milestone 2 – Module 3)

### 8.1 Why Transfer Learning?
Training a CNN from scratch requires huge data. Instead, **pretrained models** are used.

### 8.2 Models Used
- MobileNetV2
- ResNet50
- EfficientNetB0

### 8.3 Training Configuration
- Loss Function: Categorical Crossentropy
- Optimizer: Adam
- Input Size: 224 × 224 × 3
- Output: 4 classes (softmax)

### 8.4 Model Performance
| Model | Training Accuracy | Validation Accuracy |
|-----|------------------|---------------------|
| MobileNetV2 | ~90% | ~88% |
| ResNet50 | ~95% | ~89% |
| EfficientNetB0 | ~87% | ~91% |

The trained model is saved as an `.h5` file.

---

## 9. Face Detection & Prediction Pipeline (Milestone 2 – Module 4)

### 9.1 Face Detection
- Implemented using **MediaPipe Face Detection**
- Detects face bounding box
- Returns confidence score

### 9.2 Face Cropping
Only the detected face region is passed to the model, improving accuracy.

### 9.3 Prediction Output
For each face:
- Skin condition label
- Confidence percentage
- Estimated age range

### 9.4 Age Estimation Logic
Age is estimated using predefined ranges mapped to skin condition:
- Clear Skin → 20–28
- Puffy Eyes → 30–45
- Dark Spots → 35–42
- Wrinkles → 45–55

---

## 10. Backend Inference (inference.py)

### 10.1 Responsibilities
- Load trained model
- Detect face
- Preprocess image
- Run prediction
- Draw bounding boxes
- Return results

### 10.2 Output Returned
- Annotated image (RGB)
- Pandas DataFrame containing:
  - Filename
  - Condition
  - Confidence
  - Estimated Age
  - Face detector confidence
  - Bounding box coordinates

---

## 11. Frontend Web Application (app.py)

### 11.1 Why Streamlit?
- Simple
- Fast
- Python-based
- No frontend coding required

### 11.2 UI Features
- Image upload
- Side-by-side image display
- Prediction table
- Download annotated image
- Download prediction history (CSV)

### 11.3 Session State
- Prevents duplicate reprocessing
- Stores history of predictions

---

## 12. End-to-End Flow (Pin-to-Pin)

1. User uploads image
2. Image saved temporarily
3. Backend inference triggered
4. Face detected
5. Face preprocessed
6. Model predicts skin condition
7. Age estimated
8. Bounding box drawn
9. Results sent to UI
10. User views & downloads output

---

## 13. How to Run the Project

### 13.1 Install Dependencies
```bash
pip install tensorflow opencv-python mediapipe streamlit pandas numpy
```

### 13.2 Run Application
```bash
streamlit run app.py
```

---

## 14. Project Outcomes
- Fully functional AI skin analysis system
- Beginner-friendly modular code
- Accurate classification (>90%)
- Professional UI
- Ready for deployment or enhancement

---

## 15. Future Improvements
- Real age prediction model
- Multi-face detection
- Skin disease classification
- Mobile app deployment
- Cloud hosting

---

## 16. Conclusion
DermalScan demonstrates how **AI + Computer Vision + Web UI** can be combined into a complete real-world application. This project is ideal for students and freshers to understand **end-to-end AI system development**, from dataset creation to deployment.

---

### Developed By
**Ashritha Ambati**

Infosys Springboard Virtual Internship Project

