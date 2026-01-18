# 🧬 AI DermalScan Pro: Intelligent Skin & Bio-Age Analysis

![Branch](https://img.shields.io/badge/Branch-Kamsali--Niharika-purple)
![Python](https://img.shields.io/badge/Python-3.10-blue)
![Framework](https://img.shields.io/badge/Framework-Streamlit-ff4b4b)

## ⭐️ Project Overview

DermalScan is an AI-based facial skin analysis application developed to detect common skin conditions and estimate age using deep learning techniques.  
The application is implemented using Streamlit and supports image upload, prediction visualization, annotated outputs, and downloadable reports.

The project is developed in multiple milestones covering data preparation, model training, UI integration, export functionality, and testing.

---

## ⚠️ CRITICAL SETUP: READ BEFORE RUNNING

**Issue:** GitHub has a file size limit that prevents uploading the 44MB Age Prediction Model directly via the web interface.
**Solution:** The file `age_net.caffemodel` in this repository is a **dummy text file**. You must replace it for the app to work.

### **Quick Fix Instructions:**
1. **Delete** the existing `age_net.caffemodel` file from your local folder.
2. **Download** the real model weights (44MB) from here:  
   👉 [**[Direct Download Link] AgeNet Model**](https://github.com/spmallick/learnopencv/raw/master/AgeGender/age_net.caffemodel)
3. **Paste** the downloaded file into the root directory of this project.

*(If you skip this, the application will crash with an OpenCV Error).*

---

## 📅 Milestone Journey

This project was built in three distinct phases:

### **Milestone 1: Data Engineering**
* **Objective:** Build a robust dataset for skin conditions.
* **Process:** Curated and labeled images into 4 classes (`Acne/Clear`, `Dark Spots`, `Puffy Eyes`, `Wrinkles`).
* **Preprocessing:** Standardized all inputs to `224x224` pixels and normalized pixel intensity.
* **Augmentation:** Applied random rotations, zooms, and flips to prevent model overfitting.

### **Milestone 2: Model Architecture & Training**
* **Core Model:** Fine-tuned **MobileNetV2** (transfer learning) for skin classification.
* **Performance:** Achieved **>90% training accuracy** and validated against unseen test data.
* **Output:** The trained weights were exported as `mobilenet_skin.h5`.

### **Milestone 3: System Integration**
* **Frontend:** Developed a "Neon Cyberpunk" themed web UI using **Streamlit**.
* **Backend:** Integrated the Skin Model with a Caffe-based Age Estimator.
* **Logic Layer:** Implemented **Context Padding (+20%)** to fix head-cropping errors and **Heuristic Logic** to correct "Baby Face" misclassifications on adults.
* **Visualization:** Added real-time **plotly** statistical pie charts and **Batch CSV Reporting**.

---

## 🚀 Key Features

* **Multi-Face Support:** automatically detects and analyzes multiple people in a single group photo.
* **Smart Heuristics:**
    * *Rule 1:* If "Wrinkles" are detected with high confidence, the minimum age floor is raised.
    * *Rule 2:* **Context Padding** ensures the model sees the forehead and chin, improving age accuracy by ~15%.
* **Batch Processing:** Upload 5+ images at once; the system generates a consolidated **Excel/CSV Report**.
* **Privacy First:** Images are processed in memory and are not permanently stored.

---

## 🛠️ Tech Stack

This project is built on a robust stack optimized for rapid computer vision prototyping:

| Component | Technology | Purpose |
| :--- | :--- | :--- |
| **Language** | **Python 3.10** | Core logic and scripting. |
| **Frontend** | **Streamlit** | Interactive web UI, file handling, and real-time updates. |
| **Computer Vision** | **OpenCV (cv2)** | Image preprocessing, Haar Cascade detection, and drawing annotations. |
| **Deep Learning** | **TensorFlow (Keras)** | Running the custom **MobileNetV2** Skin Classification model. |
| **Inference Engine** | **Caffe (DNN)** | Running the pre-trained **AgeNet** model for age estimation. |
| **Visualization** | **Seaborn & Matplotlib** | Generating dynamic statistical charts (Horizontal Bar Plots). |
| **Data Handling** | **Pandas & NumPy** | Managing batch data and generating CSV reports. |

---

## ⚙️ Installation & Usage Guide

### **Step 1: Clone the Repository**
git clone -b Kamsali-Niharika [https://github.com/Springboard-Mentor-DermalScan/AI-DermalScan_Batch9.git](https://github.com/Springboard-Mentor-DermalScan/AI-DermalScan_Batch9.git)
cd AI-DermalScan_Batch9

Step 2: Install Dependencies
pip install -r requirements.txt

Step 3: Setup Models
mobilenet_skin.h5 (Trained Skin Model)

age_deploy.prototxt (Age Architecture)

age_net.caffemodel (Downloaded manually)

Step 4: Launch App
streamlit run app.py

AI-DermalScan_Batch9 (Branch: Kamsali-Niharika)  <-- ROOT REPOSITORY

│

├── AI-DermalScan Milestone 1 & 2.ipynb <--(Python Notebook consisting data preprocessing and model development)

├── AI DermalScan                  <-- (Documentation)

├── README.md                      <-- (The Documentation File)

├── requirements.txt               <-- (Requirements)

├── LICENSE                        <-- (Standard License file)

│

└── Milestone 3/                     <-- (MAIN PROJECT FOLDER)

    │
    
    ├── app.py                       <-- (The Main Python App)
    
    ├── mobilenet_skin.h5            <-- (Skin Model)
    ├── age_deploy.prototxt          <-- (Age Config)
    ├── age_net.caffemodel           <-- (Age Weights - Placeholder)
    ├── haarcascade_frontalface_default.xml
    │
    ├── .streamlit/                  <-- (CONFIG FOLDER)
    │   └── config.toml              <-- (Theme settings)
    │
    ├── Predicted result/            <-- (OUTPUT FOLDER)
    │   └── dermalscan_results.csv   <-- (Generated Reports)
    │
    ├── Sample Images/               <-- (UI Design and Output Images)
    │   ├── Web UI.png
    │   ├── newplot.png
    │   └── detected_...jpg
    │
    └── Sample test Images/          <-- (TEST DATASET)
        ├── shutterstock_10727980.jpg
        ├── istockphoto...jpg
        └── 360_F...jpg

👨‍💻 Developer Info

Developer: Kamsali Niharika

Program: Infosys Springboard Virtual Internship (Batch 9)
