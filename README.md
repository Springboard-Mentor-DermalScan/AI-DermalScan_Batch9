# 🧬 AI DermalScan Pro: Intelligent Skin & Bio-Age Analysis

![Status](https://img.shields.io/badge/Status-Milestone%203%20Complete-success)
![Branch](https://img.shields.io/badge/Branch-Kamsali--Niharika-purple)
![Python](https://img.shields.io/badge/Python-3.10-blue)
![Framework](https://img.shields.io/badge/Framework-Streamlit-ff4b4b)

**AI DermalScan Pro** is an advanced computer vision prototype developed during the **Infosys Springboard Virtual Internship**. It leverages Deep Learning to provide real-time facial skin analysis and biological age estimation.

Unlike standard age classifiers, this system implements a custom **"Smart Bio-Age Algorithm"** that adjusts age predictions based on detected skin health indicators (wrinkles, texture, and facial geometry), offering a more holistic analysis than simple image classification.

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

### **Milestone 3: System Integration (Current Status)**
* **Frontend:** Developed a "Neon Cyberpunk" themed web UI using **Streamlit**.
* **Backend:** Integrated the Skin Model with a Caffe-based Age Estimator.
* **Logic Layer:** Implemented **Context Padding (+20%)** to fix head-cropping errors and **Heuristic Logic** to correct "Baby Face" misclassifications on adults.
* **Visualization:** Added real-time **Seaborn** statistical charts and **Batch CSV Reporting**.

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

* **Language:** Python 3.10
* **UI Framework:** Streamlit
* **Deep Learning:** TensorFlow (Keras), OpenCV (DNN Module)
* **Data Visualization:** Seaborn, Matplotlib
* **Image Processing:** NumPy, PIL

---

## ⚙️ Installation & Usage Guide

### **Step 1: Clone the Repository**
*Note: This project is hosted on a specific branch.*
```bash
git clone -b Kamsali-Niharika [https://github.com/Springboard-Mentor-DermalScan/AI-DermalScan_Batch9.git](https://github.com/Springboard-Mentor-DermalScan/AI-DermalScan_Batch9.git)
cd AI-DermalScan_Batch9
