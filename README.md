# 🧬 AI DermalScan Pro: Intelligent Skin & Bio-Age Analysis

![Branch](https://img.shields.io/badge/Branch-Kamsali--Niharika-purple)
![Python](https://img.shields.io/badge/Python-3.10-blue)
![Framework](https://img.shields.io/badge/Framework-Streamlit-ff4b4b)
![Framework](https://img.shields.io/badge/Framework-Tensorflow-ff4b4b)
![Library](https://img.shields.io/badge/Framework-OpenCV-ff4b4b)

## ⭐️ Project Overview

**AI DermalScan Pro** is an AI-based facial skin analysis application developed to detect common skin conditions and estimate biological age using deep learning techniques. The application is implemented using **Streamlit** and supports real-time image upload, prediction visualization, annotated outputs, and downloadable reports.

The project is developed in multiple milestones covering data preparation, model training, UI integration, export functionality, and testing.

---

## ⚠️ CRITICAL SETUP: READ BEFORE RUNNING

**Issue:** GitHub has a file size limit that prevents uploading the 44MB Age Prediction Model directly via the web interface.
**Solution:** The file `age_net.caffemodel` in this repository is a **dummy text file**. You must replace it for the app to work.

### **Quick Fix Instructions:**
1. **Delete** the existing `age_net.caffemodel` file from your local folder (inside the `Milestone 3` directory).
2. **Download** the real model weights (44MB) from here:  
   👉 [**[Direct Download Link] AgeNet Model**](https://github.com/spmallick/learnopencv/raw/master/AgeGender/age_net.caffemodel)
3. **Paste** the downloaded file into the `Milestone 3` directory of this project.

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
* **Visualization:** Added real-time **Plotly** statistical charts and **Batch CSV Reporting**.

---

## 🏗️ System Architecture

```mermaid
graph TD
    %% 1. Frontend Layer
    User([User]) -->|Uploads Image| UI[Streamlit Frontend]
    UI -->|Raw Bytes| Preproc[Image Preprocessing<br/>OpenCV: Resize & Normalize]

    %% 2. Detection Layer
    Preproc -->|BGR Array| Detect[Face Detection<br/>Haar Cascade Classifier]

    %% 3. Dual Inference Layer (The Core)
    Detect -->|Face ROI| Branch1{Processing Paths}
    
    %% Path A: Skin Analysis
    Branch1 -->|Crop 224x224| SkinModel[Skin Classification<br/>MobileNetV2]
    SkinModel -->|Probabilities| SkinResult(Acne / Wrinkles / Spots)

    %% Path B: Age Analysis
    Branch1 -->|Crop + 20% Padding| AgeModel[Age Estimation<br/>Caffe AgeNet]
    AgeModel -->|Raw Age| Logic[Heuristic Logic Layer]
    SkinResult -.->|Correction Rule| Logic
    Logic -->|Refined Age| AgeResult(Bio-Age Prediction)

    %% 4. Output Layer
    SkinResult & AgeResult -->|Data Merge| visual[Visualization Engine<br/>Plotly & OpenCV]
    
    visual -->|Display| FinalImg[Annotated Image]
    visual -->|Charts| Charts[3D Interactive Charts]
    visual -->|Download| Report[Batch CSV Report]
