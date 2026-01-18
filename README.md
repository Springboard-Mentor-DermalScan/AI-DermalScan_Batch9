# 🧬 AI DermalScan Pro: Intelligent Skin & Bio-Age Analysis

![Branch](https://img.shields.io/badge/Branch-Kamsali--Niharika-purple)
![Python](https://img.shields.io/badge/Python-3.10-blue)
![Framework](https://img.shields.io/badge/Framework-Streamlit-ff4b4b)
![Framework](https://img.shields.io/badge/Framework-Tensorflow-ff4b4b)
![Library](https://img.shields.io/badge/Library-OpenCV-ff4b4b)

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
---
###🚀 Key FeaturesMulti-Face Support: Automatically detects and analyzes multiple people in a single group photo.Smart Heuristics:Rule 1: If "Wrinkles" are detected with high confidence, the minimum age floor is raised.Rule 2: Context Padding ensures the model sees the forehead and chin, improving age accuracy by ~15%.Batch Processing: Upload 5+ images at once; the system generates a consolidated Excel/CSV Report.Privacy First: Images are processed in memory and are not permanently stored.🛠️ Tech StackThis project is built on a robust stack optimized for rapid computer vision prototyping:ComponentTechnologyPurposeLanguagePython 3.10Core logic and scripting.FrontendStreamlitInteractive web UI, file handling, and real-time updates.Computer VisionOpenCV (cv2)Image preprocessing, Haar Cascade detection, and drawing annotations.Deep LearningTensorFlow (Keras)Running the custom MobileNetV2 Skin Classification model.Inference EngineCaffe (DNN)Running the pre-trained AgeNet model for age estimation.VisualizationPlotlyGenerating interactive 3D charts and visualizations.Data HandlingPandas & NumPyManaging batch data and generating CSV reports.⚙️ Installation & Usage GuideStep 1: Clone the RepositoryBashgit clone -b Kamsali-Niharika [https://github.com/Springboard-Mentor-DermalScan/AI-DermalScan_Batch9.git](https://github.com/Springboard-Mentor-DermalScan/AI-DermalScan_Batch9.git)
cd AI-DermalScan_Batch9
Step 2: Install DependenciesBashpip install -r requirements.txt
Step 3: Setup ModelsNavigate to the Milestone 3 folder:Bashcd "Milestone 3"
Important: Ensure the age_net.caffemodel file in this folder is the real 44MB file (see Critical Setup section above).Step 4: Launch AppBashstreamlit run app.py
📂 Project Directory StructurePlaintextAI-DermalScan_Batch9 (Branch: Kamsali-Niharika)    <-- ROOT REPOSITORY
│
├── AI-DermalScan Milestone 1 & 2.ipynb            <-- (Project Notebook: Data Prep & Training)
├── AI DermalScan.pdf                              <-- (Project Documentation)
├── LICENSE                                        <-- (Standard License File)
├── README.md                                      <-- (Master Documentation Guide)
├── requirements.txt                               <-- (Python Dependencies)
│
└── Milestone 3/                                   <-- (MAIN APPLICATION FOLDER)
    │
    ├── app.py                                     <-- (Main Application Script)
    ├── haarcascade_frontalface_default.xml        <-- (Face Detection Model)
    │
    ├── mobilenet_skin.h5                          <-- (Skin Classification Model)
    ├── age_deploy.prototxt                        <-- (Age Model Configuration)
    ├── age_net.caffemodel                         <-- (Age Model Weights - Placeholder)
    │
    ├── .streamlit/                                <-- (UI Configuration)
    │   └── config.toml                            <-- (Theme Settings)
    │
    ├── Predicted result/                          <-- (Generated Reports)
    │   └── dermalscan_results.csv                 <-- (Batch Analysis Output)
    │
    ├── Sample Images/                             <-- (UI Assets & Output Examples)
    │   ├── Web UI.png
    │   ├── newplot.png
    │   └── detected_output.jpg
    │
    └── Sample test Images/                        <-- (Testing Dataset)
        ├── shutterstock_10727980.jpg
        ├── istockphoto_1919265357.jpg
        └── 360_F_235640074.jpg
👨‍💻 Developer InfoDeveloper: Kamsali Niharika Program: Infosys Springboard Virtual Internship (Batch 9)
