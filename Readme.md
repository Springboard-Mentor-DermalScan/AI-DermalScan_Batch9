#  DermalScan – AI-Based Facial Skin Aging Detection System

![Python](https://img.shields.io/badge/Python-3.10-blue)
![Framework](https://img.shields.io/badge/Framework-Streamlit%20%7C%20TensorFlow-red)
![Model](https://img.shields.io/badge/Model-MobileNetV2-brightgreen)
![Library](https://img.shields.io/badge/Library-OpenCV-yellow)



##  Project Overview

DermalScan is an AI-powered facial skin analysis system that detects visible skin aging indicators from facial images. The system automatically detects faces, analyzes facial skin, classifies skin conditions (**Clear Skin, Dark Spots, Puffy Eyes, and Wrinkles**), estimates an age range, and displays confidence scores. All results are presented through a simple and interactive web interface.



##  Key Features

- AI-based skin condition classification  
- Face detection using **Haar Cascade** and **MediaPipe**  
- Deep learning model based on **MobileNetV2**  
- Real-time web interface using **Streamlit**  
- Downloadable annotated images  
- Exportable prediction history (**CSV**)  



##  Technology Stack

### Programming Language
- **Python 3.10**

### Libraries & Frameworks

| Category | Tools Used |
|-------|-----------|
| Deep Learning | TensorFlow, Keras |
| CNN Models | MobileNetV2, EfficientNetB0, ResNet50 |
| Image Processing | OpenCV, NumPy |
| Face Detection | Haar Cascade, MediaPipe |
| Visualization | Matplotlib |
| Web UI | Streamlit |
| Data Handling | Pandas |



##  Project Architecture


DermalScan/
│
├── app.py                          # Main application file
├── inference.py                    # Backend inference logic
│
├── requirements.txt                # Python dependencies
├── README.md                       # Project README
├── LICENSE                         # License information
│
├── DermalScan.ipynb                # Model training & experimentation notebook
├── DermalscanProject Documentation.pdf  # Complete project documentation
│
├── Annotated image.jpg             # Sample annotated output image
├── CSV.csv                         # Sample prediction results (CSV)
├── UserInterface.png               # Application UI screenshot
```



##  How to Run the Project

### 🔹 Clone the Repository
```bash
git clone https://github.com/Springboard-Mentor-DermalScan/AI-DermalScan_Batch9.git
cd AI-DermalScan_Batch9
```

### 🔹 Create a Virtual Environment
```bash
python -m venv venv
venv\Scripts\activate
```

### 🔹 Install Required Dependencies
```bash
pip install -r requirements.txt
```

### 🔹 Run the Application
```bash
streamlit run app.py
```



##  Project Outcomes

- Fully functional AI skin analysis system  
- Beginner-friendly and modular code structure  
- High classification accuracy (**>90%**)  
- Professional and responsive user interface  
- Ready for deployment and future enhancements  



##  Author

**Ashritha Ambati**  
Infosys Springboard Virtual Internship Project
