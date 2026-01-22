DermalScan – AI-Based Facial Skin Aging Detection System
 Project Overview
DermalScan is an AI-powered facial skin analysis system that detects visible skin aging indicators from facial images. The system automatically detects faces, analyzes facial skin, classifies skin conditions (Clear Skin, Dark Spots, Puffy Eyes, and Wrinkles), estimates age range, and displays confidence scores. All results are presented through a simple and interactive web interface.
 Key Features
•	AI-based skin condition classification
•	Face detection using  Haar Cascade and MediaPipe
•	Deep learning model MobileNetV2 
•	Real-time web interface using Streamlit
•	Downloadable annotated images
•	Exportable prediction history (CSV)
 Technology Stack
Programming Language
•	Python 3.10
Libraries & Frameworks
Category	Tools Used
Deep Learning	TensorFlow, Keras
CNN Models	MobileNetV2, EfficientNetB0, ResNet50
Image Processing	OpenCV, NumPy
Face Detection	 Haar Cascade, MediaPipe
Visualization	Matplotlib
Web UI	Streamlit\
Data Handling	Pandas

Project Architecture
 
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

 How to Run the Project
🔹 Clone the Repository
Clone the project from GitHub using the command below:
 https://github.com/Springboard-Mentor-DermalScan/AI-DermalScan_Batch9.git
🔹 Create a Virtual Environment 
Create a virtual environment to manage dependencies:
python -m venv venv
venv\Scripts\activate  # On Windows
🔹 Install Required Dependencies
Install all required Python libraries using:
pip install -r requirements.txt
🔹 Run the Application
Start the Streamlit application using: streamlit run app.py
User Interface: 
![UserInterface](UserInterface.png) 

Project Outcomes
•	Fully functional AI skin analysis system
•	Beginner-friendly modular code
•	Accurate classification (>90%)
•	Professional UI
•	Ready for deployment or enhancement
Author
Ashritha Ambati
Infosys Springboard Virtual Internship Project

