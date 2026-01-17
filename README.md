# DermalScan AI: Facial Skin Aging Detection App

## Overview
DermalScan AI is an advanced web application that uses deep learning to analyze facial images for skin aging detection. It classifies skin conditions into categories such as "clear skin", "dark spots", "puffy eyes", and "wrinkles", and estimates the apparent age based on these conditions. The app provides real-time analysis, annotated image outputs, CSV export of predictions, and logging for tracking results.

This project leverages a fine-tuned MobileNetV2 model trained on a custom dataset of facial images. The frontend is built with HTML, CSS, and JavaScript, while the backend uses Flask for API handling and TensorFlow/Keras for model inference.

## Features
- **Image Upload and Analysis**: Upload facial images for instant skin condition classification and age estimation.
- **Real-time Results**: Displays detected skin problems, confidence levels, estimated age, and analysis time.
- **Annotated Outputs**: Generates annotated images with bounding boxes and labels highlighting detected areas.
- **Download Options**: Download original and annotated images, plus CSV files containing prediction details.
- **Prediction History**: View and manage a history of past analyses with timestamps.
- **Logging**: Automatic logging of analysis results to `analysis.log` for auditing and debugging.
- **Face Detection**: Uses OpenCV's Haar cascades for face detection to focus analysis on facial regions.
- **Responsive UI**: Clean, modern interface with a dark theme for better user experience.

## Technologies Used
- **Backend**: Flask, TensorFlow/Keras, OpenCV, NumPy
- **Frontend**: HTML, CSS, JavaScript (Vanilla JS)
- **Model**: MobileNetV2 (pre-trained on ImageNet, fine-tuned on custom dataset)
- **Dataset**: Custom dataset with images categorized into 4 classes (clear skin, dark spots, puffy eyes, wrinkles)
- **Deployment**: Local Flask server (can be extended to cloud deployment)

## Installation

### Prerequisites
- Python 3.8+
- pip (Python package manager)
- Git

### Steps
1. **Clone the Repository**:
   ```bash
   git clone https://github.com/your-username/dermalscan-ai.git
   cd dermalscan-ai
   ```

2. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Download Model and Haar Cascade**:
   - Ensure `MobileNetV2_SkinAging_Model.keras` is in the root directory (trained model file).
   - Ensure `haarcascade_frontalface_default.xml` is present (OpenCV Haar cascade for face detection).

4. **Run the Application**:
   ```bash
   python backend.py
   ```
   - Open your browser and navigate to `http://127.0.0.1:5000/`.

## How to Run the Application
1. Ensure all prerequisites are installed and the repository is cloned.
2. Install dependencies: `pip install -r requirements.txt`.
3. Run the backend server: `python backend.py`.
4. Open a web browser and go to `http://127.0.0.1:5000/`.
5. The application will load the frontend interface.

## What is Shown on Screen
- **Main Interface**: A dark-themed page with a title "DermalScan AI – Skin & Age Analysis", file upload input, and "Start Analysis" button.
- **Results Section** (appears after analysis):
  - Two cards: "Before" (original image with download button) and "After" (annotated image with problem details: Problem, Age, Confidence, Analysis Time, and download button).
  - Prediction Table: A table listing bounding box coordinates (X1, Y1, X2, Y2), Class, Confidence, and Age for each detection.
  - Download CSV button for exporting predictions.
  - History Section: A table of past analyses with Time, Image (clickable thumbnail), Class, Confidence, and Age. Includes a "Clear History" button.
- The UI is responsive and uses a grid layout for results.

## Modules
- **Frontend Module** (`index.html`): Handles user interactions, image upload, displays results, and provides download options. Built with vanilla HTML, CSS, and JavaScript for simplicity and fast loading.
- **Backend Module** (`backend.py`): Flask-based API that processes uploaded images using OpenCV for face detection, TensorFlow/Keras for model inference, and logs results. Returns JSON responses with detections, annotated images, and metadata.
- **Model Module** (via `Ai_DermalScan1.ipynb` and `MobileNetV2_SkinAging_Model.keras`): Pre-trained MobileNetV2 fine-tuned on a custom dataset for classifying skin conditions. Includes data preprocessing, augmentation, training scripts, and evaluation metrics.
- **Logging and Export Module**: Integrated into backend for saving logs to `analysis.log` and annotated images to `results/`. Frontend handles CSV generation and downloads.

## Model Details
- **Architecture**: MobileNetV2 with global average pooling, dropout layers, and a dense output layer for 4 classes.
- **Training**: Fine-tuned on a dataset with data augmentation (rotation, zoom, flip). Achieved ~95% validation accuracy.
- **Inference**: Processes 224x224 resized face crops, outputs class probabilities.
- **Age Estimation**: Rule-based estimation based on detected skin condition.

## Dataset
- The dataset consists of facial images divided into 4 categories.
- Preprocessing includes resizing, normalization, and augmentation.
- Refer to `Ai_DermalScan1.ipynb` for data exploration, training code, and evaluation.

## Project Structure
```
dermalscan-ai/
├── index.html              # Frontend UI
├── backend.py              # Flask backend API
├── Ai_DermalScan1.ipynb    # Jupyter notebook for model training and testing
├── requirements.txt        # Python dependencies
├── MobileNetV2_SkinAging_Model.keras  # Trained model
├── haarcascade_frontalface_default.xml  # Face detection cascade
├── analysis.log            # Log file for analyses
├── results/                # Folder for saved annotated images
├── DATASET/                # Dataset folder (if included)
└── README.md               # This file
```

## Contributing
1. Fork the repository.
2. Create a feature branch: `git checkout -b feature-name`.
3. Commit changes: `git commit -m 'Add feature'`.
4. Push to branch: `git push origin feature-name`.
5. Open a Pull Request.

## License
This project is licensed under the MIT License. See LICENSE for details.

## Acknowledgments
- MobileNetV2 model from TensorFlow/Keras.
- OpenCV for computer vision tasks.
- Custom dataset for training.

## Contact
For questions or support, contact [narlawargauri5@example.com].
