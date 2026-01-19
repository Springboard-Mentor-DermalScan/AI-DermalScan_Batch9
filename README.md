🧬 DermalScan – AI-Powered Facial Skin Analysis System

📌 Project Overview

DermalScan is designed to analyze facial images and classify skin conditions such as Wrinkles, Dark Spots, Puffy Eyes, and Clear Skin.
The system uses transfer learning and fine-tuning on MobileNetV2, combined with a lightweight backend inference pipeline and an interactive Streamlit frontend.
Users can upload images, view annotated predictions, check confidence levels, and export results for documentation or analysis.
✨ Key Features

Image upload via Streamlit UI

CNN-based skin condition classification

Fine-tuned MobileNetV2 model

Confidence-based prediction levels

Age estimation with stable mapping logic

Annotated output images

Prediction summary table

CSV export of results

Fast inference time per image

🧠 Technologies Used
Frontend

Streamlit

Backend

Python 3.10

TensorFlow / Keras

OpenCV

NumPy

Pandas

🧪 Model Details

Base Model: MobileNetV2 (ImageNet pretrained)

Approach: Transfer Learning + Fine Tuning

Loss Function: Categorical Crossentropy

Optimizer: Adam (low learning rate during fine tuning)

Input Size: 224 × 224

Classes:

Clear Skin

Dark Spots

Puffy Eyes

Wrinkles

👤 User Guide
1️⃣ Run the Application
streamlit run frontend/app.py

2️⃣ Upload Image

Upload JPG / PNG facial image

Multiple images supported

3️⃣ View Results

For each image:

Annotated image with bounding box

Predicted skin condition

Confidence percentage

Estimated age

Risk status (Normal / Moderate / Risk)

4️⃣ Download Outputs

Annotated image

CSV prediction report

🧑‍💻 Developer Guide
📁 Project Structure

AI-DermalScan/
│
├── Dataset/
│   ├── train/
│   ├── val/
│   └── test/
│
├── models/
│   └── MobileNetV2_Module3_finetuned.h5
│
├── backend/
│   └── inference.py
│
├── frontend/
│   └── app.py
│
├── results/
│   ├── correct/
│   ├── wrong/
│   └── uncertain/
│
├── Module4_Predictions.csv
├── requirements.txt
└── README.md


