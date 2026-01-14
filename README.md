# Facial Age & Skin Concern Detection

**AI-Facial-Aging-Detection-App** is a lightweight computer vision project that detects **facial skin concerns associated with aging** from images.  
It combines **face detection**, **transfer learning–based image classification**, and an optional **Streamlit web interface** for interactive inference and result export.

---

## Project Overview

- **Objective:**  
  Automatically classify facial images into common skin concern categories related to aging.

- **Skin Concern Classes:**
  - `clear skin`
  - `dark spots`
  - `puffy eyes`
  - `wrinkles`

- **Core Techniques:**
  - Face detection using **OpenCV Haar Cascades**
  - Image classification using **TensorFlow/Keras**
  - Transfer learning with **ResNet**, **MobileNetV2**, and **EfficientNetB2**
  - Dataset splitting into `train`, `validation`, and `test` sets
  - Optional Streamlit-based UI for inference, logging, and export

---

## Repository Structure

```text
├── app.py
├── dataset/
│   ├── clear skin/
│   ├── dark spots/
│   ├── puffy eyes/
│   └── wrinkles/
├── splittedset/
│   ├── train/
│   ├── validation/
│   └── test/
├── models/
│   ├── resnet_best_model.h5
│   ├── mobilenetv2_best_model.h5
│   └── efficientnetb2_best_model.h5
├── notebooks/
│   └── preprocessandtrain.ipynb
├── scripts/
│   ├── facedetection.py
│   └── haarcascade_frontalface_default.xml
├── documentation/
├── requirements.txt
├── LICENSE
└── README.md


---
## Requirements

It is recommended to use a virtual environment.

pip install -r requirements.txt

### Main Dependencies
- Python 3.8+
- TensorFlow / Keras
- OpenCV (opencv-python)
- NumPy, Pandas
- Matplotlib
- Scikit-learn

(Refer to requirements.txt for exact versions.)

---

## Running Inference / Demo

Ensure a trained model is present in the models/ directory  
(e.g. resnet_best_model.h5).

If using Streamlit:

streamlit run app.py

### Demo Features
- Upload facial images
- Automatic face detection
- Skin concern classification
- Estimated age display
- Confidence score per prediction
- Download annotated images
- Export prediction logs as CSV

---

## Training Pipeline

The notebook notebooks/preprocessandtrain.ipynb contains the full training workflow:

- Image loading and preprocessing (resize, normalization)
- Data augmentation for better generalization
- Transfer learning using pre-trained CNN backbones
- Training with early stopping and model checkpointing
- Evaluation on the test set
- Saving the best-performing model to models/

---

## Evaluation & Results

- Training and validation performance is visualized in the notebook
- Multiple CNN backbones were evaluated:
  - ResNet
  - MobileNetV2
  - EfficientNetB2
- Best-performing models are stored in models/ for inference

Results can be reproduced by running the training notebook.

---

## Notes & Possible Improvements

- Dataset follows a folder-per-class structure compatible with Keras utilities
- Potential enhancements:
  - Landmark-based face alignment
  - Improved age estimation model
  - Class imbalance handling
  - Larger and more diverse datasets
  - Model optimization for faster inference

---

## License

This project is distributed under the terms specified in the LICENSE file.

