# AI DermaScan – Skin Condition Detection

AI DermaScan is a deep learning–based application that detects facial skin conditions from images and displays annotated results with export functionality.

---

## Features

- Facial skin condition detection
- Multi-face detection
- Supported classes:
  - Clear Skin
  - Dark Spots
  - Puffy Eyes
  - Wrinkles
- Bounding box visualization
- Age estimation per detected class
- Prediction summary table
- Download annotated image
- Export prediction history as CSV
- Interactive Streamlit UI

---

## Sample UI Screenshots

## Sample UI Screenshots

![Home Screen](screenshots/home.png)
![Annotated Output](screenshots/annotated.png)
![CSV Export](screenshots/export.png)

## Model Details

Architecture: MobileNetV2

Framework: TensorFlow / Keras

Input Size: 224 × 224

Output Classes: 4

Model File: mobilenetv2_module3.h5

## Project Structure
AI_DermaScan/
├── app.py
├── model/
│   └── mobilenetv2_module3.h5
├── exports/
│   ├── images/
│   └── logs/
├── haarcascade_frontalface_default.xml
├── requirements.txt
└── README.md

## How to Run

Install dependencies:

pip install -r requirements.txt


Run the application:

streamlit run app.py

## Important Code Snippets

Load model:

model = tf.keras.models.load_model(
    "model/mobilenetv2_module3.h5",
    compile=False
)


Face detection:

face_detector = cv2.CascadeClassifier(
    "haarcascade_frontalface_default.xml"
)
faces = face_detector.detectMultiScale(gray, 1.3, 5)


Save annotated image:

cv2.imwrite(
    "exports/images/annotated.jpg",
    cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR)
)


Export CSV:

df.to_csv("exports/logs/prediction_history.csv", index=False)

Author

Rounak Kumar Mishra