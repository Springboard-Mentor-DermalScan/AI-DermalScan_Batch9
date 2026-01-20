\# AI DermalScan – Skin Condition Detection System



AI DermalScan is a deep learning–based facial skin analysis system that detects multiple faces in an image, classifies skin conditions, estimates age ranges, and provides annotated visual outputs with export and logging features.



---



\## 📌 Features



\- Upload facial images via Streamlit UI

\- Multi-face detection using OpenCV Haar Cascade

\- Skin condition classification using MobileNetV2 (.h5 model)

\- Supported classes:

&nbsp; - Clear Skin

&nbsp; - Dark Spots

&nbsp; - Puffy Eyes

&nbsp; - Wrinkles

\- Random age estimation within class-defined ranges

\- Annotated image generation with bounding boxes

\- Prediction summary table

\- Prediction history logging

\- Export annotated images and CSV reports

\- Dark-themed interactive UI with animations



---



\## 🧠 Model Information



\- Architecture: MobileNetV2

\- Framework: TensorFlow / Keras

\- Input size: 224 × 224

\- Output: 4-class softmax classification

\- Model file: `model/mobilenetv2\_module3.h5`



---



\## 📁 Project Structure



AI\_DermalScan/

├── app.py

├── model/

├── DATASET/

├── exports/

├── README.md

├── requirements.txt

└── docs/



---



\## 🚀 How to Run the Project



\### 1️⃣ Install dependencies

```bash

pip install -r requirements.txt



\## Run the Streamlit app

streamlit run app.py



📤 Export \& Logging



-Annotated images are saved automatically

-CSV logs contain:

&nbsp;	-Timestamp

&nbsp;	-Face ID

&nbsp;	-Predicted class

&nbsp;	-Estimated age

&nbsp;	-Confidence

&nbsp;	-Bounding box coordinates

-Export options available in UI



📊 Evaluation Readiness



-Clean UI

-Real-time predictions

-No mock outputs

-Consistent logs

-Demo-ready application



👨‍💻 Author



Rounak Kumar Mishra

AI / Data Science Project

