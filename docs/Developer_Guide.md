```md

\\\\# Developer Guide – AI DermalScan



\\\\## Tech Stack

\\\\- Python

\\\\- Streamlit

\\\\- TensorFlow / Keras

\\\\- OpenCV

\\\\- Pandas



\\\\## Core Files

\\\\- app.py → UI + inference + export

\\\\- mobilenetv2\\\\\\\_module3.h5 → trained model

\\\\- haarcascade\\\\\\\_frontalface\\\\\\\_default.xml → face detection



\\\\## Pipeline Flow

1\\\\. Image upload

2\\\\. Face detection

3\\\\. Face preprocessing

4\\\\. Model inference

5\\\\. Annotation drawing

6\\\\. Logging \\\\\\\& export



\\\\## Extending the Project

\\\\- Replace Haar Cascade with YOLO

\\\\- Add real age regression model

\\\\- Cloud deployment (Streamlit Cloud / AWS)



