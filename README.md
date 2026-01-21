## 1. Project Overview

The goal of DermalScan is to provide an automated, non-invasive solution for facial skin aging analysis. Users can upload facial images and instantly receive predictions with confidence scores and visual annotations, eliminating the need for immediate clinical consultation.

---

## 2. System Architecture

**Input → Preprocessing → Face Detection → CNN Model Prediction → Annotation → Web UI Output**

* Image uploaded via web interface
* Face detected using OpenCV Haar Cascades
* Face region preprocessed and resized
* EfficientNetB0 CNN predicts skin condition
* Results displayed with bounding boxes and confidence scores

---

## 3. Technologies Used

### Programming & ML

* Python 3.10+
* TensorFlow / Keras
* NumPy
* Pandas
* Scikit-learn

### Image Processing & Visualization

* OpenCV
* Matplotlib
* Seaborn

### Frontend & Backend

* Streamlit

### Development Tools

* VS Code
* Jupyter Notebook

---

## 4. Project Folder Structure

```
DermalScan/
│── dataset/
│   ├── wrinkles/
│   ├── dark_spots/
│   ├── puffy_eyes/
│   └── clear_skin/
│
│── model/
│   └── dermalscan_efficientnet.h5
│
│── app.py                   # Streamlit web application
│── dermalscan_AI.ipynb      # Model training notebook
│── frontend_new.py
│── backend_new.py            # Image preprocessing utilities
│── requirements.txt          # Python dependencies
│── README.md                 # Project documentation
```

---

## 5. Step-by-Step Installation (Virtual Environment Setup)

### ✅ Step 1: Clone the Repository

```bash
git clone https://github.com/your-username/DermalScan.git
cd DermalScan
```

### ✅ Step 2: Create a Virtual Environment

#### On Windows

```bash
python -m venv venv
```


### ✅ Step 3: Activate the Virtual Environment

#### Windows

```bash
venv\Scripts\activate
```

> 🔹 After activation, your terminal will show `(venv)`

### ✅ Step 4: Install Required Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

---

## 6. Detailed System Workflow

### 🔹 Step 1: Image Upload

* User uploads a facial image through the Streamlit web interface.

### 🔹 Step 2: Face Detection

* OpenCV Haar Cascade classifier detects the face region in the image.
* Bounding box is created around detected face.

### 🔹 Step 3: Image Preprocessing

* Cropped face is resized to **224×224 pixels**.
* Pixel values are normalized.
* Image is converted into a tensor format.

### 🔹 Step 4: Model Inference

* Preprocessed image is passed to the trained **EfficientNetB0 CNN**.
* Model predicts one of four classes:

  * Wrinkles
  * Dark Spots
  * Puffy Eyes
  * Clear Skin

### 🔹 Step 5: Prediction Output

* Model returns:

  * Predicted class label
  * Confidence percentage

### 🔹 Step 6: Visualization

* Bounding box drawn on detected face
* Label and confidence score annotated
* Results displayed on the web UI

### 🔹 Step 7: Export & Logging (Optional)

* Annotated image can be downloaded
* Prediction history saved as CSV file

---

## 7. Running the Application

### ▶️ Start the Streamlit App

```bash
streamlit run app.py
```

* Open the URL shown in the terminal (`http://localhost:8002`)
* Upload an image and view predictions

---

## 8. Output & Results

* Annotated image with detected face and skin condition
* Confidence score for prediction
* CSV file containing prediction history
* Model accuracy: **~90.76%**
<img width="1920" height="907" alt="image" src="https://github.com/user-attachments/assets/9c29641b-12d0-4562-94e8-15ba8484f638" />
<img width="1920" height="912" alt="image" src="https://github.com/user-attachments/assets/f5b26429-f282-4198-b4fc-3dd507930fb5" />
<img width="1920" height="448" alt="image" src="https://github.com/user-attachments/assets/e5979504-8f82-488f-b90e-0616624ff34b" />

---

## 9. Future Scope

* Age estimation with numeric age prediction
* Mobile app integration
* Cloud deployment (AWS / Azure)
* Dermatologist recommendation system
* Expanded skin condition categories

---

## Conclusion

DermalScan successfully demonstrates the application of deep learning and computer vision in healthcare and cosmetic analysis. The project provides a reliable, real-time, and user-friendly system for facial skin aging detection using modern AI techniques.

---

✨ *Developed by Gargi Kulkarni*
