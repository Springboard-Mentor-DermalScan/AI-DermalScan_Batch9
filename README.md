**📋 Project Overview**

The primary objective of this project is to detect and locate age-related facial characteristics and categorize them into four specific classes: **Wrinkles, Dark Spots, Puffy Eyes, and Clear Skin.** 

The system uses a **Convolutional Neural Network (CNN)** for classification and integrates a web-based frontend for user interaction.

---------------------------------------------------------------------------------------------------------------------------------------

**🛠️ Tech Stack**

-Language: Python (v3.10.1) 

-Deep Learning: TensorFlow/Keras using the EfficientNetB0 architecture 

-Computer Vision: OpenCV (Haar Cascades) for face detection 

-Web Framework: Streamlit 

-Data Science Libraries: Pandas, NumPy, Scikit-learn, Matplotlib, and Seaborn

-IDEs: Visual Studio Code and Jupyter Notebook

---------------------------------------------------------------------------------------------------------------------------------------

**🚀 Key Modules & Methodology**

**1. Dataset Preparation & Preprocessing**

-Data Aggregation: Compiled a balanced dataset of 1,203 facial images.

-Augmentation: Applied rotations, flips, and zooms to artificially expand the dataset and improve model generalization.

-Normalization: Resized images to 224 x 224 pixels and normalized pixel values for consistent input.


**2. Model Development**

-Architecture: Leveraged transfer learning with EfficientNetB0 and the Adam optimizer.

-Performance: Achieved a training accuracy of 90.76% over 10 epochs.

-Evaluation: Monitored training/validation loss and accuracy curves to ensure stability and prevent overfitting.


**3. Inference Pipeline & UI**

-Face Detection: Combines OpenCV for initial localization with the CNN for analysis of cropped face regions.

-Real-time Feedback: The UI provides bounding boxes annotated with predicted class labels, confidence percentages, and estimated age ranges.

-Logging & Export: Implemented a CSV logging system to record session history and a download feature for annotated results.

---------------------------------------------------------------------------------------------------------------------------------------

**📈 Results**

-High Precision: Successfully identifies multiple faces and provides granular skin health metrics.

-Accessibility: Provides a non-invasive tool for real-time age estimation and skin condition monitoring.
