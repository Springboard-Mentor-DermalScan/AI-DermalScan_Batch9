Dermal AI – AI-Based Facial Skin Analysis System

Dermal AI is a deep learning–powered web application that analyzes facial images to identify skin conditions such as Clear Skin, Puffy Eyes, Dark Spots, and Wrinkles.
The system provides visual annotations, confidence scores, and estimated age through an interactive web interface.

📌 Features

📤 Upload facial images (JPG / PNG)

🖼️ Display original and annotated images

🔲 Bounding box visualization on detected face

🧠 AI-based skin condition classification

📊 Confidence percentage for all skin classes

📈 Interactive bar chart for detailed analysis

🧮 Tabular prediction data (box coordinates, age, confidence)

⏱️ Processing time display

⬇️ Download annotated image

🎨 Clean, modern, and responsive UI


🧠 Skin Conditions Detected

1.Clear Skin

2.Puffy Eyes

3.Dark Spots

4.Wrinkles

🛠️ Tech Stack
1.Frontend

2.Streamlit

3.Backend / AI

4.Python 3.10

5.TensorFlow / Keras

6.OpenCV

7.NumPy

8.Pandas


⚙️ Installation & Setup

1️⃣ Create and Activate Environment

conda create -n dermal_ai python=3.10 -y

conda activate dermal_ai


2️⃣ Install Dependencies

pip install -r requirements.txt

If installing manually:

pip install streamlit tensorflow opencv-python numpy pandas pillow matplotlib

▶️ Run the Application

Navigate to the project directory:

cd Dermal_AI2

Start the Streamlit app:

streamlit run app.py

The app will open in your browser at:

http://localhost:8501


🧪 How It Works

1.User uploads a facial image

2.Image is preprocessed and passed to the trained model

3.Model predicts probabilities for all skin classes

4.Primary condition is selected based on highest confidence

5.Bounding box and labels are drawn on the image

6.Results are displayed via tables, charts, and visuals


📊 Output Details

1.Primary Skin Condition

2.Confidence Score (%)

3.Estimated Age

4.Bounding Box Coordinates

5.Processing Time

6.Class-wise Confidence Distribution

⚠️ Disclaimer

This application is developed for educational and research purposes only.

It is not a medical diagnostic tool and should not be used for clinical decisions.

🚀 Future Enhancements

1.Multi-face detection support

2.Real-time webcam analysis

3.Skin care product recommendations

4.Cloud deployment

5.Mobile-friendly UI

6.User history & reports
