import streamlit as st
import cv2
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from PIL import Image
import io 
import plotly.graph_objects as go 

# CONFIGURATION & NEON UI 
st.set_page_config(page_title="AI DermalScan", layout="wide")

st.markdown("""
<style>
    /* Dark Theme */
    .stApp { background-color: #050505; color: #ffffff; }
    h1 { color: #00ffcc; text-shadow: 0 0 10px #00ffcc; }
    
    /* Stats Container */
    .css-1r6slb0 {
        background: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(0, 255, 204, 0.2);
        border-radius: 10px;
        padding: 10px;
    }
</style>
""", unsafe_allow_html=True)

# LOAD RESOURCES 
@st.cache_resource
def load_resources():
    # 1. Skin Model
    try:
        skin_model = load_model('mobilenet_skin.h5')
    except:
        st.error("Error: mobilenet_skin.h5 not found.")
        return None, None, None, None

    # 2. Age Model
    try:
        age_net = cv2.dnn.readNet("age_net.caffemodel", "age_deploy.prototxt")
    except:
        st.error("Error: AgeNet files missing.")
        return None, None, None, None
        
    # 3. Haar Cascade
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    labels = ['Clear Skin', 'Dark Spots', 'Puffy Eyes', 'Wrinkles']
    return skin_model, age_net, face_cascade, labels

skin_model, age_net, face_cascade, class_labels = load_resources()

AGE_MEANS = [4, 8, 15, 22, 30, 43, 55, 80]
NEON_COLORS = ['#00ff00', '#ff0055', '#00ccff', '#ffff00'] 

# PROCESS IMAGE
def process_image(uploaded_file):
    # Convert to OpenCV format
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img_bgr = cv2.imdecode(file_bytes, 1)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    
    h_img, w_img, _ = img_rgb.shape
    
    # SCALING VARIABLES
    thick = max(2, int(h_img / 300))
    font_scale = max(0.5, h_img / 1000)
    
    # HAAR CASCADE DETECTION 
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    
    faces = face_cascade.detectMultiScale(
        gray, 
        scaleFactor=1.2, 
        minNeighbors=6, 
        minSize=(int(w_img*0.1), int(h_img*0.1))
    )
    
    annotated_img = img_rgb.copy()
    faces_found_data = []
    
    for (x, y, w, h) in faces:
        ratio = w / h
        if ratio < 0.7 or ratio > 1.3:
            continue 

        face_roi_bgr = img_bgr[y:y+h, x:x+w]
        face_roi_rgb = img_rgb[y:y+h, x:x+w]
        
        # 1. Skin Prediction
        try:
            face_resized = cv2.resize(face_roi_rgb, (224, 224))
            face_norm = face_resized / 255.0
            face_batch = np.expand_dims(face_norm, axis=0)
            
            preds = skin_model.predict(face_batch)[0]
            max_index = np.argmax(preds)
            winner_label = class_labels[max_index]
            winner_conf = preds[max_index] * 100
            face_stats = {class_labels[j]: float(preds[j] * 100) for j in range(4)}
        except:
            winner_label = "Error"
            winner_conf = 0
            face_stats = {}

        # 2. Age Prediction
        try:
            face_blob = cv2.dnn.blobFromImage(face_roi_bgr, 1.0, (227, 227), (78.4, 87.8, 114.9), swapRB=False)
            age_net.setInput(face_blob)
            age_preds = age_net.forward()[0]
            predicted_age = int(sum(prob * mean for prob, mean in zip(age_preds, AGE_MEANS)))
            
            if winner_label == 'Wrinkles' and winner_conf > 50:
                if predicted_age < 45:
                    predicted_age = 50 + (45 - predicted_age) 
                    predicted_age = min(predicted_age, 85) 
            
            if winner_label == 'Puffy Eyes' and predicted_age < 18:
                predicted_age = 25

        except:
            predicted_age = "?"

        cv2.rectangle(annotated_img, (x, y), (x+w, y+h), (0, 255, 127), thick)
        
        label_text = f"{winner_label} ({winner_conf:.0f}%) | Age: {predicted_age}"
        (w_text, h_text), baseline = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thick)
        
        if y - h_text - 10 < 0:
            text_y = y + h + h_text + 15 
        else:
            text_y = y - 10 
            
        text_x = x
        if text_x + w_text > w_img:
            text_x = w_img - w_text - 10

        cv2.rectangle(annotated_img, (text_x - 5, text_y - h_text - 5), (text_x + w_text + 5, text_y + 5), (0, 0, 0), -1)
        cv2.putText(annotated_img, label_text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thick, cv2.LINE_AA)
        
        faces_found_data.append({
            "Condition": winner_label,
            "Age": predicted_age,
            "Confidence": winner_conf,
            "Stats": face_stats
        })

    return annotated_img, faces_found_data

def create_neon_pie_chart(stats_dict):
    labels = list(stats_dict.keys())
    values = list(stats_dict.values())
    max_val = max(values)
    pull_values = [0.15 if v == max_val else 0 for v in values]
    
    fig = go.Figure(data=[go.Pie(
        labels=labels, values=values, pull=pull_values, hole=.4,
        marker=dict(colors=NEON_COLORS, line=dict(color='#000000', width=2)),
        textinfo='label+percent', hoverinfo='label+value+percent'
    )])
    
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
        showlegend=False, margin=dict(t=0, b=0, l=0, r=0),
        font=dict(color='white', size=14)
    )
    return fig

# UI LAYOUT 
st.title("🧬 AI DermalScan")

uploaded_files = st.file_uploader("Upload Images", type=['jpg', 'png', 'jpeg'], accept_multiple_files=True)

if uploaded_files:
    if len(uploaded_files) > 4:
        st.warning("Analyzing first 4 images only.")
        uploaded_files = uploaded_files[:4]
        
    if st.button("Analyze Batch"):
        all_results_for_csv = []
        cols = st.columns(len(uploaded_files))
        
        for idx, file in enumerate(uploaded_files):
            with cols[idx]:
                img, faces_data = process_image(file)
                st.image(img, use_column_width=True, caption=f"Found {len(faces_data)} Faces")
                
                img_pil = Image.fromarray(img)
                buf = io.BytesIO()
                img_pil.save(buf, format="PNG")
                byte_im = buf.getvalue()
                
                st.download_button(
                    label="📥 Download Image",
                    data=byte_im,
                    file_name=f"detected_{file.name}",
                    mime="image/png",
                    key=f"dl_btn_{idx}"
                )
                
                if faces_data:
                    for i, face in enumerate(faces_data):
                        with st.expander(f"Face {i+1}: {face['Condition']}"):
                            st.write(f"**Predicted Age:** {face['Age']}")
                            fig = create_neon_pie_chart(face['Stats'])
                            st.plotly_chart(fig, use_container_width=True)
                        
                        # ADDING FULL STATS TO THE REPORT 
                        row_data = {
                            "Filename": file.name,
                            "Face ID": i+1,
                            "Condition": face['Condition'],
                            "Age": face['Age'],
                            "Confidence": f"{face['Confidence']:.1f}%"
                        }
                        # Add individual condition probabilities to the table/CSV
                        for cond, prob in face['Stats'].items():
                            row_data[f"{cond} %"] = f"{prob:.1f}%"
                            
                        all_results_for_csv.append(row_data)
                else:
                    st.error("No faces detected.")

        if all_results_for_csv:
            st.markdown("---")
            st.subheader("📄 Batch Report")
            report_df = pd.DataFrame(all_results_for_csv)
            
            st.dataframe(report_df, use_container_width=True)
            
            csv = report_df.to_csv(index=False).encode('utf-8')

            st.download_button("📥 Download Batch CSV", csv, "dermalscan_results.csv", "text/csv")
