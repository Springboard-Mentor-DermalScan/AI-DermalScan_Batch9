import streamlit as st
import requests
import pandas as pd
from PIL import Image, ImageDraw
import base64
import io
import time # Added for tracking prediction speed

# --- CONFIGURATION ---
API_URL = "http://127.0.0.1:8002/predict" 

st.set_page_config(page_title="DermalScan AI", layout="wide")

if "history" not in st.session_state:
    st.session_state.history = []

def get_base64(bin_file):
    try:
        with open(bin_file, 'rb') as f:
            return base64.b64encode(f.read()).decode()
    except: return None

def set_ui_style(bg_path):
    bin_str = get_base64(bg_path)
    if bin_str:
        st.markdown(f"""
            <style>
            .stApp {{ background: rgba(0,0,0,0); }}
            #bgVideo {{
                position: fixed; right: 0; bottom: 0;
                min-width: 100%; min-height: 100%;
                z-index: -1; opacity: 0.8;
            }}
            .table-container {{
                background-color: rgba(255, 255, 255, 0.95);
                padding: 20px; border-radius: 15px; margin-top: 30px;
                color: black;
            }}
            </style>
            <video autoplay muted loop id="bgVideo">
                <source src="data:video/mp4;base64,{bin_str}" type="video/mp4">
            </video>
        """, unsafe_allow_html=True)

set_ui_style(r"c:\Users\Admin\Downloads\Fancy_Face_Detection_Video_Generated.mp4")

st.markdown('<div style="text-align:center; color:white;"><h1>DermalScan AI</h1><p>Skin Condition & Age Analysis</p></div>', unsafe_allow_html=True)

uploaded_file = st.file_uploader("Upload Patient Scan", type=["jpg", "png", "jpeg"])

if uploaded_file:
    img = Image.open(uploaded_file).convert("RGB")
    col1, col2 = st.columns(2)
    
    with col1:
        st.image(img, caption="Original Image", use_container_width=True)

    if st.button("Analyze", use_container_width=True):
        with st.spinner("Analyzing..."):
            try:
                # Track Prediction Time
                start_timer = time.time()
                
                img_byte_arr = io.BytesIO()
                img.save(img_byte_arr, format='JPEG')
                files = {"file": ("image.jpg", img_byte_arr.getvalue(), "image/jpeg")}
                
                response = requests.post(API_URL, files=files)
                
                prediction_time = round(time.time() - start_timer, 2)
                
                if response.status_code == 200:
                    data = response.json()
                    
                    # DRAW LABELS (Condition, Age, Conf) ABOVE BOX
                    result_img = img.copy()
                    draw = ImageDraw.Draw(result_img)
                    for det in data.get("detections", []):
                        bbox = det["bbox"]
                        draw.rectangle(bbox, outline="#00FF11", width=6)
                        
                        # Requirement: Condition, Age, Confidence above box
                        label = f"{data['class']} | Age: {data['age']} | Confidence score: {data['confidence']}%"
                        draw.rectangle([bbox[0], bbox[1]-35, bbox[0]+450, bbox[1]], fill="#00FF11")
                        draw.text((bbox[0] + 5, bbox[1] - 30), label, fill="black")

                    with col2:
                        st.image(result_img, caption=f"Analyzed in {prediction_time}s", use_container_width=True)

                    # SAVE HISTORY WITH PREDICTION TIME
                    st.session_state.history.append({
                        "Prediction Time": f"{prediction_time}s",
                        "File": uploaded_file.name,
                        "Condition": data["class"],
                        "Age": data["age"],
                        "Confidence score": f"{data['confidence']}%"
                    })
                else:
                    st.error("Server Error")
            except Exception as e:
                st.error(f"Error: {e}")

# --- HISTORY TABLE & DOWNLOAD ---
if st.session_state.history:
    # 1. Custom CSS for the Dark Container and Neon Button
    st.markdown("""
        <style>
                

        /* Styling the Download Button to match Analyze button */
        div.stDownloadButton > button {
            background-color: #00FF11 !important;
            color: black !important;
            font-weight: bold !important;
            border: none !important;
            width: 100% !important;
            border-radius: 5px !important;
            padding: 10px !important;
            margin-bottom: 15px !important;
        }

        div.stDownloadButton > button:hover {
            background-color: #00cc0e !important;
            box-shadow: 0 0 10px #00FF11 !important;
        }
        </style>
    """, unsafe_allow_html=True)

    # 2. Start of Styled Container
    st.markdown('<div class="history-section">', unsafe_allow_html=True)
    st.subheader("Prediction History")

    # 3. CSV Download Logic
    history_df = pd.DataFrame(st.session_state.history)
    csv = history_df.to_csv(index=False).encode('utf-8')

    # 4. Displaying the Dataframe with the dark theme
    # We use st.dataframe because it inherits the app's dark theme better than st.table
    st.dataframe(
        history_df.iloc[::-1], 
        use_container_width=True,
        hide_index=True
    )
    
    st.markdown('</div>', unsafe_allow_html=True)

    st.download_button(
        label="📥 Download Prediction History as CSV",
        data=csv,
        file_name="dermalscan_history.csv",
        mime="text/csv",
    )