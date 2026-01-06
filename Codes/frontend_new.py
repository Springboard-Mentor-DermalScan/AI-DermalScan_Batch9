import streamlit as st
import requests
import pandas as pd
from PIL import Image, ImageDraw
import base64
import io

# --- CONFIGURATION ---
API_URL = "http://127.0.0.1:8002/predict"

st.set_page_config(page_title="DermalScan AI", layout="wide")

if "history" not in st.session_state:
    st.session_state.history = []

# --- BACKGROUND LOGIC ---
def get_base64(bin_file):
    try:
        with open(bin_file, 'rb') as f:
            return base64.b64encode(f.read()).decode()
    except:
        return None

def set_ui_style(bg_path):
    # 1. VIDEO BACKGROUND
    if bg_path.endswith(".mp4"):
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
                </style>
                <video autoplay muted loop id="bgVideo">
                    <source src="data:video/mp4;base64,{bin_str}" type="video/mp4">
                </video>
            """, unsafe_allow_html=True)
            
    # 2. IMAGE BACKGROUND
    else:
        bin_str = get_base64(bg_path)
        if bin_str:
            ext = "gif" if bg_path.endswith(".gif") else "png"
            st.markdown(f"""
                <style>
                .stApp {{
                    background-image: url("data:image/{ext};base64,{bin_str}");
                    background-attachment: fixed; background-size: cover;
                }}
                </style>
            """, unsafe_allow_html=True)

    # 3. CSS STYLES (Hide Deploy Button + Styling)
    st.markdown("""
        <style>
        /* --- HIDE STREAMLIT MENU & FOOTER --- */
        [data-testid="stToolbar"] { visibility: hidden !important; display: none !important; }
        [data-testid="stDecoration"] { visibility: hidden !important; display: none !important; }
        [data-testid="stStatusWidget"] { visibility: hidden !important; }
        #MainMenu { visibility: hidden; display: none; }
        header { visibility: hidden; display: none; }
        footer { visibility: hidden; display: none; }

        /* --- UI STYLES --- */
        .main-header { font-size: 40px; color: white; text-align: center; }
        
        .dark-box {
            background-color: rgba(20, 30, 48, 0.9);
            padding: 20px; border-radius: 20px;
            color: white; text-align: center;
            margin-bottom: 20px;
            box-shadow: 0 4px 15px rgba(0,0,0,0.5);
        }
        
        /* History Section */
        .history-container {
            margin-top: 30px;
            background: transparent !important;
            border: none !important;
            box-shadow: none !important;
        }
        .history-container h3 {
            color: white;
            text-shadow: 2px 2px 4px #000;
            margin-bottom: 15px;
            background: transparent !important;
        }
        
        /* Table Styling */
        .history-container table {
            width: 100%;
            background-color: rgba(0, 0, 0, 0.7);
            color: white;
            border-collapse: collapse;
            font-family: sans-serif;
            border-radius: 10px;
            overflow: hidden;
        }
        .history-container th {
            background-color: #2c3e50;
            color: white;
            padding: 12px;
            text-align: left;
        }
        .history-container td {
            padding: 10px;
            border-bottom: 1px solid #444;
        }
        </style>
    """, unsafe_allow_html=True)

# --- UPDATE PATH HERE ---
set_ui_style(r"c:\Users\Admin\Downloads\Fancy_Face_Detection_Video_Generated.mp4")

# --- MAIN APP UI ---
st.markdown('<div class="dark-box"><h1 style="margin:0;">DermalScan AI</h1><p>Skin Condition & Age Analysis</p></div>', unsafe_allow_html=True)

with st.container():
    uploaded_file = st.file_uploader("Upload Patient Scan", type=["jpg", "png", "jpeg"])

if uploaded_file:
    img = Image.open(uploaded_file).convert("RGB")
    col1, col2 = st.columns(2)
    with col1:
        st.image(img, caption="Original Image", use_container_width=True)

    if st.button("Analyze", use_container_width=True):
        with st.spinner("Processing..."):
            try:
                img_byte_arr = io.BytesIO()
                img.save(img_byte_arr, format='JPEG')
                files = {"file": ("image.jpg", img_byte_arr.getvalue(), "image/jpeg")}
                
                response = requests.post(API_URL, files=files)
                
                if response.status_code == 200:
                    data = response.json()
                    
                    # Draw Box
                    result_img = img.copy()
                    draw = ImageDraw.Draw(result_img)
                    for det in data.get("detections", []):
                        draw.rectangle(det["bbox"], outline="#00FF11", width=6)

                    with col2:
                        st.image(result_img, caption=f"Detected Face Condition: {data['class']}", use_container_width=True)

                    st.markdown("---")
                    m1, m2, m3 = st.columns(3)
                    m1.metric("Condition", data['class'])
                    m2.metric("Confidence", f"{data['confidence']}%")
                    m3.metric("Estimated Age", f"{data['age']}")

                    st.session_state.history.append({
                        "File": uploaded_file.name,
                        "Condition": data["class"],
                        "Age": data["age"],
                        "Confidence score": f"{data['confidence']}%"
                    })
                else:
                    st.error(f"Server Error: {response.status_code}")
            except Exception as e:
                st.error(f"Connection Failed: {e}")

# --- FIXED HISTORY SECTION ---
if st.session_state.history:
    df = pd.DataFrame(st.session_state.history)
    html_table = df.to_html(index=False, border=0, classes="table-style")
    
    st.markdown(f"""
        <div class="history-container">
            <h3>Prediction History</h3>
            {html_table}
        </div>
    """, unsafe_allow_html=True)