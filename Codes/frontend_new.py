import streamlit as st
import requests
import pandas as pd
from PIL import Image, ImageDraw
import base64
import io

# --- CONFIGURATION ---
API_URL = "http://127.0.0.1:8002/predict"  # Matches Backend Port 8002

st.set_page_config(page_title="DermalScan AI", layout="wide")

if "history" not in st.session_state:
    st.session_state.history = []

# --- BACKGROUND LOGIC (Supports MP4, GIF, PNG, JPG) ---
def get_base64(bin_file):
    try:
        with open(bin_file, 'rb') as f:
            return base64.b64encode(f.read()).decode()
    except:
        return None

def set_ui_style(bg_path):
    # 1. VIDEO BACKGROUND (MP4)
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
            
    # 2. IMAGE BACKGROUND (GIF / JPG / PNG)
    else:
        bin_str = get_base64(bg_path)
        if bin_str:
            ext = "gif" if bg_path.endswith(".gif") else "png"
            st.markdown(f"""
                <style>
                .stApp {{
                    background-image: url("data:image/{ext};base64,{bin_str}");
                    background-attachment: fixed;
                    background-size: cover;
                }}
                </style>
            """, unsafe_allow_html=True)

    # 3. COMMON CSS STYLES (Boxes, Text, Tables)
    st.markdown("""
        <style>
        .main-header { font-size: 40px; color: white; text-shadow: 2px 2px 4px #000; text-align: center; }
        .dark-box {
            background-color: rgba(20, 30, 48, 0.9);
            padding: 20px; border-radius: 20px;
            color: white; text-align: center;
            box-shadow: 0 4px 15px rgba(0,0,0,0.5);
            margin-bottom: 20px;
        }
        .table-container {
            background-color: rgba(255, 255, 255, 0.95);
            padding: 20px; border-radius: 15px; margin-top: 30px;
            color: black;
        }
        </style>
    """, unsafe_allow_html=True)

# --- ⚠️ UPDATE THIS PATH TO YOUR FILE ---
# You can use a .gif OR an .mp4 file here.
set_ui_style(r"c:\Users\Admin\Downloads\Fancy_Face_Detection_Video_Generated.mp4")

# --- MAIN APP UI ---
st.markdown('<div class="dark-box"><h1 style="margin:0;">DermalScan AI</h1><p>Advanced Age detection & Skin Condition Analysis</p></div>', unsafe_allow_html=True)

with st.container():
    uploaded_file = st.file_uploader("Upload Patient Scan", type=["jpg", "png", "jpeg"])

if uploaded_file:
    img = Image.open(uploaded_file).convert("RGB")
    col1, col2 = st.columns(2)
    
    with col1:
        st.image(img, caption="Original Image", use_container_width=True)

    # ANALYZE BUTTON
    if st.button("Analyize", use_container_width=True):
        with st.spinner("Processing..."):
            try:
                # Send to Backend
                img_byte_arr = io.BytesIO()
                img.save(img_byte_arr, format='JPEG')
                files = {"file": ("image.jpg", img_byte_arr.getvalue(), "image/jpeg")}
                
                response = requests.post(API_URL, files=files)
                
                if response.status_code == 200:
                    data = response.json()
                    
                    # DRAW BOX
                    result_img = img.copy()
                    draw = ImageDraw.Draw(result_img)
                    for det in data.get("detections", []):
                        draw.rectangle(det["bbox"], outline="#00FF11", width=6)

                    with col2:
                        st.image(result_img, caption=f"Identified: {data['class']}", use_container_width=True)

                    # SHOW METRICS
                    st.markdown("---")
                    m1, m2, m3 = st.columns(3)
                    m1.metric("Condition", data['class'])
                    m2.metric("Confidence", f"{data['confidence']}%")
                    m3.metric("Est. Age", f"{data['age']}")

                    # SAVE HISTORY
                    st.session_state.history.append({
                        "File": uploaded_file.name,
                        "Condition": data["class"],
                        "Age": data["age"],
                        "Conf": f"{data['confidence']}%"
                    })
                else:
                    st.error(f"Server Error: {response.status_code}")
                    
            except Exception as e:
                st.error(f"Connection Failed: {e}. Is backend_new.py running?")

# HISTORY TABLE
if st.session_state.history:
    st.markdown('<div class="table-container">', unsafe_allow_html=True)
    st.subheader("Prediction History")
    st.table(pd.DataFrame(st.session_state.history))
    st.markdown('</div>', unsafe_allow_html=True)