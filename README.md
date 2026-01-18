# 🧬 AI DermalScan Pro: Facial Skin & Age Analysis System

![Status](https://img.shields.io/badge/Status-Milestone%203%20Complete-success)
![Python](https://img.shields.io/badge/Python-3.10-blue)
![Streamlit](https://img.shields.io/badge/Framework-Streamlit-ff4b4b)

**AI DermalScan Pro** is a computer vision application designed to analyze facial skin conditions and estimate biological age. It combines a custom-trained **MobileNetV2** model for skin classification with a **Caffe-based AgeNet** model, enhanced by smart heuristic logic for higher accuracy.

---

## ⚠️ CRITICAL SETUP INSTRUCTION ⚠️

**Please Read Before Running:**
To comply with GitHub file size limits, the `age_net.caffemodel` file included in this repository is a **dummy placeholder**. The application will **crash** if you do not replace it with the real model weights.

**Steps to Fix:**
1. **Delete** the existing `age_net.caffemodel` file in your root folder.
2. **Download** the real model (44MB) from this link:  
   👉 [**Download Real AgeNet Model**](https://github.com/spmallick/learnopencv/raw/master/AgeGender/age_net.caffemodel)
3. **Place** the downloaded file in the root directory of this project.

---

## 🚀 Key Features

* **Multi-Face Detection:** Automatically detects and processes multiple faces in group photos.
* **Smart Age Estimation:** Uses **Context Padding (+20%)** to fix head-shape errors.
* **Skin Condition Analysis:** Classifies skin into 4 categories:
    * *Acne / Clear Skin*
    * *Dark Spots*
    * *Puffy Eyes*
    * *Wrinkles*
* **Batch Processing:** Supports uploading multiple images simultaneously.
* **Dynamic Reporting:** Generates downloadable **CSV Batch Reports**.
* **Advanced Visualization:** Features a "Neon Cyberpunk" UI with dynamic **Seaborn** charts.

---

## ⚙️ Installation & Usage

### **1. Clone the Repository**
```bash
git clone [https://github.com/YOUR_USERNAME/AI-DermalScan-Pro.git](https://github.com/YOUR_USERNAME/AI-DermalScan-Pro.git)
cd AI-DermalScan-Pro
