import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(page_title="Medical AI System", layout="wide")

# =========================
# 🎨 CLEAN PREMIUM UI CSS
# =========================
st.markdown("""
<style>

/* Background */
.stApp {
    background-color: black;
}

/* Header */
.header {
    font-size: 40px;
    font-weight: 700;
    margin-bottom: 5px;
}

/* Subtitle */
.sub {
    color: #6b7280;
    margin-bottom: 20px;
}

/* Result text */
.result {
    font-size: 30px;
    font-weight: bold;
    margin-top: 15px;
}

/* ================= SIDEBAR ================= */
section[data-testid="stSidebar"] {
    background-color: #0f172a;
}

/* Sidebar text */
section[data-testid="stSidebar"] * {
    color: white !important;
}

/* ================= FILE UPLOADER ================= */

/* OUTER BOX */
[data-testid="stFileUploader"] {
    background: rgba(255,255,255,0.06);
    border: 1px solid rgba(255,255,255,0.15);
    border-radius: 14px;
    padding: 18px;
}

/* REMOVE INNER WHITE BOX (IMPORTANT FIX) */
[data-testid="stFileUploader"] > div {
    background: transparent !important;
    border: none !important;
    box-shadow: none !important;
}

/* Upload button */
[data-testid="stFileUploader"] button {
    width: 100%;
    background: linear-gradient(90deg, #22c55e, #16a34a);
    color: white;
    border-radius: 8px;
    font-weight: 600;
    border: none;
    padding: 10px;
}

/* Hover */
[data-testid="stFileUploader"] button:hover {
    transform: scale(1.03);
}

/* ================= MAIN BUTTON ================= */
.stButton>button {
    width: 100%;
    height: 55px;
    font-size: 18px;
    font-weight: 600;
    border-radius: 10px;
    border: none;
    background: linear-gradient(90deg, #2563eb, #1e3a8a);
    color: white;
}

.stButton>button:hover {
    transform: scale(1.03);
}

</style>
""", unsafe_allow_html=True)

# =========================
# HEADER
# =========================
st.markdown("<div class='header'>🩺 Medical Diagnosis System</div>", unsafe_allow_html=True)
st.markdown("<div class='sub'>AI-powered Chest X-ray Pneumonia Detection</div>", unsafe_allow_html=True)

# =========================
# LOAD MODEL
# =========================
model = tf.keras.models.load_model("model/model.h5")

# =========================
# SIDEBAR
# =========================
st.sidebar.title("⚙️ Controls")
st.sidebar.markdown("Upload chest X-ray image below 👇")

uploaded_file = st.sidebar.file_uploader(
    "📤 Drag & Drop or Browse",
    type=["jpg", "png", "jpeg"]
)

run = st.sidebar.button("🚀 Run Diagnosis")

# =========================
# MAIN LAYOUT
# =========================
col1, col2 = st.columns([1, 2])

# IMAGE
with col1:
    st.subheader("X-ray Preview")

    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, use_container_width=True)
    else:
        st.info("Upload image from sidebar")

# RESULT
with col2:
    st.subheader("Diagnosis Result")

    if uploaded_file and run:

        img = np.array(image)
        img = cv2.resize(img, (224, 224))
        img = img / 255.0
        img = np.expand_dims(img, axis=0)

        with st.spinner("Analyzing..."):
            result = model.predict(img)

        confidence = float(result[0][0]) * 100

        if result > 0.5:
            st.markdown(
                "<div class='result' style='color:#ef4444;'>⚠️ Pneumonia Detected</div>",
                unsafe_allow_html=True
            )
            st.progress(int(confidence))
            st.write(f"Confidence: {confidence:.2f}%")
        else:
            st.markdown(
                "<div class='result' style='color:#22c55e;'>✅ Normal</div>",
                unsafe_allow_html=True
            )
            st.progress(int(100-confidence))
            st.write(f"Confidence: {100-confidence:.2f}%")

    else:
        st.info("Upload image and click 'Run Diagnosis'")

# =========================
# FOOTER
# =========================
st.markdown("---")
st.caption("Developed by Sneha • AI Medical Imaging Project")