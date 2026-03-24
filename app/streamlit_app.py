import streamlit as st
from ultralytics import YOLO
import cv2
import numpy as np
from PIL import Image
import time

# ------------------ PAGE CONFIG ------------------
st.set_page_config(page_title="AI Object Detection", layout="wide")

# ------------------ CUSTOM CSS ------------------
st.markdown("""
    <style>
    .main-title {
        font-size: 45px;
        font-weight: bold;
        text-align: center;
        background: -webkit-linear-gradient(45deg, #00c6ff, #0072ff);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .subtitle {
        text-align: center;
        font-size: 18px;
        color: gray;
        margin-bottom: 30px;
    }
    .card {
        padding: 20px;
        border-radius: 12px;
        background: #f1f3f6;
        text-align: center;
        box-shadow: 0px 4px 10px rgba(0,0,0,0.05);
        color:black !important;
    }
    </style>
""", unsafe_allow_html=True)

# ------------------ HEADER ------------------
st.markdown('<div class="main-title">🔍 AI Object Detection Dashboard</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Real-time Object Detection using YOLOv8</div>', unsafe_allow_html=True)

# ------------------ SIDEBAR ------------------
st.sidebar.title("⚙️ Settings")

option = st.sidebar.radio("Choose Input", ["Upload Image", "Use Webcam"])
confidence = st.sidebar.slider("Confidence Threshold", 0.1, 1.0, 0.5)

# ------------------ LOAD MODEL ------------------
model = YOLO("yolov8m.pt")

image = None
image_np = None

# ------------------ INPUT ------------------
if option == "Upload Image":
    uploaded_file = st.file_uploader("📤 Upload Image", type=["jpg", "png", "jpeg"])
    
    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")
        image_np = np.array(image)

elif option == "Use Webcam":
    camera_image = st.camera_input("📸 Capture Image")
    
    if camera_image:
        image = Image.open(camera_image).convert("RGB")
        image_np = np.array(image)

# ------------------ PROCESS ------------------
if image_np is not None:

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### 📷 Input")
        st.image(image, width="stretch")

    # Convert RGB → BGR
    image_np = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

    # Loading spinner
    with st.spinner("🔍 Detecting objects..."):
        time.sleep(1)
        results = model(image_np, conf=confidence)

    for r in results:
        boxes = r.boxes
        count = len(boxes)

        names = model.names
        detected_labels = []

        for box in boxes:
            cls_id = int(box.cls[0])
            label = names[cls_id]
            detected_labels.append(label)

        annotated_frame = r.plot()

    # Convert back BGR → RGB
    annotated_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)

    with col2:
        st.markdown("### 📊 Detection Output")
        st.image(annotated_frame, width="stretch")

    # ------------------ METRICS ------------------
    st.markdown("## 📈 Detection Summary")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown(f'<div class="card"><h2>🔢 {count}</h2><p>Objects Detected</p></div>', unsafe_allow_html=True)

    unique_labels = list(set(detected_labels))

    with col2:
        st.markdown(f'<div class="card"><h3>{" , ".join(unique_labels)}</h3><p>Detected Labels</p></div>', unsafe_allow_html=True)

else:
    st.info("👈 Upload image or use webcam from sidebar to start")