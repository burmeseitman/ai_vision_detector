# Streamlit demo app for Toy Car Detection using Ultralytics YOLOv8
import streamlit as st
from ultralytics import YOLO
from PIL import Image
import io, os

st.set_page_config(page_title="AI VISION — Demo", layout="wide")
st.title("🤖 AI VISION — Toy Car Detection (YOLO)")

model_path = "outputs/runs/cv_yolov8/weights/best.pt"
try:
    model = YOLO(model_path)
except Exception:
    model = YOLO("yolov8n.pt")

uploaded = st.file_uploader("Upload an image", type=["jpg","jpeg","png"])
if uploaded:
    img = Image.open(uploaded).convert("RGB")
    res = model.predict(img, conf=0.25, verbose=False)
    st.image(img, caption="Input", use_column_width=True)
    for r in res:
        st.json({"detections": [
            {"class": r.names.get(int(r.boxes.cls[i].item()), str(int(r.boxes.cls[i].item()))),
             "conf": float(r.boxes.conf[i].item()),
             "bbox": list(map(float, r.boxes.xyxy[i].tolist()))}
            for i in range(len(r.boxes))
        ]})
