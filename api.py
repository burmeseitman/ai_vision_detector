# Minimal REST API using FastAPI and Ultralytics YOLOv8
from fastapi import FastAPI, UploadFile, File
from ultralytics import YOLO
from PIL import Image
import io

app = FastAPI(title="AI VISION — REST API")

# Use pretrained if best.pt missing
try:
    model = YOLO("outputs/runs/cv_yolov8/weights/best.pt")
except Exception:
    model = YOLO("yolov8n.pt")

@app.post("/predict_image")
async def predict_image(file: UploadFile = File(...)):
    data = await file.read()
    img = Image.open(io.BytesIO(data)).convert("RGB")
    results = model.predict(img, conf=0.25, verbose=False)
    dets = []
    for r in results:
        boxes = r.boxes
        names = r.names
        for i in range(len(boxes)):
            cls = int(boxes.cls[i].item())
            conf = float(boxes.conf[i].item())
            x1, y1, x2, y2 = map(float, boxes.xyxy[i].tolist())
            dets.append({"class": names.get(cls, str(cls)), "conf": conf, "bbox": [x1,y1,x2,y2]})
    return {"detections": dets}
