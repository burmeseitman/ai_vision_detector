# AI Vision Detector — Toy Car Detection (YOLO + Python)

This repository contains the **prototype** and **product** scaffolding for an AI that checks **Toy Cars** from camera feeds (mercedes benz g63, mini cooper, dc comics, forklift, 70 camaro rs, gmc hummer).

## Features
- Real-time Toy Car detection with **YOLOv8** (via `ultralytics`).
- Webcam/CCTV/RTSP/Video inference with on-screen overlays and JSON events.
- Training pipeline with dataset config, splits, and evaluation metrics.
- Simple REST API (FastAPI) for live checks, plus a minimal Streamlit demo.
- Clear structure to meet **Project** (prototype) and **Final Project** deliverables.

## Quickstart
```bash
# 1) Create env
python -m venv .venv
source .venv/bin/activate  # on Windows: .venv\Scripts\activate

# 2) Install
pip install -r requirements.txt

# 3) (Optional) Split raw dataset into train/val/test
python split_dataset.py

# 4) Validate dataset structure
python tools/verify_dataset.py datasets/toy_car

# 5) Train
python train.py --data configs/cv_dataset.yaml --epochs 50 --img 640

# 5) Inference on webcam (index 0) or RTSP/cctv url
python detect.py --source 0
# or
python detect.py --source rtsp://username:password@IP:554/Streaming/Channels/101

# 6) REST API
uvicorn api:app --host 0.0.0.0 --port 8000

# 7) Streamlit demo
streamlit run app.py
```

## Labels (suggested minimum)
- **mercedes benz g63**, **mini cooper**, **dc comics**, **forklift**, **70 camaro rs**, **gmc hummer**

## Folder Structure
```
ai_vision_detector/
  api.py
  app.py
  detect.py
  train.py
  requirements.txt
  README.md
  LICENSE
  configs/
    cv_dataset.yaml
    yolo_hyperparams.yaml
  datasets/
    raw_data/
      images/
      labels/
    toy_car/
      images/{train,valid,test}/
      labels/{train,valid,test}/
  tools/
    verify_dataset.py
    export_samples.py
  outputs/
    runs/   # training runs (models, metrics)
    logs/
```