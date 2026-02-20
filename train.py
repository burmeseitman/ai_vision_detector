# Train YOLOv8 for Toy Car Detection
# Usage:
#   python train.py --data configs/cv_dataset.yaml --epochs 50 --img 640
import argparse, yaml, os
from ultralytics import YOLO

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="configs/cv_dataset.yaml")
    parser.add_argument("--model", type=str, default="yolov8n.pt")  # nano by default
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--img", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--project", type=str, default="outputs/runs")
    parser.add_argument("--name", type=str, default="cv_yolov8")
    args = parser.parse_args()

    assert os.path.exists(args.data), f"Dataset config not found: {args.data}"

    model = YOLO(args.model)
    results = model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.img,
        batch=args.batch,
        project=args.project,
        name=args.name,
        pretrained=True,
        workers=4,
        optimizer="auto"
    )
    print(results)

if __name__ == "__main__":
    main()