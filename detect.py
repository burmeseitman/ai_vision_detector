# Run inference on CCTV/RTSP/Video for Toy Car detection
# Usage:
#   python detect.py --source 0
#   python detect.py --source rtsp://user:pass@ip/...
import argparse, cv2, time, json
from ultralytics import YOLO

def overlay_info(frame, detections, fps):
    h, w = frame.shape[:2]
    label_y = 30
    import cv2 as _cv2
    _cv2.putText(frame, f"AI VISION — FPS: {fps:.1f}", (10, label_y), _cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
    for r in detections:
        for i in range(len(r.boxes)):
            cls = int(r.boxes.cls[i].item())
            conf = float(r.boxes.conf[i].item())
            x1,y1,x2,y2 = map(int, r.boxes.xyxy[i].tolist())
            name = r.names.get(cls, str(cls))
            _cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 2)
            _cv2.putText(frame, f"{name}:{conf:.2f}", (x1, max(15,y1-10)), _cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)
    return frame

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=str, default="outputs/runs/cv_yolov8/weights/best.pt")
    parser.add_argument("--source", type=str, default="0")
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--show", action="store_true", help="Show window")
    parser.add_argument("--save", action="store_true", help="Save output video")
    args = parser.parse_args()

    model = YOLO(args.weights) if args.weights.endswith(".pt") else YOLO("yolov8n.pt")

    import cv2
    cap = cv2.VideoCapture(0 if args.source == "0" else args.source)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = None
    if args.save:
        out = cv2.VideoWriter("outputs/runs/cv_yolov8/inference.mp4", fourcc, 20.0, (int(cap.get(3)), int(cap.get(4))))

    last = time.time()
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        results = model.predict(frame, conf=args.conf, verbose=False)
        fps = 1.0 / max(1e-6, (time.time() - last))
        last = time.time()

        frame = overlay_info(frame, results, fps)

        if args.show:
            cv2.imshow("AI VISION", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        if out:
            out.write(frame)

    cap.release()
    if out:
        out.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
