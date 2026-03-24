from ultralytics import YOLO

model = YOLO("models/yolov8s.pt")

model.export(
    format="engine",
    device=0,
    half=True,
    imgsz=640
)