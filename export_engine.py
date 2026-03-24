from ultralytics import YOLO

model = YOLO("models/yolo26s.pt")

model.export(
    format="engine",
    device=0,
    half=True,
    imgsz=640
)