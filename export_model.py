from ultralytics import YOLO

def export_system_model(model_name="yolo26s.pt"):
    """
    Exports the YOLO model to ONNX format.
    ONNX is the 'Universal Language' of AI models. It allows you to
    use this model in Android, iOS, Windows C++, and even web browsers.
    """
    print(f"--- Exporting {model_name} to ONNX ---")
    
    # 1. Load the model
    model = YOLO(model_name)
    
    # 2. Export to ONNX
    # imgsz=1280 (matching system settings)
    # opset=12 (standard for cross-platform compatibility)
    path = model.export(format="onnx", imgsz=1280, opset=12)
    
    print(f"\n✅ Export Complete: {path}")

if __name__ == "__main__":
    export_system_model()
