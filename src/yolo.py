import os

from ultralytics import YOLO

os.environ["YOLO_AUTOINSTALL"] = "false"
model = YOLO("yolo11n.pt")

# Export the model to ONNX format
model.export(format="onnx", quantize=16, device=0, imgsz=(704, 1280), batch=2)
