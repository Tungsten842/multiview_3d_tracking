import os
import shutil
from pathlib import Path

import torch
from ultralytics import YOLO


def main():
    os.environ["YOLO_AUTOINSTALL"] = "false"

    model = YOLO("yolo11-p2.yaml").load("yolo11n.pt")

    model.train(
        data="final_dataset/dataset.yaml",
        rect=True,
        cache="ram",
        workers=1,
        imgsz=1280,
        deterministic=False,
        batch=8,
        device="cuda",
        shear=4,
        scale=0.3,
        cls_pw=1,
        erasing=0.0,
        epochs=10,
        warmup_epochs=1,
        close_mosaic=1,
    )

    onnx_path = model.export(
        format="onnx", quantize=16, device=0, imgsz=(704, 1280), batch=2
    )
    final_destination = Path.cwd() / "yolo11-p2.onnx"

    shutil.copy(onnx_path, final_destination)


if __name__ == "__main__":
    main()
