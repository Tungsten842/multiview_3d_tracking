import os
import shutil
from pathlib import Path

from ultralytics import YOLO


def main():
    os.environ["YOLO_AUTOINSTALL"] = "false"

    model = YOLO("src/yolo11-p2.yaml").load("yolo11n.pt")

    # Manually copy P3 and P4 head
    pre = YOLO("yolo11n.pt").model.state_dict()
    cur = model.model.state_dict()
    remap = {"21.cv2.1": "23.cv2.0", "21.cv2.2": "23.cv2.1", "21.dfl": "23.dfl"}
    for k in cur:
        for c, p in remap.items():
            if c in k:
                cur[k] = pre[k.replace(c, p)]

    model.model.load_state_dict(cur)

    model.train(
        data="final_dataset/dataset.yaml",
        rect=True,
        cache="ram",
        workers=1,
        imgsz=1280,
        deterministic=False,
        batch=10,
        device="cuda",
        scale=0.3,
        cls_pw=1,
        erasing=0.0,
        epochs=20,
        warmup_epochs=1,
        close_mosaic=2,
    )

    onnx_path = model.export(
        format="onnx", quantize=16, device=0, imgsz=(704, 1280), batch=2
    )
    final_destination = Path.cwd() / "yolo11-p2.onnx"

    shutil.copy(onnx_path, final_destination)


if __name__ == "__main__":
    main()
