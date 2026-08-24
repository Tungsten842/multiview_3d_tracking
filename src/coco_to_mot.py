import json
import os
from collections import defaultdict


def convert_coco_to_mot(coco_path, out_path):
    with open(coco_path) as f:
        coco = json.load(f)

    cams = defaultdict(list)
    for img in coco["images"]:
        cam_id = img["file_name"].split("_frame_")[0]
        cams[cam_id].append(img)

    base, ext = os.path.splitext(out_path)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    for cam_id, imgs in cams.items():
        frame_map = {
            img["id"]: i + 1
            for i, img in enumerate(sorted(imgs, key=lambda x: x["file_name"]))
        }
        anns = sorted(
            [a for a in coco["annotations"] if a["image_id"] in frame_map],
            key=lambda a: frame_map[a["image_id"]],
        )

        with open(f"{base}_{cam_id}{ext}", "w") as f:
            for a in anns:
                fid = frame_map[a["image_id"]]
                x, y, w, h = a["bbox"]
                x = x / 3.0
                y = y / 3.0
                w = w / 3.0
                h = h / 3.0
                f.write(
                    f"{fid},{a['category_id']},{x:.2f},{y:.2f},{w:.2f},{h:.2f},1,1,1.0\n"
                )


convert_coco_to_mot("annotations/train/_annotations.coco.json", "annotations/gt.txt")
