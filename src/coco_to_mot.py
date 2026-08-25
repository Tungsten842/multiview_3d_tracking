import json
import os
from collections import defaultdict


def convert_coco_to_mot(coco_path, out_path):
    with open(coco_path) as f:
        coco = json.load(f)

    img_dict = {img["id"]: img for img in coco["images"]}

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

        zoom_factor = 1.20 if cam_id == "out13" else 1.0

        with open(f"{base}_{cam_id}{ext}", "w") as f:
            for a in anns:
                fid = frame_map[a["image_id"]] - 1
                x, y, w, h = a["bbox"]

                x /= 3.0
                y /= 3.0
                w /= 3.0
                h /= 3.0

                # Deal with camera zoom
                if zoom_factor != 1.0:
                    img_info = img_dict[a["image_id"]]
                    img_w = img_info["width"] / 3.0
                    img_h = img_info["height"] / 3.0
                    cx, cy = img_w / 2.0, img_h / 2.0

                    x = (x - cx) * zoom_factor + cx
                    y = (y - cy) * zoom_factor + cy
                    w = w * zoom_factor
                    h = h * zoom_factor

                f.write(
                    f"{fid},{a['category_id']},{x:.2f},{y:.2f},{w:.2f},{h:.2f},1,1,1.0\n"
                )


convert_coco_to_mot("annotations/train/_annotations.coco.json", "annotations/gt.txt")
