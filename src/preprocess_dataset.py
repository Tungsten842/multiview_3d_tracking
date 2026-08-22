import os
import zipfile

import fiftyone as fo
from huggingface_hub import hf_hub_download


def main():
    zip_path = hf_hub_download(
        repo_id="GabrieleGiudici/E-BARD-detection",
        filename="all.zip",
        repo_type="dataset",
    )
    output_dir = "ebard"
    os.makedirs(output_dir, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(output_dir)

    for split in ["train", "valid"]:
        ds1 = fo.Dataset.from_dir(
            dataset_dir="ebard/coco/",
            data_path=split,
            labels_path=f"annotations/instances_{split}.json",
            dataset_type=fo.types.COCODetectionDataset,
        )

        ds2 = fo.Dataset.from_dir(
            dataset_dir="roboflowdataset/",
            data_path=split,
            labels_path=f"{split}/_annotations.coco.json",
            dataset_type=fo.types.COCODetectionDataset,
        )
        ds1.merge_samples(ds2)

        class_mapping = {
            "basketball": "sports ball",
            "Ball": "sports ball",
            "referee": "person",
            "Refree_F": "person",
            "Refree_M": "person",
            "Refree_1": "person",
            "Refree_2": "person",
            "player": "person",
            "Red_0": "person",
            "Red_11": "person",
            "Red_12": "person",
            "Red_13": "person",
            "Red_16": "person",
            "Red_2": "person",
            "Red_23": "person",
            "Red_4": "person",
            "Red_7": "person",
            "Red_9": "person",
            "White_10": "person",
            "White_11": "person",
            "White_13": "person",
            "White_14": "person",
            "White_16": "person",
            "White_2": "person",
            "White_22": "person",
            "White_25": "person",
            "White_27": "person",
            "White_34": "person",
        }

        ds1 = ds1.map_labels("detections", class_mapping)

        print(ds1.distinct("detections.detections.label"))

        split = "val" if split == "valid" else split

        ds1.export(
            export_dir="final_dataset",
            dataset_type=fo.types.YOLOv5Dataset,
            label_field="detections",
            classes=["person", "sports ball"],
            split=split,
        )


if __name__ == "__main__":
    main()
