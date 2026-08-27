# Project Structure

| File(s)                     | Description        |
| --------------------------- | ---------------    |
| `video/convert.sh`          | Preprocesses camera footage. |
| `src/coco_to_mot.py`        | Converts annotations from Roboflow COCO to MOT. |
| `src/preprocess_dataset.py` | Merges annotations and preprocesses yolo training dataset. |
| `src/finetune_yolo.py`      | Finetunes yolo.     |
| `src/main.py`               | Starts pipeline.    |
| `src/video.py`              | Multi camera video publisher process. |
| `src/inference.py`          | Inference process. |
| `src/tracking.py`           | Tracking process with plotting. |
| `src/track2d.py`            | 2D tracking logic. |
| `src/track3d.py`            | 3D tracking logic. |
| `src/eval.sh`               | Runs evaluation.   |

# Getting Started
## Prerequisites & Environment Setup

### Docker Container
```bash
docker build .
```

### Execute with devcontainer cli or any supported text editor
```bash
devcontainer up --workspace-folder .
```

## Data Preparation & Preprocessing
- Save camera footage to "video" folder.
- Save Roboflow train annotations to "roboflowdataset" folder.
- Save Roboflow test annotations to "annotations" folder.

### Preprocess Camera Footage
```bash
bash video/convert.sh
```

### Convert Annotations
```bash
python src/coco_to_mot.py
```

### Build & Preprocess Dataset
```bash
python src/preprocess_dataset.py
```

## Model Finetuning
```bash
python src/finetune_yolo.py
```

## Start Visualization
```bash
rerun
```

## Run Pipeline
```bash
python src/main.py
```

## Run Evaluation

```bash
bash src/eval.sh
```
