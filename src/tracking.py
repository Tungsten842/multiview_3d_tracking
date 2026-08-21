import sys
from multiprocessing.shared_memory import SharedMemory

import numpy as np
import rerun as rr
import supervision as sv

from track2d import Tracker2D
from track3d import Tracker3D


def plot_tracks_3d_rerun(tracks_3d: list) -> None:
    if len(tracks_3d) == 0:
        return
    positions = np.array([t["pos"] for t in tracks_3d])
    track_ids = np.array([t["id"] for t in tracks_3d])
    labels = [tid for tid in track_ids]
    colors = np.column_stack(
        [
            (track_ids * 17) % 200 + 100,
            (track_ids * 31) % 200 + 100,
            (track_ids * 47) % 200 + 100,
        ]
    ).astype(np.uint8)
    rr.log(
        "world/tracks3d/positions",
        rr.Points3D(positions=positions, colors=colors, radii=0.15),
        static=True,
    )
    sizes = np.full((len(positions), 3), [0.6, 0.6, 1.7])
    rr.log(
        "world/tracks3d/boxes",
        rr.Boxes3D(centers=positions, sizes=sizes, colors=colors, labels=labels),
        static=True,
    )


class TrackSaver:
    def __init__(self, num_cams=2):
        self.annotations_2d = [[] for _ in range(num_cams)]

    def save(self, tracks, frame_index):
        for cam_idx in range(len(self.annotations_2d)):
            cam_tracks = tracks[cam_idx]
            detections = sv.Detections(
                xyxy=cam_tracks[:, :4],
                class_id=cam_tracks[:, 6].astype(int),
                tracker_id=cam_tracks[:, 4].astype(int),
            )
            mask = np.isin(detections.class_id, [0, 1])
            detections = detections[mask]

            self.annotations_2d[cam_idx].append(detections)

        if frame_index == 525:
            images_dict = {}
            annotations_dict = {}
            dummy_image = np.zeros((1280, 720, 3), dtype=np.uint8)

            for cam_idx in range(len(self.annotations_2d)):
                for i, detections in enumerate(self.annotations_2d[cam_idx]):
                    name = f"cam{cam_idx}_item{i}.png"
                    images_dict[name] = dummy_image

                    annotations_dict[name] = detections

            dataset = sv.DetectionDataset(
                classes=["player", "sports balls"],
                images=images_dict,
                annotations=annotations_dict,
            )

            dataset.as_coco(
                annotations_path="annotations.json",
                images_directory_path=None,
            )
            sys.exit()


def plot_tracks_rerun(
    camera_idx: int,
    frame: np.ndarray,
    tracks: np.ndarray | None,
) -> None:
    entity_path = f"cameras/camera_{camera_idx}"

    boxes = tracks[:, :4]
    track_ids = tracks[:, 4]
    class_ids = tracks[:, 6]

    labels = [
        f"id={int(track_id)} class={int(class_id)}"
        for track_id, class_id in zip(track_ids, class_ids)
    ]

    colors = np.column_stack(
        [
            (track_ids * 17) % 200 + 100,
            (track_ids * 31) % 200 + 100,
            (track_ids * 47) % 200 + 100,
        ]
    ).astype(np.uint8)

    centroids = np.column_stack(
        [
            (boxes[:, 0] + boxes[:, 2]) / 2.0,
            (boxes[:, 1] + boxes[:, 3]) / 2.0,
        ]
    )

    rr.log(
        f"{entity_path}/tracks",
        rr.Boxes2D(
            array=boxes,
            array_format=rr.Box2DFormat.XYXY,
            colors=colors,
            labels=labels,
            show_labels=True,
            class_ids=class_ids,
        ),
        static=True,
    )
    rr.log(
        f"{entity_path}/centroids",
        rr.Points2D(positions=centroids, colors=colors, radii=8),
        static=True,
    )
    rr.log(f"{entity_path}/image", rr.Image(frame), static=True)


def run_tracking(
    pred_ready_queue,
    pred_free_queue,
    pred_shm_names,
    pred_shape,
    frame_free_queue,
    frame_shm_names,
    frame_shape,
    calib_dir="camera",
):
    pred_shms = [SharedMemory(name=name) for name in pred_shm_names]
    frame_shms = [SharedMemory(name=name) for name in frame_shm_names]

    pred_arrays = [
        np.ndarray(
            pred_shape,
            dtype=np.float32,
            buffer=shm.buf,
        )
        for shm in pred_shms
    ]
    frame_arrays = [
        np.ndarray(
            frame_shape,
            dtype=np.float16,
            buffer=shm.buf,
        )
        for shm in frame_shms
    ]

    tracker_2d = Tracker2D(
        num_cams=pred_shape[0],
        conf_threshold=0.20,
        nms_threshold=0.75,
    )

    tracker_3d = Tracker3D(calib_dir)

    for i, cam in enumerate(tracker_3d.cameras):
        rr.log(
            f"cameras/camera_{i}",
            rr.Transform3D(translation=cam["center"], mat3x3=cam["R"].T),
            static=True,
        )

    rr.init(
        "multiview_3d_tracking",
    )
    rr.connect_grpc()

    rr.log(
        "world/origin",
        rr.Arrows3D(
            origins=[[0, 0, 0], [0, 0, 0], [0, 0, 0]],
            vectors=[[2.0, 0, 0], [0, 2.0, 0], [0, 0, 2.0]],
            colors=[[255, 0, 0], [0, 255, 0], [0, 0, 255]],
        ),
        static=True,
    )

    rr.log(
        "world/floor",
        rr.Mesh3D(
            vertex_positions=[
                [-22, -10.4, 0],
                [22, -10.4, 0],
                [22, 10.4, 0],
                [-22, 10.4, 0],
            ],
            triangle_indices=[
                [0, 1, 2],
                [0, 2, 3],
            ],
            vertex_colors=[[70, 70, 70]] * 4,
        ),
        static=True,
    )

    frame_index = 0
    track_saver = TrackSaver()
    try:
        while True:
            frame_index += 1
            pred_slot, frame_slot = pred_ready_queue.get()

            predictions = pred_arrays[pred_slot]
            frames = frame_arrays[frame_slot]
            frames = np.transpose(frames, (0, 2, 3, 1))

            tracks_2d = tracker_2d.update(
                predictions,
                frames,
            )

            camera_idx = 1
            plot_tracks_rerun(
                1,
                frames[camera_idx],
                tracks_2d[camera_idx],
            )
            tracks_3d = tracker_3d.update(tracks_2d)
            plot_tracks_3d_rerun(tracks_3d)

            pred_free_queue.put(pred_slot)
            frame_free_queue.put(frame_slot)

            track_saver.save(tracks_2d, frame_index)

    finally:
        for shm in pred_shms:
            shm.close()

        for shm in frame_shms:
            shm.close()
