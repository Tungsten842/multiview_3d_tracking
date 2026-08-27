import os
import sys
from multiprocessing.shared_memory import SharedMemory

import numpy as np
import rerun as rr

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
    sizes = np.array(
        [
            [0.24, 0.24, 0.24] if t.get("class_id") == 1 else [0.6, 0.6, 1.7]
            for t in tracks_3d
        ]
    )
    rr.log(
        "world/tracks3d/boxes",
        rr.Boxes3D(centers=positions, sizes=sizes, colors=colors, labels=labels),
        static=True,
    )


class TrackSaver:
    CAM_IDS = (2, 13)

    def __init__(
        self,
        num_cams=2,
        output_dir="predictions",
        orig_shape=(720, 1280),
        crop_shape=(704, 1280),
    ):
        self.num_cams = num_cams
        self.output_dir = output_dir
        self.mot_lines = {self.CAM_IDS[i]: [] for i in range(num_cams)}
        self.y_offset = (orig_shape[0] - crop_shape[0]) // 2
        self.x_offset = (orig_shape[1] - crop_shape[1]) // 2

        os.makedirs(self.output_dir, exist_ok=True)

    def save(self, tracks, frame_index):
        if (frame_index - 2) % 5 == 0:
            mot_frame_idx = ((frame_index - 2) // 5) + 1

            for cam_idx in range(self.num_cams):
                cam_tracks = tracks[cam_idx]
                if len(cam_tracks) == 0:
                    continue

                class_ids = cam_tracks[:, 6].astype(int)
                mask = np.isin(class_ids, [0, 1])
                filtered_tracks = cam_tracks[mask]

                cidx = self.CAM_IDS[cam_idx]

                for track in filtered_tracks:
                    x1, y1, x2, y2 = track[:4]
                    # Adjust coordinates crop
                    x1 += self.x_offset
                    y1 += self.y_offset
                    x2 += self.x_offset
                    y2 += self.y_offset

                    w = x2 - x1
                    h = y2 - y1

                    track_id = int(track[4])
                    conf = float(track[5])

                    line = f"{mot_frame_idx},{track_id},{x1:.2f},{y1:.2f},{w:.2f},{h:.2f},{conf:.2f},-1,-1,-1\n"
                    self.mot_lines[cidx].append(line)

        if frame_index == 500:
            for cidx, lines in self.mot_lines.items():
                mot_path = os.path.join(self.output_dir, f"out{cidx}_mot.txt")
                with open(mot_path, "w") as f:
                    f.writelines(lines)
                print(f"MOT predictions saved to {mot_path}", flush=True)

            sys.exit()


def plot_tracks_rerun(camera_idx, frame, tracks, frame_index):
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
    if frame_index % 2 == 0:
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

    tracker_2d = Tracker2D(num_cams=pred_shape[0])

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
            vectors=[[-2.0, 0, 0], [0, 2.0, 0], [0, 0, 2.0]],
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
    # track_saver = TrackSaver()
    try:
        while True:
            pred_slot, frame_slot = pred_ready_queue.get()

            predictions = pred_arrays[pred_slot]
            frames = frame_arrays[frame_slot]
            frames = np.transpose(frames, (0, 2, 3, 1))

            tracks_2d = tracker_2d.update(
                predictions,
                frames,
            )
            # track_saver.save(tracks_2d, frame_index)

            camera_idx = 1
            plot_tracks_rerun(
                camera_idx, frames[camera_idx], tracks_2d[camera_idx], frame_index
            )
            tracks_3d = tracker_3d.update(tracks_2d)
            plot_tracks_3d_rerun(tracks_3d)

            pred_free_queue.put(pred_slot)
            frame_free_queue.put(frame_slot)
            frame_index += 1

    finally:
        for shm in pred_shms:
            shm.close()

        for shm in frame_shms:
            shm.close()
