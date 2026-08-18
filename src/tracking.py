from multiprocessing.shared_memory import SharedMemory

import numpy as np
import rerun as rr

from track2d import Tracker2D


def plot_tracks_rerun(
    camera_idx: int,
    frame: np.ndarray,
    tracks: np.ndarray | None,
) -> None:
    entity_path = f"cameras/camera_{camera_idx}"

    frame = (frame * 255).astype(np.uint8)

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

    rr.init(
        "multiview_3d_tracking",
    )
    rr.connect_grpc()

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

            camera_idx = 1
            plot_tracks_rerun(
                1,
                frames[camera_idx],
                tracks_2d[camera_idx],
            )

            pred_free_queue.put(pred_slot)
            frame_free_queue.put(frame_slot)

    finally:
        for shm in pred_shms:
            shm.close()

        for shm in frame_shms:
            shm.close()
