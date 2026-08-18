import multiprocessing as mp
from multiprocessing.shared_memory import SharedMemory
from pathlib import Path

import numpy as np

from inference import run_inference
from tracking import run_tracking
from video import run_producer


def main():
    mp.set_start_method("spawn", force=True)

    video_names = ["out4d.mp4", "out13d.mp4"]
    video_path = Path("video")
    video_sources = [str(video_path / name) for name in video_names]

    model_file = "yolo11n.onnx"
    target_size = (1280, 704)
    batch_size = len(video_sources)
    num_slots = 6

    batch_shape = (batch_size, 3, target_size[1], target_size[0])
    frame_bytes = int(np.prod(batch_shape) * np.dtype(np.float16).itemsize)

    frame_shm_pools = [
        SharedMemory(create=True, size=frame_bytes) for _ in range(num_slots)
    ]
    frame_shm_names = [shm.name for shm in frame_shm_pools]

    frame_free_queue = mp.Queue()
    frame_ready_queue = mp.Queue(maxsize=num_slots)
    for i in range(num_slots):
        frame_free_queue.put(i)

    pred_shape = (batch_size, 84, 18480)
    pred_bytes = int(np.prod(pred_shape) * np.dtype(np.float32).itemsize)

    pred_shm_pools = [
        SharedMemory(create=True, size=pred_bytes) for _ in range(num_slots)
    ]

    pred_shm_names = [shm.name for shm in pred_shm_pools]

    pred_free_queue = mp.Queue()
    pred_ready_queue = mp.Queue(maxsize=num_slots)
    for i in range(num_slots):
        pred_free_queue.put(i)

    p1_producer = mp.Process(
        target=run_producer,
        args=(
            video_sources,
            frame_free_queue,
            frame_ready_queue,
            frame_shm_names,
            batch_shape,
            target_size,
        ),
    )
    p2_inference = mp.Process(
        target=run_inference,
        args=(
            frame_ready_queue,
            frame_free_queue,
            frame_shm_names,
            batch_shape,
            pred_ready_queue,
            pred_free_queue,
            pred_shm_names,
            pred_shape,
            model_file,
        ),
    )
    p3_tracking = mp.Process(
        target=run_tracking,
        args=(
            pred_ready_queue,
            pred_free_queue,
            pred_shm_names,
            pred_shape,
            frame_free_queue,
            frame_shm_names,
            batch_shape,
        ),
    )

    p1_producer.start()
    p2_inference.start()
    p3_tracking.start()

    p1_producer.join()
    p2_inference.join()
    p3_tracking.join()

    for shm in frame_shm_pools + pred_shm_pools:
        shm.close()
        shm.unlink()


if __name__ == "__main__":
    main()
