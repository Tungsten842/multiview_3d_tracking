from multiprocessing.shared_memory import SharedMemory
from queue import Queue
from threading import Thread

import cv2
import numpy as np


class VideoWorker(Thread):
    def __init__(self, video_path, target_size, queue_size=3):
        super().__init__(daemon=True)
        self.video_path = video_path
        self.target_size = target_size
        self.queue = Queue(maxsize=queue_size)
        self.inv_255 = np.float32(1.0 / 255.0)

    def run(self):
        cap = cv2.VideoCapture(
            self.video_path,
            cv2.CAP_FFMPEG,
            [cv2.CAP_PROP_HW_ACCELERATION, cv2.VIDEO_ACCELERATION_ANY],
        )
        if not cap.isOpened():
            self.queue.put(None)
            return

        while True:
            ret, frame = cap.read()
            if not ret:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue

            resized = cv2.resize(
                frame, self.target_size, interpolation=cv2.INTER_LINEAR
            )
            rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
            chw = rgb.transpose(2, 0, 1)

            processed = chw.astype(np.float32) * self.inv_255

            self.queue.put(processed)

        cap.release()


class BatchProducer(Thread):
    def __init__(self, workers, free_queue, ready_queue, shm_names, batch_shape):
        super().__init__(daemon=True)
        self.workers = workers
        self.free_queue = free_queue
        self.ready_queue = ready_queue
        self.shm_names = shm_names
        self.batch_shape = batch_shape

    def run(self):
        shm_blocks = [SharedMemory(name=name) for name in self.shm_names]
        buffers = [
            # np.ndarray(self.batch_shape, dtype=np.float16, buffer=shm.buf)
            np.ndarray(self.batch_shape, dtype=np.float32, buffer=shm.buf)
            for shm in shm_blocks
        ]

        while True:
            frames = []
            for worker in self.workers:
                frame = worker.queue.get()
                frames.append(frame)

            slot_idx = self.free_queue.get()

            for i, frame in enumerate(frames):
                buffers[slot_idx][i] = frame

            self.ready_queue.put(slot_idx)


class VideoPipeline:
    def __init__(
        self,
        video_paths,
        free_queue,
        ready_queue,
        shm_names,
        batch_shape,
        target_size,
        queue_size=2,
    ):

        self.workers = [
            VideoWorker(path, target_size, queue_size=queue_size)
            for path in video_paths
        ]
        self.producer = BatchProducer(
            self.workers, free_queue, ready_queue, shm_names, batch_shape
        )

    def run(self):
        for worker in self.workers:
            worker.start()
        self.producer.start()

    def stop(self):
        for worker in self.workers:
            worker.join()
        self.producer.join()


def run_producer(
    video_sources, free_queue, ready_queue, shm_names, batch_shape, target_size
):
    pipeline = VideoPipeline(
        video_sources, free_queue, ready_queue, shm_names, batch_shape, target_size
    )
    pipeline.run()
    pipeline.stop()
