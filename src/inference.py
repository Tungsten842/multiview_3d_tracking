import os
from multiprocessing.shared_memory import SharedMemory
from time import time

import numpy as np
import onnxruntime as ort


def run_inference(
    frame_ready_queue,
    frame_free_queue,
    frame_shm_names,
    frame_shape,
    pred_ready_queue,
    pred_free_queue,
    pred_shm_names,
    pred_shape,
    model_file,
):
    frame_shms = [SharedMemory(name=name) for name in frame_shm_names]
    pred_shms = [SharedMemory(name=name) for name in pred_shm_names]

    frame_arrays = [
        np.ndarray(frame_shape, dtype=np.float32, buffer=shm.buf) for shm in frame_shms
    ]
    pred_arrays = [
        np.ndarray(pred_shape, dtype=np.float32, buffer=shm.buf) for shm in pred_shms
    ]

    try:
        cache_dir = "/tmp/migraph_cache"
        os.makedirs(cache_dir, exist_ok=True)
        os.environ["ORT_MIGRAPHX_MODEL_CACHE_PATH"] = cache_dir

        session = ort.InferenceSession(
            model_file,
            providers=["MIGraphXExecutionProvider"],
        )

        input_name = session.get_inputs()[0].name

        iter = 0
        while True:
            t = time()
            start_time = t
            frame_slot = frame_ready_queue.get()
            wait_for_frame = time() - t

            t = time()
            pred_slot = pred_free_queue.get()
            wait_for_pred_slot = time() - t

            frame_array = frame_arrays[frame_slot]
            pred_array = pred_arrays[pred_slot]

            t = time()
            raw_output = session.run(
                None,
                {input_name: frame_array},
            )[0]
            inference_time = time() - t

            t = time()
            np.copyto(pred_array, raw_output)
            wait_for_copy = time() - t

            pred_ready_queue.put((pred_slot, frame_slot))
            if iter % 8 == 0:
                fps = 1 / (time() - start_time)
                print(
                    f"\rInfer={inference_time * 1000:.1f}ms | "
                    f"Frame_wait={wait_for_frame * 1000:.1f}ms | "
                    f"Pred_wait={wait_for_pred_slot * 1000:.1f}ms |",
                    f"Copy_wait={wait_for_copy * 1000:.1f}ms |",
                    f"FPS={fps:.1f}",
                    end="",
                )
            iter += 1

    finally:
        for shm in frame_shms:
            shm.close()

        for shm in pred_shms:
            shm.close()
