import time
import threading
import multiprocessing as mp
from queue import Queue, Full, Empty
from src.processing.inference import InferenceEngine
from src.utils.tools import pre_processing
from src.processing.post_processing import post_processing


def inferencing_process(read_queue: mp.Queue, result_queue: mp.Queue, model_path: str, stop_event: mp.Event):  # type: ignore
    model = InferenceEngine(model_path=model_path)
    while not stop_event.is_set():
        try:
            start_time = time.time()
            frame = read_queue.get(timeout=0.1)
            frame_prep = pre_processing(frame[0])
            result = model.forward(frame_prep)
            result = post_processing(result[0].reshape(84, 8400), 0.5, 0.25)
            infer_time = time.time() - start_time
            infer_fps = 1.0 / infer_time if infer_time > 0 else 0
            frame.append((result, infer_fps))
            result_queue.put(frame, timeout=0.1)
        except Empty:
            continue
        except Full:
            continue
        except KeyboardInterrupt:
            break
    model.clean()
    time.sleep(1)
    print("进程正常退出")


def inferencing_processes(read_queues: list, result_queues: list, model_path: str, stop_event: mp.Event):  # type: ignore
    processes = []
    print()
    for i in range(len(read_queues)):
        p = mp.Process(
            target=inferencing_process,
            args=(read_queues[i], result_queues[i], model_path, stop_event),
            daemon=True,
        )
        processes.append(p)
        p.start()
        print(f"启动进程{i}")
    return processes


def inferencing_thread(
    read_queue: Queue, result_queue: Queue, model_path: str, stop_event: threading.Event
):
    model = InferenceEngine(model_path=model_path)
    while not stop_event.is_set():
        try:
            start_time = time.time()
            frame = read_queue.get(timeout=0.1)
            frame_prep = pre_processing(frame[0])
            result = model.forward(frame_prep)
            result = post_processing(result[0].reshape(84, 8400), 0.5, 0.25)
            infer_time = time.time() - start_time
            infer_fps = 1.0 / infer_time if infer_time > 0 else 0
            frame.append((result, infer_fps))
            result_queue.put(frame, timeout=0.1)
        except Empty:
            continue
        except Full:
            continue
        except KeyboardInterrupt:
            break
    model.clean()
    time.sleep(1)
    print("线程正常退出")


def inferencing_threads(
    read_queues: list, result_queues: list, model_path: str, stop_event: threading.Event
):
    processes = []
    for i in range(len(read_queues)):
        p = threading.Thread(
            target=inferencing_thread,
            args=(read_queues[i], result_queues[i], model_path, stop_event),
            daemon=True,
        )
        processes.append(p)
        p.start()
        print(f"启动线程{i}")
    return processes
