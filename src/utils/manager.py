import os
import sys
import time
from queue import Queue, Empty
import threading
import multiprocessing as mp
from src.utils.tools import get_video_path
from src.capture.video_reader_pipeline import reading_threads
from src.processing.inference_pipeline import inferencing_threads, inferencing_processes
from src.utils.displaying import display_multi_stream, close_all_windows


def manager(video_path: str, model_path: str, infer_type: str):
    video_paths = get_video_path(video_path)
    if infer_type == "thread" or infer_type == "t":
        read_queues = [Queue(maxsize=5) for _ in range(len(video_paths))]
        result_queues = [Queue(maxsize=5) for _ in range(len(video_paths))]
        stop_event = threading.Event()
    elif infer_type == "process" or infer_type == "p":
        read_queues = [mp.Queue(maxsize=5) for _ in range(len(video_paths))]
        result_queues = [mp.Queue(maxsize=5) for _ in range(len(video_paths))]
        stop_event = mp.Event()

    read_threads = reading_threads(
        video_paths=video_paths, read_queues=read_queues, stop_event=stop_event
    )

    if infer_type == "thread" or infer_type == "t":
        infer_threads = inferencing_threads(
            read_queues=read_queues,
            result_queues=result_queues,
            model_path=model_path,
            stop_event=stop_event,
        )
    elif infer_type == "process" or infer_type == "p":
        infer_processes = inferencing_processes(
            read_queues=read_queues,
            result_queues=result_queues,
            model_path=model_path,
            stop_event=stop_event,
        )

    fps_start_time = time.time()
    total_frames = 0
    fps_update_interval = 5
    is_fullscreen = True

    try:
        while True:
            all_results = []
            for i in range(len(read_queues)):
                try:
                    result_data = result_queues[i].get(timeout=0.1)
                    all_results.append(result_data)
                    total_frames += 1
                except Empty:
                    continue

            if all_results:
                is_fullscreen, key = display_multi_stream(
                    all_results, is_fullscreen, fps=30
                )
                if key == ord("q"):
                    break

            elapsed = time.time() - fps_start_time
            if elapsed >= fps_update_interval:
                current_fps = total_frames / elapsed
                print(
                    f"[FPS] 总推理 FPS: {current_fps:.2f} (已处理 {total_frames} 帧, 耗时 {elapsed:.2f}s)"
                )
                fps_start_time = time.time()
                total_frames = 0

    except KeyboardInterrupt:
        print(f"正在退出……")
    finally:
        stop_event.set()
        close_all_windows()
        for t in read_threads:
            t.join(timeout=1)
        if infer_type == "thread" or infer_type == "t":
            for t in infer_threads:
                t.join(timeout=1)
        elif infer_type == "process" or infer_type == "p":
            for p in infer_processes:
                p.terminate()
                p.join(timeout=1)
                if p.is_alive():
                    p.kill()
                    p.join()
            os._exit(0)
