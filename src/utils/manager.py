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
        stop_event.clear()

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
    fps_update_interval = 1
    is_fullscreen = True

    try:
        while not stop_event.is_set():
            all_results = []
            for i in range(len(read_queues)):
                try:
                    result_data = result_queues[i].get_nowait()
                    all_results.append(result_data)
                    total_frames += 1
                except Empty:
                    continue

            if all_results:
                print(len(all_results))
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
        stop_event.set()
    finally:
        close_all_windows()
        for t in read_threads:
            t.join(timeout=1)
        if infer_type == "thread" or infer_type == "t":
            for t in infer_threads:
                t.join(timeout=1)
        elif infer_type == "process" or infer_type == "p":
            print("等待推理进程关闭……")
            for p in infer_processes:
                if p.is_alive():
                    p.join(timeout=2)
                    if p.is_alive():
                        print(f"退出{p.pid}")
                        p.terminate()
            for q in read_queues + result_queues:
                q.cancel_join_thread()
                q.close()
            print("全部清理完成")
            time.sleep(0.5)
            os.system("stty sane")
