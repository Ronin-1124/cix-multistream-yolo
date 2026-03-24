import time
import threading
import multiprocessing as mp
from queue import Queue, Full, Empty
from src.capture.video_reader_pipeline import reading_threads
from src.processing.inference import InferenceEngine
from src.utils.tools import get_video_path, pre_processing
from src.processing.post_processing import post_processing
from src.processing.displaying import display_multi_stream, close_all_windows
import os

def inferencing_process(read_queue: mp.Queue, result_queue: mp.Queue, model_path: str, stop_event: mp.Event): # type: ignore
    model = InferenceEngine(model_path=model_path)
    try:
        while not stop_event.is_set():
            try:
                frame = read_queue.get(timeout=0.1)
                frame_prep = pre_processing(frame[0])
                result = model.forward(frame_prep)
                result = post_processing(result[0].reshape(84,8400), 0.5, 0.25)
                frame.append(result)
                result_queue.put(frame, timeout=0.1)
            except Empty:
                continue
            except Full:
                continue
            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"NPU forward failed: {e}")
                break
    finally:
        time.sleep(1)
        model.clean()
        print("进程正常退出")


def inferencing_processes(read_queues: list, result_queues: list, model_path: str, stop_event: mp.Event): # type: ignore
    processes = []
    for i in range(len(read_queues)):
        p = mp.Process(
            target=inferencing_process,
            args=(read_queues[i], result_queues[i], model_path, stop_event),
            daemon=True
        )
        processes.append(p)
        p.start()
        print(f"启动进程{i}")
    return processes
    
    
def inferencing_thread(read_queue: Queue, result_queue: Queue, model_path: str, stop_event: threading.Event):
    model = InferenceEngine(model_path=model_path)
    while not stop_event.is_set():
        try:
            frame = read_queue.get(timeout=0.1)
            frame_prep = pre_processing(frame[0])
            result = model.forward(frame_prep)
            result = post_processing(result[0].reshape(84,8400), 0.5, 0.25)
            frame.append(result)
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


def inferencing_threads(read_queues: list, result_queues: list, model_path: str, stop_event: threading.Event):
    processes = []
    for i in range(len(read_queues)):
        p = threading.Thread(
            target=inferencing_thread,
            args=(read_queues[i], result_queues[i], model_path, stop_event),
            daemon=True
        )
        processes.append(p)
        p.start()
        print(f"启动线程{i}")
    return processes


# if __name__ == "__main__":
    # video_paths = get_video_path("data/test_videos_360P/test9.mp4")
    video_paths = get_video_path("data/test_videos_360P")
    read_queues = [Queue(maxsize=5) for _ in range(len(video_paths))]
    result_queues = [Queue(maxsize=5) for _ in range(len(video_paths))]

    stop_event = threading.Event()

    read_threads = reading_threads(video_paths=video_paths, read_queues=read_queues, stop_event=stop_event)

    model_path = "models/yolov8n.cix"

    infer_threads = inferencing_threads(
        read_queues=read_queues,
        result_queues=result_queues,
        model_path=model_path,
        stop_event=stop_event
    )

    fps_start_time = time.time()
    total_frames = 0
    fps_update_interval = 2.0
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
                is_fullscreen, key = display_multi_stream(all_results, is_fullscreen, fps=30)
                if key == ord('q'):
                    break

            elapsed = time.time() - fps_start_time
            if elapsed >= fps_update_interval:
                current_fps = total_frames / elapsed
                print(f"[FPS] 总推理 FPS: {current_fps:.2f} (已处理 {total_frames} 帧, 耗时 {elapsed:.2f}s)")
                fps_start_time = time.time()
                total_frames = 0

    except KeyboardInterrupt:
        print(f"正在退出……")
    finally:
        stop_event.set()
        close_all_windows()
        for t in read_threads:
            t.join(timeout=2)
        for t in infer_threads:
            t.join(timeout=2)


if __name__ == "__main__":
    stop_event = mp.Event()
    stop_event.clear()
    video_paths = get_video_path("data/test_videos_360P")
    read_queues = [mp.Queue(maxsize=5) for _ in range(len(video_paths))]
    result_queues = [mp.Queue(maxsize=5) for _ in range(len(video_paths))]


    read_threads = reading_threads(video_paths=video_paths, read_queues=read_queues, stop_event=stop_event)

    model_path = "models/yolov8n.cix"


    infer_processes = inferencing_processes(
        read_queues=read_queues,
        result_queues=result_queues,
        model_path=model_path,
        stop_event=stop_event
    )

    fps_start_time = time.time()
    total_frames = 0
    fps_update_interval = 2.0

    try:
        while not stop_event.is_set():
            for i in range(len(read_queues)):
                try:
                    result_data = result_queues[i].get_nowait()
                    frame, thread_id, frame_id, detections = result_data
                    total_frames += 1
                except Empty:
                    continue

            elapsed = time.time() - fps_start_time
            if elapsed >= fps_update_interval:
                current_fps = total_frames / elapsed
                print(f"[FPS] 总推理 FPS: {current_fps:.2f} (已处理 {total_frames} 帧, 耗时 {elapsed:.2f}s)")
                fps_start_time = time.time()
                total_frames = 0

    
    except KeyboardInterrupt:
        print("Stopping")
        stop_event.set()

    finally:
        print("Waiting for inference processes to clean up...")

        for p in infer_processes:
            if not p.is_alive():
                continue

            p.join(timeout=1)

            if p.is_alive():
                print(f"[WARN] Force terminating process {p.pid}")
                p.terminate()
                time.sleep(0.1)
                p.join()

        for q in read_queues + result_queues:
            try:
                q.cancel_join_thread()
                q.close()
            except:
                pass

        print("All cleaned up.")
        time.sleep(0.5)
        os.system("stty sane")
            
