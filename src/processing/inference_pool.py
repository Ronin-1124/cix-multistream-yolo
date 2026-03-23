import time
import threading
import concurrent.futures
import multiprocessing as mp
from queue import Queue, Empty
from src.capture.video_reader_pipeline import reading_threads
from src.processing.inference import InferenceEngine
from src.utils.tools import get_video_path, pre_processing
from src.processing.post_processing import post_processing


def inferencing_threadpool(read_queues: list, result_queues: list, model_path: str, stop_event: threading.Event):
    def worker_task(queue_index):
        read_queue = read_queues[queue_index]
        result_queue = result_queues[queue_index]
        model = InferenceEngine(model_path=model_path)
        
        thread_name = threading.current_thread().name
        frame_count = 0
        print(f"线程池工作线程 {thread_name} 开始处理队列 {queue_index}")
        
        while not stop_event.is_set():
            try:
                frame = read_queue.get(timeout=0.1)
                frame_prep = pre_processing(frame[0])
                result = model.forward(frame_prep)
                result = post_processing(result[0].reshape(84, 8400), 0.5, 0.25)
                frame.append(result)
                result_queue.put(frame, timeout=0.1)
                frame_count += 1
            except Empty:
                continue
            except KeyboardInterrupt:
                break
        
        model.clean()
        print(f"线程池工作线程 {thread_name} (队列 {queue_index}) 正常退出，共处理 {frame_count} 帧")
        return queue_index, frame_count, thread_name
    
    executor = concurrent.futures.ThreadPoolExecutor(
        max_workers=len(read_queues),
        thread_name_prefix="InferPool"
    )
    
    futures = []
    for i in range(len(read_queues)):
        future = executor.submit(worker_task, i)
        futures.append(future)
        print(f"提交队列 {i} 的任务到线程池")
    
    return executor, futures


def inferencing_threadpool_with_callback(read_queues: list, result_queues: list, model_path: str, stop_event: threading.Event):
    def worker_task(queue_index):
        read_queue = read_queues[queue_index]
        result_queue = result_queues[queue_index]
        model = InferenceEngine(model_path=model_path)
        
        thread_name = threading.current_thread().name
        frame_count = 0
        print(f"线程池工作线程 {thread_name} 开始处理队列 {queue_index}")
        
        while not stop_event.is_set():
            try:
                frame = read_queue.get(timeout=0.1)
                frame_prep = pre_processing(frame[0])
                result = model.forward(frame_prep)
                result = post_processing(result[0].reshape(84, 8400), 0.5, 0.25)
                frame.append(result)
                result_queue.put(frame, timeout=0.1)
                frame_count += 1
            except Empty:
                continue
            except KeyboardInterrupt:
                break
        
        model.clean()
        return queue_index, frame_count, thread_name
    
    def task_done_callback(future):
        try:
            queue_index, frame_count, thread_name = future.result()
            print(f"回调: 线程 {thread_name} (队列 {queue_index}) 处理完成，共处理 {frame_count} 帧")
        except Exception as e:
            print(f"回调: 任务执行出错: {e}")
    
    executor = concurrent.futures.ThreadPoolExecutor(
        max_workers=len(read_queues),
        thread_name_prefix="InferPool"
    )
    
    futures = []
    for i in range(len(read_queues)):
        future = executor.submit(worker_task, i)
        future.add_done_callback(task_done_callback)
        futures.append(future)
        print(f"提交队列 {i} 的任务到线程池（带回调）")
    
    return executor, futures


def shutdown_threadpool(executor, futures, timeout=2):
    print("正在关闭线程池...")
    executor.shutdown(wait=False)
    
    for future in futures:
        try:
            future.result(timeout=timeout)
        except concurrent.futures.TimeoutError:
            print(f"任务 {future} 等待超时")
        except Exception as e:
            print(f"任务 {future} 执行出错: {e}")
    
    print("线程池已关闭")


def _process_worker(queue_index, read_queue, result_queue, model_path, stop_event):
    """进程池工作函数（模块级，可被 pickle）"""
    model = InferenceEngine(model_path=model_path)
    process_name = mp.current_process().name
    frame_count = 0
    print(f"进程池工作进程 {process_name} 开始处理队列 {queue_index}")

    while not stop_event.is_set():
        try:
            frame = read_queue.get(timeout=0.1)
            frame_prep = pre_processing(frame[0])
            result = model.forward(frame_prep)
            result = post_processing(result[0].reshape(84, 8400), 0.5, 0.25)
            frame.append(result)
            result_queue.put(frame, timeout=0.1)
            frame_count += 1
        except Empty:
            continue
        except mp.queues.Full:
            continue
        except KeyboardInterrupt:
            break

    model.clean()
    print(f"进程池工作进程 {process_name} (队列 {queue_index}) 正常退出，共处理 {frame_count} 帧")
    return queue_index, frame_count, process_name


def inferencing_processpool(read_queues: list, result_queues: list, model_path: str, stop_event: mp.Event): # type: ignore
    executor = concurrent.futures.ProcessPoolExecutor(
        max_workers=len(read_queues)
    )

    futures = []
    for i in range(len(read_queues)):
        future = executor.submit(_process_worker, i, read_queues[i], result_queues[i], model_path, stop_event)
        futures.append(future)
        print(f"提交队列 {i} 的任务到进程池")

    return executor, futures


def inferencing_processpool_with_callback(read_queues: list, result_queues: list, model_path: str, stop_event: mp.Event): # type: ignore
    def task_done_callback(future):
        try:
            queue_index, frame_count, process_name = future.result()
            print(f"回调: 进程 {process_name} (队列 {queue_index}) 处理完成，共处理 {frame_count} 帧")
        except Exception as e:
            print(f"回调: 任务执行出错: {e}")

    executor = concurrent.futures.ProcessPoolExecutor(
        max_workers=len(read_queues)
    )

    futures = []
    for i in range(len(read_queues)):
        future = executor.submit(_process_worker, i, read_queues[i], result_queues[i], model_path, stop_event)
        future.add_done_callback(task_done_callback)
        futures.append(future)
        print(f"提交队列 {i} 的任务到进程池（带回调）")

    return executor, futures


def shutdown_processpool(executor, futures, timeout=2):
    print("正在关闭进程池...")
    executor.shutdown(wait=False)
    
    for future in futures:
        try:
            future.result(timeout=timeout)
        except concurrent.futures.TimeoutError:
            print(f"任务 {future} 等待超时")
        except Exception as e:
            print(f"任务 {future} 执行出错: {e}")
    
    print("进程池已关闭")


if __name__ == "__main__":
    video_paths = get_video_path("data/test_videos_360P/test9.mp4")
    # video_paths = get_video_path("data/test_videos_360P")

    # True测试进程池，False测试线程池
    use_processpool = False

    if use_processpool:
        manager = mp.Manager()
        read_queues = [manager.Queue(maxsize=5) for _ in range(len(video_paths))]
        result_queues = [manager.Queue(maxsize=5) for _ in range(len(video_paths))]
        stop_event = manager.Event()
    else:
        read_queues = [Queue(maxsize=5) for _ in range(len(video_paths))]
        result_queues = [Queue(maxsize=5) for _ in range(len(video_paths))]
        stop_event = threading.Event()

    read_threads = reading_threads(
        video_paths=video_paths, 
        read_queues=read_queues, 
        stop_event=stop_event
    )
    
    model_path = "models/yolov8n.cix"
    
    if use_processpool:
        executor, futures = inferencing_processpool_with_callback(
            read_queues=read_queues,
            result_queues=result_queues,
            model_path=model_path,
            stop_event=stop_event
        )
    else:
        executor, futures = inferencing_threadpool_with_callback(
            read_queues=read_queues,
            result_queues=result_queues,
            model_path=model_path,
            stop_event=stop_event
        )
    
    fps_start_time = time.time()
    total_frames = 0
    fps_update_interval = 2.0
    frame_count_per_source = [0] * len(video_paths)

    try:
        while True:
            any_data = False
            for i in range(len(video_paths)):
                try:
                    result_data = result_queues[i].get(timeout=0.05)
                    frame, thread_id, frame_id, detections = result_data
                    total_frames += 1
                    frame_count_per_source[i] += 1
                    any_data = True
                        
                except Empty:
                    continue
            
            if not any_data:
                all_done = all(future.done() for future in futures)
                if all_done:
                    print("所有工作线程已完成，退出主循环")
                    break
            
            elapsed = time.time() - fps_start_time
            if elapsed >= fps_update_interval:
                current_fps = total_frames / elapsed if elapsed > 0 else 0
                print(f"[FPS] 总推理 FPS: {current_fps:.2f} (已处理 {total_frames} 帧, 耗时 {elapsed:.2f}s)")
                print(f"各源处理帧数: {frame_count_per_source}")
                fps_start_time = time.time()
                total_frames = 0
                    
    except KeyboardInterrupt:
        print(f"正在退出……")
    finally:
        stop_event.set()
        
        for i, t in enumerate(read_threads):
            t.join(timeout=2)
        
        if use_processpool:
            shutdown_processpool(executor, futures, timeout=2)
        else:
            shutdown_threadpool(executor, futures, timeout=2)
        
        print("程序退出完成")