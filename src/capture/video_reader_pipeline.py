import threading
from queue import Queue, Full
from src.capture.video_reader import VideoReader


def reading_thread(
    video_path: str, read_queue: Queue, video_id: int, stop_event: threading.Event
):
    frame_id = 0
    reader = VideoReader(video_path=video_path)
    for frame in reader.read_frames():
        if stop_event.is_set():
            break
        try:
            read_queue.put([frame, video_id, frame_id])
            frame_id += 1
        except Full:
            print("错误，读取队列满")



def reading_threads(video_paths: list, read_queues: list, stop_event: threading.Event):
    threads = []
    for i, path in enumerate(video_paths):
        t = threading.Thread(
            target=reading_thread,
            args=(path, read_queues[i], i, stop_event),
            daemon=True,
        )
        threads.append(t)
        t.start()
    return threads
