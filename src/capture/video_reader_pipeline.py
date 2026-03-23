import time
import threading
from queue import Queue, Full
from src.capture.video_reader import VideoReader
from src.utils.tools import get_video_path


def reading_thread(video_path: str, read_queue: Queue, video_id: int, stop_event: threading.Event):
    frame_id = 0
    while not stop_event.is_set():
        reader = VideoReader(video_path=video_path)
        for frame in reader.read_frames():
            if stop_event.is_set():
                break
            try:
                read_queue.put([frame, video_id, frame_id], timeout=0.1)
                frame_id += 1
            except Full:
                print("ERROR: READ QUEUE FULL")
        frame_id = 0


def reading_threads(video_paths: list, read_queues: list, stop_event: threading.Event):
    threads = []
    for i, path in enumerate(video_paths):
        t = threading.Thread(
            target=reading_thread,
            args=(path, read_queues[i], i, stop_event),
            daemon=True
        )
        threads.append(t)
        t.start()
    return threads


if __name__ == "__main__":
    video_paths = get_video_path("data/test_videos_720P")
    test_queues = [Queue(maxsize=10) for i in range(len(video_paths))]
    stop_event = threading.Event()
    threads = reading_threads(video_paths=video_paths, read_queues=test_queues, stop_event=stop_event)

    try:
        while True:
            for i in range(len(video_paths)):
                frame = test_queues[i].get()
                if frame[2] % 1000 == 0:
                    print(f"Get a frame successfully, frame shape: {frame[0].shape}")
    except KeyboardInterrupt:
        print("\n手动停止")

    stop_event.set()
    for t in threads:
        t.join()

