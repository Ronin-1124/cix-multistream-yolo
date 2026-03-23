import os
import logging
from queue import Queue
from src.utils.tools import get_video_path
LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO').upper()
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL),
    format='%(asctime)s - %(levelname)s - %(message)s'
)

video_paths = get_video_path("data")

read_queues = [Queue() for i in range(len(video_paths))]

result_queues = []

if __name__ == "__main__":
    print(len(read_queues))
