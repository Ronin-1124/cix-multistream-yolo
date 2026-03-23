import os
import subprocess
import cv2
import numpy as np


def get_video_path(paths: str) -> list:
    video_paths = []
    if os.path.isfile(paths):
        video_paths.append(paths)
    elif os.path.isdir(paths):
        for root, _, files in os.walk(paths):
            for file in files:
                if file.endswith(('.mp4', '.avi', '.mkv', '.mov')):
                    video_paths.append(os.path.join(root, file))
    return video_paths


def get_video_info(video_path: str) -> dict:
    cmd = [
        'ffprobe',
        '-v', 'error',
        '-select_streams', 'v:0',
        '-show_entries', 'stream=width,height,r_frame_rate,nb_frames,duration,codec_name',
        '-of', 'default=noprint_wrappers=1',
        video_path
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    video_info = {}
    for line in result.stdout.strip().split('\n'):
        key, value = line.split('=')
        video_info[key] = value
    return video_info
    
def pre_processing(frame: np.ndarray) -> np.ndarray:
    frame = cv2.resize(frame, (640, 640))
    frame = frame.astype(np.float32) / 255.0
    frame = np.transpose(frame, (2, 0, 1))
    frame = np.expand_dims(frame, axis=0)
    return frame

