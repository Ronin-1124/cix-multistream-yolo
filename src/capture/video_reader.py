import numpy as np
import subprocess
from src.utils.tools import get_video_info

import logging
logger = logging.getLogger(__name__)


class VideoReader:
    def __init__(self, video_path: str):
        video_info = get_video_info(video_path)
        frame, time = map(float, video_info['r_frame_rate'].split('/'))
        self.codec = video_info['codec_name']
        self.fps = frame / time
        self.width = int(video_info['width'])
        self.height = int(video_info['height'])
        self.nb_frames = int(video_info['nb_frames'])
        self.duration = float(video_info['duration'])

        logger.debug(f"视频编码格式：{video_info['codec_name']}")
        logger.debug(f"帧率：{frame/time:.2f} fps")
        logger.debug(f"分辨率：{video_info['width']}x{video_info['height']}")
        logger.debug(f"总帧数：{video_info['nb_frames']}")
        logger.debug(f"视频时长：{video_info['duration']}秒")

        self.decoder = self.codec + "_v4l2m2m"
        self.frame_size = self.width * self.height * 3

        self.cmd = [
            'ffmpeg',
            # '-c:v',
            # self.decoder,
            '-i',
            video_path,
            '-f',
            'rawvideo',
            '-pix_fmt',
            'bgr24',
            '-r',
            str(self.fps),
            '-'
        ]

        self.process = None
        # self.frame = 0
    
    def read_frames(self, stop_event=None):
        self.process = subprocess.Popen(
            self.cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
            )
        try:
            while True:
                if stop_event and stop_event.is_set():
                    break
                raw_frame = self.process.stdout.read(self.frame_size)
                if not raw_frame:
                    break
                if len(raw_frame) < self.frame_size:
                    break
                frame = np.frombuffer(raw_frame, dtype=np.uint8).reshape((self.height, self.width, 3))
                yield frame
        finally:
            if self.process:
                self.process.kill()
                self.process.communicate(timeout=1)