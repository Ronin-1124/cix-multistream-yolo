import numpy as np
import subprocess
from src.utils.tools import get_video_info


class VideoReader:
    def __init__(self, video_path: str):
        video_info = get_video_info(video_path)
        frame, time = map(float, video_info["r_frame_rate"].split("/"))
        self.codec = video_info["codec_name"]
        self.fps = frame / time
        self.width = int(video_info["width"])
        self.height = int(video_info["height"])
        self.nb_frames = int(video_info["nb_frames"])
        self.duration = float(video_info["duration"])
        self.decoder = self.codec + "_v4l2m2m"
        self.frame_size = self.width * self.height * 3
        self.cmd = [
            "ffmpeg",
            # '-c:v',
            # self.decoder,
            "-i",
            video_path,
            "-f",
            "rawvideo",
            "-pix_fmt",
            "bgr24",
            "-r",
            str(self.fps),
            "-",
        ]

    def read_frames(self):
        process = subprocess.Popen(
            self.cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )
        try:
            while True:
                raw_frame = process.stdout.read(self.frame_size)
                if not raw_frame:
                    break
                if len(raw_frame) < self.frame_size:
                    break
                frame = np.frombuffer(raw_frame, dtype=np.uint8).reshape(
                    (self.height, self.width, 3)
                )
                yield frame
        finally:
            process.terminate()
