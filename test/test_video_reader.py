import cv2
import numpy as np
import pytest

from src.capture.video_reader import VideoReader


def _write_tiny_mp4(path, *, width: int = 64, height: int = 48, frames: int = 5, fps: int = 5) -> int:
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, float(fps), (width, height))
    if not writer.isOpened():
        pytest.skip("当前环境无法打开 mp4v 编码器，跳过视频写入/读取测试")

    for i in range(frames):
        img = np.zeros((height, width, 3), dtype=np.uint8)
        img[:] = (i * 30) % 255
        writer.write(img)
    writer.release()
    return frames


def test_video_reader_reads_frames(tmp_path):
    video_path = tmp_path / "tiny.mp4"
    expected = _write_tiny_mp4(video_path, frames=6)

    reader = VideoReader(str(video_path))
    got = 0
    while True:
        frame = reader.read_frame()
        if frame is None:
            break
        assert frame.ndim == 3
        assert frame.shape[2] == 3
        got += 1
    reader.release()

    # 有些 OpenCV/编码器组合可能会少 1 帧；这里保证“至少读到大部分帧”
    assert got >= expected - 1

