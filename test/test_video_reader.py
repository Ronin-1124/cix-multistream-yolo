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
    """VideoReader 使用 read_frames 生成器读取帧"""
    video_path = tmp_path / "tiny.mp4"
    expected = _write_tiny_mp4(video_path, frames=6)

    reader = VideoReader(str(video_path))
    got = 0
    for frame in reader.read_frames():
        assert frame.ndim == 3
        assert frame.shape[2] == 3
        got += 1

    assert got >= expected - 1


def test_video_reader_properties(tmp_path):
    """VideoReader 属性应正确解析视频元数据"""
    video_path = tmp_path / "test_props.mp4"
    width, height, fps_val = 128, 96, 10
    _write_tiny_mp4(video_path, width=width, height=height, frames=5, fps=fps_val)

    reader = VideoReader(str(video_path))

    assert reader.width == width
    assert reader.height == height
    assert reader.fps == fps_val
    assert reader.codec in ['h264', 'hevc', 'mpeg4', 'vp90', 'av01']
    assert reader.nb_frames >= 1
    assert reader.duration > 0


def test_video_reader_read_frames_generator(tmp_path):
    """read_frames 应作为生成器工作"""
    video_path = tmp_path / "test_gen.mp4"
    _write_tiny_mp4(video_path, frames=5)

    reader = VideoReader(str(video_path))
    frames = list(reader.read_frames())

    assert len(frames) >= 4
    for frame in frames:
        assert frame.ndim == 3
        assert frame.shape[2] == 3


def test_video_reader_frame_shape(tmp_path):
    """读取的帧形状应与视频分辨率匹配"""
    video_path = tmp_path / "test_shape.mp4"
    width, height = 160, 120
    _write_tiny_mp4(video_path, width=width, height=height, frames=3)

    reader = VideoReader(str(video_path))
    frame = next(reader.read_frames())

    assert frame.shape == (height, width, 3)
