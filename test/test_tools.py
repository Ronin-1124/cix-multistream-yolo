import os
import numpy as np
import pytest
from src.utils.tools import get_video_path, pre_processing


def test_get_video_path():
    # 测试单个视频文件路径
    video_file = "test_video.mp4"
    with open(video_file, 'w') as f:
        f.write("dummy video content")
    assert get_video_path(video_file) == [video_file]
    os.remove(video_file)

    # 测试目录路径
    os.mkdir("test_dir")
    video_files = ["video1.mp4", "video2.avi", "not_a_video.txt"]
    for file in video_files:
        with open(os.path.join("test_dir", file), 'w') as f:
            f.write("dummy content")
    
    expected_paths = [os.path.join("test_dir", "video1.mp4"), os.path.join("test_dir", "video2.avi")]
    assert set(get_video_path("test_dir")) == set(expected_paths)

    # 清理测试目录
    for file in video_files:
        os.remove(os.path.join("test_dir", file))
    os.rmdir("test_dir")


def test_get_video_path_empty():
    """无效路径应返回空列表"""
    assert get_video_path("nonexistent_path_12345") == []


def test_get_video_path_single_file_returns_as_is():
    """单独文件路径会被直接返回（函数不检查扩展名）"""
    with open("test_video.xyz", "w") as f:
        f.write("dummy content")
    # get_video_path 对单独文件不做扩展名检查
    assert get_video_path("test_video.xyz") == ["test_video.xyz"]
    os.remove("test_video.xyz")


class TestPreProcessing:
    """pre_processing 函数测试"""

    def test_pre_processing_output_shape(self):
        """预处理输出形状应为 (1, 3, 640, 640)"""
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        result = pre_processing(frame)
        assert result.shape == (1, 3, 640, 640)

    def test_pre_processing_output_dtype(self):
        """预处理输出应为 float32"""
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        result = pre_processing(frame)
        assert result.dtype == np.float32

    def test_pre_processing_value_range(self):
        """预处理后像素值应在 [0, 1] 范围内"""
        frame = np.full((480, 640, 3), 255, dtype=np.uint8)  # 全白图像
        result = pre_processing(frame)
        assert result.min() >= 0.0
        assert result.max() <= 1.0

    def test_pre_processing_channel_order(self):
        """预处理后通道顺序应为 (B, G, R) -> (C, H, W)"""
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        frame[:, :, 0] = 255  # B 通道为 255
        result = pre_processing(frame)
        # 结果应该是 (1, 3, 640, 640)，第一通道应该是 255/255=1.0
        assert result[0, 0, 0, 0] == 1.0

    def test_pre_processing_resize(self):
        """不同尺寸图像都应调整为 640x640"""
        small_frame = np.zeros((240, 320, 3), dtype=np.uint8)
        large_frame = np.zeros((1080, 1920, 3), dtype=np.uint8)

        result_small = pre_processing(small_frame)
        result_large = pre_processing(large_frame)

        assert result_small.shape == (1, 3, 640, 640)
        assert result_large.shape == (1, 3, 640, 640)