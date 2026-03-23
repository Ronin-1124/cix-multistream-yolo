import os
import pytest
from src.utils.tools import get_video_path


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