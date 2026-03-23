import argparse
import os
import subprocess
from src.utils.manager import manager


def ensure_model(model_path: str):
    if not os.path.exists(model_path):
        print(f"模型不存在，正在下载: {model_path}")
        script = os.path.join(os.path.dirname(__file__), "src/utils/download_model.sh")
        subprocess.run(["bash", script, model_path], check=True)
    else:
        print(f"模型已存在: {model_path}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="多路视频流 YOLO 推理程序",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python main.py  # 使用默认参数
        """,
    )
    parser.add_argument(
        "-i",
        "--input",
        nargs="+",
        default=["data/test_videos_360P"],
        help=f"视频文件路径、目录路径，或多个路径（空格分隔）, 默认: data/test_videos_360P",
    )
    parser.add_argument(
        "-m",
        "--model",
        default="models/yolov8n.cix",
        help=f"模型文件路径 (.cix), 默认: models/yolov8n.cix",
    )
    parser.add_argument(
        "-t",
        "--type",
        choices=["thread", "t", "process", "p"],
        default="thread",
        help="推理模式: thread (线程) 或 process (进程), 默认 thread",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    ensure_model(args.model)
    manager(args.input, args.model, args.type)
