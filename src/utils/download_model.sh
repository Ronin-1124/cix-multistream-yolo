#!/bin/bash

MODEL_PATH="$1"
MODEL_DIR=$(dirname "$MODEL_PATH")

if [ -z "$MODEL_PATH" ]; then
    echo "用法: $0 <model_path>"
    echo "示例: $0 models/yolov8n.cix"
    exit 1
fi

if [ -f "$MODEL_PATH" ]; then
    echo "模型已存在: $MODEL_PATH"
    exit 0
fi

MODEL_NAME=$(basename "$MODEL_PATH" .cix)

case "$MODEL_NAME" in
    yolov8n)
        DOWNLOAD_URL="https://www.modelscope.cn/models/cix/ai_model_hub/resolve/master/models/ComputeVision/Object_Detection/onnx_yolov8_n/yolov8n.cix"
        ;;
    yolov8s)
        DOWNLOAD_URL="https://www.modelscope.cn/models/cix/ai_model_hub/resolve/master/models/ComputeVision/Object_Detection/onnx_yolov8_s/yolov8s.cix"
        ;;
    *)
        echo "不支持的模型: $MODEL_NAME，仅支持 yolov8n, yolov8s"
        exit 1
        ;;
esac

if [ -z "$DOWNLOAD_URL" ]; then
    echo "错误: $MODEL_NAME 的下载链接未配置，请联系维护者"
    exit 1
fi

mkdir -p "$MODEL_DIR"

echo "正在下载模型: $MODEL_NAME ..."
echo "下载链接: $DOWNLOAD_URL"
if command -v wget &> /dev/null; then
    wget -O "$MODEL_PATH" "$DOWNLOAD_URL"
elif command -v curl &> /dev/null; then
    curl -L -o "$MODEL_PATH" "$DOWNLOAD_URL"
else
    echo "错误: 需要 wget 或 curl 来下载文件"
    exit 1
fi

if [ -f "$MODEL_PATH" ]; then
    echo "模型下载成功: $MODEL_PATH"
else
    echo "模型下载失败"
    exit 1
fi
