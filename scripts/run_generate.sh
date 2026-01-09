#!/bin/bash

# 文本生成脚本
# 使用训练好的模型生成中文文本

echo "=================================="
echo "GPT2中文 - 文本生成"
echo "=================================="

# 默认参数
MODEL_PATH="outputs/models/checkpoint-best"
PROMPT="中国是一个"
NUM_SAMPLES=3

# 解析命令行参数
while [[ $# -gt 0 ]]; do
    case $1 in
        --model_path)
            MODEL_PATH="$2"
            shift 2
            ;;
        --prompt)
            PROMPT="$2"
            shift 2
            ;;
        --num_samples)
            NUM_SAMPLES="$2"
            shift 2
            ;;
        --interactive)
            INTERACTIVE="--interactive"
            shift
            ;;
        *)
            echo "未知参数: $1"
            exit 1
            ;;
    esac
done

# 运行生成
if [ -n "$INTERACTIVE" ]; then
    python src/generate.py \
        --model_path "$MODEL_PATH" \
        --config config.yaml \
        $INTERACTIVE
else
    python src/generate.py \
        --model_path "$MODEL_PATH" \
        --config config.yaml \
        --prompt "$PROMPT" \
        --num_samples "$NUM_SAMPLES"
fi

echo ""
echo "生成完成！"
