#!/bin/bash

# 训练脚本
# 训练GPT2中文模型

echo "=================================="
echo "GPT2中文 - 模型训练"
echo "=================================="

# 运行训练
python src/train.py --config config.yaml

echo ""
echo "训练完成！"
