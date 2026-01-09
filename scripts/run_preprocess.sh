#!/bin/bash

# 数据预处理脚本
# 读取维基百科数据并进行预处理

echo "=================================="
echo "GPT2中文 - 数据预处理"
echo "=================================="

# 运行预处理
python src/preprocess.py --config config.yaml

echo ""
echo "预处理完成！"
