# GPT2 中文文本生成项目

基于维基百科中文语料的GPT2文本生成系统，使用PyTorch和Transformers库实现。

## 📋 项目简介

本项目实现了一个完整的GPT2中文文本生成系统，包括数据预处理、模型训练、文本生成和模型评估等功能。项目使用维基百科中文语料进行训练，能够生成连贯流畅的中文文本。

### 主要特性

- ✅ 完整的数据预处理流程，支持大规模维基百科数据
- ✅ 基于Hugging Face Transformers的GPT2模型实现
- ✅ 支持从预训练模型微调或从头训练
- ✅ 多种文本生成策略（Greedy、Beam Search、Top-K、Top-P、Temperature）
- ✅ 混合精度训练（AMP）和梯度累积支持
- ✅ TensorBoard训练监控
- ✅ 完整的模型评估工具（困惑度计算、样本生成）
- ✅ 交互式文本生成界面

## 🗂️ 项目结构

```
gpt2-chinese-wiki/
├── README.md                          # 项目说明文档
├── requirements.txt                   # Python依赖包列表
├── config.yaml                        # 配置文件
├── .gitignore                        # Git忽略文件
├── data/                             # 数据目录（需手动创建）
│   └── wiki_zh/                      # 维基百科数据
│       ├── AA/
│       │   ├── wiki_00
│       │   └── wiki_01
│       └── AB/
├── src/                              # 源代码目录
│   ├── __init__.py                   # 包初始化
│   ├── preprocess.py                 # 数据预处理
│   ├── dataset.py                    # 数据集类
│   ├── model.py                      # GPT2模型定义
│   ├── train.py                      # 训练脚本
│   ├── generate.py                   # 文本生成脚本
│   ├── evaluate.py                   # 模型评估脚本
│   └── utils.py                      # 工具函数
├── scripts/                          # 运行脚本
│   ├── run_preprocess.sh            # 数据预处理运行脚本
│   ├── run_train.sh                 # 训练运行脚本
│   └── run_generate.sh              # 生成运行脚本
└── outputs/                          # 输出目录（自动创建）
    ├── processed_data/               # 预处理后的数据
    ├── models/                       # 保存的模型
    └── generated_texts/              # 生成的文本
```

## 🔧 环境要求

- Python 3.8+
- CUDA 11.0+ (推荐使用GPU)
- 16GB+ RAM (推荐)
- 足够的磁盘空间用于存储数据和模型

## 📦 安装步骤

### 1. 克隆项目

```bash
git clone https://github.com/zhaichen998-svg/gpt2-chinese-wiki.git
cd gpt2-chinese-wiki
```

### 2. 创建虚拟环境（推荐）

```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# 或
venv\Scripts\activate  # Windows
```

### 3. 安装依赖

```bash
pip install -r requirements.txt
```

## 📊 数据准备

### 数据格式

数据应放在 `data/wiki_zh` 目录下，包含多个子文件夹（AA, AB, AC等），每个子文件夹包含多个JSON格式的文本文件。

每个文件包含多行JSON对象，格式如下：

```json
{"id": "263", "url": "https://zh.wikipedia.org/wiki?curid=263", "title": "中国", "text": "中国\n\n中国是位于东亚的国家..."}
```

### 获取数据

可以从以下来源获取维基百科中文数据：
- [维基百科中文数据转储](https://dumps.wikimedia.org/zhwiki/)
- 使用 [WikiExtractor](https://github.com/attardi/wikiextractor) 提取文本

### 数据目录结构示例

```
data/wiki_zh/
├── AA/
│   ├── wiki_00
│   ├── wiki_01
│   └── ...
├── AB/
│   ├── wiki_00
│   └── ...
└── ...
```

## 🚀 使用方法

### 1. 数据预处理

首先需要对原始维基百科数据进行预处理：

```bash
# 使用脚本
bash scripts/run_preprocess.sh

# 或直接运行Python脚本
python src/preprocess.py --config config.yaml
```

预处理会完成以下工作：
- 读取所有wiki_zh文件夹下的JSON文件
- 提取文本内容并进行清理
- 使用tokenizer将文本转换为token IDs
- 分割训练集和验证集（默认90/10）
- 保存处理后的数据到 `outputs/processed_data/`

### 2. 模型训练

```bash
# 使用脚本
bash scripts/run_train.sh

# 或直接运行Python脚本
python src/train.py --config config.yaml
```

训练过程会：
- 加载预处理好的数据
- 初始化GPT2模型（可选从预训练模型加载）
- 进行训练并定期保存checkpoint
- 在验证集上评估性能
- 使用TensorBoard记录训练过程

#### 查看训练进度

```bash
tensorboard --logdir logs/tensorboard
```

### 3. 文本生成

#### 命令行生成

```bash
# 使用脚本
bash scripts/run_generate.sh --prompt "中国是一个" --num_samples 3

# 或直接运行Python脚本
python src/generate.py \
    --model_path outputs/models/checkpoint-best \
    --prompt "中国是一个" \
    --num_samples 3
```

#### 交互式生成

```bash
# 使用脚本
bash scripts/run_generate.sh --interactive

# 或直接运行Python脚本
python src/generate.py --interactive
```

#### 生成策略选项

```bash
# Greedy decoding (贪婪解码)
python src/generate.py --prompt "人工智能" --strategy greedy

# Beam search (束搜索)
python src/generate.py --prompt "人工智能" --strategy beam

# Top-K sampling (Top-K采样)
python src/generate.py --prompt "人工智能" --strategy top_k --top_k 50

# Top-P sampling (核采样)
python src/generate.py --prompt "人工智能" --strategy top_p --top_p 0.95

# Temperature sampling (温度采样)
python src/generate.py --prompt "人工智能" --strategy temperature --temperature 0.8
```

### 4. 模型评估

```bash
# 评估困惑度
python src/evaluate.py --eval_perplexity

# 生成样本文本
python src/evaluate.py --generate_samples

# 完整评估（困惑度 + 样本生成）
python src/evaluate.py --eval_perplexity --generate_samples
```

评估报告会保存在 `outputs/evaluation/evaluation_report.md`

## ⚙️ 配置说明

配置文件 `config.yaml` 包含所有超参数设置：

### 数据配置

```yaml
data:
  wiki_path: "data/wiki_zh"          # 维基百科数据路径
  processed_path: "outputs/processed_data"  # 处理后数据路径
  train_split: 0.9                    # 训练集比例
  max_length: 512                     # 最大序列长度
```

### 模型配置

```yaml
model:
  model_name: "uer/gpt2-chinese-cluecorpussmall"  # 预训练模型名称
  vocab_size: 21128                   # 词表大小
  hidden_size: 768                    # 隐藏层维度
  num_hidden_layers: 12               # Transformer层数
  num_attention_heads: 12             # 注意力头数
  max_position_embeddings: 512        # 最大位置编码
```

### 训练配置

```yaml
training:
  batch_size: 8                       # 批次大小
  num_epochs: 10                      # 训练轮数
  learning_rate: 5.0e-5              # 学习率
  warmup_steps: 1000                  # 预热步数
  gradient_accumulation_steps: 4      # 梯度累积步数
  max_grad_norm: 1.0                 # 梯度裁剪阈值
  save_steps: 5000                    # 保存间隔
  eval_steps: 1000                    # 评估间隔
  logging_steps: 100                  # 日志记录间隔
  output_dir: "outputs/models"        # 模型输出目录
  use_amp: true                       # 混合精度训练
  seed: 42                            # 随机种子
```

### 生成配置

```yaml
generation:
  max_length: 200                     # 生成最大长度
  temperature: 1.0                    # 温度参数
  top_k: 50                          # Top-K参数
  top_p: 0.95                        # Top-P参数
  num_return_sequences: 3             # 返回序列数
```

## 📝 示例结果

### 训练输出示例

```
==================================================
开始训练GPT2中文模型
==================================================
使用GPU: NVIDIA GeForce RTX 3090
模型参数总数: 102,267,648
可训练参数: 102,267,648
训练批次数: 1250
验证批次数: 139

==================================================
Epoch 1/10
==================================================
训练中: 100%|████████████| 1250/1250 [15:23<00:00]
训练损失: 3.2145, 困惑度: 24.87
验证损失: 2.9876, 困惑度: 19.84
保存最佳模型 (验证损失: 2.9876)
```

### 生成示例

**输入提示**: "中国是一个"

**生成结果**:

```
中国是一个历史悠久的文明古国，拥有五千年的灿烂文化。
从古代的四大发明到现代的科技创新，中国一直在人类文明
发展史上扮演着重要角色。
```

## 🔍 常见问题

### Q1: 运行预处理时提示找不到数据？

A: 请确保维基百科数据已放在 `data/wiki_zh` 目录下，目录结构正确。

### Q2: 训练时显存不足？

A: 可以尝试以下方法：
- 减小 `batch_size`
- 增大 `gradient_accumulation_steps`
- 减小 `max_length`
- 使用更小的模型

### Q3: 如何使用自己的数据？

A: 将数据转换为JSON格式，每行一个JSON对象，包含 `text` 字段，放在 `data/wiki_zh` 目录下即可。

### Q4: 如何从头训练而不使用预训练模型？

A: 在 `config.yaml` 中将 `model_name` 设置为 `"scratch"`。

### Q5: 生成的文本质量不好？

A: 可以尝试：
- 训练更多epoch
- 调整生成参数（temperature, top_k, top_p）
- 使用更大的模型
- 使用更多训练数据

## 📚 参考资料

- [GPT-2 论文](https://d4mucfpksywv.cloudfront.net/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)
- [Hugging Face Transformers](https://huggingface.co/docs/transformers/)
- [PyTorch 文档](https://pytorch.org/docs/stable/index.html)
- [维基百科数据转储](https://dumps.wikimedia.org/)

## 🤝 贡献

欢迎提交Issue和Pull Request！

## 📄 许可证

MIT License

## 👥 作者

GPT2 Chinese Wiki Team

---

**注意**: 训练大型语言模型需要大量计算资源和时间，建议使用GPU进行训练。
