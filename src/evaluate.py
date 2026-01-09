"""
模型评估脚本
计算验证集上的困惑度、生成样本文本等
"""

import os
import argparse
import yaml
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np

from dataset import WikiTextDataset
from model import GPT2ChineseModel, load_tokenizer
from generate import generate_text
from utils import (
    setup_logging, calculate_perplexity, get_device,
    create_dir, format_time, Timer
)


def evaluate_perplexity(model: GPT2ChineseModel,
                       data_loader: DataLoader,
                       device: str,
                       logger) -> float:
    """
    计算数据集上的困惑度
    
    Args:
        model: GPT2模型
        data_loader: 数据加载器
        device: 设备
        logger: 日志记录器
        
    Returns:
        平均困惑度
    """
    model.get_model().eval()
    total_loss = 0
    total_tokens = 0
    
    logger.info("计算困惑度...")
    
    with torch.no_grad():
        for batch in tqdm(data_loader, desc="评估中"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = input_ids.clone()
            
            outputs = model.forward(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            
            # 只计算非padding位置的损失
            loss = outputs.loss
            num_tokens = attention_mask.sum().item()
            
            total_loss += loss.item() * num_tokens
            total_tokens += num_tokens
    
    avg_loss = total_loss / total_tokens
    perplexity = calculate_perplexity(avg_loss)
    
    return perplexity


def generate_samples(model: GPT2ChineseModel,
                    tokenizer,
                    prompts: list,
                    config: dict,
                    logger) -> list:
    """
    生成样本文本用于定性评估
    
    Args:
        model: GPT2模型
        tokenizer: tokenizer对象
        prompts: 提示文本列表
        config: 生成配置
        logger: 日志记录器
        
    Returns:
        生成的文本列表
    """
    logger.info("生成样本文本...")
    
    all_samples = []
    
    for prompt in prompts:
        logger.info(f"\n提示: {prompt}")
        
        generated_texts = generate_text(
            model, tokenizer, prompt,
            max_length=config['max_length'],
            temperature=config['temperature'],
            top_k=config['top_k'],
            top_p=config['top_p'],
            num_return_sequences=1,
            strategy='top_p'
        )
        
        for text in generated_texts:
            logger.info(f"生成: {text}")
            all_samples.append({
                'prompt': prompt,
                'generated': text
            })
    
    return all_samples


def save_evaluation_report(perplexity: float,
                          samples: list,
                          output_path: str,
                          config: dict):
    """
    保存评估报告
    
    Args:
        perplexity: 困惑度
        samples: 生成的样本
        output_path: 输出路径
        config: 配置
    """
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("=" * 50 + "\n")
        f.write("GPT2中文模型评估报告\n")
        f.write("=" * 50 + "\n\n")
        
        f.write("## 量化指标\n\n")
        f.write(f"困惑度 (Perplexity): {perplexity:.2f}\n\n")
        
        f.write("## 模型配置\n\n")
        for key, value in config.get('model', {}).items():
            f.write(f"- {key}: {value}\n")
        
        f.write("\n## 训练配置\n\n")
        for key, value in config.get('training', {}).items():
            f.write(f"- {key}: {value}\n")
        
        f.write("\n" + "=" * 50 + "\n")
        f.write("## 生成样本 (定性评估)\n")
        f.write("=" * 50 + "\n\n")
        
        for i, sample in enumerate(samples, 1):
            f.write(f"### 样本 {i}\n\n")
            f.write(f"**提示**: {sample['prompt']}\n\n")
            f.write(f"**生成**: {sample['generated']}\n\n")
            f.write("-" * 50 + "\n\n")
    
    print(f"\n评估报告已保存: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='评估GPT2中文模型')
    parser.add_argument('--model_path', type=str, default=None,
                       help='模型路径')
    parser.add_argument('--config', type=str, default='config.yaml',
                       help='配置文件路径')
    parser.add_argument('--eval_perplexity', action='store_true',
                       help='计算困惑度')
    parser.add_argument('--generate_samples', action='store_true',
                       help='生成样本文本')
    parser.add_argument('--output_dir', type=str, default='outputs/evaluation',
                       help='输出目录')
    args = parser.parse_args()
    
    # 加载配置
    with open(args.config, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # 设置日志
    logger = setup_logging()
    
    logger.info("=" * 50)
    logger.info("GPT2中文模型评估")
    logger.info("=" * 50)
    
    # 获取设备
    device = get_device()
    
    # 确定模型路径
    if args.model_path:
        model_path = args.model_path
    else:
        model_path = os.path.join(config['training']['output_dir'], 'checkpoint-best')
    
    if not os.path.exists(model_path):
        logger.error(f"模型路径不存在: {model_path}")
        return
    
    # 加载模型和tokenizer
    logger.info(f"加载模型: {model_path}")
    model = GPT2ChineseModel(model_name=model_path, device=device)
    tokenizer = load_tokenizer(model_path)
    
    # 创建输出目录
    create_dir(args.output_dir)
    
    timer = Timer()
    timer.start()
    
    results = {}
    
    # 评估困惑度
    if args.eval_perplexity:
        logger.info("\n" + "=" * 50)
        logger.info("评估困惑度")
        logger.info("=" * 50)
        
        # 加载验证数据
        val_data_path = os.path.join(config['data']['processed_path'], 'val_data.pkl')
        
        if not os.path.exists(val_data_path):
            logger.error(f"验证数据不存在: {val_data_path}")
        else:
            val_dataset = WikiTextDataset(
                val_data_path,
                max_length=config['data']['max_length'],
                pad_token_id=tokenizer.pad_token_id if tokenizer.pad_token_id else 0
            )
            
            val_loader = DataLoader(
                val_dataset,
                batch_size=config['training']['batch_size'],
                shuffle=False,
                num_workers=4
            )
            
            perplexity = evaluate_perplexity(model, val_loader, device, logger)
            results['perplexity'] = perplexity
            
            logger.info(f"\n困惑度: {perplexity:.2f}")
    
    # 生成样本
    samples = []
    if args.generate_samples:
        logger.info("\n" + "=" * 50)
        logger.info("生成样本文本")
        logger.info("=" * 50)
        
        # 预定义的提示
        prompts = [
            "中国是一个",
            "人工智能技术",
            "在古代",
            "科学家发现",
            "这个故事讲述了"
        ]
        
        samples = generate_samples(
            model, tokenizer, prompts,
            config['generation'], logger
        )
    
    # 保存评估报告
    if results or samples:
        report_path = os.path.join(args.output_dir, 'evaluation_report.md')
        save_evaluation_report(
            results.get('perplexity', 0.0),
            samples,
            report_path,
            config
        )
    
    elapsed_time = timer.stop()
    logger.info("\n" + "=" * 50)
    logger.info("评估完成！")
    logger.info(f"耗时: {format_time(elapsed_time)}")
    logger.info("=" * 50)


if __name__ == '__main__':
    main()
