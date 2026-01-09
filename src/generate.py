"""
文本生成脚本
使用训练好的GPT2模型生成中文文本
"""

import os
import argparse
import yaml
import torch
from typing import List

from model import GPT2ChineseModel, load_tokenizer
from utils import setup_logging, get_device, create_dir


def generate_text(model: GPT2ChineseModel,
                 tokenizer,
                 prompt: str,
                 max_length: int = 200,
                 temperature: float = 1.0,
                 top_k: int = 50,
                 top_p: float = 0.95,
                 num_return_sequences: int = 1,
                 strategy: str = 'top_p') -> List[str]:
    """
    生成文本
    
    Args:
        model: GPT2模型
        tokenizer: tokenizer对象
        prompt: 输入提示文本
        max_length: 生成的最大长度
        temperature: 温度参数（控制随机性）
        top_k: Top-k采样参数
        top_p: Top-p (nucleus)采样参数
        num_return_sequences: 返回序列数量
        strategy: 生成策略 ('greedy', 'beam', 'top_k', 'top_p', 'temperature')
        
    Returns:
        生成的文本列表
    """
    # 编码输入
    input_ids = tokenizer.encode(prompt, return_tensors='pt').to(model.device)
    
    # 根据策略设置参数
    if strategy == 'greedy':
        do_sample = False
        top_k = 0
        top_p = 1.0
        temperature = 1.0
    elif strategy == 'beam':
        # Beam search
        outputs = model.get_model().generate(
            input_ids=input_ids,
            max_length=max_length,
            num_beams=5,
            num_return_sequences=min(num_return_sequences, 5),
            early_stopping=True,
            pad_token_id=tokenizer.pad_token_id if tokenizer.pad_token_id else 0,
            eos_token_id=tokenizer.sep_token_id if tokenizer.sep_token_id else 102
        )
        generated_texts = [tokenizer.decode(output, skip_special_tokens=True) 
                          for output in outputs]
        return generated_texts
    elif strategy == 'top_k':
        do_sample = True
        top_p = 1.0
    elif strategy == 'top_p':
        do_sample = True
    elif strategy == 'temperature':
        do_sample = True
        top_k = 0
        top_p = 1.0
    else:
        do_sample = True
    
    # 生成
    outputs = model.generate(
        input_ids=input_ids,
        max_length=max_length,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        num_return_sequences=num_return_sequences,
        do_sample=do_sample
    )
    
    # 解码
    generated_texts = [tokenizer.decode(output, skip_special_tokens=True) 
                      for output in outputs]
    
    return generated_texts


def interactive_generate(model: GPT2ChineseModel,
                        tokenizer,
                        config: dict,
                        logger):
    """
    交互式生成模式
    
    Args:
        model: GPT2模型
        tokenizer: tokenizer对象
        config: 生成配置
        logger: 日志记录器
    """
    logger.info("=" * 50)
    logger.info("进入交互式生成模式")
    logger.info("输入 'quit' 或 'exit' 退出")
    logger.info("=" * 50)
    
    while True:
        prompt = input("\n请输入提示文本: ").strip()
        
        if prompt.lower() in ['quit', 'exit', 'q']:
            logger.info("退出交互模式")
            break
        
        if not prompt:
            continue
        
        try:
            generated_texts = generate_text(
                model, tokenizer, prompt,
                max_length=config['max_length'],
                temperature=config['temperature'],
                top_k=config['top_k'],
                top_p=config['top_p'],
                num_return_sequences=config['num_return_sequences'],
                strategy='top_p'
            )
            
            print("\n生成结果:")
            print("-" * 50)
            for i, text in enumerate(generated_texts, 1):
                print(f"\n[{i}] {text}")
            print("-" * 50)
            
        except Exception as e:
            logger.error(f"生成失败: {e}")


def main():
    parser = argparse.ArgumentParser(description='使用GPT2模型生成中文文本')
    parser.add_argument('--model_path', type=str, default=None,
                       help='模型路径')
    parser.add_argument('--config', type=str, default='config.yaml',
                       help='配置文件路径')
    parser.add_argument('--prompt', type=str, default=None,
                       help='输入提示文本')
    parser.add_argument('--num_samples', type=int, default=None,
                       help='生成样本数量')
    parser.add_argument('--max_length', type=int, default=None,
                       help='最大生成长度')
    parser.add_argument('--temperature', type=float, default=None,
                       help='温度参数')
    parser.add_argument('--top_k', type=int, default=None,
                       help='Top-k参数')
    parser.add_argument('--top_p', type=float, default=None,
                       help='Top-p参数')
    parser.add_argument('--strategy', type=str, default='top_p',
                       choices=['greedy', 'beam', 'top_k', 'top_p', 'temperature'],
                       help='生成策略')
    parser.add_argument('--interactive', action='store_true',
                       help='交互式生成模式')
    parser.add_argument('--output_file', type=str, default=None,
                       help='输出文件路径')
    args = parser.parse_args()
    
    # 加载配置
    with open(args.config, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # 设置日志
    logger = setup_logging()
    
    logger.info("=" * 50)
    logger.info("GPT2中文文本生成")
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
        logger.info("请先训练模型或指定正确的模型路径")
        return
    
    # 加载模型和tokenizer
    logger.info(f"加载模型: {model_path}")
    model = GPT2ChineseModel(model_name=model_path, device=device)
    tokenizer = load_tokenizer(model_path)
    
    # 更新生成配置
    gen_config = config['generation'].copy()
    if args.num_samples:
        gen_config['num_return_sequences'] = args.num_samples
    if args.max_length:
        gen_config['max_length'] = args.max_length
    if args.temperature:
        gen_config['temperature'] = args.temperature
    if args.top_k:
        gen_config['top_k'] = args.top_k
    if args.top_p:
        gen_config['top_p'] = args.top_p
    
    logger.info(f"生成配置: {gen_config}")
    logger.info(f"生成策略: {args.strategy}")
    
    # 交互式模式
    if args.interactive:
        interactive_generate(model, tokenizer, gen_config, logger)
        return
    
    # 单次生成模式
    if not args.prompt:
        logger.error("请提供提示文本 (--prompt) 或使用交互模式 (--interactive)")
        return
    
    logger.info(f"提示文本: {args.prompt}")
    logger.info("生成中...")
    
    generated_texts = generate_text(
        model, tokenizer, args.prompt,
        max_length=gen_config['max_length'],
        temperature=gen_config['temperature'],
        top_k=gen_config['top_k'],
        top_p=gen_config['top_p'],
        num_return_sequences=gen_config['num_return_sequences'],
        strategy=args.strategy
    )
    
    # 输出结果
    logger.info("\n生成结果:")
    logger.info("=" * 50)
    for i, text in enumerate(generated_texts, 1):
        logger.info(f"\n[样本 {i}]")
        logger.info(text)
        logger.info("-" * 50)
    
    # 保存到文件
    if args.output_file:
        output_dir = os.path.dirname(args.output_file)
        if output_dir:
            create_dir(output_dir)
        
        with open(args.output_file, 'w', encoding='utf-8') as f:
            f.write(f"提示文本: {args.prompt}\n")
            f.write(f"生成策略: {args.strategy}\n")
            f.write(f"配置: {gen_config}\n")
            f.write("=" * 50 + "\n\n")
            
            for i, text in enumerate(generated_texts, 1):
                f.write(f"[样本 {i}]\n")
                f.write(text + "\n")
                f.write("-" * 50 + "\n\n")
        
        logger.info(f"\n生成结果已保存到: {args.output_file}")


if __name__ == '__main__':
    main()
