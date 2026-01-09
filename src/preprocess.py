"""
数据预处理脚本
读取维基百科中文语料，进行分词、构建词表、转换为token IDs
"""

import os
import json
import argparse
import yaml
from typing import List, Dict, Tuple
from pathlib import Path
from tqdm import tqdm
import pickle

from transformers import BertTokenizer
from sklearn.model_selection import train_test_split

from utils import setup_logging, set_seed, create_dir


def load_wiki_texts(wiki_path: str, max_files: int = None) -> List[Dict[str, str]]:
    """
    从wiki_zh目录加载所有JSON文件
    
    Args:
        wiki_path: 维基百科数据路径
        max_files: 最大文件数量限制（用于测试）
        
    Returns:
        包含文本内容的字典列表
    """
    logger = setup_logging()
    all_texts = []
    
    if not os.path.exists(wiki_path):
        logger.error(f"数据路径不存在: {wiki_path}")
        logger.info("请将维基百科数据放在 data/wiki_zh 目录下")
        return all_texts
    
    # 遍历所有子目录
    wiki_path = Path(wiki_path)
    json_files = list(wiki_path.rglob("wiki_*"))
    
    if max_files:
        json_files = json_files[:max_files]
    
    logger.info(f"找到 {len(json_files)} 个wiki文件")
    
    for json_file in tqdm(json_files, desc="加载wiki文件"):
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        try:
                            data = json.loads(line)
                            if 'text' in data and data['text'].strip():
                                all_texts.append({
                                    'id': data.get('id', ''),
                                    'title': data.get('title', ''),
                                    'text': data['text']
                                })
                        except json.JSONDecodeError:
                            continue
        except Exception as e:
            logger.warning(f"读取文件 {json_file} 时出错: {e}")
            continue
    
    logger.info(f"总共加载了 {len(all_texts)} 篇文章")
    return all_texts


def preprocess_texts(texts: List[Dict[str, str]], 
                    tokenizer,
                    max_length: int = 512) -> List[List[int]]:
    """
    预处理文本并转换为token IDs
    
    Args:
        texts: 文本字典列表
        tokenizer: tokenizer对象
        max_length: 最大序列长度
        
    Returns:
        token IDs列表
    """
    logger = setup_logging()
    all_token_ids = []
    
    logger.info("开始tokenize文本...")
    
    for item in tqdm(texts, desc="Tokenizing"):
        text = item['text']
        
        # 分段处理长文本
        # 将文本按段落分割
        paragraphs = [p.strip() for p in text.split('\n') if p.strip()]
        
        for paragraph in paragraphs:
            if len(paragraph) < 10:  # 跳过太短的段落
                continue
            
            # Tokenize
            encoded = tokenizer.encode(
                paragraph,
                add_special_tokens=True,
                max_length=max_length,
                truncation=True
            )
            
            if len(encoded) > 10:  # 只保留足够长的序列
                all_token_ids.append(encoded)
    
    logger.info(f"生成了 {len(all_token_ids)} 个训练样本")
    return all_token_ids


def save_processed_data(train_data: List[List[int]], 
                       val_data: List[List[int]],
                       output_path: str) -> None:
    """
    保存处理后的数据
    
    Args:
        train_data: 训练数据
        val_data: 验证数据
        output_path: 输出路径
    """
    logger = setup_logging()
    create_dir(output_path)
    
    # 保存为pickle文件
    train_file = os.path.join(output_path, 'train_data.pkl')
    val_file = os.path.join(output_path, 'val_data.pkl')
    
    with open(train_file, 'wb') as f:
        pickle.dump(train_data, f)
    
    with open(val_file, 'wb') as f:
        pickle.dump(val_data, f)
    
    logger.info(f"训练数据已保存: {train_file} ({len(train_data)} 样本)")
    logger.info(f"验证数据已保存: {val_file} ({len(val_data)} 样本)")


def main():
    parser = argparse.ArgumentParser(description='数据预处理')
    parser.add_argument('--config', type=str, default='config.yaml', 
                       help='配置文件路径')
    parser.add_argument('--max_files', type=int, default=None,
                       help='最大处理文件数（用于测试）')
    args = parser.parse_args()
    
    # 加载配置
    with open(args.config, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # 设置随机种子
    set_seed(config['training'].get('seed', 42))
    
    # 设置日志
    logger = setup_logging()
    logger.info("=" * 50)
    logger.info("开始数据预处理")
    logger.info("=" * 50)
    
    # 加载tokenizer
    model_name = config['model']['model_name']
    logger.info(f"加载tokenizer: {model_name}")
    
    try:
        tokenizer = BertTokenizer.from_pretrained(model_name)
    except:
        logger.warning(f"无法加载 {model_name}，使用bert-base-chinese")
        tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
    
    # 加载原始文本
    wiki_path = config['data']['wiki_path']
    all_texts = load_wiki_texts(wiki_path, max_files=args.max_files)
    
    if len(all_texts) == 0:
        logger.error("未找到任何文本数据，退出")
        logger.info("提示: 请将维基百科数据放在 data/wiki_zh 目录下")
        logger.info("数据格式: 每行一个JSON对象，包含id, title, text字段")
        return
    
    # 预处理文本
    max_length = config['data']['max_length']
    all_token_ids = preprocess_texts(all_texts, tokenizer, max_length)
    
    # 分割训练集和验证集
    train_split = config['data']['train_split']
    train_data, val_data = train_test_split(
        all_token_ids, 
        train_size=train_split, 
        random_state=config['training'].get('seed', 42)
    )
    
    logger.info(f"训练集大小: {len(train_data)}")
    logger.info(f"验证集大小: {len(val_data)}")
    
    # 保存处理后的数据
    output_path = config['data']['processed_path']
    save_processed_data(train_data, val_data, output_path)
    
    # 保存tokenizer
    tokenizer_path = os.path.join(output_path, 'tokenizer')
    create_dir(tokenizer_path)
    tokenizer.save_pretrained(tokenizer_path)
    logger.info(f"Tokenizer已保存: {tokenizer_path}")
    
    logger.info("=" * 50)
    logger.info("数据预处理完成！")
    logger.info("=" * 50)


if __name__ == '__main__':
    main()
