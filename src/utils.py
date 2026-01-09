"""
Utility functions for GPT2 Chinese text generation project.
包含常用工具函数，如设置随机种子、保存/加载模型、日志记录等。
"""

import os
import random
import logging
import time
from typing import Optional, Dict, Any
import numpy as np
import torch
from pathlib import Path


def set_seed(seed: int) -> None:
    """
    设置所有随机种子以确保可重现性
    
    Args:
        seed: 随机种子值
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # 确保CUDA操作的确定性
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def setup_logging(log_file: Optional[str] = None, level: int = logging.INFO) -> logging.Logger:
    """
    配置日志系统
    
    Args:
        log_file: 日志文件路径，如果为None则只输出到控制台
        level: 日志级别
        
    Returns:
        配置好的logger对象
    """
    logger = logging.getLogger("GPT2-Chinese")
    logger.setLevel(level)
    
    # 清除已有的handlers
    logger.handlers.clear()
    
    # 创建格式器
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # 控制台handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # 文件handler（如果指定了日志文件）
    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


def save_model(model: torch.nn.Module, 
               tokenizer: Any,
               output_dir: str,
               optimizer: Optional[torch.optim.Optimizer] = None,
               scheduler: Optional[Any] = None,
               epoch: Optional[int] = None,
               step: Optional[int] = None,
               **kwargs) -> None:
    """
    保存模型检查点
    
    Args:
        model: PyTorch模型
        tokenizer: Tokenizer对象
        output_dir: 输出目录
        optimizer: 优化器（可选）
        scheduler: 学习率调度器（可选）
        epoch: 当前epoch（可选）
        step: 当前步数（可选）
        **kwargs: 其他要保存的信息
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # 保存模型和tokenizer
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    # 保存训练状态
    if optimizer or scheduler or epoch is not None or step is not None:
        checkpoint = {
            'epoch': epoch,
            'step': step,
        }
        if optimizer:
            checkpoint['optimizer_state_dict'] = optimizer.state_dict()
        if scheduler:
            checkpoint['scheduler_state_dict'] = scheduler.state_dict()
        checkpoint.update(kwargs)
        
        torch.save(checkpoint, os.path.join(output_dir, 'training_state.pt'))
    
    print(f"模型已保存到: {output_dir}")


def load_model(model_path: str, 
               model_class: Any,
               tokenizer_class: Any,
               device: str = 'cuda') -> tuple:
    """
    加载模型和tokenizer
    
    Args:
        model_path: 模型路径
        model_class: 模型类
        tokenizer_class: Tokenizer类
        device: 设备
        
    Returns:
        (model, tokenizer) 元组
    """
    model = model_class.from_pretrained(model_path)
    tokenizer = tokenizer_class.from_pretrained(model_path)
    model.to(device)
    model.eval()
    
    print(f"模型已从 {model_path} 加载")
    return model, tokenizer


def load_checkpoint(checkpoint_path: str,
                   model: torch.nn.Module,
                   optimizer: Optional[torch.optim.Optimizer] = None,
                   scheduler: Optional[Any] = None) -> Dict[str, Any]:
    """
    加载训练检查点
    
    Args:
        checkpoint_path: 检查点文件路径
        model: 模型对象
        optimizer: 优化器（可选）
        scheduler: 学习率调度器（可选）
        
    Returns:
        包含训练状态的字典
    """
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    if optimizer and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    if scheduler and 'scheduler_state_dict' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
    return checkpoint


def calculate_perplexity(loss: float) -> float:
    """
    从损失值计算困惑度
    
    Args:
        loss: 交叉熵损失值
        
    Returns:
        困惑度值
    """
    return np.exp(loss)


def format_time(seconds: float) -> str:
    """
    将秒数格式化为可读的时间字符串
    
    Args:
        seconds: 秒数
        
    Returns:
        格式化的时间字符串
    """
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    
    if hours > 0:
        return f"{hours}h {minutes}m {secs}s"
    elif minutes > 0:
        return f"{minutes}m {secs}s"
    else:
        return f"{secs}s"


class Timer:
    """计时器类，用于测量代码执行时间"""
    
    def __init__(self):
        self.start_time = None
        self.end_time = None
    
    def start(self) -> None:
        """开始计时"""
        self.start_time = time.time()
    
    def stop(self) -> float:
        """
        停止计时
        
        Returns:
            经过的时间（秒）
        """
        self.end_time = time.time()
        return self.elapsed()
    
    def elapsed(self) -> float:
        """
        获取经过的时间
        
        Returns:
            经过的时间（秒）
        """
        if self.start_time is None:
            return 0.0
        end = self.end_time if self.end_time else time.time()
        return end - self.start_time
    
    def __enter__(self):
        self.start()
        return self
    
    def __exit__(self, *args):
        self.stop()


def clean_text(text: str) -> str:
    """
    清理文本，去除多余空格和特殊字符
    
    Args:
        text: 原始文本
        
    Returns:
        清理后的文本
    """
    # 去除多余空格
    text = ' '.join(text.split())
    # 去除特殊控制字符
    text = ''.join(char for char in text if char.isprintable() or char in '\n\t')
    return text.strip()


def get_device() -> str:
    """
    获取可用的设备（GPU或CPU）
    
    Returns:
        设备名称字符串
    """
    if torch.cuda.is_available():
        device = 'cuda'
        print(f"使用GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = 'cpu'
        print("使用CPU")
    return device


def count_parameters(model: torch.nn.Module) -> int:
    """
    计算模型的可训练参数数量
    
    Args:
        model: PyTorch模型
        
    Returns:
        可训练参数数量
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def create_dir(directory: str) -> None:
    """
    创建目录（如果不存在）
    
    Args:
        directory: 目录路径
    """
    Path(directory).mkdir(parents=True, exist_ok=True)
