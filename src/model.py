"""
GPT2模型定义和封装
使用Hugging Face的transformers库
"""

import torch
import torch.nn as nn
from transformers import GPT2LMHeadModel, GPT2Config, BertTokenizer
from typing import Optional, Tuple


class GPT2ChineseModel:
    """
    GPT2中文模型封装类
    
    封装了模型的加载、保存和配置
    """
    
    def __init__(self, 
                 model_name: str = "uer/gpt2-chinese-cluecorpussmall",
                 config: Optional[dict] = None,
                 device: str = 'cuda'):
        """
        初始化GPT2模型
        
        Args:
            model_name: 预训练模型名称或路径
            config: 模型配置字典（用于从头训练）
            device: 设备
        """
        self.device = device
        self.model_name = model_name
        
        if model_name == "scratch" and config:
            # 从头开始训练
            print("从头创建GPT2模型...")
            model_config = GPT2Config(
                vocab_size=config.get('vocab_size', 21128),
                n_positions=config.get('max_position_embeddings', 512),
                n_embd=config.get('hidden_size', 768),
                n_layer=config.get('num_hidden_layers', 12),
                n_head=config.get('num_attention_heads', 12),
                pad_token_id=0,
                bos_token_id=101,
                eos_token_id=102,
            )
            self.model = GPT2LMHeadModel(model_config)
        else:
            # 从预训练模型加载
            print(f"加载预训练模型: {model_name}")
            try:
                self.model = GPT2LMHeadModel.from_pretrained(model_name)
            except Exception as e:
                print(f"加载模型失败: {e}")
                print("尝试加载默认配置...")
                model_config = GPT2Config(
                    vocab_size=21128,
                    n_positions=512,
                    n_embd=768,
                    n_layer=12,
                    n_head=12,
                )
                self.model = GPT2LMHeadModel(model_config)
        
        self.model.to(device)
        
        # 打印模型信息
        num_params = sum(p.numel() for p in self.model.parameters())
        num_trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"模型参数总数: {num_params:,}")
        print(f"可训练参数: {num_trainable_params:,}")
    
    def get_model(self) -> GPT2LMHeadModel:
        """返回模型对象"""
        return self.model
    
    def save(self, save_path: str) -> None:
        """
        保存模型
        
        Args:
            save_path: 保存路径
        """
        self.model.save_pretrained(save_path)
        print(f"模型已保存到: {save_path}")
    
    def load(self, model_path: str) -> None:
        """
        加载模型
        
        Args:
            model_path: 模型路径
        """
        self.model = GPT2LMHeadModel.from_pretrained(model_path)
        self.model.to(self.device)
        print(f"模型已从 {model_path} 加载")
    
    def forward(self, 
                input_ids: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None,
                labels: Optional[torch.Tensor] = None) -> Tuple:
        """
        前向传播
        
        Args:
            input_ids: 输入token IDs
            attention_mask: 注意力掩码
            labels: 标签（用于计算损失）
            
        Returns:
            模型输出
        """
        return self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )
    
    def generate(self,
                input_ids: torch.Tensor,
                max_length: int = 100,
                temperature: float = 1.0,
                top_k: int = 50,
                top_p: float = 0.95,
                num_return_sequences: int = 1,
                do_sample: bool = True,
                **kwargs) -> torch.Tensor:
        """
        生成文本
        
        Args:
            input_ids: 输入token IDs
            max_length: 生成的最大长度
            temperature: 温度参数
            top_k: Top-k采样参数
            top_p: Top-p (nucleus)采样参数
            num_return_sequences: 返回序列数量
            do_sample: 是否采样
            **kwargs: 其他生成参数
            
        Returns:
            生成的token IDs
        """
        self.model.eval()
        
        with torch.no_grad():
            outputs = self.model.generate(
                input_ids=input_ids,
                max_length=max_length,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                num_return_sequences=num_return_sequences,
                do_sample=do_sample,
                pad_token_id=0,
                eos_token_id=102,
                **kwargs
            )
        
        return outputs


def load_tokenizer(tokenizer_path: str) -> BertTokenizer:
    """
    加载tokenizer
    
    Args:
        tokenizer_path: tokenizer路径
        
    Returns:
        BertTokenizer对象
    """
    try:
        tokenizer = BertTokenizer.from_pretrained(tokenizer_path)
    except:
        print(f"无法从 {tokenizer_path} 加载tokenizer，使用默认")
        tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
    
    return tokenizer
