"""
PyTorch Dataset类实现
用于加载和处理预处理后的数据
"""

import torch
from torch.utils.data import Dataset
from typing import List, Dict
import pickle


class WikiTextDataset(Dataset):
    """
    维基百科文本数据集类
    
    用于GPT2语言模型训练的数据集
    """
    
    def __init__(self, 
                 data_path: str,
                 max_length: int = 512,
                 pad_token_id: int = 0):
        """
        初始化数据集
        
        Args:
            data_path: 预处理后的数据文件路径（.pkl文件）
            max_length: 最大序列长度
            pad_token_id: padding token的ID
        """
        super().__init__()
        
        self.max_length = max_length
        self.pad_token_id = pad_token_id
        
        # 加载数据
        with open(data_path, 'rb') as f:
            self.data = pickle.load(f)
        
        print(f"加载数据集: {data_path}")
        print(f"样本数量: {len(self.data)}")
    
    def __len__(self) -> int:
        """返回数据集大小"""
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        获取单个样本
        
        Args:
            idx: 样本索引
            
        Returns:
            包含input_ids和attention_mask的字典
        """
        token_ids = self.data[idx]
        
        # 截断或padding到固定长度
        if len(token_ids) > self.max_length:
            token_ids = token_ids[:self.max_length]
        
        # 创建attention mask (1表示真实token，0表示padding)
        attention_mask = [1] * len(token_ids)
        
        # Padding
        padding_length = self.max_length - len(token_ids)
        if padding_length > 0:
            token_ids = token_ids + [self.pad_token_id] * padding_length
            attention_mask = attention_mask + [0] * padding_length
        
        # 转换为tensor
        return {
            'input_ids': torch.tensor(token_ids, dtype=torch.long),
            'attention_mask': torch.tensor(attention_mask, dtype=torch.long)
        }


class DynamicPaddingDataset(Dataset):
    """
    动态padding数据集类
    
    在DataLoader的collate_fn中进行动态padding，更节省内存
    """
    
    def __init__(self, data_path: str):
        """
        初始化数据集
        
        Args:
            data_path: 预处理后的数据文件路径（.pkl文件）
        """
        super().__init__()
        
        # 加载数据
        with open(data_path, 'rb') as f:
            self.data = pickle.load(f)
        
        print(f"加载数据集: {data_path}")
        print(f"样本数量: {len(self.data)}")
    
    def __len__(self) -> int:
        """返回数据集大小"""
        return len(self.data)
    
    def __getitem__(self, idx: int) -> List[int]:
        """
        获取单个样本的token IDs
        
        Args:
            idx: 样本索引
            
        Returns:
            token IDs列表
        """
        return self.data[idx]


def collate_fn(batch: List[List[int]], 
               pad_token_id: int = 0,
               max_length: int = 512) -> Dict[str, torch.Tensor]:
    """
    自定义collate函数，用于DataLoader
    
    对batch中的样本进行动态padding到batch内的最大长度
    
    Args:
        batch: 样本列表，每个样本是token IDs列表
        pad_token_id: padding token的ID
        max_length: 最大序列长度
        
    Returns:
        包含input_ids和attention_mask的字典
    """
    # 找到batch中的最大长度
    batch_max_length = min(max(len(seq) for seq in batch), max_length)
    
    input_ids_list = []
    attention_mask_list = []
    
    for token_ids in batch:
        # 截断
        if len(token_ids) > batch_max_length:
            token_ids = token_ids[:batch_max_length]
        
        # 创建attention mask
        attention_mask = [1] * len(token_ids)
        
        # Padding
        padding_length = batch_max_length - len(token_ids)
        if padding_length > 0:
            token_ids = token_ids + [pad_token_id] * padding_length
            attention_mask = attention_mask + [0] * padding_length
        
        input_ids_list.append(token_ids)
        attention_mask_list.append(attention_mask)
    
    return {
        'input_ids': torch.tensor(input_ids_list, dtype=torch.long),
        'attention_mask': torch.tensor(attention_mask_list, dtype=torch.long)
    }
