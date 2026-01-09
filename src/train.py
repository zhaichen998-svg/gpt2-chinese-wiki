"""
训练脚本
实现完整的GPT2模型训练流程
"""

import os
import argparse
import yaml
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.cuda.amp import autocast, GradScaler
from transformers import get_linear_schedule_with_warmup
from torch.utils.tensorboard import SummaryWriter

from dataset import WikiTextDataset, DynamicPaddingDataset, collate_fn
from model import GPT2ChineseModel, load_tokenizer
from utils import (
    setup_logging, set_seed, save_model, calculate_perplexity,
    format_time, Timer, get_device, create_dir
)


def train_epoch(model, 
                train_loader, 
                optimizer, 
                scheduler,
                scaler,
                device,
                gradient_accumulation_steps,
                max_grad_norm,
                use_amp,
                logger):
    """
    训练一个epoch
    
    Args:
        model: GPT2模型
        train_loader: 训练数据加载器
        optimizer: 优化器
        scheduler: 学习率调度器
        scaler: 梯度缩放器（用于混合精度训练）
        device: 设备
        gradient_accumulation_steps: 梯度累积步数
        max_grad_norm: 梯度裁剪阈值
        use_amp: 是否使用混合精度训练
        logger: 日志记录器
        
    Returns:
        平均损失
    """
    model.train()
    total_loss = 0
    optimizer.zero_grad()
    
    progress_bar = tqdm(train_loader, desc="训练中")
    
    for step, batch in enumerate(progress_bar):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        
        # GPT2的labels就是input_ids（自回归语言模型）
        labels = input_ids.clone()
        
        # 混合精度训练
        if use_amp:
            with autocast():
                outputs = model.forward(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                loss = outputs.loss / gradient_accumulation_steps
            
            scaler.scale(loss).backward()
        else:
            outputs = model.forward(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            loss = outputs.loss / gradient_accumulation_steps
            loss.backward()
        
        total_loss += loss.item() * gradient_accumulation_steps
        
        # 梯度累积
        if (step + 1) % gradient_accumulation_steps == 0:
            if use_amp:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.get_model().parameters(), max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
            else:
                torch.nn.utils.clip_grad_norm_(model.get_model().parameters(), max_grad_norm)
                optimizer.step()
            
            scheduler.step()
            optimizer.zero_grad()
        
        # 更新进度条
        current_lr = scheduler.get_last_lr()[0]
        progress_bar.set_postfix({
            'loss': f'{loss.item() * gradient_accumulation_steps:.4f}',
            'lr': f'{current_lr:.2e}'
        })
    
    avg_loss = total_loss / len(train_loader)
    return avg_loss


def validate(model, val_loader, device, logger):
    """
    验证模型
    
    Args:
        model: GPT2模型
        val_loader: 验证数据加载器
        device: 设备
        logger: 日志记录器
        
    Returns:
        平均损失
    """
    model.get_model().eval()
    total_loss = 0
    
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="验证中"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = input_ids.clone()
            
            outputs = model.forward(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            
            total_loss += outputs.loss.item()
    
    avg_loss = total_loss / len(val_loader)
    return avg_loss


def main():
    parser = argparse.ArgumentParser(description='训练GPT2中文模型')
    parser.add_argument('--config', type=str, default='config.yaml',
                       help='配置文件路径')
    parser.add_argument('--resume', type=str, default=None,
                       help='从checkpoint恢复训练')
    args = parser.parse_args()
    
    # 加载配置
    with open(args.config, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # 设置随机种子
    seed = config['training'].get('seed', 42)
    set_seed(seed)
    
    # 设置日志
    log_dir = 'logs'
    create_dir(log_dir)
    logger = setup_logging(os.path.join(log_dir, 'train.log'))
    
    logger.info("=" * 50)
    logger.info("开始训练GPT2中文模型")
    logger.info("=" * 50)
    
    # 获取设备
    device = get_device()
    
    # 创建输出目录
    output_dir = config['training']['output_dir']
    create_dir(output_dir)
    
    # 加载tokenizer
    tokenizer_path = os.path.join(config['data']['processed_path'], 'tokenizer')
    if not os.path.exists(tokenizer_path):
        logger.warning(f"Tokenizer路径不存在: {tokenizer_path}")
        logger.info("尝试使用模型名称加载tokenizer...")
        tokenizer_path = config['model']['model_name']
    
    tokenizer = load_tokenizer(tokenizer_path)
    logger.info(f"Tokenizer词表大小: {len(tokenizer)}")
    
    # 加载数据集
    logger.info("加载数据集...")
    train_data_path = os.path.join(config['data']['processed_path'], 'train_data.pkl')
    val_data_path = os.path.join(config['data']['processed_path'], 'val_data.pkl')
    
    if not os.path.exists(train_data_path):
        logger.error(f"训练数据不存在: {train_data_path}")
        logger.info("请先运行数据预处理: python src/preprocess.py")
        return
    
    train_dataset = WikiTextDataset(
        train_data_path,
        max_length=config['data']['max_length'],
        pad_token_id=tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
    )
    
    val_dataset = WikiTextDataset(
        val_data_path,
        max_length=config['data']['max_length'],
        pad_token_id=tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
    )
    
    # 创建DataLoader
    batch_size = config['training']['batch_size']
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    logger.info(f"训练批次数: {len(train_loader)}")
    logger.info(f"验证批次数: {len(val_loader)}")
    
    # 初始化模型
    logger.info("初始化模型...")
    model = GPT2ChineseModel(
        model_name=config['model']['model_name'],
        config=config['model'],
        device=device
    )
    
    # 初始化优化器
    optimizer = AdamW(
        model.get_model().parameters(),
        lr=config['training']['learning_rate'],
        eps=1e-8
    )
    
    # 计算总训练步数
    num_epochs = config['training']['num_epochs']
    gradient_accumulation_steps = config['training']['gradient_accumulation_steps']
    total_steps = len(train_loader) * num_epochs // gradient_accumulation_steps
    
    # 初始化学习率调度器
    warmup_steps = config['training']['warmup_steps']
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps
    )
    
    # 混合精度训练
    use_amp = config['training'].get('use_amp', True) and device == 'cuda'
    scaler = GradScaler() if use_amp else None
    
    if use_amp:
        logger.info("使用混合精度训练 (AMP)")
    
    # TensorBoard
    writer = SummaryWriter(log_dir=os.path.join(log_dir, 'tensorboard'))
    
    # 训练循环
    logger.info("开始训练...")
    logger.info(f"总epochs: {num_epochs}")
    logger.info(f"总步数: {total_steps}")
    
    best_val_loss = float('inf')
    global_step = 0
    
    timer = Timer()
    timer.start()
    
    for epoch in range(num_epochs):
        logger.info(f"\n{'='*50}")
        logger.info(f"Epoch {epoch + 1}/{num_epochs}")
        logger.info(f"{'='*50}")
        
        # 训练
        train_loss = train_epoch(
            model, train_loader, optimizer, scheduler, scaler,
            device, gradient_accumulation_steps,
            config['training']['max_grad_norm'],
            use_amp, logger
        )
        
        train_ppl = calculate_perplexity(train_loss)
        logger.info(f"训练损失: {train_loss:.4f}, 困惑度: {train_ppl:.2f}")
        
        # 验证
        val_loss = validate(model, val_loader, device, logger)
        val_ppl = calculate_perplexity(val_loss)
        logger.info(f"验证损失: {val_loss:.4f}, 困惑度: {val_ppl:.2f}")
        
        # TensorBoard记录
        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Loss/val', val_loss, epoch)
        writer.add_scalar('Perplexity/train', train_ppl, epoch)
        writer.add_scalar('Perplexity/val', val_ppl, epoch)
        
        # 保存最佳模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_path = os.path.join(output_dir, 'checkpoint-best')
            save_model(
                model.get_model(),
                tokenizer,
                best_model_path,
                optimizer,
                scheduler,
                epoch,
                global_step
            )
            logger.info(f"保存最佳模型 (验证损失: {val_loss:.4f})")
        
        # 定期保存checkpoint
        if (epoch + 1) % 5 == 0:
            checkpoint_path = os.path.join(output_dir, f'checkpoint-epoch-{epoch+1}')
            save_model(
                model.get_model(),
                tokenizer,
                checkpoint_path,
                optimizer,
                scheduler,
                epoch,
                global_step
            )
            logger.info(f"保存checkpoint: {checkpoint_path}")
    
    # 训练结束
    elapsed_time = timer.stop()
    logger.info("=" * 50)
    logger.info("训练完成！")
    logger.info(f"总耗时: {format_time(elapsed_time)}")
    logger.info(f"最佳验证损失: {best_val_loss:.4f}")
    logger.info(f"最佳模型保存在: {os.path.join(output_dir, 'checkpoint-best')}")
    logger.info("=" * 50)
    
    writer.close()


if __name__ == '__main__':
    main()
