"""
神经网络训练模块
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from typing import List, Tuple, Dict, Any, Optional
import logging
import time
import os
from .model import AlphaZeroNet, AlphaZeroLoss


class ChessDataset(Dataset):
    """
    国际象棋训练数据集
    """
    
    def __init__(self, boards: np.ndarray, policies: np.ndarray, values: np.ndarray):
        """
        初始化数据集
        
        Args:
            boards: 棋盘状态 (N, 20, 8, 8)
            policies: 策略分布 (N, 4096)
            values: 价值标签 (N,)
        """
        assert len(boards) == len(policies) == len(values), "数据长度不匹配"
        
        self.boards = torch.FloatTensor(boards)
        self.policies = torch.FloatTensor(policies)
        self.values = torch.FloatTensor(values)
    
    def __len__(self) -> int:
        return len(self.boards)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.boards[idx], self.policies[idx], self.values[idx]


class NetworkTrainer:
    """
    神经网络训练器
    """
    
    def __init__(self, 
                 model: AlphaZeroNet,
                 config,
                 device: Optional[torch.device] = None):
        """
        初始化训练器
        
        Args:
            model: 神经网络模型
            config: 配置对象
            device: 计算设备
        """
        self.model = model
        self.config = config
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 将模型移动到指定设备
        self.model.to(self.device)
        
        # 损失函数
        self.criterion = AlphaZeroLoss()
        
        # 优化器
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=config.LEARNING_RATE,
            weight_decay=config.L2_REG
        )
        
        # 学习率调度器
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=5,
            verbose=True
        )
        
        # TensorBoard
        self.writer = None
        self.setup_tensorboard()
        
        # 训练历史
        self.training_history = {
            'total_loss': [],
            'policy_loss': [],
            'value_loss': [],
            'learning_rate': [],
            'epoch_time': []
        }
        
        # 日志
        self.logger = logging.getLogger(__name__)
        
        # 训练统计
        self.global_step = 0
        self.epoch_count = 0
    
    def setup_tensorboard(self):
        """设置TensorBoard日志"""
        try:
            log_dir = os.path.join(self.config.LOG_DIR, 'tensorboard')
            os.makedirs(log_dir, exist_ok=True)
            self.writer = SummaryWriter(log_dir)
            self.logger.info(f"TensorBoard日志目录: {log_dir}")
        except Exception as e:
            self.logger.warning(f"TensorBoard设置失败: {e}")
    
    def train_epoch(self, 
                   train_loader: DataLoader,
                   val_loader: Optional[DataLoader] = None) -> Dict[str, float]:
        """
        训练一个epoch
        
        Args:
            train_loader: 训练数据加载器
            val_loader: 验证数据加载器
            
        Returns:
            Dict[str, float]: 训练统计信息
        """
        self.model.train()
        
        total_loss = 0.0
        total_policy_loss = 0.0
        total_value_loss = 0.0
        num_batches = 0
        
        start_time = time.time()
        
        for batch_idx, (boards, target_policies, target_values) in enumerate(train_loader):
            # 移动数据到设备
            boards = boards.to(self.device)
            target_policies = target_policies.to(self.device)
            target_values = target_values.to(self.device)
            
            # 前向传播
            policy_logits, value_pred = self.model(boards)
            
            # 计算损失
            loss, loss_dict = self.criterion(
                policy_logits, value_pred,
                target_policies, target_values
            )
            
            # 反向传播
            self.optimizer.zero_grad()
            loss.backward()
            
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            # 更新参数
            self.optimizer.step()
            
            # 累积损失
            total_loss += loss_dict['total_loss']
            total_policy_loss += loss_dict['policy_loss']
            total_value_loss += loss_dict['value_loss']
            num_batches += 1
            
            # 记录到TensorBoard
            if self.writer and self.global_step % 10 == 0:
                self.writer.add_scalar('Loss/Train_Total', loss_dict['total_loss'], self.global_step)
                self.writer.add_scalar('Loss/Train_Policy', loss_dict['policy_loss'], self.global_step)
                self.writer.add_scalar('Loss/Train_Value', loss_dict['value_loss'], self.global_step)
                self.writer.add_scalar('Learning_Rate', self.optimizer.param_groups[0]['lr'], self.global_step)
            
            self.global_step += 1
            
            # 打印进度
            if batch_idx % 50 == 0:
                self.logger.info(
                    f"Batch {batch_idx}/{len(train_loader)}, "
                    f"Loss: {loss_dict['total_loss']:.4f}, "
                    f"Policy: {loss_dict['policy_loss']:.4f}, "
                    f"Value: {loss_dict['value_loss']:.4f}"
                )
        
        # 计算平均损失
        avg_total_loss = total_loss / num_batches
        avg_policy_loss = total_policy_loss / num_batches
        avg_value_loss = total_value_loss / num_batches
        
        # 验证
        val_stats = {}
        if val_loader:
            val_stats = self.validate(val_loader)
            # 调整学习率
            self.scheduler.step(val_stats['total_loss'])
        
        epoch_time = time.time() - start_time
        
        # 训练统计
        train_stats = {
            'total_loss': avg_total_loss,
            'policy_loss': avg_policy_loss,
            'value_loss': avg_value_loss,
            'learning_rate': self.optimizer.param_groups[0]['lr'],
            'epoch_time': epoch_time
        }
        
        # 更新历史记录
        for key, value in train_stats.items():
            self.training_history[key].append(value)
        
        # 记录到TensorBoard
        if self.writer:
            self.writer.add_scalar('Loss/Train_Epoch_Total', avg_total_loss, self.epoch_count)
            self.writer.add_scalar('Loss/Train_Epoch_Policy', avg_policy_loss, self.epoch_count)
            self.writer.add_scalar('Loss/Train_Epoch_Value', avg_value_loss, self.epoch_count)
            
            if val_stats:
                self.writer.add_scalar('Loss/Val_Total', val_stats['total_loss'], self.epoch_count)
                self.writer.add_scalar('Loss/Val_Policy', val_stats['policy_loss'], self.epoch_count)
                self.writer.add_scalar('Loss/Val_Value', val_stats['value_loss'], self.epoch_count)
        
        self.epoch_count += 1
        
        return {**train_stats, **val_stats}
    
    def validate(self, val_loader: DataLoader) -> Dict[str, float]:
        """
        验证模型
        
        Args:
            val_loader: 验证数据加载器
            
        Returns:
            Dict[str, float]: 验证统计信息
        """
        self.model.eval()
        
        total_loss = 0.0
        total_policy_loss = 0.0
        total_value_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for boards, target_policies, target_values in val_loader:
                # 移动数据到设备
                boards = boards.to(self.device)
                target_policies = target_policies.to(self.device)
                target_values = target_values.to(self.device)
                
                # 前向传播
                policy_logits, value_pred = self.model(boards)
                
                # 计算损失
                _, loss_dict = self.criterion(
                    policy_logits, value_pred,
                    target_policies, target_values
                )
                
                # 累积损失
                total_loss += loss_dict['total_loss']
                total_policy_loss += loss_dict['policy_loss']
                total_value_loss += loss_dict['value_loss']
                num_batches += 1
        
        # 计算平均损失
        return {
            'val_total_loss': total_loss / num_batches,
            'val_policy_loss': total_policy_loss / num_batches,
            'val_value_loss': total_value_loss / num_batches
        }
    
    def train(self, 
              train_data: Tuple[np.ndarray, np.ndarray, np.ndarray],
              val_data: Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]] = None,
              epochs: Optional[int] = None) -> Dict[str, List[float]]:
        """
        训练模型
        
        Args:
            train_data: 训练数据 (boards, policies, values)
            val_data: 验证数据
            epochs: 训练轮数
            
        Returns:
            Dict[str, List[float]]: 训练历史
        """
        epochs = epochs or self.config.EPOCHS
        
        # 创建数据集
        train_dataset = ChessDataset(*train_data)
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.BATCH_SIZE,
            shuffle=True,
            num_workers=2,
            pin_memory=True if self.device.type == 'cuda' else False
        )
        
        val_loader = None
        if val_data:
            val_dataset = ChessDataset(*val_data)
            val_loader = DataLoader(
                val_dataset,
                batch_size=self.config.BATCH_SIZE,
                shuffle=False,
                num_workers=2,
                pin_memory=True if self.device.type == 'cuda' else False
            )
        
        self.logger.info(f"开始训练，总轮数: {epochs}")
        self.logger.info(f"训练数据: {len(train_dataset)} 样本")
        if val_data:
            self.logger.info(f"验证数据: {len(val_dataset)} 样本")
        
        # 训练循环
        for epoch in range(epochs):
            self.logger.info(f"\n开始训练 Epoch {epoch + 1}/{epochs}")
            
            # 训练一个epoch
            epoch_stats = self.train_epoch(train_loader, val_loader)
            
            # 打印统计信息
            self.logger.info(
                f"Epoch {epoch + 1} 完成 - "
                f"训练损失: {epoch_stats['total_loss']:.4f}, "
                f"验证损失: {epoch_stats.get('val_total_loss', 'N/A')}, "
                f"学习率: {epoch_stats['learning_rate']:.6f}, "
                f"时间: {epoch_stats['epoch_time']:.2f}s"
            )
        
        self.logger.info("训练完成")
        return self.training_history
    
    def evaluate_model(self, test_data: Tuple[np.ndarray, np.ndarray, np.ndarray]) -> Dict[str, float]:
        """
        评估模型性能
        
        Args:
            test_data: 测试数据
            
        Returns:
            Dict[str, float]: 评估结果
        """
        test_dataset = ChessDataset(*test_data)
        test_loader = DataLoader(
            test_dataset,
            batch_size=self.config.BATCH_SIZE,
            shuffle=False
        )
        
        self.model.eval()
        
        total_loss = 0.0
        total_policy_loss = 0.0
        total_value_loss = 0.0
        policy_accuracy = 0.0
        value_mae = 0.0
        num_samples = 0
        
        with torch.no_grad():
            for boards, target_policies, target_values in test_loader:
                boards = boards.to(self.device)
                target_policies = target_policies.to(self.device)
                target_values = target_values.to(self.device)
                
                # 前向传播
                policy_logits, value_pred = self.model(boards)
                
                # 计算损失
                _, loss_dict = self.criterion(
                    policy_logits, value_pred,
                    target_policies, target_values
                )
                
                # 累积损失
                batch_size = boards.size(0)
                total_loss += loss_dict['total_loss'] * batch_size
                total_policy_loss += loss_dict['policy_loss'] * batch_size
                total_value_loss += loss_dict['value_loss'] * batch_size
                
                # 计算准确率
                policy_probs = torch.softmax(policy_logits, dim=1)
                pred_actions = torch.argmax(policy_probs, dim=1)
                target_actions = torch.argmax(target_policies, dim=1)
                policy_accuracy += (pred_actions == target_actions).float().sum().item()
                
                # 计算价值MAE
                value_mae += torch.abs(value_pred.squeeze() - target_values).sum().item()
                
                num_samples += batch_size
        
        return {
            'test_total_loss': total_loss / num_samples,
            'test_policy_loss': total_policy_loss / num_samples,
            'test_value_loss': total_value_loss / num_samples,
            'policy_accuracy': policy_accuracy / num_samples,
            'value_mae': value_mae / num_samples
        }
    
    def save_checkpoint(self, filepath: str, additional_info: Optional[Dict] = None) -> bool:
        """
        保存训练检查点
        
        Args:
            filepath: 保存路径
            additional_info: 额外信息
            
        Returns:
            bool: 是否成功保存
        """
        try:
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            
            checkpoint = {
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'scheduler_state_dict': self.scheduler.state_dict(),
                'training_history': self.training_history,
                'global_step': self.global_step,
                'epoch_count': self.epoch_count,
                'config': self.config.to_dict() if hasattr(self.config, 'to_dict') else {},
            }
            
            if additional_info:
                checkpoint.update(additional_info)
            
            torch.save(checkpoint, filepath)
            self.logger.info(f"检查点已保存到: {filepath}")
            return True
            
        except Exception as e:
            self.logger.error(f"保存检查点失败: {e}")
            return False
    
    def load_checkpoint(self, filepath: str) -> bool:
        """
        加载训练检查点
        
        Args:
            filepath: 检查点文件路径
            
        Returns:
            bool: 是否成功加载
        """
        try:
            if not os.path.exists(filepath):
                self.logger.error(f"检查点文件不存在: {filepath}")
                return False
            
            checkpoint = torch.load(filepath, map_location=self.device)
            
            # 加载模型状态
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            
            # 加载训练状态
            self.training_history = checkpoint.get('training_history', self.training_history)
            self.global_step = checkpoint.get('global_step', 0)
            self.epoch_count = checkpoint.get('epoch_count', 0)
            
            self.logger.info(f"检查点已从 {filepath} 加载")
            return True
            
        except Exception as e:
            self.logger.error(f"加载检查点失败: {e}")
            return False
    
    def get_training_stats(self) -> Dict[str, Any]:
        """
        获取训练统计信息
        
        Returns:
            Dict[str, Any]: 训练统计
        """
        if not self.training_history['total_loss']:
            return {}
        
        return {
            'total_epochs': self.epoch_count,
            'global_steps': self.global_step,
            'best_loss': min(self.training_history['total_loss']),
            'current_lr': self.optimizer.param_groups[0]['lr'],
            'avg_epoch_time': np.mean(self.training_history['epoch_time']),
            'total_training_time': sum(self.training_history['epoch_time'])
        }
    
    def close(self):
        """关闭训练器，清理资源"""
        if self.writer:
            self.writer.close()
        self.logger.info("训练器已关闭") 