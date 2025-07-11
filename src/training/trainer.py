"""
训练模块
"""

import torch
import numpy as np
import logging
from typing import Dict, List, Optional, Tuple
import time
from tqdm import tqdm
from torch.cuda.amp import autocast, GradScaler

from ..neural_network.model import AlphaZeroNet
from ..self_play.self_play import SelfPlay
from ..evaluation.visualizer import TrainingVisualizer


class Trainer:
    """AlphaZero训练器"""
    
    def __init__(self, model: AlphaZeroNet, config):
        """
        初始化训练器
        
        Args:
            model: 神经网络模型
            config: 配置对象
        """
        self.model = model
        self.config = config
        self.device = torch.device(config.DEVICE)
        self.model.to(self.device)
        
        # 优化器
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=config.LEARNING_RATE,
            weight_decay=config.WEIGHT_DECAY
        )
        
        # 学习率调度器
        self.scheduler = torch.optim.lr_scheduler.MultiStepLR(
            self.optimizer,
            milestones=config.LR_MILESTONES,
            gamma=config.LR_GAMMA
        )
        
        # 混合精度训练
        self.scaler = GradScaler() if config.USE_AMP else None
        
        # 自我对弈
        self.self_play = SelfPlay(model, config)
        
        # 可视化器
        self.visualizer = TrainingVisualizer(config)
        
        # 训练统计
        self.stats = {
            'total_loss': [],
            'policy_loss': [],
            'value_loss': [],
            'learning_rate': []
        }
        
        self.logger = logging.getLogger(__name__)
        
        # CUDA优化
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True
            if hasattr(torch.cuda, 'empty_cache'):
                torch.cuda.empty_cache()
    
    def train(self, num_iterations: int) -> None:
        """
        训练模型
        
        Args:
            num_iterations: 训练迭代次数
        """
        self.logger.info(f"开始训练: {num_iterations} 次迭代")
        
        for iteration in range(num_iterations):
            iteration_start = time.time()
            self.logger.info(f"迭代 {iteration + 1}/{num_iterations}")
            
            # 1. 自我对弈生成数据
            states, policies, values = self.self_play.generate_games(
                self.config.NUM_SELF_PLAY_GAMES
            )
            
            # 2. 训练模型
            total_loss, policy_loss, value_loss = self._train_epoch(
                states, policies, values
            )
            
            # 3. 更新学习率
            self.scheduler.step()
            current_lr = self.scheduler.get_last_lr()[0]
            
            # 4. 更新统计信息
            self.stats['total_loss'].append(total_loss)
            self.stats['policy_loss'].append(policy_loss)
            self.stats['value_loss'].append(value_loss)
            self.stats['learning_rate'].append(current_lr)
            
            # 5. 可视化训练过程
            self.visualizer.plot_training_stats(self.stats)
            
            # 6. 输出进度
            iteration_time = time.time() - iteration_start
            self.logger.info(
                f"迭代 {iteration + 1} 完成 "
                f"(用时: {iteration_time:.2f}s): "
                f"总损失={total_loss:.4f}, "
                f"策略损失={policy_loss:.4f}, "
                f"价值损失={value_loss:.4f}, "
                f"学习率={current_lr:.6f}"
            )
            
            # 7. 清理GPU内存
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    
    def _train_epoch(self,
                     states: np.ndarray,
                     policies: np.ndarray,
                     values: np.ndarray) -> Tuple[float, float, float]:
        """
        训练一个epoch
        
        Args:
            states: 状态数据
            policies: 策略数据
            values: 价值数据
            
        Returns:
            Tuple[float, float, float]: (总损失, 策略损失, 价值损失)
        """
        self.model.train()
        
        # 转换为张量并固定内存
        states = torch.from_numpy(states).float().pin_memory().to(self.device, non_blocking=True)
        policies = torch.from_numpy(policies).float().pin_memory().to(self.device, non_blocking=True)
        values = torch.from_numpy(values).float().pin_memory().to(self.device, non_blocking=True)
        
        # 计算批次数
        num_samples = len(states)
        batch_size = self.config.BATCH_SIZE
        num_batches = (num_samples + batch_size - 1) // batch_size
        
        total_loss = 0.0
        total_policy_loss = 0.0
        total_value_loss = 0.0
        
        # 批次训练
        for i in tqdm(range(num_batches), desc="训练批次"):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, num_samples)
            
            batch_states = states[start_idx:end_idx]
            batch_policies = policies[start_idx:end_idx]
            batch_values = values[start_idx:end_idx]
            
            # 使用混合精度训练
            if self.config.USE_AMP:
                with autocast():
                    # 前向传播
                    policy_out, value_out = self.model(batch_states)
                    
                    # 计算损失
                    policy_loss = -torch.sum(batch_policies * torch.log(policy_out + 1e-8)) / len(batch_states)
                    value_loss = torch.mean((value_out - batch_values) ** 2)
                    loss = policy_loss + value_loss
                
                # 反向传播
                self.optimizer.zero_grad()
                self.scaler.scale(loss).backward()
                
                # 梯度裁剪
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config.MAX_GRAD_NORM
                )
                
                # 更新参数
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                # 前向传播
                policy_out, value_out = self.model(batch_states)
                
                # 计算损失
                policy_loss = -torch.sum(batch_policies * torch.log(policy_out + 1e-8)) / len(batch_states)
                value_loss = torch.mean((value_out - batch_values) ** 2)
                loss = policy_loss + value_loss
                
                # 反向传播
                self.optimizer.zero_grad()
                loss.backward()
                
                # 梯度裁剪
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config.MAX_GRAD_NORM
                )
                
                # 更新参数
                self.optimizer.step()
            
            # 累计损失
            total_loss += loss.item()
            total_policy_loss += policy_loss.item()
            total_value_loss += value_loss.item()
        
        # 计算平均损失
        avg_loss = total_loss / num_batches
        avg_policy_loss = total_policy_loss / num_batches
        avg_value_loss = total_value_loss / num_batches
        
        return avg_loss, avg_policy_loss, avg_value_loss
    
    def save_model(self, filepath: str) -> None:
        """
        保存模型
        
        Args:
            filepath: 保存路径
        """
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'scaler_state_dict': self.scaler.state_dict() if self.scaler else None,
            'stats': self.stats
        }, filepath)
        
        self.logger.info(f"模型已保存到: {filepath}")
    
    def load_model(self, filepath: str) -> None:
        """
        加载模型
        
        Args:
            filepath: 模型文件路径
        """
        checkpoint = torch.load(filepath, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        if self.scaler and checkpoint['scaler_state_dict']:
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
        self.stats = checkpoint['stats']
        
        self.logger.info(f"模型已从 {filepath} 加载") 