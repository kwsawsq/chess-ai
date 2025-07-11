"""
AlphaZero神经网络模型
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Dict, Any, Optional
import logging
import os


class ResidualBlock(nn.Module):
    """
    残差块
    """
    
    def __init__(self, channels: int, dropout: float = 0.0):
        """
        初始化残差块
        
        Args:
            channels: 通道数
            dropout: dropout率
        """
        super(ResidualBlock, self).__init__()
        
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
        self.dropout = nn.Dropout2d(dropout) if dropout > 0 else None
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播"""
        residual = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)
        
        if self.dropout:
            out = self.dropout(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        out += residual
        out = F.relu(out)
        
        return out


class AlphaZeroNet(nn.Module):
    """
    AlphaZero神经网络
    
    架构：
    - 输入层：20个通道的8x8棋盘
    - 残差块：多个残差块
    - 策略头：输出动作概率分布
    - 价值头：输出局面评估值
    """
    
    def __init__(self, config):
        """
        初始化神经网络
        
        Args:
            config: 配置对象
        """
        super(AlphaZeroNet, self).__init__()
        
        # 从配置中获取参数
        self.input_channels = config.IN_CHANNELS
        self.channels = config.NUM_CHANNELS
        self.num_res_layers = config.NUM_RESIDUAL_BLOCKS
        self.action_size = config.ACTION_SIZE
        self.dropout = config.DROPOUT_RATE
        
        # 输入层
        self.input_conv = nn.Conv2d(self.input_channels, self.channels, kernel_size=3, padding=1, bias=False)
        self.input_bn = nn.BatchNorm2d(self.channels)
        
        # 残差块
        self.res_layers = nn.ModuleList([
            ResidualBlock(self.channels, self.dropout) for _ in range(self.num_res_layers)
        ])
        
        # 策略头
        self.policy_conv = nn.Conv2d(self.channels, 32, kernel_size=1, bias=False)
        self.policy_bn = nn.BatchNorm2d(32)
        self.policy_fc = nn.Linear(32 * 8 * 8, self.action_size)
        
        # 价值头
        self.value_conv = nn.Conv2d(self.channels, 3, kernel_size=1, bias=False)
        self.value_bn = nn.BatchNorm2d(3)
        self.value_fc1 = nn.Linear(3 * 8 * 8, 256)
        self.value_fc2 = nn.Linear(256, 1)
        
        # 初始化权重
        self._initialize_weights()
        
        # 日志
        self.logger = logging.getLogger(__name__)
        
        # 设备配置
        self.device = torch.device('cuda' if torch.cuda.is_available() and config.USE_GPU else 'cpu')
        self.to(self.device)
    
    def _initialize_weights(self):
        """初始化权重"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        前向传播
        
        Args:
            x: 输入张量 (batch_size, 20, 8, 8)
            
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: (策略logits, 价值)
        """
        # 输入层
        x = self.input_conv(x)
        x = self.input_bn(x)
        x = F.relu(x)
        
        # 残差块
        for res_layer in self.res_layers:
            x = res_layer(x)
        
        # 策略头
        policy = self.policy_conv(x)
        policy = self.policy_bn(policy)
        policy = F.relu(policy)
        policy = policy.view(policy.size(0), -1)  # 展平
        policy = self.policy_fc(policy)
        
        # 价值头
        value = self.value_conv(x)
        value = self.value_bn(value)
        value = F.relu(value)
        value = value.view(value.size(0), -1)  # 展平
        value = F.relu(self.value_fc1(value))
        value = torch.tanh(self.value_fc2(value))
        
        return policy, value
    
    def predict(self, board: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        预测单个棋盘状态
        
        Args:
            board: 棋盘状态 (20, 8, 8)
            
        Returns:
            Tuple[np.ndarray, float]: (策略概率, 价值)
        """
        self.eval()
        with torch.no_grad():
            # 转换为张量
            board_tensor = torch.FloatTensor(board).unsqueeze(0).to(self.device)
            
            # 前向传播
            policy_logits, value = self.forward(board_tensor)
            
            # 转换为概率
            policy_probs = F.softmax(policy_logits, dim=1)
            
            # 转换回numpy
            policy_probs = policy_probs.cpu().numpy().squeeze()
            value = value.cpu().numpy().squeeze()
            
            return policy_probs, value
    
    def predict_batch(self, boards: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        批量预测
        
        Args:
            boards: 棋盘状态批量 (batch_size, 20, 8, 8)
            
        Returns:
            Tuple[np.ndarray, np.ndarray]: (策略概率批量, 价值批量)
        """
        self.eval()
        with torch.no_grad():
            # 转换为张量
            board_tensor = torch.FloatTensor(boards).to(self.device)
            
            # 前向传播
            policy_logits, value = self.forward(board_tensor)
            
            # 转换为概率
            policy_probs = F.softmax(policy_logits, dim=1)
            
            # 转换回numpy
            policy_probs = policy_probs.cpu().numpy()
            value = value.cpu().numpy()
            
            return policy_probs, value
    
    def train_step(self,
                  states: np.ndarray,
                  policy_targets: np.ndarray,
                  value_targets: np.ndarray,
                  optimizer: torch.optim.Optimizer,
                  criterion: nn.Module) -> Dict[str, float]:
        """
        执行一步训练
        
        Args:
            states: 状态批量 (batch_size, 20, 8, 8)
            policy_targets: 策略目标 (batch_size, action_size)
            value_targets: 价值目标 (batch_size,)
            optimizer: 优化器
            criterion: 损失函数
            
        Returns:
            Dict[str, float]: 训练统计信息
        """
        self.train()
        
        # 转换为张量
        states = torch.FloatTensor(states).to(self.device)
        policy_targets = torch.FloatTensor(policy_targets).to(self.device)
        value_targets = torch.FloatTensor(value_targets).to(self.device)
        
        # 前向传播
        policy_logits, value_pred = self.forward(states)
        
        # 计算损失
        loss, loss_dict = criterion(policy_logits, value_pred, policy_targets, value_targets)
        
        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        return loss_dict
    
    def save(self, filepath: str, additional_data: Optional[Dict] = None) -> bool:
        """
        保存模型
        
        Args:
            filepath: 保存路径
            additional_data: 额外数据
            
        Returns:
            bool: 是否成功保存
        """
        try:
            save_data = {
                'model_state_dict': self.state_dict(),
                'model_config': {
                    'input_channels': self.input_channels,
                    'channels': self.channels,
                    'num_res_layers': self.num_res_layers,
                    'action_size': self.action_size,
                    'dropout': self.dropout
                }
            }
            
            if additional_data:
                save_data.update(additional_data)
            
            torch.save(save_data, filepath)
            self.logger.info(f"模型已保存到: {filepath}")
            return True
            
        except Exception as e:
            self.logger.error(f"保存模型失败: {str(e)}")
            return False
    
    def load(self, filepath: str) -> Optional[Dict]:
        """
        加载模型
        
        Args:
            filepath: 模型文件路径
            
        Returns:
            Optional[Dict]: 额外数据（如果存在）
        """
        try:
            checkpoint = torch.load(filepath, map_location=self.device)
            self.load_state_dict(checkpoint['model_state_dict'])
            self.logger.info(f"模型已加载: {filepath}")
            
            # 返回除模型状态之外的数据
            del checkpoint['model_state_dict']
            return checkpoint if checkpoint else None
            
        except Exception as e:
            self.logger.error(f"加载模型失败: {str(e)}")
            return None
    
    def get_info(self) -> Dict[str, Any]:
        """
        获取模型信息
        
        Returns:
            Dict[str, Any]: 模型信息
        """
        return {
            'input_channels': self.input_channels,
            'channels': self.channels,
            'num_res_layers': self.num_res_layers,
            'action_size': self.action_size,
            'dropout': self.dropout,
            'num_parameters': sum(p.numel() for p in self.parameters()),
            'device': str(self.device)
        }
    
    def __str__(self) -> str:
        """字符串表示"""
        return f"AlphaZeroNet(in_channels={self.input_channels}, " \
               f"channels={self.channels}, " \
               f"res_layers={self.num_res_layers}, " \
               f"action_size={self.action_size})"


class AlphaZeroLoss(nn.Module):
    """
    AlphaZero损失函数
    
    损失 = 价值损失 + 策略损失 + L2正则化
    """
    
    def __init__(self, value_weight: float = 1.0, policy_weight: float = 1.0):
        """
        初始化损失函数
        
        Args:
            value_weight: 价值损失权重
            policy_weight: 策略损失权重
        """
        super(AlphaZeroLoss, self).__init__()
        self.value_weight = value_weight
        self.policy_weight = policy_weight
        
        # 损失函数
        self.value_loss_fn = nn.MSELoss()
        self.policy_loss_fn = nn.CrossEntropyLoss()
    
    def forward(self, 
                policy_logits: torch.Tensor, 
                value_pred: torch.Tensor,
                target_policy: torch.Tensor, 
                target_value: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        计算损失
        
        Args:
            policy_logits: 策略logits (batch_size, action_size)
            value_pred: 预测价值 (batch_size, 1)
            target_policy: 目标策略 (batch_size, action_size)
            target_value: 目标价值 (batch_size, 1)
            
        Returns:
            Tuple[torch.Tensor, Dict[str, float]]: (总损失, 损失详情)
        """
        # 价值损失
        value_loss = self.value_loss_fn(value_pred.squeeze(), target_value.squeeze())
        
        # 策略损失（使用KL散度）
        log_policy = F.log_softmax(policy_logits, dim=1)
        policy_loss = -torch.sum(target_policy * log_policy, dim=1).mean()
        
        # 总损失
        total_loss = self.value_weight * value_loss + self.policy_weight * policy_loss
        
        # 损失详情
        loss_dict = {
            'total_loss': total_loss.item(),
            'value_loss': value_loss.item(),
            'policy_loss': policy_loss.item()
        }
        
        return total_loss, loss_dict


def create_model(config) -> AlphaZeroNet:
    """
    根据配置创建模型
    
    Args:
        config: 配置对象
        
    Returns:
        AlphaZeroNet: 神经网络模型
    """
    model = AlphaZeroNet(config)
    
    # 如果有GPU，移到GPU上
    if torch.cuda.is_available():
        model = model.cuda()
        logging.info("模型已移动到GPU")
    else:
        logging.info("使用CPU运行模型")
    
    return model


def load_model_from_checkpoint(filepath: str, config) -> Optional[AlphaZeroNet]:
    """
    从检查点加载模型
    
    Args:
        filepath: 检查点文件路径
        config: 配置对象
        
    Returns:
        Optional[AlphaZeroNet]: 加载的模型，失败时返回None
    """
    try:
        model = create_model(config)
        
        if model.load(filepath):
            return model
        else:
            return None
            
    except Exception as e:
        logging.error(f"从检查点加载模型失败: {e}")
        return None


def model_summary(model: AlphaZeroNet) -> str:
    """
    获取模型摘要
    
    Args:
        model: 神经网络模型
        
    Returns:
        str: 模型摘要信息
    """
    info = model.get_info()
    
    summary = "AlphaZero神经网络模型摘要\n"
    summary += "=" * 50 + "\n"
    summary += f"输入通道数: {info['input_channels']}\n"
    summary += f"隐藏层通道数: {info['channels']}\n"
    summary += f"残差块数量: {info['num_res_layers']}\n"
    summary += f"动作空间大小: {info['action_size']}\n"
    summary += f"Dropout率: {info['dropout']}\n"
    summary += f"总参数数量: {info['num_parameters']:,}\n"
    summary += f"设备: {info['device']}\n"
    
    return summary 