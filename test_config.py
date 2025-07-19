#!/usr/bin/env python3
"""
配置测试脚本 - 验证环境和配置是否正确
"""

import os
import sys
import torch
import psutil
from pathlib import Path

# 添加项目根目录到Python路径
project_root = str(Path(__file__).parent)
if project_root not in sys.path:
    sys.path.append(project_root)

def test_imports():
    """测试所有必要的模块是否可以正常导入"""
    print("=== 测试模块导入 ===")
    
    try:
        from config.config import Config
        print("✓ 配置模块导入成功")
    except Exception as e:
        print(f"✗ 配置模块导入失败: {e}")
        return False
    
    try:
        from src.neural_network.model import AlphaZeroNet
        print("✓ 神经网络模型导入成功")
    except Exception as e:
        print(f"✗ 神经网络模型导入失败: {e}")
        return False
    
    try:
        from src.game import ChessGame
        print("✓ 游戏模块导入成功")
    except Exception as e:
        print(f"✗ 游戏模块导入失败: {e}")
        return False
    
    try:
        from src.mcts import MCTS
        print("✓ MCTS模块导入成功")
    except Exception as e:
        print(f"✗ MCTS模块导入失败: {e}")
        return False
    
    try:
        from src.self_play import SelfPlay
        print("✓ 自我对弈模块导入成功")
    except Exception as e:
        print(f"✗ 自我对弈模块导入失败: {e}")
        return False
    
    return True

def test_system_resources():
    """测试系统资源"""
    print("\n=== 测试系统资源 ===")
    
    # 内存检查
    memory = psutil.virtual_memory()
    print(f"系统内存: 总计 {memory.total / 1024**3:.1f}GB, 可用 {memory.available / 1024**3:.1f}GB")
    
    if memory.available < 2 * 1024**3:
        print("⚠️  可用内存不足2GB，可能会遇到问题")
    else:
        print("✓ 内存充足")
    
    # GPU检查
    if torch.cuda.is_available():
        gpu_count = torch.cuda.device_count()
        print(f"✓ 检测到 {gpu_count} 个GPU")
        
        for i in range(gpu_count):
            props = torch.cuda.get_device_properties(i)
            print(f"  GPU {i}: {props.name}, 内存: {props.total_memory / 1024**3:.1f}GB")
    else:
        print("⚠️  未检测到CUDA GPU，将使用CPU")
    
    # 磁盘空间检查
    disk = psutil.disk_usage('.')
    print(f"磁盘空间: 总计 {disk.total / 1024**3:.1f}GB, 可用 {disk.free / 1024**3:.1f}GB")
    
    if disk.free < 1 * 1024**3:
        print("⚠️  可用磁盘空间不足1GB")
    else:
        print("✓ 磁盘空间充足")

def test_model_creation():
    """测试模型创建"""
    print("\n=== 测试模型创建 ===")
    
    try:
        from config.config import Config
        from src.neural_network.model import AlphaZeroNet
        
        config = Config()
        print(f"✓ 配置加载成功，设备: {config.DEVICE}")
        
        model = AlphaZeroNet(config)
        print("✓ 模型创建成功")
        
        # 计算模型参数数量
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"  模型参数: 总计 {total_params:,}, 可训练 {trainable_params:,}")
        
        # 测试模型前向传播
        batch_size = 4
        dummy_input = torch.randn(batch_size, config.IN_CHANNELS, config.BOARD_SIZE, config.BOARD_SIZE)
        
        if config.USE_GPU and torch.cuda.is_available():
            model = model.cuda()
            dummy_input = dummy_input.cuda()
        
        with torch.no_grad():
            policy, value = model(dummy_input)
            print(f"✓ 模型前向传播成功，输出形状: policy {policy.shape}, value {value.shape}")
        
        return True
        
    except Exception as e:
        print(f"✗ 模型创建失败: {e}")
        return False

def test_game_creation():
    """测试游戏创建"""
    print("\n=== 测试游戏创建 ===")
    
    try:
        from config.config import Config
        from src.game import ChessGame
        
        config = Config()
        game = ChessGame(config)
        print("✓ 游戏创建成功")
        
        # 测试游戏状态
        state = game.get_state()
        print(f"✓ 游戏状态获取成功，形状: {state.shape}")
        
        # 测试合法走法
        legal_moves = game.get_legal_moves()
        print(f"✓ 合法走法获取成功，数量: {len(legal_moves)}")
        
        return True
        
    except Exception as e:
        print(f"✗ 游戏创建失败: {e}")
        return False

def main():
    """主函数"""
    print("=== 国际象棋AI配置测试 ===\n")
    
    success = True
    
    # 测试模块导入
    if not test_imports():
        success = False
    
    # 测试系统资源
    test_system_resources()
    
    # 测试模型创建
    if not test_model_creation():
        success = False
    
    # 测试游戏创建
    if not test_game_creation():
        success = False
    
    print("\n=== 测试结果 ===")
    if success:
        print("✓ 所有测试通过，可以开始训练！")
        return 0
    else:
        print("✗ 部分测试失败，请检查配置")
        return 1

if __name__ == "__main__":
    exit(main())
