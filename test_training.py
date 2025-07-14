#!/usr/bin/env python3
"""
测试训练修复效果
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.neural_network.model import AlphaZeroNet
from src.training.trainer import Trainer
from config.config import config
import torch
import numpy as np

def test_training():
    """测试训练功能"""
    print("开始测试训练修复效果...")
    
    # 创建模型
    model = AlphaZeroNet(config)
    print(f"模型创建成功，设备: {model.device}")
    
    # 创建训练器
    trainer = Trainer(model, config)
    print("训练器创建成功")
    
    # 测试少量迭代
    print(f"开始测试训练 ({config.NUM_ITERATIONS} 次迭代)...")
    try:
        trainer.train(config.NUM_ITERATIONS)
        print("训练测试完成!")
        
        # 保存模型
        model_path = 'models/test_model.pth'
        trainer.save_model(model_path)
        print(f"模型已保存到: {model_path}")
        
    except Exception as e:
        print(f"训练测试失败: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_training() 