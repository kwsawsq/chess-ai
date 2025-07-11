"""
训练入口文件
"""

import torch
import sys
from pathlib import Path

# 添加项目根目录到Python路径
project_root = str(Path(__file__).parent.parent)
if project_root not in sys.path:
    sys.path.append(project_root)

from src.neural_network.model import AlphaZeroNet
from src.training.trainer import Trainer
from config.config import Config

def main():
    # 加载配置
    config = Config()
    
    # 创建模型
    model = AlphaZeroNet(config)
    
    # 创建训练器
    trainer = Trainer(model, config)
    
    # 开始训练
    try:
        trainer.train(config.NUM_ITERATIONS)
    except KeyboardInterrupt:
        print("\n训练被用户中断")
        # 保存检查点
        trainer.save_model('models/checkpoint_interrupted.pth')
        print("已保存中断时的检查点")

if __name__ == "__main__":
    main() 