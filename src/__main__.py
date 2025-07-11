"""
训练入口文件
"""

import torch
from neural_network.model import AlphaZeroNet
from training.trainer import Trainer
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