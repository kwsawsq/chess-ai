#!/usr/bin/env python3
"""
测试训练脚本 - 50次迭代
"""

import sys
import os

# 使用相对路径而不是硬编码的绝对路径
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

from config.config import config
from src.training.training_pipeline import TrainingPipeline

def main():
    print("=== 开始50次迭代测试训练 ===")
    print(f"配置信息:")
    print(f"  - 迭代次数: {config.NUM_ITERATIONS}")
    print(f"  - 评估间隔: {config.EVAL_INTERVAL}")
    print(f"  - 评估对局数: {config.EVAL_EPISODES}")
    print(f"  - 评估MCTS搜索: {config.NUM_MCTS_SIMS_EVAL}")
    print(f"  - 训练MCTS搜索: {config.NUM_MCTS_SIMS}")
    print(f"")
    print(f"预期总时间: 约4.3小时")
    print(f"评估时间点: 第10, 20, 30, 40, 50次迭代")
    print(f"")
    
    # 创建训练流水线
    pipeline = TrainingPipeline(config)
    
    try:
        # 开始训练
        pipeline.train(config.NUM_ITERATIONS)
        print("\n🎉 测试训练完成！")
        
        # 显示统计信息
        stats = pipeline.get_statistics()
        print(f"\n=== 最终统计 ===")
        print(f"总游戏数: {stats.get('total_games', 0)}")
        print(f"最佳胜率: {stats.get('best_win_rate', 0):.2%}")
        
    except KeyboardInterrupt:
        print("\n⏹️ 训练被手动停止")
    except Exception as e:
        print(f"\n❌ 训练出错: {e}")
        raise

if __name__ == "__main__":
    main() 