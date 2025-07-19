#!/usr/bin/env python3
"""
修复的测试训练脚本 - 解决进程池问题并支持续训
"""

import sys
import os
import glob
import logging
import psutil
from datetime import datetime
import multiprocessing as mp
import torch

# 将关键的import移到main函数内部，确保在设置多进程启动方式之后再执行
# sys.path.append('/root/chess-ai')
# from config.config import config
# from src.training.training_pipeline import TrainingPipeline

def setup_logging(config):
    """设置全局日志"""
    os.makedirs(config.LOG_DIR, exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = os.path.join(config.LOG_DIR, f'alphazero_{timestamp}.log')
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    
    return logging.getLogger(__name__), log_file

def find_latest_checkpoint(config):
    """查找最新的检查点文件"""
    checkpoint_dir = os.path.join(config.MODEL_DIR, 'checkpoints')
    if not os.path.exists(checkpoint_dir):
        return None
    
    checkpoint_files = glob.glob(os.path.join(checkpoint_dir, 'checkpoint_*.pth'))
    if not checkpoint_files:
        return None
    
    # 按修改时间排序，返回最新的
    latest = max(checkpoint_files, key=os.path.getmtime)
    return latest

def get_optimal_workers_for_gpu():
    """根据GPU内存和CPU核心数，为自我对弈任务建议工作进程数"""
    cpu_count = psutil.cpu_count(logical=True)
    print(f"系统检测到 {cpu_count} 个CPU核心")
    
    # 对于GPU上的自我对弈，每个worker都会加载一个模型，因此主要瓶颈是GPU显存。
    # 一个比较安全的值是4-8，具体取决于模型大小和GPU显存。
    # 我们这里推荐一个保守值，以避免显存不足。
    recommended_workers = 4
    
    print("考虑到每个工作进程都需要在GPU上加载模型，推荐使用较少的工作进程以避免显存不足。")
    return recommended_workers

def main():
    # 关键修复：将触发CUDA初始化的import移到此处
    # 使用相对路径而不是硬编码的绝对路径
    current_dir = os.path.dirname(os.path.abspath(__file__))
    sys.path.append(current_dir)
    from config.config import config
    from src.training.training_pipeline import TrainingPipeline

    logger, log_file = setup_logging(config)
    
    print("=== 修复的测试训练脚本 ===")
    print(f"日志文件: {log_file}")
    
    # 优化工作进程数
    optimal_workers = get_optimal_workers_for_gpu()
    if optimal_workers != config.NUM_WORKERS:
        use_optimal = input(f"是否使用建议的工作进程数（{optimal_workers}）? (y/n): ").lower() == 'y'
        if use_optimal:
            config.NUM_WORKERS = optimal_workers
    
    print(f"\n配置信息:")
    print(f"  - 迭代次数: {config.NUM_ITERATIONS}")
    print(f"  - 自我对弈游戏数: {config.NUM_SELF_PLAY_GAMES}")
    print(f"  - 工作进程数: {config.NUM_WORKERS}")
    print(f"  - 评估间隔: {config.EVAL_INTERVAL}")
    print(f"")
    
    # 查找是否有检查点文件
    latest_checkpoint = find_latest_checkpoint(config)
    if latest_checkpoint:
        print(f"发现检查点文件: {latest_checkpoint}")
        print("⚠️  注意：由于模型架构已优化，旧检查点不兼容。")
        print("建议从头开始训练以获得最佳性能。")
        use_checkpoint = input("是否强制尝试加载检查点? (建议选择n) (y/n): ").lower() == 'y'
    else:
        print("未发现检查点文件，将重新开始训练")
        use_checkpoint = False
    
    # 创建训练流水线
    pipeline = TrainingPipeline(config)
    
    # 如果有检查点，尝试加载
    start_iteration = 1
    if use_checkpoint and latest_checkpoint:
        logger.info(f"尝试加载检查点: {latest_checkpoint}")
        if pipeline.load_checkpoint(latest_checkpoint):
            start_iteration = pipeline.stats['iteration'] + 1
            logger.info(f"成功加载检查点，将从第 {start_iteration} 次迭代开始")
        else:
            logger.warning("检查点加载失败，将重新开始训练")
    
    try:
        # 开始训练
        remaining_iterations = config.NUM_ITERATIONS - start_iteration + 1
        if remaining_iterations > 0:
            logger.info(f"开始训练，剩余迭代次数: {remaining_iterations}")
            pipeline.train(start_iteration)  # 传递起始迭代次数，让train方法处理到配置的总次数
        else:
            logger.info("训练已完成！")
        
        print("\n🎉 测试训练完成！")
        
        # 显示统计信息
        stats = pipeline.get_statistics()
        print(f"\n=== 最终统计 ===")
        print(f"总迭代数: {stats.get('iteration', 0)}")
        print(f"总游戏数: {stats.get('total_games', 0)}")
        print(f"最佳胜率: {stats.get('best_win_rate', 0):.2%}")
        print(f"当前数据量: {stats.get('current_data_size', 0)}")
        
    except KeyboardInterrupt:
        print("\n⏹️ 训练被手动停止")
        logger.info("训练被用户中断")
    except Exception as e:
        print(f"\n❌ 训练出错: {e}")
        logger.error(f"训练出错: {e}", exc_info=True)
        raise
    finally:
        # 恢复原始配置
        # config.NUM_WORKERS = original_workers
        pass

if __name__ == "__main__":
    # 为保证CUDA在多进程中的稳定性，在程序入口处设置启动方式为'spawn'
    # 这必须在任何与CUDA相关的操作或子进程启动之前完成。
    if mp.get_start_method(allow_none=True) != 'spawn':
        try:
            # 必须在任何CUDA调用之前设置
            mp.set_start_method('spawn', force=True)
            print("INFO: Multiprocessing start method set to 'spawn'.")
        except RuntimeError:
            # 如果已经设置，可能会报错，可以安全忽略
            pass
    main() 