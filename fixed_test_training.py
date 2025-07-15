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

sys.path.append('/root/chess-ai')

from config.config import config
from src.training.training_pipeline import TrainingPipeline

def setup_logging():
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

def find_latest_checkpoint():
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

def get_optimal_workers():
    """根据系统CPU核心数确定最优工作进程数"""
    try:
        # 尝试获取物理CPU核心数
        cpu_count = psutil.cpu_count(logical=False)  # 只计算物理核心
        if cpu_count is None or cpu_count > 32:  # 如果检测不准确或数值异常
            # 在AutoDL等云环境下使用固定的安全值
            if os.path.exists('/etc/autodl'):  # 检测是否在AutoDL环境
                return 12  # AutoDL环境下使用12个进程
            return 8  # 其他环境下的默认值
    except Exception as e:
        print(f"警告: CPU核心检测失败 ({e})，使用默认值")
        return 8
    
    # 为系统和其他进程预留核心
    if cpu_count >= 16:
        return 12  # 16核及以上使用12个进程
    elif cpu_count >= 8:
        return cpu_count - 2  # 8-15核预留2个核心
    else:
        return max(1, cpu_count - 1)  # 8核以下预留1个核心

def main():
    logger, log_file = setup_logging()
    
    print("=== 修复的测试训练脚本 ===")
    print(f"日志文件: {log_file}")
    
    # 优化工作进程数
    optimal_workers = get_optimal_workers()
    if optimal_workers != config.NUM_WORKERS:
        print(f"系统检测到 {psutil.cpu_count()} 个CPU核心")
        print(f"建议的工作进程数: {optimal_workers}")
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
    latest_checkpoint = find_latest_checkpoint()
    if latest_checkpoint:
        print(f"发现检查点文件: {latest_checkpoint}")
        use_checkpoint = input("是否从检查点继续训练? (y/n): ").lower() == 'y'
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

if __name__ == "__main__":
    main() 