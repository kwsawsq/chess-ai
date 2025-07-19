#!/usr/bin/env python3
"""
改进的训练启动脚本，包含错误处理和内存监控
"""

import os
import sys
import torch
import psutil
import logging
from pathlib import Path

# 添加项目根目录到Python路径
project_root = str(Path(__file__).parent)
if project_root not in sys.path:
    sys.path.append(project_root)

from src.neural_network.model import AlphaZeroNet
from src.training.trainer import Trainer
from config.config import Config

def setup_logging():
    """设置日志"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('logs/training_startup.log'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

def check_system_resources():
    """检查系统资源"""
    logger = logging.getLogger(__name__)

    # 检查内存
    memory = psutil.virtual_memory()
    logger.info(f"系统内存: 总计 {memory.total / 1024**3:.1f}GB, 可用 {memory.available / 1024**3:.1f}GB, 使用率 {memory.percent:.1f}%")

    if memory.available < 4 * 1024**3:  # 少于4GB可用内存
        logger.warning("可用内存不足4GB，可能会遇到内存问题")

    # 检查磁盘空间
    disk = psutil.disk_usage('.')
    logger.info(f"磁盘空间: 总计 {disk.total / 1024**3:.1f}GB, 可用 {disk.free / 1024**3:.1f}GB, 使用率 {(disk.used/disk.total)*100:.1f}%")

    if disk.free < 2 * 1024**3:  # 少于2GB可用磁盘空间
        logger.warning("可用磁盘空间不足2GB，可能会遇到存储问题")

    # 检查GPU
    if torch.cuda.is_available():
        gpu_memory = torch.cuda.get_device_properties(0).total_memory
        logger.info(f"GPU内存: {gpu_memory / 1024**3:.1f}GB")

        # 清理GPU缓存
        torch.cuda.empty_cache()
        logger.info("已清理GPU缓存")

        # 检查GPU内存使用情况
        if hasattr(torch.cuda, 'memory_allocated'):
            allocated = torch.cuda.memory_allocated() / 1024**3
            cached = torch.cuda.memory_reserved() / 1024**3
            logger.info(f"GPU内存使用: 已分配 {allocated:.2f}GB, 已缓存 {cached:.2f}GB")
    else:
        logger.warning("未检测到CUDA GPU，将使用CPU训练")

def find_latest_checkpoint():
    """查找最新的检查点文件"""
    checkpoint_dir = Path("models/checkpoints")
    if not checkpoint_dir.exists():
        return None
    
    checkpoint_files = list(checkpoint_dir.glob("checkpoint_iter_*.pth"))
    if not checkpoint_files:
        return None
    
    # 按迭代次数排序，返回最新的
    checkpoint_files.sort(key=lambda x: int(x.stem.split('_')[-1]))
    return checkpoint_files[-1]

def main():
    """主函数"""
    logger = setup_logging()
    logger.info("开始启动训练...")
    
    try:
        # 检查系统资源
        check_system_resources()
        
        # 加载配置
        config = Config()
        logger.info(f"配置加载完成，设备: {config.DEVICE}")
        
        # 创建模型
        model = AlphaZeroNet(config)
        logger.info("模型创建完成")
        
        # 创建训练器
        trainer = Trainer(model, config)
        logger.info("训练器创建完成")
        
        # 检查是否有检查点可以恢复
        checkpoint_path = find_latest_checkpoint()
        if checkpoint_path:
            logger.info(f"发现检查点: {checkpoint_path}")
            try:
                trainer.load_model(str(checkpoint_path))
                logger.info("检查点加载成功，将继续训练")
            except Exception as e:
                logger.warning(f"检查点加载失败: {e}，将重新开始训练")
        else:
            logger.info("未发现检查点，将重新开始训练")
        
        # 开始训练
        logger.info("开始训练...")
        trainer.train(config.NUM_ITERATIONS)
        
    except KeyboardInterrupt:
        logger.info("训练被用户中断")
        # 保存检查点
        try:
            trainer.save_model('models/checkpoint_interrupted.pth')
            logger.info("已保存中断时的检查点")
        except Exception as e:
            logger.error(f"保存检查点失败: {e}")
    
    except Exception as e:
        logger.error(f"训练过程中发生错误: {e}", exc_info=True)
        # 尝试保存紧急检查点
        try:
            if 'trainer' in locals():
                trainer.save_model('models/checkpoint_emergency.pth')
                logger.info("已保存紧急检查点")
        except Exception as save_error:
            logger.error(f"保存紧急检查点失败: {save_error}")
        
        raise
    
    finally:
        # 清理资源
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            logger.info("已清理GPU缓存")

if __name__ == "__main__":
    main()
