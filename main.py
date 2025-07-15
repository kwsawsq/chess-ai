"""
国际象棋AI训练主程序 - 使用优化后的训练流水线
"""

import logging
from config.config import config
from src.training.training_pipeline import TrainingPipeline

def main():
    # 设置日志
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger(__name__)
    
    logger.info("开始训练流程")
    
    # 创建训练流水线
    pipeline = TrainingPipeline(config)
    
    try:
        # 开始训练
        pipeline.train(config.NUM_ITERATIONS)
        logger.info("训练完成")
        
    except KeyboardInterrupt:
        logger.info("训练被用户中断")
    except Exception as e:
        logger.error(f"训练过程出错: {str(e)}", exc_info=True)

if __name__ == "__main__":
    main() 