"""
AlphaZero国际象棋AI主程序
"""

import os
import argparse
import logging
from datetime import datetime

from config.config import AlphaZeroConfig
from .game import ChessGame
from .neural_network import AlphaZeroNet
from .training import TrainingPipeline
from .evaluation import Evaluator


def setup_logging(config):
    """
    设置日志
    """
    # 创建日志目录
    os.makedirs(config.LOG_DIR, exist_ok=True)
    
    # 设置日志格式
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
    
    return logging.getLogger(__name__)


def train(config, logger):
    """
    训练模型
    
    Args:
        config: 配置对象
        logger: 日志记录器
    """
    logger.info("开始训练流程")
    
    # 创建训练流水线
    pipeline = TrainingPipeline(config)
    
    # 如果存在检查点，加载最新的检查点
    if os.path.exists(config.MODEL_DIR):
        checkpoints = [f for f in os.listdir(config.MODEL_DIR) 
                      if f.startswith('checkpoint_iter_')]
        if checkpoints:
            latest_checkpoint = sorted(checkpoints)[-1]
            checkpoint_path = os.path.join(config.MODEL_DIR, latest_checkpoint)
            logger.info(f"加载检查点: {checkpoint_path}")
            pipeline.load_checkpoint(checkpoint_path)
    
    # 开始训练
    try:
        pipeline.train(config.NUM_ITERATIONS)
    except KeyboardInterrupt:
        logger.info("训练被用户中断")
    except Exception as e:
        logger.error(f"训练过程出错: {str(e)}", exc_info=True)
    
    logger.info("训练完成")


def evaluate(config, logger, model_path):
    """
    评估模型
    
    Args:
        config: 配置对象
        logger: 日志记录器
        model_path: 模型文件路径
    """
    logger.info(f"开始评估模型: {model_path}")
    
    # 加载模型
    model = AlphaZeroNet(config)
    if not model.load(model_path):
        logger.error("加载模型失败")
        return
    
    # 创建评估器
    evaluator = Evaluator(config)
    
    # 评估模型
    try:
        eval_result = evaluator.evaluate_model(
            model,
            num_games=config.EVAL_GAMES,
            save_games=True
        )
        
        # 生成评估报告
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        report_path = os.path.join(
            config.LOG_DIR,
            'evaluation',
            f'evaluation_report_{timestamp}.md'
        )
        
        evaluator.generate_evaluation_report(
            os.path.basename(model_path),
            eval_result,
            save_path=report_path
        )
        
        logger.info(f"评估报告已保存到: {report_path}")
        
    except Exception as e:
        logger.error(f"评估过程出错: {str(e)}", exc_info=True)
    
    logger.info("评估完成")


def play(config, logger, model_path):
    """
    人机对弈
    
    Args:
        config: 配置对象
        logger: 日志记录器
        model_path: 模型文件路径
    """
    logger.info("开始人机对弈")
    
    # 加载模型
    model = AlphaZeroNet(config)
    if not model.load(model_path):
        logger.error("加载模型失败")
        return
    
    # 创建游戏
    game = ChessGame()
    
    # 创建MCTS搜索器
    from .mcts import MCTS
    mcts = MCTS(model, config, game)
    
    try:
        while not game.is_game_over():
            # 显示棋盘
            print("\n当前棋盘:")
            print(game.board)
            
            # 获取当前玩家
            current_player = game.get_current_player()
            
            if current_player == 1:  # 人类玩家（白方）
                # 获取合法走法
                legal_moves = game.get_legal_moves()
                if not legal_moves:
                    break
                
                # 显示合法走法
                print("\n合法走法:")
                for i, move in enumerate(legal_moves):
                    print(f"{i+1}: {move.uci()}")
                
                # 获取玩家输入
                while True:
                    try:
                        choice = int(input("\n请选择走法 (输入序号): ")) - 1
                        if 0 <= choice < len(legal_moves):
                            move = legal_moves[choice]
                            break
                        else:
                            print("无效的选择，请重试")
                    except ValueError:
                        print("请输入有效的数字")
                
                # 执行走法
                game.make_move(move)
                
            else:  # AI玩家（黑方）
                print("\nAI思考中...")
                
                # 执行MCTS搜索
                action_probs, _ = mcts.search(game.board)
                action = int(action_probs.argmax())
                move = game.board.action_to_move(action)
                
                if move:
                    print(f"AI选择走法: {move.uci()}")
                    game.make_move(move)
                else:
                    logger.error(f"AI无法执行动作 {action}")
                    break
        
        # 游戏结束
        print("\n游戏结束!")
        print(game.board)
        
        result = game.get_result()
        if result == 1:
            print("白方（人类）胜利!")
        elif result == -1:
            print("黑方（AI）胜利!")
        else:
            print("平局!")
            
    except KeyboardInterrupt:
        logger.info("游戏被用户中断")
    except Exception as e:
        logger.error(f"游戏过程出错: {str(e)}", exc_info=True)


def main():
    """主函数"""
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='AlphaZero国际象棋AI')
    parser.add_argument('mode', choices=['train', 'evaluate', 'play'],
                       help='运行模式：训练/评估/对弈')
    parser.add_argument('--model', type=str,
                       help='模型文件路径（评估和对弈模式需要）')
    parser.add_argument('--config', type=str, default='config/config.py',
                       help='配置文件路径')
    
    args = parser.parse_args()
    
    # 加载配置
    config = AlphaZeroConfig()
    
    # 设置日志
    logger = setup_logging(config)
    
    # 根据模式执行相应功能
    if args.mode == 'train':
        train(config, logger)
    elif args.mode == 'evaluate':
        if not args.model:
            logger.error("评估模式需要指定模型文件路径")
            return
        evaluate(config, logger, args.model)
    else:  # play
        if not args.model:
            logger.error("对弈模式需要指定模型文件路径")
            return
        play(config, logger, args.model)


if __name__ == '__main__':
    main() 