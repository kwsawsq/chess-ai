"""
训练流水线模块
实现完整的训练循环系统
"""

import os
import time
import logging
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
from datetime import datetime

from ..game import ChessGame
from ..neural_network import AlphaZeroNet
from ..self_play import SelfPlay
from ..evaluation import Evaluator


class TrainingPipeline:
    """
    AlphaZero训练流水线
    管理整个训练过程，包括：
    1. 自我对弈生成数据
    2. 神经网络训练
    3. 模型评估
    4. 检查点保存
    """
    
    def __init__(self, config):
        """
        初始化训练流水线
        
        Args:
            config: 配置对象
        """
        self.config = config
        
        # 创建游戏引擎
        self.game = ChessGame(config)
        
        # 创建神经网络
        self.current_net = AlphaZeroNet(config)
        
        # 创建自我对弈模块
        self.self_play = SelfPlay(self.current_net, config)
        
        # 创建评估器
        self.evaluator = Evaluator(config)
        
        # 训练数据
        self.training_data: List[Tuple[np.ndarray, np.ndarray, float]] = []
        self.training_history: List[Dict[str, float]] = []
        
        # 检查点目录
        self.checkpoint_dir = os.path.join(config.MODEL_DIR, 'checkpoints')
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        
        # 日志设置
        self.logger = logging.getLogger(__name__)
        self._setup_logging()
        
        # 训练统计
        self.stats = {
            'iteration': 0,
            'total_games': 0,
            'best_win_rate': 0.0,
            'training_time': 0.0
        }
    
    def _setup_logging(self):
        """设置日志"""
        log_dir = os.path.join(self.config.LOG_DIR, 'training')
        os.makedirs(log_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_file = os.path.join(log_dir, f'training_{timestamp}.log')
        
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)
    
    def train(self, num_iterations: int):
        """
        执行训练循环
        
        Args:
            num_iterations: 训练迭代次数
        """
        start_time = time.time()
        
        try:
            for iteration in range(self.stats['iteration'], num_iterations):
                self.stats['iteration'] = iteration + 1
                self.logger.info(f"开始训练迭代 {iteration + 1}/{num_iterations}")

                # 1. 自我对弈生成数据
                self.logger.info("开始自我对弈...")
                new_examples = self.self_play.generate_training_data(
                    num_games=self.config.NUM_SELF_PLAY_GAMES
                )

                if new_examples:
                    self.training_data.extend(new_examples)
                    
                    if len(self.training_data) > self.config.REPLAY_BUFFER_SIZE:
                        self.training_data = self.training_data[-self.config.REPLAY_BUFFER_SIZE:]
                    
                    self.stats['total_games'] += self.config.NUM_SELF_PLAY_GAMES
                    self.logger.info(f"生成了 {len(new_examples)} 个新样本, 当前总样本数: {len(self.training_data)}")
                else:
                    self.logger.warning("本次自我对弈没有生成任何有效数据，跳过本轮训练。")
                    continue

                if len(self.training_data) < self.config.MIN_REPLAY_SIZE:
                    self.logger.info(f"当前样本数 {len(self.training_data)} 不足 {self.config.MIN_REPLAY_SIZE}, 跳过本轮训练。")
                    continue

                # 2. 训练神经网络
                self.logger.info("开始网络训练...")
                train_stats = self._train_network()
                if train_stats:
                    self.training_history.append(train_stats)
                else:
                    self.logger.warning("网络训练未返回统计数据。")

                # 3. 评估新模型
                self.logger.info("开始模型评估...")
                evaluation_stats = self._evaluate_model()

                # 4. 保存检查点
                if (iteration + 1) % self.config.SAVE_INTERVAL == 0:
                    self._save_checkpoint(iteration + 1, evaluation_stats)

                # 记录本次迭代信息
                self._log_iteration_stats(iteration + 1, train_stats, evaluation_stats)

        except KeyboardInterrupt:
            self.logger.info("训练被用户中断")
        except Exception as e:
            self.logger.error(f"训练过程出错: {str(e)}", exc_info=True)
        finally:
            # 保存最终模型和训练数据
            self._save_final_model()
            self.stats['training_time'] = time.time() - start_time
            self.logger.info(f"训练结束，总用时: {self.stats['training_time']:.2f}秒")
    
    def _train_network(self) -> Dict[str, float]:
        """
        训练神经网络一个周期
        
        Returns:
            Dict[str, float]: 训练统计信息
        """
        # 准备训练数据
        states, policy_targets, value_targets = zip(*self.training_data)
        states = np.array(states)
        policy_targets = np.array(policy_targets)
        value_targets = np.array(value_targets)
        
        # 训练模型
        history = self.current_net.train(
            states,
            policy_targets,
            value_targets,
            batch_size=self.config.BATCH_SIZE,
            epochs=self.config.EPOCHS_PER_ITERATION
        )
        
        return {
            'policy_loss': float(np.mean(history['policy_loss'])),
            'value_loss': float(np.mean(history['value_loss'])),
            'total_loss': float(np.mean(history['total_loss']))
        }
    
    def _evaluate_model(self) -> Dict[str, float]:
        """
        评估当前模型
        
        Returns:
            Dict[str, float]: 评估统计信息
        """
        # 创建评估用的网络副本
        previous_net = AlphaZeroNet(self.config)
        previous_net.load_state_dict(self.current_net.state_dict())
        
        # 评估
        eval_stats = self.self_play.evaluate_against_previous(
            previous_net,
            num_games=self.config.EVAL_GAMES,
            verbose=True
        )
        
        # 更新最佳胜率
        if eval_stats['win_rate'] > self.stats['best_win_rate']:
            self.stats['best_win_rate'] = eval_stats['win_rate']
        
        return eval_stats
    
    def _save_checkpoint(self, iteration: int, eval_stats: Dict[str, float]):
        """
        保存检查点
        
        Args:
            iteration: 当前迭代次数
            eval_stats: 评估统计信息
        """
        checkpoint = {
            'iteration': iteration,
            'model_state_dict': self.current_net.state_dict(),
            'stats': self.stats,
            'eval_stats': eval_stats,
            'config': self.config
        }
        
        checkpoint_path = os.path.join(
            self.checkpoint_dir,
            f'checkpoint_iter_{iteration}.pth'
        )
        
        self.current_net.save(checkpoint_path, checkpoint)
        self.logger.info(f"保存检查点到: {checkpoint_path}")
    
    def _save_final_model(self):
        """保存最终模型"""
        final_model_path = os.path.join(
            self.config.MODEL_DIR,
            'final_model.pth'
        )
        
        self.current_net.save(final_model_path)
        self.logger.info(f"保存最终模型到: {final_model_path}")
        
        # 保存训练数据
        data_path = os.path.join(
            self.config.DATA_DIR,
            'final_training_data.npz'
        )
        
        if self.training_data:
            states, policies, values = zip(*self.training_data)
            np.savez(data_path, states=np.array(states), policies=np.array(policies), values=np.array(values))
            self.logger.info(f"保存训练数据到: {data_path}")
        else:
            self.logger.warning("没有训练数据可供保存。")
    
    def _log_iteration_stats(self,
                           iteration: int,
                           train_stats: Dict[str, float],
                           eval_stats: Dict[str, float]):
        """
        记录迭代统计信息
        
        Args:
            iteration: 当前迭代次数
            train_stats: 训练统计信息
            eval_stats: 评估统计信息
        """
        self.logger.info(
            f"\n迭代 {iteration} 统计信息:\n"
            f"训练损失:\n"
            f"  - 策略损失: {train_stats['policy_loss']:.4f}\n"
            f"  - 价值损失: {train_stats['value_loss']:.4f}\n"
            f"  - 总损失: {train_stats['total_loss']:.4f}\n"
            f"评估结果:\n"
            f"  - 胜率: {eval_stats['win_rate']:.2%}\n"
            f"  - 对弈局数: {eval_stats['num_games']}\n"
            f"  - 平均游戏长度: {eval_stats['avg_game_length']:.1f}\n"
            f"累计统计:\n"
            f"  - 总对弈局数: {self.stats['total_games']}\n"
            f"  - 最佳胜率: {self.stats['best_win_rate']:.2%}"
        )
    
    def load_checkpoint(self, checkpoint_path: str) -> bool:
        """
        加载检查点
        
        Args:
            checkpoint_path: 检查点文件路径
            
        Returns:
            bool: 是否成功加载
        """
        try:
            checkpoint = self.current_net.load(checkpoint_path)
            if checkpoint:
                self.stats = checkpoint.get('stats', self.stats)
                self.logger.info(f"成功加载检查点: {checkpoint_path}")
                return True
        except Exception as e:
            self.logger.error(f"加载检查点失败: {str(e)}")
        return False
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        获取训练统计信息
        
        Returns:
            Dict[str, Any]: 统计信息字典
        """
        return {
            'stats': self.stats,
            'training_history': self.training_history,
            'current_data_size': len(self.training_data)
        } 