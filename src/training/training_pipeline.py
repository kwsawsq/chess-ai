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
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from ..game import ChessGame
from ..neural_network import AlphaZeroNet, AlphaZeroLoss
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
    
    def run(self, start_iter: int = 1, load_model_path: str = None, load_data_path: str = None):
        """
        运行完整的训练流程
        
        Args:
            start_iter (int): 开始的迭代次数
            load_model_path (str, optional): 要加载的模型路径. Defaults to None.
            load_data_path (str, optional): 要加载的数据路径. Defaults to None.
        """
        if load_model_path and os.path.exists(load_model_path):
            if self.current_net.load_model(load_model_path):
                self.logger.info(f"成功从 {load_model_path} 加载模型。")
                # 尝试从文件名解析迭代次数
                try:
                    num_str = os.path.basename(load_model_path).split('_')[-1].split('.')[0]
                    self.stats['iteration'] = int(num_str)
                    start_iter = self.stats['iteration'] + 1
                except (ValueError, IndexError):
                    self.stats['iteration'] = start_iter -1
                    
        if load_data_path and os.path.exists(load_data_path):
            try:
                data = np.load(load_data_path, allow_pickle=True)
                self.training_data = data['data'].tolist()
                self.logger.info(f"成功从 {load_data_path} 加载 {len(self.training_data)} 条数据。")
            except Exception as e:
                self.logger.error(f"加载数据文件 {load_data_path} 失败: {e}")

        self.train(start_iter)

    def train(self, start_iteration: int):
        """
        执行训练循环
        
        Args:
            start_iteration (int): 开始的迭代次数
        """
        start_time = time.time()
        
        num_total_iterations = self.config.NUM_ITERATIONS
        
        try:
            for iteration in range(start_iteration, num_total_iterations + 1):
                self.stats['iteration'] = iteration
                self.logger.info(f"开始训练迭代 {iteration}/{num_total_iterations}")
                
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
                
                # 3. 评估新模型（按间隔进行）
                if iteration % self.config.EVAL_INTERVAL == 0:
                    self.logger.info("开始模型评估...")
                    evaluation_stats = self._evaluate_model()
                else:
                    self.logger.info(f"跳过评估（将在第 {(iteration // self.config.EVAL_INTERVAL + 1) * self.config.EVAL_INTERVAL} 次迭代时评估）")
                    evaluation_stats = None
                
                # 4. 保存检查点
                if iteration % self.config.SAVE_INTERVAL == 0:
                    self._save_checkpoint(iteration, evaluation_stats)
                
                # 记录本次迭代信息
                if evaluation_stats is not None:
                    self._log_iteration_stats(iteration, train_stats, evaluation_stats)
                else:
                    self._log_training_only_stats(iteration, train_stats)

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
        训练神经网络
        
        Returns:
            Dict[str, float]: 训练统计信息, 如果数据不足则返回None
        """
        if len(self.training_data) < self.config.BATCH_SIZE:
            self.logger.warning(f"训练数据不足 ({len(self.training_data)})，跳过训练。")
            return None

        optimizer = torch.optim.Adam(self.current_net.parameters(), lr=self.config.LEARNING_RATE, weight_decay=self.config.WEIGHT_DECAY)
        criterion = AlphaZeroLoss()

        self.current_net.train()

        all_policy_loss = []
        all_value_loss = []
        all_total_loss = []

        for epoch in range(self.config.NUM_EPOCHS):
            self.logger.info(f"开始 Epoch {epoch + 1}/{self.config.NUM_EPOCHS}")
            
            np.random.shuffle(self.training_data)
            
            progress_bar = tqdm(range(0, len(self.training_data), self.config.BATCH_SIZE), desc=f"Epoch {epoch + 1}")
            
            for i in progress_bar:
                batch_data = self.training_data[i:i+self.config.BATCH_SIZE]
                states, policy_targets, value_targets = zip(*batch_data)

                states = np.array(states)
                policy_targets = np.array(policy_targets)
                value_targets = np.array(value_targets)
                
                loss_dict = self.current_net.train_step(
                    states,
                    policy_targets,
                    value_targets,
                    optimizer, 
                    criterion
                )
                
                all_policy_loss.append(loss_dict['policy_loss'])
                all_value_loss.append(loss_dict['value_loss'])
                all_total_loss.append(loss_dict['total_loss'])
                
                progress_bar.set_postfix({
                    'policy_loss': np.mean(all_policy_loss),
                    'value_loss': np.mean(all_value_loss)
                })
        
        return {
            'policy_loss': np.mean(all_policy_loss),
            'value_loss': np.mean(all_value_loss),
            'total_loss': np.mean(all_total_loss)
        }
    
    def _evaluate_model(self) -> Dict[str, float]:
        """
        评估当前模型
        
        Returns:
            Dict[str, float]: 评估统计信息
        """
        self.logger.info("开始模型评估...")
        
        # 创建一个“旧”模型用于对比
        previous_net = AlphaZeroNet(self.config)
        # 简单地从当前模型复制权重作为“旧”模型
        # 在实际应用中, 你可能需要加载上一个最好的模型
        previous_net.load_state_dict(self.current_net.state_dict())
        
        # 使用评估器进行评估
        win_rate, draw_rate, loss_rate = self.evaluator.evaluate(
            self.current_net, 
            previous_net,
            num_games=self.config.EVAL_EPISODES
        )
        
        self.logger.info(f"评估结果 - 胜率: {win_rate:.2%}, 平局率: {draw_rate:.2%}, 败率: {loss_rate:.2%}")
        
        eval_stats = {
            'win_rate': win_rate,
            'draw_rate': draw_rate,
            'loss_rate': loss_rate,
            'num_games': self.config.EVAL_EPISODES,
            'avg_game_length': 50.0  # 估计的平均游戏长度
        }
        
        # 更新最佳胜率
        if eval_stats['win_rate'] > self.stats.get('best_win_rate', 0.0):
            self.stats['best_win_rate'] = eval_stats['win_rate']
            self.logger.info(f"新高胜率! 保存为最佳模型。")
            self.current_net.save(os.path.join(self.config.MODEL_DIR, 'best_model.pth'))
        
        return eval_stats
    
    def _save_checkpoint(self, iteration: int, eval_stats: Optional[Dict[str, float]]):
        """
        保存检查点
        
        Args:
            iteration: 当前迭代次数
            eval_stats: 评估统计信息
        """
        checkpoint_path = os.path.join(self.checkpoint_dir, f'checkpoint_iter_{iteration}.pth')
        
        self.current_net.save(checkpoint_path, {
            'iteration': iteration,
            'model_state_dict': self.current_net.state_dict(),
            'optimizer_state_dict': None, # We create optimizer on the fly
            'stats': self.stats,
            'eval_stats': eval_stats,
            'training_data_sample': self.training_data[:1000] # Save a sample
        })
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
    
    def _log_training_only_stats(self,
                               iteration: int, 
                               train_stats: Dict[str, float]):
        """
        记录只有训练统计的迭代信息
        
        Args:
            iteration: 当前迭代次数
            train_stats: 训练统计信息
        """
        self.logger.info(
            f"\n迭代 {iteration} 统计信息:\n"
            f"训练损失:\n"
            f"  - 策略损失: {train_stats['policy_loss']:.4f}\n"
            f"  - 价值损失: {train_stats['value_loss']:.4f}\n"
            f"  - 总损失: {train_stats['total_loss']:.4f}\n"
            f"评估: 跳过（将在第 {(iteration // self.config.EVAL_INTERVAL + 1) * self.config.EVAL_INTERVAL} 次迭代时评估）\n"
            f"累计统计:\n"
            f"  - 总对弈局数: {self.stats['total_games']}\n"
            f"  - 最佳胜率: {self.stats['best_win_rate']:.2%}"
        )
    
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
        加载检查点，恢复训练状态
        
        Args:
            checkpoint_path (str): 检查点文件路径
            
        Returns:
            bool: 是否加载成功
        """
        if not os.path.exists(checkpoint_path):
            self.logger.warning(f"检查点文件不存在: {checkpoint_path}")
            return False
            
        try:
            self.logger.info(f"正在从 {checkpoint_path} 加载检查点...")
            # 使用 map_location='cpu' 避免在加载时占用GPU显存，并可能加速加载
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            
            # 恢复网络权重
            self.current_net.load_state_dict(checkpoint['model_state_dict'])
            self.logger.info("成功恢复模型权重。")
            
            # 恢复训练统计数据
            self.stats = checkpoint['stats']
            self.logger.info(f"成功恢复训练统计: {self.stats}")

            # 可选：恢复训练历史
            if 'training_history' in checkpoint:
                self.training_history = checkpoint['training_history']
            
            # 注意：暂时不加载训练数据，因为这可能是导致卡顿的原因
            # 如果需要，应该从单独的文件或在训练开始时重新生成
            if 'training_data' in checkpoint:
                self.logger.info(f"检查点包含 {len(checkpoint['training_data'])} 条训练数据，本次不加载以加快启动速度。")

            return True
        except Exception as e:
            self.logger.error(f"加载检查点 {checkpoint_path} 失败: {e}", exc_info=True)
            return False
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取当前训练统计信息"""
        return {
            'iteration': self.stats['iteration'],
            'total_games': self.stats['total_games'],
            'best_win_rate': self.stats['best_win_rate'],
            'training_time': self.stats['training_time'],
            'current_data_size': len(self.training_data)
        } 