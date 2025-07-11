"""
自我对弈模块
"""

import numpy as np
import torch
import random
from typing import List, Tuple, Dict, Optional
import logging
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import time
from tqdm import tqdm
import sys

from ..game.chess_game import ChessGame
from ..mcts.mcts import MCTS

# 移除直接导入
# from ..evaluation.visualizer import TrainingVisualizer


class SelfPlay:
    """自我对弈系统"""
    
    def __init__(self, model, config):
        """
        初始化自我对弈系统
        
        Args:
            model: 神经网络模型
            config: 配置对象
        """
        self.model = model
        self.config = config
        self.mcts = MCTS(model, config)
        self.logger = logging.getLogger(__name__)
        
        # 延迟导入以避免循环依赖
        from ..evaluation.visualizer import TrainingVisualizer
        
        # 可视化器
        self.visualizer = TrainingVisualizer(config)
        
        # 训练统计
        self.game_lengths = []
        self.win_rates = []
        
    def play_game(self) -> Tuple[List[np.ndarray], List[np.ndarray], List[float]]:
        """
        进行一局自我对弈
        
        Returns:
            Tuple[List[np.ndarray], List[np.ndarray], List[float]]:
                - 状态历史
                - 策略历史
                - 价值历史
        """
        game = ChessGame()
        states, policies, values = [], [], []
        game_moves = []
        
        # 通知可视化器开始新游戏
        self.visualizer.new_game()
        
        while not game.is_over():
            # 获取当前状态
            state = game.get_state()
            
            # MCTS搜索
            policy, root = self.mcts.search(game.board)
            
            # 获取移动概率字典
            move_probs = root.get_move_probabilities()
            
            # 可视化当前局面
            _, value = self.model.predict(state)
            self.visualizer.visualize_board(
                game.board,
                policy,
                value,
                move_probs
            )
            
            # 选择动作
            if len(states) < self.config.TEMP_THRESHOLD:
                action = np.random.choice(len(policy), p=policy)
            else:
                action = np.argmax(policy)
            
            # 记录数据
            states.append(state)
            policies.append(policy)
            
            # 执行动作
            move = game.board.action_to_move(action)
            if move:
                game_moves.append(str(move))
                game.board.make_move(move)
            
            # 计算价值（从当前玩家的角度）
            if game.is_over():
                if game.board.is_checkmate():
                    reward = -1
                else:
                    reward = 0
            else:
                _, reward = self.model.predict(game.get_state())
            
            values.append(reward)
        
        # 保存游戏记录
        result = "1-0" if game.board.get_result() == 1 else "0-1" if game.board.get_result() == -1 else "1/2-1/2"
        self.visualizer.save_game_pgn(game_moves, result)
        
        # 更新统计信息
        self.game_lengths.append(len(game_moves))
        win_rate = sum(1 for v in values if v > 0) / len(values)
        self.win_rates.append(win_rate)
        
        # 绘制统计图表
        self.visualizer.plot_game_stats(self.game_lengths, self.win_rates)
        
        return states, policies, values
    
    def generate_games(self, num_games: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        生成自我对弈游戏数据
        
        Args:
            num_games: 要生成的游戏数量
            
        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray]: (状态数据, 策略数据, 价值数据)
        """
        all_states = []
        all_policies = []
        all_values = []
        
        # 创建进度条
        progress_bar = tqdm(total=num_games, desc="自我对弈进度",
                           bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]')
        
        # 使用进程池进行并行自我对弈
        with ProcessPoolExecutor(max_workers=self.config.NUM_WORKERS) as executor:
            futures = []
            for _ in range(num_games):
                future = executor.submit(self._play_game)
                futures.append(future)
            
            # 收集结果
            for future in futures:
                try:
                    states, policies, values = future.result()
                    all_states.extend(states)
                    all_policies.extend(policies)
                    all_values.extend(values)
                    progress_bar.update(1)
                    progress_bar.set_postfix({'数据量': len(all_states)})
                    sys.stdout.flush()
                except Exception as e:
                    print(f"\n游戏生成出错: {str(e)}")
                    continue
        
        progress_bar.close()
        
        # 转换为numpy数组
        print("\n处理生成的数据...")
        all_states = np.array(all_states)
        all_policies = np.array(all_policies)
        all_values = np.array(all_values)
        
        print(f"生成完成! 总数据量: {len(all_states)}")
        
        return all_states, all_policies, all_values 