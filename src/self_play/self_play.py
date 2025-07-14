"""
自我对弈模块
实现AI之间的对弈
"""

import os
import time
import logging
import numpy as np
from typing import List, Tuple, Optional, Dict, Any
from concurrent.futures import ProcessPoolExecutor
import multiprocessing
import torch

# 设置多进程启动方法为'spawn'
if torch.cuda.is_available():
    multiprocessing.set_start_method('spawn', force=True)

from ..game import ChessGame, ChessBoard
from ..neural_network import AlphaZeroNet
from ..mcts import MCTS
import random
from tqdm import tqdm
import sys
from ..evaluation.visualizer import ResultVisualizer

class SelfPlay:
    def __init__(self, model, config):
        """
        初始化自我对弈模块
        
        Args:
            model: 神经网络模型
            config: 配置对象
        """
        self.model = model
        self.config = config
        self.visualizer = ResultVisualizer()
        
    def _play_game(self) -> Tuple[List[np.ndarray], List[np.ndarray], List[float]]:
        """
        进行一局自我对弈
        
        Returns:
            Tuple[List[np.ndarray], List[np.ndarray], List[float]]:
                - 状态列表
                - 策略列表
                - 价值列表
        """
        game = ChessGame(self.config)
        states, policies, values = [], [], []
        moves = []  # 记录游戏的所有移动
        
        while not game.is_over():
            state = game.get_state()
            
            # 调试：检查state格式
            if state is None:
                print("Error: state is None")
                break
                
            try:
                policy, value = self.model.predict(state)
            except Exception as e:
                print(f"预测出错: {e}")
                break  # 或 continue 跳过当前步骤
            
            # 记录当前状态
            states.append(state)
            policies.append(policy)
            values.append(value)
            
            # 选择移动
            move = game.select_move(policy)
            moves.append(move)
            
            # 执行移动
            game.make_move(move)
        
        if not moves:  # 如果游戏未完成
            return [], [], []

        # 获取游戏结果
        result = game.get_result()
        result_str = "1-0" if result == 1 else "0-1" if result == -1 else "1/2-1/2"
        
        try:
            # 可视化游戏
            self.visualizer.display_game(moves, result_str)
        except Exception as e:
            print(f"可视化游戏出错: {str(e)}")
            # 错误不影响训练数据返回
        
        # 根据最终结果调整价值
        final_value = result
        values = [final_value if i % 2 == 0 else -final_value for i in range(len(values))]
        
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
        
        # 如果worker数量为1，使用单进程模式避免多进程问题
        if self.config.NUM_WORKERS == 1:
            for i in range(num_games):
                try:
                    states, policies, values = self._play_game()
                    if states:  # 只有当游戏成功完成时才添加数据
                        all_states.extend(states)
                        all_policies.extend(policies)
                        all_values.extend(values)
                    progress_bar.update(1)
                    progress_bar.set_postfix({'数据量': len(all_states)})
                    sys.stdout.flush()
                except Exception as e:
                    print(f"\n游戏生成出错: {str(e)}")
                    continue
        else:
            # 使用进程池进行并行自我对弈
            try:
                with ProcessPoolExecutor(max_workers=self.config.NUM_WORKERS) as executor:
                    futures = []
                    for _ in range(num_games):
                        future = executor.submit(self._play_game)
                        futures.append(future)
                    
                    # 收集结果
                    for future in futures:
                        try:
                            states, policies, values = future.result(timeout=300)  # 5分钟超时
                            if states:  # 只有当游戏成功完成时才添加数据
                                all_states.extend(states)
                                all_policies.extend(policies)
                                all_values.extend(values)
                            progress_bar.update(1)
                            progress_bar.set_postfix({'数据量': len(all_states)})
                            sys.stdout.flush()
                        except Exception as e:
                            print(f"\n游戏生成出错: {str(e)}")
                            progress_bar.update(1)  # 仍然更新进度条
                            continue
            except Exception as e:
                print(f"\n进程池执行出错: {str(e)}")
                # 回退到单进程模式
                for i in range(num_games - len(all_states)):
                    try:
                        states, policies, values = self._play_game()
                        if states:
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
        if all_states:
            print("\n处理生成的数据...")
            all_states = np.array(all_states)
            all_policies = np.array(all_policies)
            all_values = np.array(all_values)
            
            print(f"生成完成! 总数据量: {len(all_states)}")
        else:
            print("\n警告: 没有生成任何有效数据!")
            all_states = np.array([])
            all_policies = np.array([])
            all_values = np.array([])
        
        return all_states, all_policies, all_values 