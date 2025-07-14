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
        self.logger = logging.getLogger(__name__)
        
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
        
        return list(zip(states, policies, values))
    
    def generate_training_data(self, num_games: int) -> List[Tuple[np.ndarray, np.ndarray, float]]:
        """
        生成自我对弈游戏数据
        
        Args:
            num_games: 要生成的游戏数量
            
        Returns:
            List[Tuple[np.ndarray, np.ndarray, float]]: 训练样本列表 (状态, 策略, 价值)
        """
        all_examples = []
        
        progress_bar = tqdm(total=num_games, desc="自我对弈进度",
                          bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]')
        
        if self.config.NUM_WORKERS == 1:
            for _ in range(num_games):
                try:
                    examples = self._play_game()
                    if examples:
                        all_examples.extend(examples)
                    progress_bar.update(1)
                    progress_bar.set_postfix({'数据量': len(all_examples)})
                    sys.stdout.flush()
                except Exception as e:
                    self.logger.error(f"游戏生成出错: {str(e)}", exc_info=True)
                    continue
        else:
            try:
                with ProcessPoolExecutor(max_workers=self.config.NUM_WORKERS) as executor:
                    futures = [executor.submit(self._play_game) for _ in range(num_games)]
                    
                    for future in futures:
                        try:
                            examples = future.result(timeout=300)
                            if examples:
                                all_examples.extend(examples)
                            progress_bar.update(1)
                            progress_bar.set_postfix({'数据量': len(all_examples)})
                            sys.stdout.flush()
                        except Exception as e:
                            self.logger.error(f"游戏生成子进程出错: {str(e)}", exc_info=True)
                            progress_bar.update(1)
                            continue
            except Exception as e:
                self.logger.error(f"进程池执行出错: {str(e)}", exc_info=True)

        progress_bar.close()

        if all_examples:
            self.logger.info(f"生成完成! 总数据量: {len(all_examples)}")
        else:
            self.logger.warning("警告: 没有生成任何有效数据!")
            
        return all_examples 