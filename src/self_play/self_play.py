"""
自我对弈模块
实现AI之间的对弈
"""

import os
import time
import logging
import numpy as np
from typing import List, Tuple, Optional, Dict, Any
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing
import torch
import sys
from tqdm import tqdm

from ..game import ChessGame
from ..neural_network import AlphaZeroNet
from ..mcts import MCTS
from ..evaluation.visualizer import ResultVisualizer
from .worker import init_worker_self_play, play_one_game_worker


class SelfPlay:
    def __init__(self, model: AlphaZeroNet, config: Any):
        """
        初始化自我对弈模块
        """
        self.model = model
        self.config = config
        self.logger = logging.getLogger(__name__)

    def generate_training_data(self, num_games: int) -> List[Tuple[np.ndarray, np.ndarray, float]]:
        """
        通过并行自我对弈生成训练数据。
        """
        all_examples = []
        
        # 将模型移到CPU以安全地获取state_dict
        original_device = next(self.model.parameters()).device
        self.model.to('cpu')
        model_state = self.model.state_dict()
        
        # 先尝试单进程模式进行调试
        if self.config.NUM_WORKERS == 1:
            self.logger.info("使用单进程模式进行自我对弈...")
            progress_bar = tqdm(total=num_games, desc="自我对弈", leave=False)
            
            # 初始化单进程环境
            init_worker_self_play(model_state, self.config)
            
            for i in range(num_games):
                try:
                    examples = play_one_game_worker()
                    if examples:
                        all_examples.extend(examples)
                    progress_bar.update(1)
                    progress_bar.set_postfix({'数据量': len(all_examples)})
                except Exception as e:
                    self.logger.error(f"单进程自我对弈第{i+1}局出错: {e}")
                    progress_bar.update(1)
            
            progress_bar.close()
        else:
            # 多进程模式
            self.logger.info(f"使用{self.config.NUM_WORKERS}进程进行自我对弈...")
            initargs = (model_state, self.config)

            try:
                with ProcessPoolExecutor(max_workers=self.config.NUM_WORKERS,
                                         initializer=init_worker_self_play,
                                         initargs=initargs) as executor:
                    
                    # 提交所有任务
                    futures = [executor.submit(play_one_game_worker) for _ in range(num_games)]
                    
                    progress_bar = tqdm(total=num_games, desc="自我对弈", leave=False)
                    
                    # 使用 as_completed 处理完成的任务
                    completed_count = 0
                    for future in as_completed(futures, timeout=600):  # 10分钟总超时
                        try:
                            examples = future.result(timeout=60)  # 单个任务1分钟超时
                            if examples:
                                all_examples.extend(examples)
                            completed_count += 1
                            progress_bar.update(1)
                            progress_bar.set_postfix({'数据量': len(all_examples)})
                        except Exception as e:
                            self.logger.error(f"自我对弈任务出错: {e}")
                            completed_count += 1
                            progress_bar.update(1)
                    
                    progress_bar.close()
                    
            except Exception as e:
                self.logger.error(f"多进程自我对弈失败: {e}")
                # 回退到单进程模式
                self.logger.info("回退到单进程模式...")
                return self.generate_training_data_single_process(num_games, model_state)

        # 将模型移回原设备
        self.model.to(original_device)

        if all_examples:
            self.logger.info(f"生成完成! 总数据量: {len(all_examples)}")
        else:
            self.logger.warning("警告: 本轮自我对弈没有生成任何有效数据!")
        
        return all_examples
    
    def generate_training_data_single_process(self, num_games: int, model_state: dict) -> List[Tuple[np.ndarray, np.ndarray, float]]:
        """
        单进程模式生成训练数据（回退方案）
        """
        all_examples = []
        
        self.logger.info("使用单进程回退模式...")
        progress_bar = tqdm(total=num_games, desc="自我对弈(单进程)", leave=False)
        
        # 初始化单进程环境
        init_worker_self_play(model_state, self.config)
        
        for i in range(num_games):
            try:
                examples = play_one_game_worker()
                if examples:
                    all_examples.extend(examples)
                progress_bar.update(1)
                progress_bar.set_postfix({'数据量': len(all_examples)})
            except Exception as e:
                self.logger.error(f"单进程自我对弈第{i+1}局出错: {e}")
                progress_bar.update(1)
        
        progress_bar.close()
        return all_examples 