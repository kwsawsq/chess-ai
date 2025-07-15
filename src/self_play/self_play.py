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
        self.model.to('cpu')
        model_state = self.model.state_dict()
        
        initargs = (model_state, self.config)

        with ProcessPoolExecutor(max_workers=self.config.NUM_WORKERS,
                                 initializer=init_worker_self_play,
                                 initargs=initargs) as executor:
            
            futures = [executor.submit(play_one_game_worker) for _ in range(num_games)]
            
            progress_bar = tqdm(total=num_games, desc="自我对弈", leave=False)

            for future in futures:
                try:
                    examples = future.result(timeout=300) # 5分钟超时
                    if examples:
                        all_examples.extend(examples)
                    progress_bar.update(1)
                    progress_bar.set_postfix({'数据量': len(all_examples)})
                except Exception as e:
                    self.logger.error(f"自我对弈子进程出错: {e}", exc_info=True)
                    progress_bar.update(1)
        
        progress_bar.close()

        # 将模型移回原设备
        self.model.to(self.config.DEVICE)

        if all_examples:
            self.logger.info(f"生成完成! 总数据量: {len(all_examples)}")
        else:
            self.logger.warning("警告: 本轮自我对弈没有生成任何有效数据!")
        
        return all_examples 