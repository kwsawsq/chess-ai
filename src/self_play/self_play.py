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
        
        # 确保模型在CPU上，以便安全地传递state_dict
        self.model.cpu()
        model_state = self.model.state_dict()
        
        # 多进程模式
        self.logger.info(f"使用 {self.config.NUM_WORKERS} 个进程进行并行自我对弈...")
        initargs = (model_state, self.config)

        # 使用 mp.get_context("spawn") 来创建进程池，这是与CUDA交互最安全的方式
        ctx = multiprocessing.get_context("spawn")

        try:
            with ProcessPoolExecutor(max_workers=self.config.NUM_WORKERS, mp_context=ctx,
                                     initializer=init_worker_self_play,
                                     initargs=initargs) as executor:
                
                futures = [executor.submit(play_one_game_worker) for _ in range(num_games)]
                
                progress_bar = tqdm(total=num_games, desc="自我对弈", leave=False)

                for future in as_completed(futures):
                    try:
                        examples = future.result(timeout=600) # 每个任务10分钟超时
                        if examples:
                            all_examples.extend(examples)
                        progress_bar.update(1)
                        progress_bar.set_postfix({'数据量': len(all_examples)})
                    except Exception as e:
                        # 捕获通用异常并检查特定错误消息
                        if "A process in the process pool was terminated abruptly" in str(e):
                             self.logger.error("!!! 工作进程池已损坏！这很可能是因为子进程因内存不足（OOM）而被系统终止。")
                             self.logger.error("请尝试在配置中减少工作进程数（NUM_WORKERS），然后重试。")
                        else:
                            self.logger.error(f"一个自我对弈任务失败: {e}", exc_info=True)
                        progress_bar.update(1) # 即使失败也要更新进度条
            
            progress_bar.close()

        except Exception as e:
             self.logger.error(f"创建进程池时发生严重错误: {e}", exc_info=True)
        
        # 将模型移回原设备
        self.model.to(self.config.DEVICE)

        if all_examples:
            self.logger.info(f"生成完成! 总数据量: {len(all_examples)}")
        else:
            self.logger.warning("警告: 本轮自我对弈没有生成任何有效数据!")
        
        return all_examples 