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


# --- Worker Process Initialization ---
g_model: Optional[AlphaZeroNet] = None
g_config: Optional[Any] = None

def init_worker_self_play(model_state: Dict, config: Any):
    """
    初始化自我对弈工作进程。
    """
    global g_model, g_config
    
    g_config = config
    
    # 禁用梯度计算以进行推理优化
    torch.set_grad_enabled(False)

    # 在新进程中重新创建和加载模型
    g_model = AlphaZeroNet(g_config)
    g_model.load_state_dict(model_state)
    g_model.to(g_config.DEVICE)
    g_model.eval()

def _play_one_game_worker() -> Optional[List[Tuple[np.ndarray, np.ndarray, float]]]:
    """
    在工作进程中进行一局自我对弈。
    """
    global g_model, g_config

    game = ChessGame(g_config)
    mcts = MCTS(g_model, g_config)
    
    states_hist, policies_hist, values_hist = [], [], []

    while not game.is_over():
        state = game.get_state()
        policy, value = mcts.search(game.board)
        
        states_hist.append(state)
        policies_hist.append(policy)
        values_hist.append(value)
        
        # 根据温度参数选择确定性或随机性走法
        temp = 1 if len(game.board.board.move_stack) <= g_config.TEMP_THRESHOLD else 0
        move = game.select_move(policy, deterministic=(temp == 0))
        game.make_move(move)

    if not game.is_over():
        return None

    # 根据最终结果调整价值
    game_result = game.get_result()
    final_values = []
    for i in range(len(values_hist)):
        # 价值是从当前玩家的角度出发的，所以需要交替
        value = game_result if i % 2 == 0 else -game_result
        final_values.append(value)

    return list(zip(states_hist, policies_hist, final_values))

# --- End Worker Process Initialization ---


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
            
            futures = [executor.submit(_play_one_game_worker) for _ in range(num_games)]
            
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