"""
蒙特卡洛树搜索主算法
"""

import numpy as np
import math
import logging
from typing import Dict, List, Optional, Tuple, Any

import chess

from .node import MCTSNode
from ..game import ChessBoard
from ..neural_network import AlphaZeroNet


class MCTS:
    """
    蒙特卡洛树搜索（带批处理功能）
    """
    def __init__(self, model: AlphaZeroNet, config: Any):
        self.model = model
        self.config = config
        self.logger = logging.getLogger(__name__)

    def search(self, board: ChessBoard, add_noise: bool = True) -> Tuple[np.ndarray, float]:
        """
        对当前棋盘进行MCTS搜索
        """
        root_node = MCTSNode(board=board.copy())
        
        # 如果根节点是叶子节点，先进行一次评估和扩展
        if root_node.is_leaf():
            policy_batch, value_batch = self.model.predict_batch(np.array([root_node.board.get_canonical_form(root_node.player)]))
            root_node.expand(policy_batch[0])
            root_node.backup(value_batch[0][0])

        if add_noise:
            root_node.add_dirichlet_noise(epsilon=self.config.DIRICHLET_EPSILON, alpha=self.config.DIRICHLET_ALPHA)

        # 批处理循环
        for _ in range(self.config.NUM_MCTS_SIMS // self.config.MCTS_BATCH_SIZE):
            pending_evals: List[MCTSNode] = []
            
            # 1. 选择 & 收集叶子节点
            for _ in range(self.config.MCTS_BATCH_SIZE):
                leaf_node = self._select_leaf(root_node)
                pending_evals.append(leaf_node)
            
            # 2. 批处理评估
            states_batch = np.array([node.board.get_canonical_form(node.player) for node in pending_evals])
            policies, values = self.model.predict_batch(states_batch)

            # 3. 扩展 & 反向传播
            for i, node in enumerate(pending_evals):
                if not node.is_terminal:
                    node.expand(policies[i])
                    node.backup(values[i][0])
                else: # 如果是终端节点，直接用游戏结果反向传播
                    node.backup(node.terminal_value)
        
        # 从根节点获取最终策略和价值
        policy = root_node.get_action_probs(temperature=1.0) # 在训练时使用温度1
        value = root_node.get_value()
        
        return policy, value

    def _select_leaf(self, node: MCTSNode) -> MCTSNode:
        """
        从给定节点开始，选择一个叶子节点进行评估
        """
        current_node = node
        while not current_node.is_leaf():
            _, current_node = current_node.select_child(self.config.C_PUCT)
        return current_node 