"""
MCTS节点类
"""

import numpy as np
import math
from typing import Dict, List, Optional, Tuple, Any
import logging
from ..game import ChessBoard


class MCTSNode:
    """
    蒙特卡洛树搜索节点
    """
    
    def __init__(self, 
                 board: ChessBoard,
                 parent: Optional['MCTSNode'] = None,
                 action: Optional[int] = None,
                 prior_prob: float = 0.0):
        """
        初始化MCTS节点
        
        Args:
            board: 棋盘状态
            parent: 父节点
            action: 到达此节点的动作
            prior_prob: 先验概率
        """
        self.board = board
        self.parent = parent
        self.action = action
        self.prior_prob = prior_prob
        
        # 统计信息
        self.visit_count = 0
        self.value_sum = 0.0
        self.children: Dict[int, 'MCTSNode'] = {}
        
        # 节点状态
        self.is_expanded = False
        self.is_terminal = board.is_game_over()
        
        # 如果是终端节点，计算结果
        if self.is_terminal:
            self.terminal_value = board.get_result()
        else:
            self.terminal_value = None
        
        # 当前玩家
        self.player = board.get_current_player()
        
        # 合法动作缓存
        self._legal_actions = None
        self._legal_actions_mask = None
    
    def get_legal_actions(self) -> List[int]:
        """获取合法动作列表"""
        if self._legal_actions is None:
            if self.is_terminal:
                self._legal_actions = []
            else:
                legal_mask = self.board.get_legal_moves_mask()
                self._legal_actions = [i for i, is_legal in enumerate(legal_mask) if is_legal]
        return self._legal_actions
    
    def get_legal_actions_mask(self) -> np.ndarray:
        """获取合法动作掩码"""
        if self._legal_actions_mask is None:
            self._legal_actions_mask = self.board.get_legal_moves_mask()
        return self._legal_actions_mask
    
    def is_leaf(self) -> bool:
        """检查是否为叶子节点"""
        return not self.is_expanded or self.is_terminal
    
    def is_root(self) -> bool:
        """检查是否为根节点"""
        return self.parent is None
    
    def expand(self, action_probs: np.ndarray) -> bool:
        """
        扩展节点
        
        Args:
            action_probs: 动作概率分布 (4096,)
            
        Returns:
            bool: 是否成功扩展
        """
        if self.is_terminal or self.is_expanded:
            return False
        
        legal_actions = self.get_legal_actions()
        
        if not legal_actions:
            self.is_terminal = True
            return False
        
        # 为每个合法动作创建子节点
        for action in legal_actions:
            # 执行动作获得新的棋盘状态
            new_board = self.board.copy()
            move = new_board.action_to_move(action)
            
            if move and new_board.make_move(move):
                prior_prob = action_probs[action]
                child_node = MCTSNode(
                    board=new_board,
                    parent=self,
                    action=action,
                    prior_prob=prior_prob
                )
                self.children[action] = child_node
        
        self.is_expanded = True
        return True
    
    def select_child(self, c_puct: float = 1.0) -> Tuple[int, 'MCTSNode']:
        """
        使用UCB公式选择最佳子节点
        
        Args:
            c_puct: 探索参数
            
        Returns:
            Tuple[int, MCTSNode]: (动作, 子节点)
        """
        if not self.children:
            raise ValueError("节点未扩展或没有子节点")
        
        best_action = None
        best_value = -float('inf')
        
        for action, child in self.children.items():
            ucb_value = child.get_ucb_value(c_puct)
            
            if ucb_value > best_value:
                best_value = ucb_value
                best_action = action
        
        return best_action, self.children[best_action]
    
    def get_ucb_value(self, c_puct: float) -> float:
        """
        计算UCB值
        
        Args:
            c_puct: 探索参数
            
        Returns:
            float: UCB值
        """
        if self.visit_count == 0:
            return float('inf')  # 未访问的节点优先级最高
        
        # Q值（平均价值）
        q_value = self.value_sum / self.visit_count
        
        # U值（探索奖励）
        if self.parent:
            u_value = (c_puct * self.prior_prob * 
                      math.sqrt(self.parent.visit_count) / (1 + self.visit_count))
        else:
            u_value = 0
        
        return q_value + u_value
    
    def backup(self, value: float):
        """
        反向传播价值
        
        Args:
            value: 价值评估
        """
        self.visit_count += 1
        self.value_sum += value
        
        # 递归向上传播
        if self.parent:
            # 从对手视角传播价值（取负）
            self.parent.backup(-value)
    
    def get_action_probs(self, temperature: float = 1.0) -> np.ndarray:
        """
        获取基于访问次数的动作概率分布
        
        Args:
            temperature: 温度参数，控制探索程度
            
        Returns:
            np.ndarray: 动作概率分布 (4096,)
        """
        action_probs = np.zeros(4096)
        
        if not self.children:
            return action_probs
        
        # 收集访问次数
        actions = list(self.children.keys())
        visit_counts = np.array([self.children[action].visit_count for action in actions])
        
        if temperature == 0:
            # 贪婪选择
            best_action_idx = np.argmax(visit_counts)
            action_probs[actions[best_action_idx]] = 1.0
        else:
            # 温度采样
            if temperature != 1.0:
                visit_counts = visit_counts ** (1.0 / temperature)
            
            # 归一化
            total_visits = np.sum(visit_counts)
            if total_visits > 0:
                for i, action in enumerate(actions):
                    action_probs[action] = visit_counts[i] / total_visits
        
        return action_probs
    
    def get_best_action(self) -> Optional[int]:
        """
        获取访问次数最多的动作
        
        Returns:
            Optional[int]: 最佳动作
        """
        if not self.children:
            return None
        
        best_action = None
        best_visits = -1
        
        for action, child in self.children.items():
            if child.visit_count > best_visits:
                best_visits = child.visit_count
                best_action = action
        
        return best_action
    
    def get_value(self) -> float:
        """
        获取节点的平均价值
        
        Returns:
            float: 平均价值
        """
        if self.is_terminal and self.terminal_value is not None:
            # 从当前玩家视角返回终端价值
            if self.terminal_value == 0:
                return 0.0  # 平局
            elif self.terminal_value == self.player:
                return 1.0  # 当前玩家获胜
            else:
                return -1.0  # 当前玩家失败
        
        if self.visit_count == 0:
            return 0.0
        
        return self.value_sum / self.visit_count
    
    def add_dirichlet_noise(self, epsilon: float = 0.25, alpha: float = 0.3):
        """
        为根节点添加狄利克雷噪声
        
        Args:
            epsilon: 噪声权重
            alpha: 狄利克雷参数
        """
        if not self.children or not self.is_root():
            return
        
        actions = list(self.children.keys())
        noise = np.random.dirichlet([alpha] * len(actions))
        
        for i, action in enumerate(actions):
            child = self.children[action]
            child.prior_prob = (1 - epsilon) * child.prior_prob + epsilon * noise[i]
    
    def get_path_to_root(self) -> List['MCTSNode']:
        """
        获取从此节点到根节点的路径
        
        Returns:
            List[MCTSNode]: 节点路径
        """
        path = []
        node = self
        while node is not None:
            path.append(node)
            node = node.parent
        return path
    
    def get_depth(self) -> int:
        """
        获取节点深度
        
        Returns:
            int: 深度
        """
        depth = 0
        node = self.parent
        while node is not None:
            depth += 1
            node = node.parent
        return depth
    
    def get_subtree_size(self) -> int:
        """
        获取子树大小
        
        Returns:
            int: 子树节点数量
        """
        size = 1
        for child in self.children.values():
            size += child.get_subtree_size()
        return size
    
    def get_stats(self) -> Dict[str, Any]:
        """
        获取节点统计信息
        
        Returns:
            Dict[str, Any]: 统计信息
        """
        stats = {
            'visit_count': self.visit_count,
            'value_sum': self.value_sum,
            'average_value': self.get_value(),
            'prior_prob': self.prior_prob,
            'is_expanded': self.is_expanded,
            'is_terminal': self.is_terminal,
            'children_count': len(self.children),
            'depth': self.get_depth(),
            'player': self.player
        }
        
        if self.action is not None:
            stats['action'] = self.action
        
        if self.terminal_value is not None:
            stats['terminal_value'] = self.terminal_value
        
        return stats
    
    def print_tree(self, max_depth: int = 3, indent: str = "") -> str:
        """
        打印树结构（调试用）
        
        Args:
            max_depth: 最大打印深度
            indent: 缩进字符串
            
        Returns:
            str: 树结构字符串
        """
        lines = []
        
        # 当前节点信息
        node_info = f"{indent}Action: {self.action}, Visits: {self.visit_count}, Value: {self.get_value():.3f}"
        if self.is_terminal:
            node_info += " [TERMINAL]"
        lines.append(node_info)
        
        # 子节点信息
        if max_depth > 0 and self.children:
            sorted_children = sorted(
                self.children.items(),
                key=lambda x: x[1].visit_count,
                reverse=True
            )
            
            for action, child in sorted_children[:5]:  # 只显示前5个最常访问的子节点
                lines.append(child.print_tree(max_depth - 1, indent + "  "))
        
        return "\n".join(lines)
    
    def __str__(self) -> str:
        """字符串表示"""
        return (f"MCTSNode(action={self.action}, visits={self.visit_count}, "
                f"value={self.get_value():.3f}, children={len(self.children)})")
    
    def __repr__(self) -> str:
        """调试表示"""
        return self.__str__() 