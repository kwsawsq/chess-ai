"""
蒙特卡洛树搜索主算法
"""

import numpy as np
import time
from typing import Optional, Tuple, Dict, Any, List
import logging
from .node import MCTSNode
from ..game import ChessBoard
from ..neural_network import AlphaZeroNet


class MCTS:
    """
    蒙特卡洛树搜索算法
    """
    
    def __init__(self,
                 neural_net: AlphaZeroNet,
                 config,
                 game: Optional[ChessBoard] = None):
        """
        初始化MCTS
        
        Args:
            neural_net: 神经网络模型
            config: 配置对象
            game: 游戏引擎（可选）
        """
        self.neural_net = neural_net
        self.config = config
        self.game = game or ChessBoard()
        
        # MCTS参数
        self.num_simulations = config.NUM_MCTS_SIMS
        self.c_puct = config.C_PUCT
        self.dir_epsilon = config.DIRICHLET_EPSILON
        self.dir_alpha = config.DIRICHLET_ALPHA
        
        # 统计信息
        self.search_stats = {
            'total_simulations': 0,
            'total_time': 0.0,
            'evaluations': 0,
            'cache_hits': 0
        }
        
        # 评估缓存
        self.evaluation_cache: Dict[str, Tuple[np.ndarray, float]] = {}
        self.cache_size_limit = 10000
        
        # 日志
        self.logger = logging.getLogger(__name__)
        
        # 树重用
        self.root: Optional[MCTSNode] = None
    
    def search(self, 
               board: ChessBoard, 
               num_simulations: Optional[int] = None,
               add_noise: bool = True,
               temperature: float = 1.0) -> Tuple[np.ndarray, MCTSNode]:
        """
        执行MCTS搜索
        
        Args:
            board: 当前棋盘状态
            num_simulations: 模拟次数
            add_noise: 是否添加噪声（根节点）
            temperature: 温度参数
            
        Returns:
            Tuple[np.ndarray, MCTSNode]: (动作概率分布, 根节点)
        """
        num_simulations = num_simulations or self.num_simulations
        
        # 创建或重用根节点
        if self.root is None or self.root.board.get_fen() != board.get_fen():
            self.root = MCTSNode(board=board.copy())
        
        start_time = time.time()
        
        # 执行模拟
        for i in range(num_simulations):
            # 单次模拟的四个阶段
            self._simulate_once(self.root)
            
            # 定期清理缓存
            if i % 100 == 0 and len(self.evaluation_cache) > self.cache_size_limit:
                self._clean_cache()
        
        # 为根节点添加狄利克雷噪声
        if add_noise and self.root.is_expanded:
            self.root.add_dirichlet_noise(self.dir_epsilon, self.dir_alpha)
        
        # 获取动作概率分布
        action_probs = self.root.get_action_probs(temperature)
        
        # 更新统计信息
        search_time = time.time() - start_time
        self.search_stats['total_simulations'] += num_simulations
        self.search_stats['total_time'] += search_time
        
        self.logger.debug(
            f"MCTS搜索完成: {num_simulations}次模拟, "
            f"用时{search_time:.3f}s, "
            f"根节点访问{self.root.visit_count}次"
        )
        
        return action_probs, self.root
    
    def _simulate_once(self, node: MCTSNode) -> float:
        """
        执行一次完整的模拟
        
        Args:
            node: 起始节点
            
        Returns:
            float: 叶子节点的价值评估
        """
        path = []
        current = node
        
        # 阶段1: 选择 - 从根节点向下选择到叶子节点
        while not current.is_leaf():
            action, current = current.select_child(self.c_puct)
            path.append(current)
        
        # 阶段2: 扩展 - 如果叶子节点不是终端节点，扩展它
        value = 0.0
        
        if current.is_terminal:
            # 终端节点，直接获取结果
            value = current.get_value()
        else:
            # 非终端叶子节点，需要神经网络评估和扩展
            policy_probs, value = self._evaluate_node(current)
            
            # 扩展节点
            current.expand(policy_probs)
        
        # 阶段3: 回溯 - 将价值向上传播
        current.backup(value)
        
        return value
    
    def _evaluate_node(self, node: MCTSNode) -> Tuple[np.ndarray, float]:
        """
        使用神经网络评估节点
        
        Args:
            node: 要评估的节点
            
        Returns:
            Tuple[np.ndarray, float]: (策略概率, 价值评估)
        """
        # 检查缓存
        board_hash = node.board.get_board_hash()
        if board_hash in self.evaluation_cache:
            self.search_stats['cache_hits'] += 1
            return self.evaluation_cache[board_hash]
        
        # 获取棋盘的规范形式
        canonical_board = node.board.get_canonical_form(node.player)
        
        # 神经网络预测
        policy_probs, value = self.neural_net.predict(canonical_board)
        
        # 应用合法动作掩码
        legal_mask = node.get_legal_actions_mask()
        policy_probs = policy_probs * legal_mask
        
        # 重新归一化
        policy_sum = np.sum(policy_probs)
        if policy_sum > 0:
            policy_probs = policy_probs / policy_sum
        else:
            # 如果没有合法动作，均匀分布
            legal_actions = node.get_legal_actions()
            policy_probs = np.zeros(len(policy_probs))
            if legal_actions:
                for action in legal_actions:
                    policy_probs[action] = 1.0 / len(legal_actions)
        
        # 缓存结果
        self.evaluation_cache[board_hash] = (policy_probs, value)
        self.search_stats['evaluations'] += 1
        
        return policy_probs, value
    
    def get_best_action(self, 
                       board: ChessBoard,
                       num_simulations: Optional[int] = None,
                       temperature: float = 0.0) -> int:
        """
        获取最佳动作
        
        Args:
            board: 当前棋盘状态
            num_simulations: 模拟次数
            temperature: 温度参数
            
        Returns:
            int: 最佳动作索引
        """
        action_probs, root = self.search(
            board, 
            num_simulations, 
            add_noise=False, 
            temperature=temperature
        )
        
        if temperature == 0.0:
            # 贪婪选择
            return int(np.argmax(action_probs))
        else:
            # 概率采样
            legal_actions = np.where(action_probs > 0)[0]
            if len(legal_actions) == 0:
                # 备用方案：从合法动作中随机选择
                legal_actions = [i for i, is_legal in enumerate(board.get_legal_moves_mask()) if is_legal]
                if legal_actions:
                    return np.random.choice(legal_actions)
                else:
                    return 0  # 应该不会发生
            
            return np.random.choice(len(action_probs), p=action_probs)
    
    def get_action_probabilities(self, 
                               board: ChessBoard,
                               num_simulations: Optional[int] = None,
                               temperature: float = 1.0) -> np.ndarray:
        """
        获取动作概率分布
        
        Args:
            board: 当前棋盘状态
            num_simulations: 模拟次数
            temperature: 温度参数
            
        Returns:
            np.ndarray: 动作概率分布
        """
        action_probs, _ = self.search(board, num_simulations, temperature=temperature)
        return action_probs
    
    def update_with_move(self, action: int) -> bool:
        """
        使用指定动作更新树（树重用）
        
        Args:
            action: 执行的动作
            
        Returns:
            bool: 是否成功更新
        """
        if self.root is None or not self.root.is_expanded:
            self.root = None
            return False
        
        # 查找对应的子节点
        if action in self.root.children:
            # 将子节点设为新的根节点
            new_root = self.root.children[action]
            new_root.parent = None
            self.root = new_root
            return True
        else:
            # 没有找到对应的子节点，重置树
            self.root = None
            return False
    
    def reset_tree(self):
        """重置搜索树"""
        self.root = None
        self.evaluation_cache.clear()
    
    def _clean_cache(self):
        """清理评估缓存"""
        if len(self.evaluation_cache) > self.cache_size_limit:
            # 简单策略：清除一半的缓存
            items = list(self.evaluation_cache.items())
            items_to_keep = items[len(items)//2:]
            self.evaluation_cache = dict(items_to_keep)
            self.logger.debug(f"清理缓存，保留{len(self.evaluation_cache)}项")
    
    def get_search_statistics(self) -> Dict[str, Any]:
        """
        获取搜索统计信息
        
        Returns:
            Dict[str, Any]: 统计信息
        """
        stats = self.search_stats.copy()
        
        if stats['total_simulations'] > 0:
            stats['avg_time_per_simulation'] = stats['total_time'] / stats['total_simulations']
        else:
            stats['avg_time_per_simulation'] = 0.0
        
        if stats['evaluations'] > 0:
            stats['cache_hit_rate'] = stats['cache_hits'] / (stats['cache_hits'] + stats['evaluations'])
        else:
            stats['cache_hit_rate'] = 0.0
        
        stats['cache_size'] = len(self.evaluation_cache)
        
        if self.root:
            stats['root_visits'] = self.root.visit_count
            stats['root_children'] = len(self.root.children)
            stats['tree_size'] = self.root.get_subtree_size()
        
        return stats
    
    def get_principal_variation(self, max_depth: int = 10) -> List[int]:
        """
        获取主要变例（最常访问的路径）
        
        Args:
            max_depth: 最大深度
            
        Returns:
            List[int]: 动作序列
        """
        if not self.root:
            return []
        
        pv = []
        current = self.root
        depth = 0
        
        while (current.children and 
               depth < max_depth and 
               not current.is_terminal):
            best_action = current.get_best_action()
            if best_action is None:
                break
            
            pv.append(best_action)
            current = current.children[best_action]
            depth += 1
        
        return pv
    
    def analyze_position(self, board: ChessBoard, depth: int = 3) -> Dict[str, Any]:
        """
        分析当前局面
        
        Args:
            board: 棋盘状态
            depth: 分析深度
            
        Returns:
            Dict[str, Any]: 分析结果
        """
        # 执行搜索
        action_probs, root = self.search(board)
        
        # 获取最佳几个动作
        legal_actions = [i for i, prob in enumerate(action_probs) if prob > 0]
        legal_actions.sort(key=lambda x: action_probs[x], reverse=True)
        
        top_moves = []
        for i, action in enumerate(legal_actions[:5]):  # 前5个最佳动作
            child = root.children.get(action)
            if child:
                move_info = {
                    'action': action,
                    'probability': action_probs[action],
                    'visits': child.visit_count,
                    'value': child.get_value(),
                    'ucb': child.get_ucb_value(self.c_puct)
                }
                
                # 添加走法信息
                move = board.action_to_move(action)
                if move:
                    move_info['move'] = move.uci()
                
                top_moves.append(move_info)
        
        analysis = {
            'position_value': root.get_value(),
            'total_visits': root.visit_count,
            'top_moves': top_moves,
            'principal_variation': self.get_principal_variation(),
            'search_stats': self.get_search_statistics()
        }
        
        return analysis
    
    def play_random_game(self, board: ChessBoard, max_moves: int = 200) -> Tuple[int, List[int]]:
        """
        使用MCTS进行随机对弈
        
        Args:
            board: 起始棋盘状态
            max_moves: 最大步数
            
        Returns:
            Tuple[int, List[int]]: (游戏结果, 动作序列)
        """
        current_board = board.copy()
        moves = []
        
        for move_count in range(max_moves):
            if current_board.is_game_over():
                break
            
            # 获取最佳动作
            action = self.get_best_action(current_board, temperature=0.1)
            moves.append(action)
            
            # 执行动作
            move = current_board.action_to_move(action)
            if move and current_board.make_move(move):
                # 更新树
                self.update_with_move(action)
            else:
                self.logger.error(f"无法执行动作: {action}")
                break
        
        result = current_board.get_result()
        return result if result is not None else 0, moves
    
    def set_config(self, **kwargs):
        """
        更新配置参数
        
        Args:
            **kwargs: 配置参数
        """
        if 'num_simulations' in kwargs:
            self.num_simulations = kwargs['num_simulations']
        if 'c_puct' in kwargs:
            self.c_puct = kwargs['c_puct']
        if 'dir_epsilon' in kwargs:
            self.dir_epsilon = kwargs['dir_epsilon']
        if 'dir_alpha' in kwargs:
            self.dir_alpha = kwargs['dir_alpha']
    
    def get_tree_info(self) -> Dict[str, Any]:
        """
        获取搜索树信息
        
        Returns:
            Dict[str, Any]: 树信息
        """
        if not self.root:
            return {'tree_exists': False}
        
        return {
            'tree_exists': True,
            'root_visits': self.root.visit_count,
            'root_value': self.root.get_value(),
            'children_count': len(self.root.children),
            'tree_size': self.root.get_subtree_size(),
            'max_depth': self._get_max_depth(self.root),
            'is_expanded': self.root.is_expanded
        }
    
    def _get_max_depth(self, node: MCTSNode, current_depth: int = 0) -> int:
        """计算树的最大深度"""
        if not node.children:
            return current_depth
        
        max_child_depth = 0
        for child in node.children.values():
            child_depth = self._get_max_depth(child, current_depth + 1)
            max_child_depth = max(max_child_depth, child_depth)
        
        return max_child_depth
    
    def print_tree(self, max_depth: int = 3) -> str:
        """
        打印搜索树（调试用）
        
        Args:
            max_depth: 最大打印深度
            
        Returns:
            str: 树结构字符串
        """
        if not self.root:
            return "搜索树为空"
        
        return self.root.print_tree(max_depth)
    
    def __str__(self) -> str:
        """字符串表示"""
        tree_info = self.get_tree_info()
        if tree_info['tree_exists']:
            return (f"MCTS(模拟次数={self.num_simulations}, "
                   f"根节点访问={tree_info['root_visits']}, "
                   f"树大小={tree_info['tree_size']})")
        else:
            return f"MCTS(模拟次数={self.num_simulations}, 树=空)" 