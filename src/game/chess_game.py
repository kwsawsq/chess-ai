"""
国际象棋游戏类
"""

import chess
import numpy as np
from typing import List, Optional, Tuple
from .chess_board import ChessBoard

class ChessGame:
    """国际象棋游戏"""
    
    def __init__(self, config):
        """
        初始化游戏
        
        Args:
            config: 配置对象
        """
        self.board = ChessBoard()  # 使用我们的ChessBoard类
        self.config = config
        self.move_history = []  # 记录所有移动
    
    def get_state(self) -> np.ndarray:
        """
        获取当前状态
        
        Returns:
            np.ndarray: 状态数组
        """
        # 获取当前玩家
        current_player = self.board.get_current_player()
        
        # 获取规范形式的状态
        return self.board.get_canonical_form(current_player)
    
    def select_move(self, policy: np.ndarray) -> str:
        """
        根据策略选择移动
        
        Args:
            policy: 移动概率分布
            
        Returns:
            str: 选择的移动（UCI格式）
        """
        # 获取所有合法移动
        legal_moves = self.board.get_legal_moves()
        
        # 将策略限制在合法移动范围内
        legal_policy = np.zeros_like(policy)
        for move in legal_moves:
            move_idx = self.board.move_to_action(move)
            if move_idx is not None and move_idx < len(policy):
                legal_policy[move_idx] = policy[move_idx]
        
        # 归一化概率
        if legal_policy.sum() > 0:
            legal_policy /= legal_policy.sum()
        else:
            # 如果没有有效的概率，使用均匀分布
            legal_policy = np.ones_like(policy) / len(policy)
        
        # 根据温度参数选择移动
        if len(self.move_history) < self.config.TEMP_THRESHOLD:
            # 探索阶段：使用概率选择
            move_idx = np.random.choice(len(policy), p=legal_policy)
        else:
            # 利用阶段：选择最高概率的移动
            move_idx = np.argmax(legal_policy)
        
        # 将索引转换为移动
        selected_move = None
        for move in legal_moves:
            if self.board.move_to_action(move) == move_idx:
                selected_move = move
                break
        
        if selected_move is None:
            # 如果没有找到对应的移动，随机选择一个合法移动
            selected_move = np.random.choice(legal_moves)
        
        # 转换为UCI格式并记录
        move_uci = selected_move.uci()
        self.move_history.append(move_uci)
        
        return move_uci
    
    def make_move(self, move_uci: str) -> bool:
        """
        执行移动
        
        Args:
            move_uci: UCI格式的移动
            
        Returns:
            bool: 移动是否成功
        """
        try:
            # 将UCI格式转换为Move对象并执行
            move = chess.Move.from_uci(move_uci)
            return self.board.make_move(move)
        except ValueError:
            print(f"无效的移动: {move_uci}")
            return False
    
    def is_over(self) -> bool:
        """
        检查游戏是否结束
        
        Returns:
            bool: 游戏是否结束
        """
        return self.board.is_game_over()
    
    def get_result(self) -> int:
        """
        获取游戏结果
        
        Returns:
            int: 1(白胜)/-1(黑胜)/0(和棋)
        """
        result = self.board.get_result()
        return result if result is not None else 0
    
    def __str__(self) -> str:
        """字符串表示"""
        return str(self.board)
    
    def __repr__(self) -> str:
        """调试表示"""
        return self.__str__() 