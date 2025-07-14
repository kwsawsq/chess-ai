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
    
    def get_current_player(self) -> int:
        """
        获取当前玩家.

        Returns:
            int: 1 for White, -1 for Black.
        """
        return 1 if self.board.board.turn == chess.WHITE else -1

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
            policy: 策略概率分布
            
        Returns:
            str: 选择的移动（UCI格式，如 'e2e4'）
        """
        # 获取合法移动
        legal_moves = list(self.board.board.legal_moves)
        
        if not legal_moves:
            return ""
        
        # 将合法移动转换为索引
        legal_indices = [self.move_to_index(move) for move in legal_moves]
        
        # 获取合法移动的概率
        legal_probs = policy[legal_indices]
        
        # 处理 NaN 或全零的情况
        if np.any(np.isnan(legal_probs)) or np.sum(legal_probs) == 0:
            # 如果策略包含 NaN 或全为零，使用均匀分布
            legal_probs = np.ones(len(legal_moves)) / len(legal_moves)
        else:
            # 归一化概率
            legal_probs = legal_probs / np.sum(legal_probs)
        
        # 根据概率选择移动
        selected_idx = np.random.choice(len(legal_moves), p=legal_probs)
        selected_move = legal_moves[selected_idx]
        
        # 返回UCI格式的移动
        return selected_move.uci()
    
    def make_move(self, move: str) -> None:
        """
        执行移动
        
        Args:
            move: 移动（UCI格式，如 'e2e4'）
        """
        try:
            # 将字符串转换为Move对象
            chess_move = chess.Move.from_uci(move)
            
            # 检查移动是否合法
            if chess_move in self.board.board.legal_moves:
                self.board.board.push(chess_move)
            else:
                print(f"警告: 尝试执行非法移动 {move}")
        except ValueError as e:
            print(f"错误: 无效的移动格式 {move}, {str(e)}")
            return
    
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

    def move_to_index(self, move: chess.Move) -> int:
        """
        将Move对象转换为策略数组的索引
        
        Args:
            move: chess.Move对象
            
        Returns:
            int: 对应的策略数组索引
        """
        # 获取起始和目标位置
        from_square = move.from_square
        to_square = move.to_square
        
        # 计算基本移动索引 (64 * 64 = 4096种可能移动)
        move_idx = from_square * 64 + to_square
        
        # 处理升变
        if move.promotion:
            # 根据升变类型调整索引
            # 升变为后(5)、车(4)、象(3)、马(2)
            promotion_offset = {
                chess.QUEEN: 0,
                chess.ROOK: 1,
                chess.BISHOP: 2,
                chess.KNIGHT: 3
            }
            # 在基本移动之后添加升变偏移
            # 注意：这里假设策略数组大小为4096，包含了所有可能的移动
            move_idx = 4096 - 16 + promotion_offset[move.promotion]
        
        return move_idx 