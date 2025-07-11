"""
国际象棋棋盘表示类
"""

import numpy as np
import chess
import chess.engine
from typing import List, Dict, Tuple, Optional, Any
import logging


class ChessBoard:
    """
    国际象棋棋盘表示类
    使用python-chess库作为基础，提供AlphaZero所需的接口
    """
    
    def __init__(self, fen: str = None):
        """
        初始化棋盘
        
        Args:
            fen: FEN字符串，如果为None则使用初始局面
        """
        if fen:
            self.board = chess.Board(fen)
        else:
            self.board = chess.Board()
        
        # 棋盘历史记录，用于检测重复局面
        self.history = [self.board.fen()]
        
        # 缓存合法动作
        self._legal_moves_cache = None
        self._legal_moves_mask_cache = None
        
    def copy(self) -> 'ChessBoard':
        """创建棋盘副本"""
        new_board = ChessBoard()
        new_board.board = self.board.copy()
        new_board.history = self.history.copy()
        return new_board
    
    def get_canonical_form(self, player: int) -> np.ndarray:
        """
        获取棋盘的规范形式（从当前玩家视角）
        
        Args:
            player: 当前玩家 (1: 白方, -1: 黑方)
            
        Returns:
            np.ndarray: 形状为 (20, 8, 8) 的数组
        """
        # 20个通道：
        # 0-11: 12种棋子（白方6种 + 黑方6种）
        # 12-15: 重复位置历史（最近4次）
        # 16: 当前玩家颜色
        # 17: 总移动次数
        # 18: 王车易位权利
        # 19: 吃过路兵目标
        
        board_array = np.zeros((20, 8, 8), dtype=np.float32)
        
        # 从当前玩家视角获取棋盘
        if player == -1:  # 黑方
            # 旋转棋盘180度
            board_to_analyze = self.board.copy()
            board_to_analyze = board_to_analyze.mirror()
        else:  # 白方
            board_to_analyze = self.board
        
        # 棋子编码
        piece_map = {
            chess.PAWN: 0, chess.ROOK: 1, chess.KNIGHT: 2,
            chess.BISHOP: 3, chess.QUEEN: 4, chess.KING: 5
        }
        
        # 填充棋子位置
        for square in chess.SQUARES:
            piece = board_to_analyze.piece_at(square)
            if piece:
                row, col = divmod(square, 8)
                if player == -1:  # 黑方视角，需要翻转坐标
                    row, col = 7 - row, 7 - col
                
                piece_type = piece_map[piece.piece_type]
                if piece.color == chess.WHITE:
                    board_array[piece_type, row, col] = 1
                else:
                    board_array[piece_type + 6, row, col] = 1
        
        # 重复位置历史
        current_fen = self.board.fen().split()[0]  # 只要棋盘部分
        for i, hist_fen in enumerate(self.history[-4:]):
            hist_board_fen = hist_fen.split()[0]
            if hist_board_fen == current_fen:
                board_array[12 + i, :, :] = 1
        
        # 当前玩家颜色
        if self.board.turn == chess.WHITE:
            board_array[16, :, :] = 1
        
        # 总移动次数（归一化）
        board_array[17, :, :] = self.board.fullmove_number / 100.0
        
        # 王车易位权利
        castling_rights = 0
        if self.board.has_kingside_castling_rights(chess.WHITE):
            castling_rights += 1
        if self.board.has_queenside_castling_rights(chess.WHITE):
            castling_rights += 2
        if self.board.has_kingside_castling_rights(chess.BLACK):
            castling_rights += 4
        if self.board.has_queenside_castling_rights(chess.BLACK):
            castling_rights += 8
        board_array[18, :, :] = castling_rights / 15.0
        
        # 吃过路兵目标
        if self.board.ep_square:
            ep_row, ep_col = divmod(self.board.ep_square, 8)
            if player == -1:  # 黑方视角
                ep_row, ep_col = 7 - ep_row, 7 - ep_col
            board_array[19, ep_row, ep_col] = 1
        
        return board_array
    
    def get_legal_moves(self) -> List[chess.Move]:
        """获取所有合法动作"""
        if self._legal_moves_cache is None:
            self._legal_moves_cache = list(self.board.legal_moves)
        return self._legal_moves_cache
    
    def get_legal_moves_mask(self) -> np.ndarray:
        """
        获取合法动作掩码
        
        Returns:
            np.ndarray: 形状为 (4096,) 的布尔数组
        """
        if self._legal_moves_mask_cache is None:
            mask = np.zeros(4096, dtype=bool)
            for move in self.get_legal_moves():
                action_idx = self.move_to_action(move)
                if action_idx is not None:
                    mask[action_idx] = True
            self._legal_moves_mask_cache = mask
        return self._legal_moves_mask_cache
    
    def move_to_action(self, move: chess.Move) -> Optional[int]:
        """
        将走法转换为动作索引
        
        Args:
            move: 走法
            
        Returns:
            int: 动作索引 (0-4095)
        """
        from_square = move.from_square
        to_square = move.to_square
        
        # 基础动作索引：from_square * 64 + to_square
        action_idx = from_square * 64 + to_square
        
        # 处理升变
        if move.promotion:
            # 升变动作需要特殊编码
            # 这里简化处理，实际可能需要更复杂的编码
            promotion_offset = {
                chess.QUEEN: 0, chess.ROOK: 1, 
                chess.BISHOP: 2, chess.KNIGHT: 3
            }
            if move.promotion in promotion_offset:
                action_idx += promotion_offset[move.promotion] * 4096
        
        return action_idx if action_idx < 4096 else None
    
    def action_to_move(self, action_idx: int) -> Optional[chess.Move]:
        """
        将动作索引转换为走法
        
        Args:
            action_idx: 动作索引
            
        Returns:
            chess.Move: 走法
        """
        if action_idx >= 4096:
            return None
        
        from_square = action_idx // 64
        to_square = action_idx % 64
        
        # 创建基础走法
        move = chess.Move(from_square, to_square)
        
        # 检查是否为合法走法
        if move in self.board.legal_moves:
            return move
        
        # 检查升变走法
        for promotion in [chess.QUEEN, chess.ROOK, chess.BISHOP, chess.KNIGHT]:
            promo_move = chess.Move(from_square, to_square, promotion)
            if promo_move in self.board.legal_moves:
                return promo_move
        
        return None
    
    def make_move(self, move: chess.Move) -> bool:
        """
        执行走法
        
        Args:
            move: 走法
            
        Returns:
            bool: 是否成功执行
        """
        if move in self.board.legal_moves:
            self.board.push(move)
            self.history.append(self.board.fen())
            # 清除缓存
            self._legal_moves_cache = None
            self._legal_moves_mask_cache = None
            return True
        return False
    
    def undo_move(self) -> bool:
        """
        撤销最后一步走法
        
        Returns:
            bool: 是否成功撤销
        """
        if len(self.board.move_stack) > 0:
            self.board.pop()
            if len(self.history) > 1:
                self.history.pop()
            # 清除缓存
            self._legal_moves_cache = None
            self._legal_moves_mask_cache = None
            return True
        return False
    
    def is_game_over(self) -> bool:
        """检查游戏是否结束"""
        return self.board.is_game_over()
    
    def get_result(self) -> Optional[int]:
        """
        获取游戏结果
        
        Returns:
            int: 1表示白方胜利，-1表示黑方胜利，0表示平局，None表示游戏未结束
        """
        if not self.board.is_game_over():
            return None
        
        result = self.board.result()
        if result == "1-0":
            return 1  # 白方胜利
        elif result == "0-1":
            return -1  # 黑方胜利
        else:
            return 0  # 平局
    
    def get_current_player(self) -> int:
        """
        获取当前玩家
        
        Returns:
            int: 1表示白方，-1表示黑方
        """
        return 1 if self.board.turn == chess.WHITE else -1
    
    def get_fen(self) -> str:
        """获取FEN字符串"""
        return self.board.fen()
    
    def get_board_hash(self) -> str:
        """获取棋盘哈希值（用于检测重复局面）"""
        return self.board.fen().split()[0]  # 只要棋盘部分
    
    def is_repetition(self) -> bool:
        """检查是否为重复局面"""
        current_hash = self.get_board_hash()
        count = sum(1 for fen in self.history if fen.split()[0] == current_hash)
        return count >= 3
    
    def get_piece_value(self) -> float:
        """
        获取棋子价值评估（简单实现）
        
        Returns:
            float: 当前局面的棋子价值
        """
        piece_values = {
            chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3,
            chess.ROOK: 5, chess.QUEEN: 9, chess.KING: 0
        }
        
        value = 0
        for square in chess.SQUARES:
            piece = self.board.piece_at(square)
            if piece:
                piece_value = piece_values[piece.piece_type]
                value += piece_value if piece.color == chess.WHITE else -piece_value
        
        return value
    
    def __str__(self) -> str:
        """字符串表示"""
        return str(self.board)
    
    def __repr__(self) -> str:
        """调试表示"""
        return f"ChessBoard('{self.board.fen()}')" 