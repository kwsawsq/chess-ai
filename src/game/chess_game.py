"""
国际象棋游戏引擎类
"""

import numpy as np
import chess
from typing import List, Tuple, Optional, Dict, Any
import logging
from .chess_board import ChessBoard


class ChessGame:
    """
    国际象棋游戏引擎
    为AlphaZero提供标准的游戏接口
    """
    
    def __init__(self, board: Optional[ChessBoard] = None):
        """
        初始化游戏
        
        Args:
            board: 棋盘对象，如果为None则创建新的开局棋盘
        """
        self.board = board if board is not None else ChessBoard()
        self.game_history = []  # 游戏历史记录
        self.move_count = 0
        
        # 游戏设置
        self.board_size = 8
        self.action_size = 4096
        self.players = [1, -1]  # 1=白方，-1=黑方
        
        # 日志记录
        self.logger = logging.getLogger(__name__)
    
    def get_init_board(self) -> ChessBoard:
        """
        获取初始棋盘状态
        
        Returns:
            ChessBoard: 初始棋盘
        """
        return ChessBoard()
    
    def get_board_size(self) -> int:
        """获取棋盘大小"""
        return self.board_size
    
    def get_action_size(self) -> int:
        """获取动作空间大小"""
        return self.action_size
    
    def get_next_state(self, board: ChessBoard, player: int, action: int) -> Tuple[ChessBoard, int]:
        """
        获取执行动作后的下一个状态
        
        Args:
            board: 当前棋盘状态
            player: 当前玩家 (1=白方, -1=黑方)
            action: 要执行的动作索引
            
        Returns:
            Tuple[ChessBoard, int]: (新的棋盘状态, 下一个玩家)
        """
        # 复制棋盘
        new_board = board.copy()
        
        # 将动作转换为走法
        move = new_board.action_to_move(action)
        
        if move is None or not new_board.make_move(move):
            # 非法动作，返回原状态
            self.logger.warning(f"非法动作: {action}")
            return board, player
        
        # 记录历史
        self.game_history.append({
            'board': board.get_fen(),
            'player': player,
            'action': action,
            'move': move.uci()
        })
        
        # 返回新状态和下一个玩家
        return new_board, -player
    
    def get_valid_moves(self, board: ChessBoard, player: int) -> np.ndarray:
        """
        获取合法动作掩码
        
        Args:
            board: 当前棋盘状态
            player: 当前玩家
            
        Returns:
            np.ndarray: 合法动作掩码 (4096,)
        """
        # 检查当前玩家是否匹配
        if board.get_current_player() != player:
            return np.zeros(self.action_size, dtype=bool)
        
        return board.get_legal_moves_mask()
    
    def get_game_ended(self, board: ChessBoard, player: int) -> float:
        """
        检查游戏是否结束，并返回结果
        
        Args:
            board: 当前棋盘状态
            player: 当前玩家
            
        Returns:
            float: 游戏结果 (1=当前玩家胜利, -1=当前玩家失败, 0=平局或游戏未结束)
        """
        if not board.is_game_over():
            return 0  # 游戏未结束
        
        result = board.get_result()
        if result is None:
            return 0
        
        # 从当前玩家的角度返回结果
        if result == 0:
            return 0  # 平局
        elif result == player:
            return 1  # 当前玩家胜利
        else:
            return -1  # 当前玩家失败
    
    def get_canonical_form(self, board: ChessBoard, player: int) -> np.ndarray:
        """
        获取棋盘的规范形式（从当前玩家视角）
        
        Args:
            board: 棋盘状态
            player: 当前玩家
            
        Returns:
            np.ndarray: 规范形式的棋盘表示 (20, 8, 8)
        """
        return board.get_canonical_form(player)
    
    def get_symmetries(self, board: np.ndarray, pi: np.ndarray) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        获取棋盘的对称形式（用于数据增强）
        
        Args:
            board: 棋盘状态 (20, 8, 8)
            pi: 策略向量 (4096,)
            
        Returns:
            List[Tuple[np.ndarray, np.ndarray]]: 对称形式的 (棋盘, 策略) 列表
        """
        # 国际象棋的对称性比较复杂，这里实现水平翻转
        symmetries = []
        
        # 原始状态
        symmetries.append((board, pi))
        
        # 水平翻转
        flipped_board = self._flip_board_horizontal(board)
        flipped_pi = self._flip_policy_horizontal(pi)
        symmetries.append((flipped_board, flipped_pi))
        
        return symmetries
    
    def _flip_board_horizontal(self, board: np.ndarray) -> np.ndarray:
        """
        水平翻转棋盘
        
        Args:
            board: 原始棋盘 (20, 8, 8)
            
        Returns:
            np.ndarray: 翻转后的棋盘
        """
        return np.flip(board, axis=2)  # 沿着列方向翻转
    
    def _flip_policy_horizontal(self, pi: np.ndarray) -> np.ndarray:
        """
        水平翻转策略向量
        
        Args:
            pi: 原始策略向量 (4096,)
            
        Returns:
            np.ndarray: 翻转后的策略向量
        """
        flipped_pi = np.zeros_like(pi)
        
        for i in range(4096):
            from_square = i // 64
            to_square = i % 64
            
            # 翻转起始和目标位置
            from_row, from_col = divmod(from_square, 8)
            to_row, to_col = divmod(to_square, 8)
            
            flipped_from_col = 7 - from_col
            flipped_to_col = 7 - to_col
            
            flipped_from_square = from_row * 8 + flipped_from_col
            flipped_to_square = to_row * 8 + flipped_to_col
            
            flipped_action = flipped_from_square * 64 + flipped_to_square
            
            if flipped_action < 4096:
                flipped_pi[flipped_action] = pi[i]
        
        return flipped_pi
    
    def string_representation(self, board: ChessBoard) -> str:
        """
        获取棋盘的字符串表示
        
        Args:
            board: 棋盘状态
            
        Returns:
            str: 棋盘的字符串表示
        """
        return board.get_fen()
    
    def play_game(self, player1_func, player2_func, verbose: bool = False) -> Tuple[int, List[Dict]]:
        """
        进行一局完整的游戏
        
        Args:
            player1_func: 白方玩家函数
            player2_func: 黑方玩家函数
            verbose: 是否输出详细信息
            
        Returns:
            Tuple[int, List[Dict]]: (游戏结果, 游戏历史)
        """
        board = self.get_init_board()
        current_player = 1  # 白方先走
        move_history = []
        
        while True:
            # 获取当前玩家的动作
            if current_player == 1:
                action = player1_func(board, current_player)
            else:
                action = player2_func(board, current_player)
            
            if verbose:
                print(f"玩家 {current_player} 的动作: {action}")
            
            # 记录当前状态
            move_history.append({
                'board': board.get_fen(),
                'player': current_player,
                'action': action,
                'canonical_board': self.get_canonical_form(board, current_player)
            })
            
            # 执行动作
            board, current_player = self.get_next_state(board, current_player, action)
            
            # 检查游戏是否结束
            game_result = self.get_game_ended(board, current_player)
            if game_result != 0:
                if verbose:
                    print(f"游戏结束，结果: {game_result}")
                return game_result, move_history
            
            # 检查是否达到最大步数
            if len(move_history) >= 500:  # 防止无限循环
                if verbose:
                    print("达到最大步数，游戏平局")
                return 0, move_history
    
    def get_training_examples(self, game_history: List[Dict], result: int) -> List[Tuple[np.ndarray, np.ndarray, float]]:
        """
        从游戏历史中生成训练样本
        
        Args:
            game_history: 游戏历史记录
            result: 游戏结果
            
        Returns:
            List[Tuple[np.ndarray, np.ndarray, float]]: 训练样本列表 (棋盘, 策略, 价值)
        """
        training_examples = []
        
        for i, history_item in enumerate(game_history):
            board = history_item['canonical_board']
            player = history_item['player']
            
            # 计算从当前玩家视角的游戏结果
            if result == 0:
                value = 0  # 平局
            elif result == player:
                value = 1  # 当前玩家胜利
            else:
                value = -1  # 当前玩家失败
            
            # 创建均匀的策略分布（在实际实现中，这应该是MCTS的结果）
            pi = np.zeros(self.action_size)
            valid_moves = self.get_valid_moves(ChessBoard(history_item['board']), player)
            if np.any(valid_moves):
                pi[valid_moves] = 1.0 / np.sum(valid_moves)
            
            training_examples.append((board, pi, value))
        
        return training_examples
    
    def get_move_statistics(self) -> Dict[str, Any]:
        """
        获取移动统计信息
        
        Returns:
            Dict[str, Any]: 统计信息
        """
        if not self.game_history:
            return {}
        
        stats = {
            'total_moves': len(self.game_history),
            'white_moves': sum(1 for h in self.game_history if h['player'] == 1),
            'black_moves': sum(1 for h in self.game_history if h['player'] == -1),
            'game_length': len(self.game_history),
            'last_move': self.game_history[-1]['move'] if self.game_history else None
        }
        
        return stats
    
    def reset_game(self):
        """重置游戏状态"""
        self.board = ChessBoard()
        self.game_history = []
        self.move_count = 0
    
    def load_game_from_pgn(self, pgn_string: str) -> bool:
        """
        从PGN字符串加载游戏
        
        Args:
            pgn_string: PGN格式的游戏记录
            
        Returns:
            bool: 是否成功加载
        """
        try:
            # 这里需要实现PGN解析逻辑
            # 简化实现，实际需要使用python-chess的PGN解析功能
            self.logger.info("PGN加载功能待实现")
            return True
        except Exception as e:
            self.logger.error(f"加载PGN失败: {e}")
            return False
    
    def save_game_to_pgn(self) -> str:
        """
        将游戏保存为PGN格式
        
        Returns:
            str: PGN格式的游戏记录
        """
        try:
            # 简化实现，实际需要完整的PGN生成逻辑
            pgn_parts = []
            pgn_parts.append('[Event "AlphaZero Self-Play"]')
            pgn_parts.append('[Site "Computer"]')
            pgn_parts.append('[Date "?"]')
            pgn_parts.append('[Round "?"]')
            pgn_parts.append('[White "AlphaZero"]')
            pgn_parts.append('[Black "AlphaZero"]')
            pgn_parts.append('[Result "*"]')
            pgn_parts.append('')
            
            # 添加走法
            for i, history in enumerate(self.game_history):
                if i % 2 == 0:  # 白方走法
                    pgn_parts.append(f"{i//2 + 1}. {history['move']}")
                else:  # 黑方走法
                    pgn_parts.append(f" {history['move']}")
            
            return '\n'.join(pgn_parts)
        except Exception as e:
            self.logger.error(f"保存PGN失败: {e}")
            return ""
    
    def __str__(self) -> str:
        """字符串表示"""
        return f"ChessGame(moves={len(self.game_history)})"
    
    def __repr__(self) -> str:
        """调试表示"""
        return f"ChessGame(board='{self.board.get_fen()}', moves={len(self.game_history)})" 