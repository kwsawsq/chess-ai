"""
国际象棋游戏类
"""

import chess
import numpy as np
from typing import List, Optional, Tuple

class ChessGame:
    """国际象棋游戏"""
    
    def __init__(self, config):
        """
        初始化游戏
        
        Args:
            config: 配置对象
        """
        self.board = chess.Board()
        self.config = config
        self.move_history = []  # 记录所有移动
    
    def get_state(self) -> np.ndarray:
        """
        获取当前状态
        
        Returns:
            np.ndarray: 状态数组
        """
        # 实现状态转换逻辑
        pass
    
    def select_move(self, policy: np.ndarray) -> str:
        """
        根据策略选择移动
        
        Args:
            policy: 移动概率分布
            
        Returns:
            str: 选择的移动（UCI格式）
        """
        # 获取所有合法移动
        legal_moves = list(self.board.legal_moves)
        
        # 将策略限制在合法移动范围内
        legal_policy = np.zeros_like(policy)
        for move in legal_moves:
            move_idx = self.move_to_index(move)
            if move_idx < len(policy):
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
        for move in legal_moves:
            if self.move_to_index(move) == move_idx:
                selected_move = move
                break
        else:
            # 如果没有找到对应的移动，随机选择一个合法移动
            selected_move = np.random.choice(legal_moves)
        
        # 转换为SAN格式并记录
        move_san = self.board.san(selected_move)
        self.move_history.append(move_san)
        
        return move_san
    
    def make_move(self, move_san: str) -> bool:
        """
        执行移动
        
        Args:
            move_san: SAN格式的移动
            
        Returns:
            bool: 移动是否成功
        """
        try:
            # 将SAN格式转换为Move对象并执行
            move = self.board.parse_san(move_san)
            self.board.push(move)
            return True
        except ValueError:
            print(f"无效的移动: {move_san}")
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
        if not self.board.is_game_over():
            return 0
            
        result = self.board.result()
        if result == "1-0":
            return 1
        elif result == "0-1":
            return -1
        else:
            return 0
    
    def move_to_index(self, move: chess.Move) -> int:
        """
        将Move对象转换为策略数组的索引
        
        Args:
            move: 棋步
            
        Returns:
            int: 索引
        """
        # 计算起始格和目标格的索引
        from_square = move.from_square
        to_square = move.to_square
        
        # 计算基础索引
        base_index = from_square * 64 + to_square
        
        # 如果是升变，添加升变类型的偏移
        if move.promotion:
            promotion_offset = {
                chess.QUEEN: 0,
                chess.ROOK: 1,
                chess.BISHOP: 2,
                chess.KNIGHT: 3
            }
            base_index += promotion_offset[move.promotion] * 64 * 64
        
        return base_index
    
    def index_to_move(self, index: int) -> Optional[chess.Move]:
        """
        将索引转换为Move对象
        
        Args:
            index: 策略数组的索引
            
        Returns:
            Optional[chess.Move]: 棋步对象
        """
        # 提取升变类型
        promotion_type = index // (64 * 64)
        remaining_index = index % (64 * 64)
        
        # 提取起始格和目标格
        from_square = remaining_index // 64
        to_square = remaining_index % 64
        
        # 创建基础移动
        move = chess.Move(from_square, to_square)
        
        # 添加升变类型
        if promotion_type > 0:
            promotion_pieces = [chess.QUEEN, chess.ROOK, chess.BISHOP, chess.KNIGHT]
            if promotion_type <= len(promotion_pieces):
                move.promotion = promotion_pieces[promotion_type - 1]
        
        # 验证移动的合法性
        if move in self.board.legal_moves:
            return move
        return None
    
    def __str__(self) -> str:
        """字符串表示"""
        return str(self.board)
    
    def __repr__(self) -> str:
        """调试表示"""
        return self.__str__() 