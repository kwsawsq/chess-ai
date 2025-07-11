"""
自我对弈模块
"""

import numpy as np
import time
import random
from typing import List, Tuple, Dict, Any, Optional
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
from ..game import ChessBoard, ChessGame
from ..mcts import MCTS
from ..neural_network import AlphaZeroNet


class SelfPlay:
    """
    自我对弈类
    生成训练数据
    """
    
    def __init__(self,
                 neural_net: AlphaZeroNet,
                 config,
                 game: Optional[ChessGame] = None):
        """
        初始化自我对弈
        
        Args:
            neural_net: 神经网络模型
            config: 配置对象
            game: 游戏引擎（可选）
        """
        self.neural_net = neural_net
        self.config = config
        self.game = game or ChessGame()
        
        # 创建MCTS
        self.mcts = MCTS(neural_net, config, game)
        
        # 自我对弈参数
        self.num_games = config.SELF_PLAY_GAMES
        self.temp_threshold = config.TEMP_THRESHOLD
        self.temp_init = config.TEMP_INIT
        self.temp_final = config.TEMP_FINAL
        
        # 统计信息
        self.stats = {
            'games_played': 0,
            'total_moves': 0,
            'white_wins': 0,
            'black_wins': 0,
            'draws': 0,
            'avg_game_length': 0.0,
            'total_time': 0.0
        }
        
        # 训练数据收集
        self.training_examples: List[Tuple[np.ndarray, np.ndarray, float]] = []
        
        # 日志
        self.logger = logging.getLogger(__name__)
        
        # 线程锁
        self._lock = threading.Lock()
    
    def play_game(self, 
                  game_id: Optional[int] = None,
                  verbose: bool = False,
                  temperature_schedule: Optional[List[float]] = None) -> Tuple[List[Dict], int, int]:
        """
        进行一局自我对弈
        
        Args:
            game_id: 游戏ID（用于日志）
            verbose: 是否输出详细信息
            temperature_schedule: 自定义温度调度
            
        Returns:
            Tuple[List[Dict], int, int]: (游戏历史, 游戏结果, 游戏长度)
        """
        board = self.game.get_init_board()
        game_history = []
        move_count = 0
        
        start_time = time.time()
        
        if verbose and game_id is not None:
            self.logger.info(f"开始游戏 {game_id}")
        
        while not board.is_game_over() and move_count < 500:  # 最大500步防止无限循环
            current_player = board.get_current_player()
            
            # 计算温度
            temperature = self._get_temperature(move_count, temperature_schedule)
            
            # 执行MCTS搜索
            action_probs, root = self.mcts.search(
                board, 
                add_noise=True, 
                temperature=temperature
            )
            
            # 记录训练数据
            canonical_board = self.game.get_canonical_form(board, current_player)
            game_history.append({
                'board': canonical_board,
                'action_probs': action_probs,
                'player': current_player,
                'move_count': move_count
            })
            
            # 选择动作
            if temperature == 0:
                action = int(np.argmax(action_probs))
            else:
                # 基于概率采样
                valid_actions = np.where(action_probs > 0)[0]
                if len(valid_actions) > 0:
                    action = np.random.choice(valid_actions, p=action_probs[valid_actions]/np.sum(action_probs[valid_actions]))
                else:
                    # 备用方案
                    legal_moves = board.get_legal_moves()
                    if legal_moves:
                        move = random.choice(legal_moves)
                        action = board.move_to_action(move)
                        if action is None:
                            action = 0
                    else:
                        break
            
            # 执行动作
            move = board.action_to_move(action)
            if move and board.make_move(move):
                # 更新MCTS树
                self.mcts.update_with_move(action)
                move_count += 1
                
                if verbose:
                    self.logger.debug(f"游戏 {game_id}, 第{move_count}步: {move.uci()}")
            else:
                self.logger.error(f"无法执行动作 {action} (move: {move})")
                break
        
        # 游戏结束
        game_result = board.get_result()
        if game_result is None:
            game_result = 0  # 平局
        
        game_time = time.time() - start_time
        
        if verbose and game_id is not None:
            result_str = {1: "白方胜利", -1: "黑方胜利", 0: "平局"}
            self.logger.info(
                f"游戏 {game_id} 结束: {result_str[game_result]}, "
                f"{move_count} 步, 用时 {game_time:.2f}s"
            )
        
        return game_history, game_result, move_count
    
    def _get_temperature(self, move_count: int, 
                        temperature_schedule: Optional[List[float]] = None) -> float:
        """
        获取当前步数对应的温度
        
        Args:
            move_count: 当前步数
            temperature_schedule: 自定义温度调度
            
        Returns:
            float: 温度值
        """
        if temperature_schedule:
            if move_count < len(temperature_schedule):
                return temperature_schedule[move_count]
            else:
                return temperature_schedule[-1]
        
        # 默认温度调度：前期高温度，后期低温度
        if move_count < self.temp_threshold:
            return self.temp_init
        else:
            # 线性递减
            progress = min(1.0, (move_count - self.temp_threshold) / 50.0)
            return self.temp_init * (1 - progress) + self.temp_final * progress
    
    def generate_training_data(self, 
                             num_games: Optional[int] = None,
                             verbose: bool = False,
                             parallel: bool = True,
                             max_workers: Optional[int] = None) -> List[Tuple[np.ndarray, np.ndarray, float]]:
        """
        生成训练数据
        
        Args:
            num_games: 游戏数量
            verbose: 是否输出详细信息
            parallel: 是否并行执行
            max_workers: 最大工作线程数
            
        Returns:
            List[Tuple[np.ndarray, np.ndarray, float]]: 训练样本列表
        """
        num_games = num_games or self.num_games
        max_workers = max_workers or min(4, num_games)
        
        self.logger.info(f"开始生成训练数据: {num_games} 局游戏")
        start_time = time.time()
        
        all_training_examples = []
        game_results = []
        game_lengths = []
        
        if parallel and num_games > 1:
            # 并行执行
            all_training_examples = self._generate_parallel(
                num_games, verbose, max_workers
            )
        else:
            # 串行执行
            for game_id in range(num_games):
                game_history, result, length = self.play_game(
                    game_id=game_id, 
                    verbose=verbose
                )
                
                # 转换为训练样本
                training_examples = self._process_game_history(game_history, result)
                all_training_examples.extend(training_examples)
                
                game_results.append(result)
                game_lengths.append(length)
                
                if verbose and (game_id + 1) % 10 == 0:
                    self.logger.info(f"完成 {game_id + 1}/{num_games} 局游戏")
        
        # 更新统计信息
        self._update_stats(game_results, game_lengths, time.time() - start_time)
        
        # 数据增强
        if len(all_training_examples) > 0:
            augmented_examples = self._augment_data(all_training_examples)
            all_training_examples.extend(augmented_examples)
        
        self.logger.info(
            f"训练数据生成完成: {len(all_training_examples)} 个样本, "
            f"用时 {time.time() - start_time:.2f}s"
        )
        
        self.training_examples = all_training_examples
        return all_training_examples
    
    def _generate_parallel(self, 
                          num_games: int, 
                          verbose: bool, 
                          max_workers: int) -> List[Tuple[np.ndarray, np.ndarray, float]]:
        """
        并行生成训练数据
        
        Args:
            num_games: 游戏数量
            verbose: 是否输出详细信息
            max_workers: 最大工作线程数
            
        Returns:
            List[Tuple[np.ndarray, np.ndarray, float]]: 训练样本列表
        """
        all_training_examples = []
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # 提交所有任务
            futures = []
            for game_id in range(num_games):
                future = executor.submit(
                    self._play_single_game_wrapper,
                    game_id, verbose
                )
                futures.append(future)
            
            # 收集结果
            completed_games = 0
            for future in as_completed(futures):
                try:
                    training_examples = future.result()
                    all_training_examples.extend(training_examples)
                    completed_games += 1
                    
                    if verbose and completed_games % 10 == 0:
                        self.logger.info(f"完成 {completed_games}/{num_games} 局游戏")
                        
                except Exception as e:
                    self.logger.error(f"游戏执行失败: {e}")
        
        return all_training_examples
    
    def _play_single_game_wrapper(self, 
                                 game_id: int, 
                                 verbose: bool) -> List[Tuple[np.ndarray, np.ndarray, float]]:
        """
        单局游戏包装器（用于并行执行）
        
        Args:
            game_id: 游戏ID
            verbose: 是否输出详细信息
            
        Returns:
            List[Tuple[np.ndarray, np.ndarray, float]]: 训练样本列表
        """
        # 为每个线程创建独立的MCTS实例
        thread_mcts = MCTS(self.neural_net, self.config, self.game)
        
        # 执行游戏
        board = self.game.get_init_board()
        game_history = []
        move_count = 0
        
        while not board.is_game_over() and move_count < 500:
            current_player = board.get_current_player()
            temperature = self._get_temperature(move_count)
            
            action_probs, _ = thread_mcts.search(
                board, 
                add_noise=True, 
                temperature=temperature
            )
            
            canonical_board = self.game.get_canonical_form(board, current_player)
            game_history.append({
                'board': canonical_board,
                'action_probs': action_probs,
                'player': current_player,
                'move_count': move_count
            })
            
            # 选择并执行动作
            if temperature == 0:
                action = int(np.argmax(action_probs))
            else:
                valid_actions = np.where(action_probs > 0)[0]
                if len(valid_actions) > 0:
                    action = np.random.choice(valid_actions, p=action_probs[valid_actions]/np.sum(action_probs[valid_actions]))
                else:
                    break
            
            move = board.action_to_move(action)
            if move and board.make_move(move):
                thread_mcts.update_with_move(action)
                move_count += 1
            else:
                break
        
        game_result = board.get_result()
        if game_result is None:
            game_result = 0
        
        return self._process_game_history(game_history, game_result)
    
    def _process_game_history(self, 
                             game_history: List[Dict], 
                             game_result: int) -> List[Tuple[np.ndarray, np.ndarray, float]]:
        """
        处理游戏历史，转换为训练样本
        
        Args:
            game_history: 游戏历史
            game_result: 游戏结果
            
        Returns:
            List[Tuple[np.ndarray, np.ndarray, float]]: 训练样本列表
        """
        training_examples = []
        
        for i, history_item in enumerate(game_history):
            board = history_item['board']
            action_probs = history_item['action_probs']
            player = history_item['player']
            
            # 计算从当前玩家视角的游戏结果
            if game_result == 0:
                value = 0.0  # 平局
            elif game_result == player:
                value = 1.0  # 当前玩家获胜
            else:
                value = -1.0  # 当前玩家失败
            
            training_examples.append((board, action_probs, value))
        
        return training_examples
    
    def _augment_data(self, 
                     training_examples: List[Tuple[np.ndarray, np.ndarray, float]]) -> List[Tuple[np.ndarray, np.ndarray, float]]:
        """
        数据增强（使用棋盘对称性）
        
        Args:
            training_examples: 原始训练样本
            
        Returns:
            List[Tuple[np.ndarray, np.ndarray, float]]: 增强后的训练样本
        """
        augmented_examples = []
        
        for board, action_probs, value in training_examples:
            # 获取对称形式
            symmetries = self.game.get_symmetries(board, action_probs)
            
            # 添加对称样本（除了原始样本）
            for sym_board, sym_probs in symmetries[1:]:  # 跳过第一个（原始样本）
                augmented_examples.append((sym_board, sym_probs, value))
        
        return augmented_examples
    
    def _update_stats(self, 
                     results: List[int], 
                     lengths: List[int], 
                     total_time: float):
        """
        更新统计信息
        
        Args:
            results: 游戏结果列表
            lengths: 游戏长度列表
            total_time: 总时间
        """
        with self._lock:
            self.stats['games_played'] += len(results)
            self.stats['total_moves'] += sum(lengths)
            self.stats['white_wins'] += sum(1 for r in results if r == 1)
            self.stats['black_wins'] += sum(1 for r in results if r == -1)
            self.stats['draws'] += sum(1 for r in results if r == 0)
            self.stats['total_time'] += total_time
            
            if len(lengths) > 0:
                self.stats['avg_game_length'] = sum(lengths) / len(lengths)
    
    def evaluate_against_previous(self, 
                                 previous_net: AlphaZeroNet,
                                 num_games: int = 100,
                                 verbose: bool = False) -> Dict[str, Any]:
        """
        与之前版本的模型对弈评估
        
        Args:
            previous_net: 之前的神经网络
            num_games: 评估游戏数量
            verbose: 是否输出详细信息
            
        Returns:
            Dict[str, Any]: 评估结果
        """
        self.logger.info(f"开始模型对比评估: {num_games} 局游戏")
        
        # 创建对手MCTS
        opponent_mcts = MCTS(previous_net, self.config, self.game)
        
        results = []
        game_lengths = []
        
        for game_id in range(num_games):
            board = self.game.get_init_board()
            move_count = 0
            
            # 随机决定谁是白方
            current_is_white = (game_id % 2 == 0)
            
            while not board.is_game_over() and move_count < 500:
                current_player = board.get_current_player()
                
                # 决定使用哪个模型
                if (current_player == 1 and current_is_white) or (current_player == -1 and not current_is_white):
                    # 当前模型
                    action = self.mcts.get_best_action(board, temperature=0.0)
                else:
                    # 对手模型
                    action = opponent_mcts.get_best_action(board, temperature=0.0)
                
                # 执行动作
                move = board.action_to_move(action)
                if move and board.make_move(move):
                    move_count += 1
                else:
                    break
            
            # 记录结果
            game_result = board.get_result()
            if game_result is None:
                game_result = 0
            
            # 从当前模型的视角计算结果
            if current_is_white:
                model_result = game_result
            else:
                model_result = -game_result
            
            results.append(model_result)
            game_lengths.append(move_count)
            
            if verbose and (game_id + 1) % 20 == 0:
                wins = sum(1 for r in results if r == 1)
                win_rate = wins / len(results)
                self.logger.info(f"评估进度: {game_id + 1}/{num_games}, 胜率: {win_rate:.3f}")
        
        # 计算统计结果
        wins = sum(1 for r in results if r == 1)
        losses = sum(1 for r in results if r == -1)
        draws = sum(1 for r in results if r == 0)
        
        evaluation_result = {
            'total_games': num_games,
            'wins': wins,
            'losses': losses,
            'draws': draws,
            'win_rate': wins / num_games,
            'loss_rate': losses / num_games,
            'draw_rate': draws / num_games,
            'avg_game_length': sum(game_lengths) / len(game_lengths)
        }
        
        self.logger.info(
            f"评估完成: 胜率 {evaluation_result['win_rate']:.3f}, "
            f"平局率 {evaluation_result['draw_rate']:.3f}"
        )
        
        return evaluation_result
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        获取统计信息
        
        Returns:
            Dict[str, Any]: 统计信息
        """
        stats = self.stats.copy()
        
        if stats['games_played'] > 0:
            stats['white_win_rate'] = stats['white_wins'] / stats['games_played']
            stats['black_win_rate'] = stats['black_wins'] / stats['games_played']
            stats['draw_rate'] = stats['draws'] / stats['games_played']
            stats['avg_time_per_game'] = stats['total_time'] / stats['games_played']
        else:
            stats['white_win_rate'] = 0.0
            stats['black_win_rate'] = 0.0
            stats['draw_rate'] = 0.0
            stats['avg_time_per_game'] = 0.0
        
        stats['total_training_examples'] = len(self.training_examples)
        
        return stats
    
    def reset_statistics(self):
        """重置统计信息"""
        self.stats = {
            'games_played': 0,
            'total_moves': 0,
            'white_wins': 0,
            'black_wins': 0,
            'draws': 0,
            'avg_game_length': 0.0,
            'total_time': 0.0
        }
        self.training_examples.clear()
    
    def save_training_data(self, filepath: str) -> bool:
        """
        保存训练数据
        
        Args:
            filepath: 保存路径
            
        Returns:
            bool: 是否成功保存
        """
        try:
            import pickle
            import os
            
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            
            with open(filepath, 'wb') as f:
                pickle.dump(self.training_examples, f)
            
            self.logger.info(f"训练数据已保存到: {filepath}")
            return True
            
        except Exception as e:
            self.logger.error(f"保存训练数据失败: {e}")
            return False
    
    def load_training_data(self, filepath: str) -> bool:
        """
        加载训练数据
        
        Args:
            filepath: 文件路径
            
        Returns:
            bool: 是否成功加载
        """
        try:
            import pickle
            
            with open(filepath, 'rb') as f:
                self.training_examples = pickle.load(f)
            
            self.logger.info(f"训练数据已从 {filepath} 加载, {len(self.training_examples)} 个样本")
            return True
            
        except Exception as e:
            self.logger.error(f"加载训练数据失败: {e}")
            return False
    
    def __str__(self) -> str:
        """字符串表示"""
        stats = self.get_statistics()
        return (f"SelfPlay(游戏数={stats['games_played']}, "
                f"训练样本={stats['total_training_examples']}, "
                f"平均长度={stats['avg_game_length']:.1f})")
    
    def __repr__(self) -> str:
        """调试表示"""
        return self.__str__() 