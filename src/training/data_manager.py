"""
训练数据管理器
"""
import os
import pickle
import numpy as np
from typing import List, Dict, Tuple, Optional, Any
from collections import deque
import random
import time


class DataManager:
    """训练数据管理器"""
    
    def __init__(self, 
                 max_buffer_size: int = 10000,
                 data_dir: str = "data",
                 auto_save: bool = True,
                 save_interval: int = 100):
        """
        初始化数据管理器
        
        Args:
            max_buffer_size: 最大缓冲区大小
            data_dir: 数据目录
            auto_save: 是否自动保存
            save_interval: 保存间隔（游戏数）
        """
        self.max_buffer_size = max_buffer_size
        self.data_dir = data_dir
        self.auto_save = auto_save
        self.save_interval = save_interval
        
        # 创建数据目录
        os.makedirs(data_dir, exist_ok=True)
        
        # 数据缓冲区
        self.replay_buffer = deque(maxlen=max_buffer_size)
        
        # 统计信息
        self.stats = {
            'total_games': 0,
            'total_samples': 0,
            'buffer_size': 0,
            'saved_files': 0
        }
        
        # 文件索引
        self.file_index = 0
        
    def add_games_data(self, games_data: List[Dict[str, Any]]):
        """
        添加游戏数据到缓冲区
        
        Args:
            games_data: 游戏数据列表
        """
        for game_data in games_data:
            self.replay_buffer.append(game_data)
            self.stats['total_games'] += 1
            self.stats['total_samples'] += len(game_data['states'])
        
        self.stats['buffer_size'] = len(self.replay_buffer)
        
        # 自动保存
        if self.auto_save and self.stats['total_games'] % self.save_interval == 0:
            self.save_buffer()
    
    def get_training_data(self, 
                         num_samples: Optional[int] = None,
                         recent_games_only: bool = False,
                         recent_ratio: float = 0.5) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        获取训练数据
        
        Args:
            num_samples: 样本数量，None表示使用所有数据
            recent_games_only: 是否只使用最近的游戏
            recent_ratio: 最近游戏的比例
            
        Returns:
            (states, policies, values) 训练数据
        """
        if not self.replay_buffer:
            return np.array([]), np.array([]), np.array([])
        
        # 选择游戏数据
        if recent_games_only:
            num_recent = int(len(self.replay_buffer) * recent_ratio)
            selected_games = list(self.replay_buffer)[-num_recent:]
        else:
            selected_games = list(self.replay_buffer)
        
        # 提取训练样本
        all_states = []
        all_policies = []
        all_values = []
        
        for game_data in selected_games:
            states = game_data['states']
            policies = game_data['mcts_policies']
            values = game_data['values']
            
            for i in range(len(states)):
                all_states.append(states[i])
                all_policies.append(policies[i])
                all_values.append(values[i])
        
        # 转换为numpy数组
        states_array = np.array(all_states)
        policies_array = np.array(all_policies)
        values_array = np.array(all_values)
        
        # 随机采样
        if num_samples is not None and num_samples < len(states_array):
            indices = np.random.choice(len(states_array), num_samples, replace=False)
            states_array = states_array[indices]
            policies_array = policies_array[indices]
            values_array = values_array[indices]
        
        return states_array, policies_array, values_array
    
    def get_batch(self, batch_size: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        获取一个批次的数据
        
        Args:
            batch_size: 批次大小
            
        Returns:
            (states, policies, values) 批次数据
        """
        states, policies, values = self.get_training_data()
        
        if len(states) == 0:
            return np.array([]), np.array([]), np.array([])
        
        # 随机采样
        if batch_size < len(states):
            indices = np.random.choice(len(states), batch_size, replace=False)
            return states[indices], policies[indices], values[indices]
        else:
            return states, policies, values
    
    def split_data(self, 
                   train_ratio: float = 0.8,
                   val_ratio: float = 0.1,
                   test_ratio: float = 0.1) -> Tuple[Tuple[np.ndarray, np.ndarray, np.ndarray], ...]:
        """
        分割数据为训练、验证、测试集
        
        Args:
            train_ratio: 训练集比例
            val_ratio: 验证集比例
            test_ratio: 测试集比例
            
        Returns:
            (train_data, val_data, test_data)
        """
        states, policies, values = self.get_training_data()
        
        if len(states) == 0:
            empty_data = (np.array([]), np.array([]), np.array([]))
            return empty_data, empty_data, empty_data
        
        # 随机打乱
        indices = np.random.permutation(len(states))
        states = states[indices]
        policies = policies[indices]
        values = values[indices]
        
        # 计算分割点
        n_samples = len(states)
        train_end = int(n_samples * train_ratio)
        val_end = int(n_samples * (train_ratio + val_ratio))
        
        # 分割数据
        train_data = (states[:train_end], policies[:train_end], values[:train_end])
        val_data = (states[train_end:val_end], policies[train_end:val_end], values[train_end:val_end])
        test_data = (states[val_end:], policies[val_end:], values[val_end:])
        
        return train_data, val_data, test_data
    
    def save_buffer(self, filename: Optional[str] = None):
        """
        保存缓冲区数据
        
        Args:
            filename: 文件名，None表示自动生成
        """
        if not self.replay_buffer:
            return
        
        if filename is None:
            timestamp = int(time.time())
            filename = f"games_data_{timestamp}_{self.file_index}.pkl"
            self.file_index += 1
        
        filepath = os.path.join(self.data_dir, filename)
        
        # 保存数据
        with open(filepath, 'wb') as f:
            pickle.dump(list(self.replay_buffer), f)
        
        self.stats['saved_files'] += 1
        print(f"保存数据到: {filepath}")
        
        return filepath
    
    def load_buffer(self, filepath: str):
        """
        加载缓冲区数据
        
        Args:
            filepath: 文件路径
        """
        if not os.path.exists(filepath):
            print(f"文件不存在: {filepath}")
            return
        
        with open(filepath, 'rb') as f:
            games_data = pickle.load(f)
        
        # 添加到缓冲区
        self.add_games_data(games_data)
        print(f"加载数据从: {filepath}, 游戏数: {len(games_data)}")
    
    def load_all_data(self):
        """加载目录中的所有数据文件"""
        if not os.path.exists(self.data_dir):
            print(f"数据目录不存在: {self.data_dir}")
            return
        
        data_files = [f for f in os.listdir(self.data_dir) if f.endswith('.pkl')]
        
        for filename in data_files:
            filepath = os.path.join(self.data_dir, filename)
            self.load_buffer(filepath)
    
    def clear_buffer(self):
        """清空缓冲区"""
        self.replay_buffer.clear()
        self.stats['buffer_size'] = 0
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        获取统计信息
        
        Returns:
            统计信息字典
        """
        self.stats['buffer_size'] = len(self.replay_buffer)
        
        # 计算数据分布
        if self.replay_buffer:
            game_lengths = [len(game['states']) for game in self.replay_buffer]
            results = [game['result'] for game in self.replay_buffer]
            
            self.stats.update({
                'avg_game_length': np.mean(game_lengths),
                'min_game_length': np.min(game_lengths),
                'max_game_length': np.max(game_lengths),
                'white_wins': sum(1 for r in results if r == 1.0),
                'black_wins': sum(1 for r in results if r == -1.0),
                'draws': sum(1 for r in results if r == 0.0),
            })
        
        return self.stats.copy()
    
    def get_recent_games(self, num_games: int) -> List[Dict[str, Any]]:
        """
        获取最近的游戏数据
        
        Args:
            num_games: 游戏数量
            
        Returns:
            最近的游戏数据列表
        """
        if not self.replay_buffer:
            return []
        
        return list(self.replay_buffer)[-num_games:]
    
    def sample_games(self, num_games: int) -> List[Dict[str, Any]]:
        """
        随机采样游戏数据
        
        Args:
            num_games: 游戏数量
            
        Returns:
            随机采样的游戏数据列表
        """
        if not self.replay_buffer:
            return []
        
        if num_games >= len(self.replay_buffer):
            return list(self.replay_buffer)
        
        return random.sample(list(self.replay_buffer), num_games)
    
    def remove_old_games(self, keep_recent: int):
        """
        移除旧的游戏数据
        
        Args:
            keep_recent: 保留最近的游戏数量
        """
        if len(self.replay_buffer) > keep_recent:
            # 转换为列表，保留最近的数据
            recent_games = list(self.replay_buffer)[-keep_recent:]
            self.replay_buffer.clear()
            self.replay_buffer.extend(recent_games)
            
            self.stats['buffer_size'] = len(self.replay_buffer)
    
    def export_data(self, filepath: str, format: str = 'numpy'):
        """
        导出数据到文件
        
        Args:
            filepath: 文件路径
            format: 文件格式 ('numpy', 'pickle', 'json')
        """
        states, policies, values = self.get_training_data()
        
        if format == 'numpy':
            np.savez_compressed(filepath, 
                              states=states, 
                              policies=policies, 
                              values=values)
        elif format == 'pickle':
            with open(filepath, 'wb') as f:
                pickle.dump({
                    'states': states,
                    'policies': policies,
                    'values': values
                }, f)
        elif format == 'json':
            import json
            data = {
                'states': states.tolist(),
                'policies': policies.tolist(),
                'values': values.tolist()
            }
            with open(filepath, 'w') as f:
                json.dump(data, f)
        else:
            raise ValueError(f"不支持的格式: {format}")
    
    def import_data(self, filepath: str, format: str = 'numpy'):
        """
        从文件导入数据
        
        Args:
            filepath: 文件路径
            format: 文件格式 ('numpy', 'pickle', 'json')
        """
        if format == 'numpy':
            data = np.load(filepath)
            states = data['states']
            policies = data['policies']
            values = data['values']
        elif format == 'pickle':
            with open(filepath, 'rb') as f:
                data = pickle.load(f)
            states = data['states']
            policies = data['policies']
            values = data['values']
        elif format == 'json':
            import json
            with open(filepath, 'r') as f:
                data = json.load(f)
            states = np.array(data['states'])
            policies = np.array(data['policies'])
            values = np.array(data['values'])
        else:
            raise ValueError(f"不支持的格式: {format}")
        
        # 转换为游戏数据格式
        games_data = []
        for i in range(len(states)):
            game_data = {
                'states': [states[i]],
                'mcts_policies': [policies[i]],
                'values': [values[i]],
                'players': [1],  # 假设为白方
                'result': values[i],
                'game_length': 1,
                'game_id': i
            }
            games_data.append(game_data)
        
        self.add_games_data(games_data)
    
    def get_data_quality_metrics(self) -> Dict[str, float]:
        """
        获取数据质量指标
        
        Returns:
            数据质量指标
        """
        if not self.replay_buffer:
            return {}
        
        states, policies, values = self.get_training_data()
        
        metrics = {
            'policy_entropy': np.mean([-np.sum(p * np.log(p + 1e-8)) for p in policies]),
            'value_mean': np.mean(values),
            'value_std': np.std(values),
            'value_range': np.max(values) - np.min(values),
            'policy_sparsity': np.mean(np.sum(policies > 0.01, axis=1)),
        }
        
        return metrics

    def get_last_iteration_data(self) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """获取最后一个迭代（批次）的数据"""
        if not self.replay_buffer:
            return None
        
        last_game_data = self.replay_buffer[-1]
        states = np.array(last_game_data['states'])
        policies = np.array(last_game_data['mcts_policies'])
        values = np.array(last_game_data['values'])
        
        return states, policies, values

    def __len__(self) -> int:
        """返回缓冲区大小"""
        return self.stats['buffer_size']
    
    def __str__(self) -> str:
        """字符串表示"""
        return f"DataManager(buffer_size={len(self.replay_buffer)}, total_games={self.stats['total_games']})"
    
    def __repr__(self) -> str:
        """调试表示"""
        return self.__str__() 