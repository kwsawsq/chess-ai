"""
AlphaZero 训练器
"""
import os
import time
import json
from typing import Dict, List, Any, Optional, Tuple
import numpy as np
from tqdm import tqdm

from ..neural_network import AlphaZeroNet, NetworkTrainer
from ..self_play import SelfPlay
from ..evaluation import ModelEvaluator
from .data_manager import DataManager
from config.config import config


class AlphaZeroTrainer:
    """AlphaZero 主训练器"""
    
    def __init__(self, 
                 model: Optional[AlphaZeroNet] = None,
                 device: str = 'cuda',
                 config_dict: Optional[Dict] = None):
        """
        初始化训练器
        
        Args:
            model: 初始模型，None表示创建新模型
            device: 训练设备
            config_dict: 配置字典，覆盖默认配置
        """
        # 应用配置
        self.config = config
        if config_dict:
            for key, value in config_dict.items():
                setattr(self.config, key, value)
        
        # 创建必要的目录
        self.config.create_dirs()
        
        # 初始化模型
        if model is None:
            self.model = AlphaZeroNet(
                input_channels=20,
                num_channels=self.config.NUM_CHANNELS,
                num_residual_blocks=self.config.NUM_RESIDUAL_BLOCKS,
                action_size=self.config.ACTION_SIZE,
                dropout_rate=self.config.DROPOUT_RATE
            )
        else:
            self.model = model
        
        # 设备配置
        self.device = device
        if device == 'cuda' and not torch.cuda.is_available():
            self.device = 'cpu'
            print("CUDA不可用，使用CPU训练")
        
        # 初始化组件
        self.network_trainer = NetworkTrainer(
            model=self.model,
            device=self.device,
            learning_rate=self.config.LEARNING_RATE,
            weight_decay=self.config.WEIGHT_DECAY
        )
        
        self.self_play = SelfPlay(
            model=self.model,
            mcts_simulations=self.config.NUM_MCTS_SIMS,
            c_puct=self.config.CPUCT,
            temperature_threshold=self.config.TEMPERATURE_THRESHOLD,
            dirichlet_alpha=self.config.DIRICHLET_ALPHA,
            dirichlet_epsilon=self.config.DIRICHLET_EPSILON,
            num_workers=self.config.NUM_WORKERS
        )
        
        self.data_manager = DataManager(
            max_buffer_size=self.config.MAX_REPLAY_BUFFER_SIZE,
            data_dir=self.config.DATA_DIR,
            auto_save=True,
            save_interval=self.config.SAVE_INTERVAL
        )
        
        self.evaluator = ModelEvaluator(
            num_eval_games=self.config.NUM_EVAL_GAMES,
            mcts_simulations=self.config.NUM_MCTS_SIMS,
            c_puct=self.config.CPUCT
        )
        
        # 训练状态
        self.iteration = 0
        self.best_model_path = None
        self.training_history = {
            'iterations': [],
            'self_play_games': [],
            'training_loss': [],
            'win_rates': [],
            'model_updates': []
        }
    
    def train(self, 
              num_iterations: int = 1000,
              self_play_games_per_iteration: int = 100,
              training_epochs_per_iteration: int = 10,
              evaluation_interval: int = 10,
              save_interval: int = 50,
              verbose: bool = True) -> Dict[str, Any]:
        """
        执行完整的训练循环
        
        Args:
            num_iterations: 训练迭代次数
            self_play_games_per_iteration: 每次迭代的自我对弈游戏数
            training_epochs_per_iteration: 每次迭代的训练轮数
            evaluation_interval: 评估间隔
            save_interval: 保存间隔
            verbose: 是否输出详细信息
            
        Returns:
            训练历史
        """
        print(f"开始AlphaZero训练")
        print(f"设备: {self.device}")
        print(f"模型参数: {self.model.get_model_size():,}")
        print(f"训练迭代: {num_iterations}")
        print(f"每次迭代游戏数: {self_play_games_per_iteration}")
        print("-" * 50)
        
        start_time = time.time()
        
        for iteration in range(num_iterations):
            self.iteration = iteration
            iteration_start_time = time.time()
            
            if verbose:
                print(f"\n=== 迭代 {iteration + 1}/{num_iterations} ===")
            
            # 1. 自我对弈
            self_play_start_time = time.time()
            games_data = self._self_play_phase(
                self_play_games_per_iteration, 
                verbose
            )
            self_play_time = time.time() - self_play_start_time
            
            # 2. 训练神经网络
            training_start_time = time.time()
            training_stats = self._training_phase(
                training_epochs_per_iteration, 
                verbose
            )
            training_time = time.time() - training_start_time
            
            # 3. 评估和模型更新
            evaluation_stats = {}
            if (iteration + 1) % evaluation_interval == 0:
                evaluation_start_time = time.time()
                evaluation_stats = self._evaluation_phase(verbose)
                evaluation_time = time.time() - evaluation_start_time
            
            # 4. 保存模型和数据
            if (iteration + 1) % save_interval == 0:
                self._save_checkpoint(iteration + 1)
            
            # 5. 记录历史
            iteration_time = time.time() - iteration_start_time
            self._record_iteration_stats(
                iteration + 1, 
                games_data, 
                training_stats, 
                evaluation_stats,
                {
                    'self_play_time': self_play_time,
                    'training_time': training_time,
                    'evaluation_time': evaluation_stats.get('evaluation_time', 0),
                    'total_time': iteration_time
                }
            )
            
            if verbose:
                self._print_iteration_summary(
                    iteration + 1, 
                    training_stats, 
                    evaluation_stats,
                    iteration_time
                )
        
        total_time = time.time() - start_time
        
        # 保存最终模型
        final_model_path = os.path.join(
            self.config.MODEL_DIR, 
            f"final_model_iteration_{num_iterations}.pth"
        )
        self.model.save_checkpoint(final_model_path)
        
        # 保存训练历史
        self._save_training_history()
        
        print(f"\n训练完成！总时间: {total_time:.2f}秒")
        print(f"最终模型保存到: {final_model_path}")
        
        return self.training_history
    
    def _self_play_phase(self, num_games: int, verbose: bool) -> List[Dict[str, Any]]:
        """
        自我对弈阶段
        
        Args:
            num_games: 游戏数量
            verbose: 是否输出详细信息
            
        Returns:
            游戏数据
        """
        if verbose:
            print(f"开始自我对弈 ({num_games} 局)...")
        
        # 更新自我对弈模型
        self.self_play.update_model(self.model)
        
        # 进行自我对弈
        games_data = self.self_play.play_games(num_games, verbose=False)
        
        # 生成训练数据
        states, policies, values = self.self_play.generate_training_data(games_data)
        
        # 添加到数据管理器
        self.data_manager.add_games_data(games_data)
        
        if verbose:
            stats = self.self_play.get_statistics()
            print(f"自我对弈完成: 白方胜率={stats['white_win_rate']:.3f}, "
                  f"黑方胜率={stats['black_win_rate']:.3f}, "
                  f"平局率={stats['draw_rate']:.3f}")
            print(f"平均游戏长度: {stats['average_game_length']:.1f}")
            print(f"训练样本数: {len(states)}")
        
        return games_data
    
    def _training_phase(self, epochs: int, verbose: bool) -> Dict[str, Any]:
        """
        训练阶段
        
        Args:
            epochs: 训练轮数
            verbose: 是否输出详细信息
            
        Returns:
            训练统计信息
        """
        if verbose:
            print(f"开始训练神经网络 ({epochs} 轮)...")
        
        # 获取训练数据
        train_data, val_data, _ = self.data_manager.split_data(
            train_ratio=0.8, 
            val_ratio=0.2, 
            test_ratio=0.0
        )
        
        if len(train_data[0]) == 0:
            print("警告: 没有训练数据，跳过训练")
            return {}
        
        # 训练网络
        history = self.network_trainer.train(
            train_data=train_data,
            val_data=val_data,
            epochs=epochs,
            batch_size=self.config.BATCH_SIZE,
            tensorboard_dir=self.config.TENSORBOARD_LOG_DIR,
            save_dir=self.config.CHECKPOINT_DIR,
            save_interval=epochs  # 每轮迭代结束后保存
        )
        
        if verbose and history:
            print(f"训练完成: 损失={history['train_loss'][-1]:.4f}")
            if history['val_loss']:
                print(f"验证损失={history['val_loss'][-1]:.4f}")
        
        return history
    
    def _evaluation_phase(self, verbose: bool) -> Dict[str, Any]:
        """
        评估阶段
        
        Args:
            verbose: 是否输出详细信息
            
        Returns:
            评估统计信息
        """
        if verbose:
            print("开始模型评估...")
        
        evaluation_stats = {}
        
        # 如果有之前的最佳模型，进行对比
        if self.best_model_path and os.path.exists(self.best_model_path):
            try:
                # 加载之前的最佳模型
                previous_model = AlphaZeroNet.load_from_checkpoint(self.best_model_path)
                
                # 进行对弈评估
                battle_results = self.self_play.play_against_previous_version(
                    previous_model, 
                    num_games=self.config.NUM_EVAL_GAMES
                )
                
                evaluation_stats.update(battle_results)
                
                # 如果新模型表现更好，更新最佳模型
                if battle_results['win_rate'] > self.config.EVAL_WIN_RATE_THRESHOLD:
                    self.best_model_path = self._save_best_model()
                    evaluation_stats['model_updated'] = True
                    
                    if verbose:
                        print(f"新模型表现更好！胜率: {battle_results['win_rate']:.3f}")
                        print(f"更新最佳模型")
                else:
                    evaluation_stats['model_updated'] = False
                    
                    if verbose:
                        print(f"新模型表现一般。胜率: {battle_results['win_rate']:.3f}")
                        
            except Exception as e:
                print(f"评估过程中出错: {e}")
                evaluation_stats['error'] = str(e)
        else:
            # 没有之前的模型，直接保存当前模型为最佳
            self.best_model_path = self._save_best_model()
            evaluation_stats['model_updated'] = True
            
            if verbose:
                print("保存当前模型为最佳模型")
        
        return evaluation_stats
    
    def _save_best_model(self) -> str:
        """
        保存最佳模型
        
        Returns:
            保存路径
        """
        best_model_path = os.path.join(
            self.config.MODEL_DIR, 
            f"best_model_iteration_{self.iteration + 1}.pth"
        )
        self.model.save_checkpoint(best_model_path)
        return best_model_path
    
    def _save_checkpoint(self, iteration: int):
        """
        保存检查点
        
        Args:
            iteration: 迭代次数
        """
        checkpoint_path = os.path.join(
            self.config.CHECKPOINT_DIR, 
            f"checkpoint_iteration_{iteration}.pth"
        )
        self.model.save_checkpoint(checkpoint_path)
        
        # 保存训练状态
        training_state = {
            'iteration': iteration,
            'model_path': checkpoint_path,
            'best_model_path': self.best_model_path,
            'training_history': self.training_history,
            'data_manager_stats': self.data_manager.get_statistics(),
            'config': vars(self.config)
        }
        
        state_path = os.path.join(
            self.config.CHECKPOINT_DIR, 
            f"training_state_iteration_{iteration}.json"
        )
        
        with open(state_path, 'w') as f:
            json.dump(training_state, f, indent=2)
    
    def _record_iteration_stats(self, 
                               iteration: int, 
                               games_data: List[Dict[str, Any]], 
                               training_stats: Dict[str, Any],
                               evaluation_stats: Dict[str, Any],
                               timing_stats: Dict[str, float]):
        """
        记录迭代统计信息
        
        Args:
            iteration: 迭代次数
            games_data: 游戏数据
            training_stats: 训练统计
            evaluation_stats: 评估统计
            timing_stats: 时间统计
        """
        self.training_history['iterations'].append(iteration)
        self.training_history['self_play_games'].append(len(games_data))
        
        if training_stats and 'train_loss' in training_stats:
            self.training_history['training_loss'].append(training_stats['train_loss'][-1])
        
        if evaluation_stats and 'win_rate' in evaluation_stats:
            self.training_history['win_rates'].append(evaluation_stats['win_rate'])
        
        if evaluation_stats and 'model_updated' in evaluation_stats:
            self.training_history['model_updates'].append(evaluation_stats['model_updated'])
    
    def _print_iteration_summary(self, 
                                iteration: int, 
                                training_stats: Dict[str, Any],
                                evaluation_stats: Dict[str, Any],
                                total_time: float):
        """
        打印迭代总结
        
        Args:
            iteration: 迭代次数
            training_stats: 训练统计
            evaluation_stats: 评估统计
            total_time: 总时间
        """
        print(f"迭代 {iteration} 完成 (耗时: {total_time:.2f}秒)")
        
        if training_stats:
            print(f"  训练损失: {training_stats.get('train_loss', [0])[-1]:.4f}")
        
        if evaluation_stats:
            if 'win_rate' in evaluation_stats:
                print(f"  评估胜率: {evaluation_stats['win_rate']:.3f}")
            if 'model_updated' in evaluation_stats:
                print(f"  模型更新: {'是' if evaluation_stats['model_updated'] else '否'}")
        
        # 数据统计
        data_stats = self.data_manager.get_statistics()
        print(f"  数据缓冲区: {data_stats['buffer_size']} 局游戏")
        print(f"  总训练样本: {data_stats['total_samples']}")
    
    def _save_training_history(self):
        """保存训练历史"""
        history_path = os.path.join(
            self.config.LOG_DIR, 
            f"training_history_{int(time.time())}.json"
        )
        
        with open(history_path, 'w') as f:
            json.dump(self.training_history, f, indent=2)
    
    def resume_training(self, checkpoint_path: str):
        """
        从检查点恢复训练
        
        Args:
            checkpoint_path: 检查点路径
        """
        # 加载模型
        self.model.load_checkpoint(checkpoint_path)
        
        # 加载训练状态
        state_path = checkpoint_path.replace('.pth', '_state.json')
        if os.path.exists(state_path):
            with open(state_path, 'r') as f:
                training_state = json.load(f)
            
            self.iteration = training_state['iteration']
            self.best_model_path = training_state['best_model_path']
            self.training_history = training_state['training_history']
            
            print(f"从迭代 {self.iteration} 恢复训练")
        else:
            print("警告: 无法找到训练状态文件")
    
    def get_current_model(self) -> AlphaZeroNet:
        """获取当前模型"""
        return self.model
    
    def get_best_model(self) -> Optional[AlphaZeroNet]:
        """获取最佳模型"""
        if self.best_model_path and os.path.exists(self.best_model_path):
            return AlphaZeroNet.load_from_checkpoint(self.best_model_path)
        return None
    
    def get_training_statistics(self) -> Dict[str, Any]:
        """获取训练统计信息"""
        return {
            'iteration': self.iteration,
            'training_history': self.training_history,
            'data_manager_stats': self.data_manager.get_statistics(),
            'self_play_stats': self.self_play.get_statistics(),
            'model_info': self.model.get_model_info()
        }
    
    def __str__(self) -> str:
        """字符串表示"""
        return f"AlphaZeroTrainer(iteration={self.iteration}, device={self.device})"
    
    def __repr__(self) -> str:
        """调试表示"""
        return self.__str__() 