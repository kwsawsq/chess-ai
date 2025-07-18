"""
评估模块
实现模型评估和可视化功能
"""

import os
import json
import logging
from typing import Dict, List, Any, Optional, Tuple
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
import torch
import multiprocessing as mp

from ..neural_network import AlphaZeroNet
from ..game import ChessGame
from ..mcts import MCTS

# --- Worker Process Initialization ---
g_new_model: Optional[AlphaZeroNet] = None
g_old_model: Optional[AlphaZeroNet] = None
g_config: Optional[Any] = None

def init_worker_evaluate(new_model_state: Dict, old_model_state: Dict, config: Any):
    """
    初始化评估工作进程。
    这个函数在每个工作进程启动时只运行一次。
    """
    global g_new_model, g_old_model, g_config
    
    g_config = config
    
    # 禁用梯度计算以进行推理优化
    torch.set_grad_enabled(False)

    # 在新进程中重新创建和加载模型
    g_new_model = AlphaZeroNet(g_config)
    g_new_model.load_state_dict(new_model_state)
    g_new_model.to(g_config.DEVICE)
    g_new_model.eval()

    g_old_model = AlphaZeroNet(g_config)
    g_old_model.load_state_dict(old_model_state)
    g_old_model.to(g_config.DEVICE)
    g_old_model.eval()

def _play_one_game_worker(new_model_plays_white: bool) -> int:
    """
    在工作进程中使用已初始化的模型进行一局对弈。

    Args:
        new_model_plays_white: 新模型是否执白

    Returns:
        int: 从白方角度看的游戏结果 (1: 白胜, 0: 平局, -1: 黑胜)
    """
    global g_new_model, g_old_model, g_config
    
    game = ChessGame(g_config)
    
    # 为评估创建MCTS实例
    eval_mcts_config = g_config
    eval_mcts_config.NUM_MCTS_SIMS = g_config.NUM_MCTS_SIMS_EVAL
    
    mcts_new = MCTS(g_new_model, eval_mcts_config)
    mcts_old = MCTS(g_old_model, eval_mcts_config)
    
    models = {1: mcts_new, -1: mcts_old} if new_model_plays_white else {1: mcts_old, -1: mcts_new}

    while not game.is_over():
        player = game.get_current_player()
        mcts = models[player]
        
        policy, _ = mcts.search(game.board)
        
        # 在评估中使用确定性走法（选择概率最高的）
        move = game.select_move(policy, deterministic=True)
        game.make_move(move)

    return game.get_result()

# --- End Worker Process Initialization ---

class Evaluator:
    """
    模型评估器
    负责评估模型性能并生成可视化报告
    """
    
    def __init__(self, config):
        """
        初始化评估器
        
        Args:
            config: 配置对象
        """
        self.config = config
        
        # 创建评估结果目录
        self.eval_dir = os.path.join(config.LOG_DIR, 'evaluation')
        os.makedirs(self.eval_dir, exist_ok=True)
        
        # 日志设置
        self.logger = logging.getLogger(__name__)
        self._setup_logging()
        
        # 评估历史
        self.history: List[Dict[str, Any]] = []

    def evaluate(self, 
                 model: AlphaZeroNet, 
                 benchmark_model: AlphaZeroNet, 
                 num_games: int = 10) -> Tuple[float, float, float]:
        """
        评估模型性能 (使用优化的并行工作进程)

        Args:
            model: 待评估的新模型
            benchmark_model: 基准模型
            num_games: 评估游戏局数
            
        Returns:
            Tuple[float, float, float]: (新模型胜率, 平局率, 新模型败率)
        """
        wins, draws, losses = 0, 0, 0
        
        # 将模型移到CPU以安全地获取state_dict，避免多进程fork问题
        model.to('cpu')
        benchmark_model.to('cpu')
        new_model_state = model.state_dict()
        old_model_state = benchmark_model.state_dict()
            
        initargs = (new_model_state, old_model_state, self.config)

        with ProcessPoolExecutor(max_workers=self.config.NUM_WORKERS,
                                 initializer=init_worker_evaluate,
                                 initargs=initargs) as executor:
            
            # 创建任务，一半新模型执白，一半执黑
            futures = {executor.submit(_play_one_game_worker, i < num_games / 2): (i < num_games / 2) for i in range(num_games)}

            progress_bar = tqdm(as_completed(futures), total=num_games, desc="模型评估")

            for future in progress_bar:
                new_model_was_white = futures[future]
                try:
                    # result 是从白方角度的结果 (1=白胜, -1=黑胜)
                    result = future.result()
                    
                    # 根据新模型执白/黑来判断胜负
                    if (new_model_was_white and result == 1) or \
                       (not new_model_was_white and result == -1):
                        wins += 1
                    elif result == 0:
                        draws += 1
                    else:
                        losses += 1

                    progress_bar.set_postfix({'wins': wins, 'draws': draws, 'losses': losses})
                except Exception as e:
                    self.logger.error(f"评估子进程出错: {e}", exc_info=True)

        # 评估结束后将模型移回原设备，以进行后续训练
        model.to(self.config.DEVICE)
        
        total = wins + draws + losses
        if total == 0:
            return 0.0, 0.0, 0.0
            
        return wins / total, draws / total, losses / total
    
    def _setup_logging(self):
        """设置日志"""
        log_dir = os.path.join(self.config.LOG_DIR, 'evaluation')
        os.makedirs(log_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_file = os.path.join(log_dir, f'evaluation_{timestamp}.log')
        
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)
    
    def evaluate_model(self,
                      model: AlphaZeroNet,
                      benchmark_model: Optional[AlphaZeroNet] = None,
                      num_games: int = 100,
                      save_games: bool = True) -> Dict[str, Any]:
        """
        评估模型性能
        
        Args:
            model: 待评估的模型
            benchmark_model: 基准模型（可选）
            num_games: 评估游戏局数
            save_games: 是否保存对弈记录
            
        Returns:
            Dict[str, Any]: 评估结果
        """
        # 延迟导入以避免循环依赖
        from ..self_play import SelfPlay
        
        self.logger.info(f"开始评估模型，对弈局数: {num_games}")
        
        # 创建游戏环境
        game = ChessGame(self.config)
        
        # 如果没有基准模型，创建一个新的随机初始化模型
        if benchmark_model is None:
            benchmark_model = AlphaZeroNet(self.config)
        
        win_rate, draw_rate, loss_rate = self.evaluate(model, benchmark_model, num_games)
        
        # 记录评估结果
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        eval_result = {
            'timestamp': timestamp,
            'num_games': num_games,
            'stats': {
                'win_rate': win_rate,
                'draw_rate': draw_rate,
                'loss_rate': loss_rate
            },
            'model_config': self.config
        }
        
        self.history.append(eval_result)
        
        # 保存评估结果
        if save_games:
            self._save_evaluation_result(eval_result)
        
        return eval_result
    
    def evaluate_model_strength(self,
                              model: AlphaZeroNet,
                              opponent_models: List[AlphaZeroNet],
                              num_games: int = 50) -> Dict[str, Any]:
        """
        评估模型相对强度
        
        Args:
            model: 待评估的模型
            opponent_models: 对手模型列表
            num_games: 每个对手的对弈局数
            
        Returns:
            Dict[str, Any]: 评估结果
        """
        self.logger.info(f"开始评估模型相对强度，对手数量: {len(opponent_models)}")
        
        results = []
        for i, opponent in enumerate(opponent_models):
            self.logger.info(f"对战对手 {i+1}/{len(opponent_models)}")
            result = self.evaluate_model(
                model,
                opponent,
                num_games=num_games,
                save_games=False
            )
            results.append(result)
        
        # 汇总结果
        summary = {
            'timestamp': datetime.now().strftime('%Y%m%d_%H%M%S'),
            'num_opponents': len(opponent_models),
            'num_games_per_opponent': num_games,
            'results': results,
            'average_win_rate': np.mean([r['stats']['win_rate'] for r in results])
        }
        
        return summary
    
    def plot_training_progress(self,
                             training_history: List[Dict[str, float]],
                             save_path: Optional[str] = None):
        """
        绘制训练进度图表
        
        Args:
            training_history: 训练历史数据
            save_path: 图表保存路径（可选）
        """
        # 提取数据
        iterations = list(range(1, len(training_history) + 1))
        policy_loss = [h['policy_loss'] for h in training_history]
        value_loss = [h['value_loss'] for h in training_history]
        total_loss = [h['total_loss'] for h in training_history]
        
        # 创建图表
        plt.figure(figsize=(12, 8))
        
        # 绘制损失曲线
        plt.subplot(2, 1, 1)
        plt.plot(iterations, policy_loss, label='策略损失', marker='o')
        plt.plot(iterations, value_loss, label='价值损失', marker='s')
        plt.plot(iterations, total_loss, label='总损失', marker='^')
        plt.xlabel('迭代次数')
        plt.ylabel('损失值')
        plt.title('训练损失曲线')
        plt.legend()
        plt.grid(True)
        
        # 如果有评估历史，绘制胜率曲线
        if self.history:
            plt.subplot(2, 1, 2)
            eval_iterations = [h['stats']['iteration'] for h in self.history]
            win_rates = [h['stats']['win_rate'] for h in self.history]
            plt.plot(eval_iterations, win_rates, label='胜率', marker='o', color='green')
            plt.xlabel('迭代次数')
            plt.ylabel('胜率')
            plt.title('模型胜率变化')
            plt.legend()
            plt.grid(True)
        
        plt.tight_layout()
        
        # 保存图表
        if save_path:
            plt.savefig(save_path)
            self.logger.info(f"训练进度图表已保存到: {save_path}")
        else:
            plt.show()
        
        plt.close()
    
    def _save_evaluation_result(self, result: Dict[str, Any]):
        """
        保存评估结果
        
        Args:
            result: 评估结果
        """
        # 创建评估结果目录
        timestamp = result['timestamp']
        result_dir = os.path.join(self.eval_dir, f'eval_{timestamp}')
        os.makedirs(result_dir, exist_ok=True)
        
        # 保存评估统计信息
        stats_file = os.path.join(result_dir, 'stats.json')
        with open(stats_file, 'w', encoding='utf-8') as f:
            json.dump(result['stats'], f, indent=2, ensure_ascii=False)
        
        self.logger.info(f"评估结果已保存到: {result_dir}")
    
    def generate_evaluation_report(self,
                                 model_name: str,
                                 eval_result: Dict[str, Any],
                                 save_path: Optional[str] = None) -> str:
        """
        生成评估报告
        
        Args:
            model_name: 模型名称
            eval_result: 评估结果
            save_path: 报告保存路径（可选）
            
        Returns:
            str: 报告文本
        """
        stats = eval_result['stats']
        timestamp = eval_result['timestamp']
        
        report = f"""
# AlphaZero模型评估报告

## 基本信息
- 模型名称: {model_name}
- 评估时间: {timestamp}
- 对弈局数: {stats['num_games']}

## 性能指标
- 胜率: {stats['win_rate']:.2%}
- 平均游戏长度: {stats['avg_game_length']:.1f}步
- 对弈结果:
  - 胜: {stats.get('wins', 0)}局
  - 负: {stats.get('losses', 0)}局
  - 平: {stats.get('draws', 0)}局

## 模型配置
- 神经网络结构:
  - 输入通道数: {self.config.NUM_CHANNELS}
  - 残差块数量: {self.config.NUM_RESIDUAL_BLOCKS}
  - Dropout率: {self.config.DROPOUT_RATE}
- MCTS参数:
  - 模拟次数: {self.config.NUM_MCTS_SIMS}
  - 探索常数: {self.config.C_PUCT}
  - Dirichlet噪声: α={self.config.DIRICHLET_ALPHA}

## 评估说明
本次评估通过自我对弈方式进行，每局游戏都采用相同的参数设置。
评估过程中，模型同时扮演黑白双方，以消除先后手的影响。

## 结论
模型展现出{self._get_performance_description(stats['win_rate'])}的性能表现。
"""
        
        if save_path:
            with open(save_path, 'w', encoding='utf-8') as f:
                f.write(report)
            self.logger.info(f"评估报告已保存到: {save_path}")
        
        return report
    
    def _get_performance_description(self, win_rate: float) -> str:
        """
        根据胜率生成性能描述
        
        Args:
            win_rate: 胜率
            
        Returns:
            str: 性能描述
        """
        if win_rate >= 0.7:
            return "优秀"
        elif win_rate >= 0.6:
            return "良好"
        elif win_rate >= 0.5:
            return "一般"
        else:
            return "有待提升"
    
    def save_history(self, filepath: str):
        """
        保存评估历史
        
        Args:
            filepath: 保存路径
        """
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(self.history, f, indent=2, ensure_ascii=False)
        self.logger.info(f"评估历史已保存到: {filepath}")
    
    def load_history(self, filepath: str) -> bool:
        """
        加载评估历史
        
        Args:
            filepath: 历史文件路径
            
        Returns:
            bool: 是否成功加载
        """
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                self.history = json.load(f)
            self.logger.info(f"成功加载评估历史: {filepath}")
            return True
        except Exception as e:
            self.logger.error(f"加载评估历史失败: {str(e)}")
            return False 