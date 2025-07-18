"""
结果可视化工具
"""
import matplotlib
matplotlib.use('agg')  # 在导入pyplot之前设置后端
matplotlib.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Helvetica', 'sans-serif']  # 设置字体
matplotlib.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, List, Any, Optional
import seaborn as sns
import pandas as pd
from tqdm import tqdm
import chess
import chess.svg
import os
#from IPython.display import display, HTML, SVG
import time
import json


class ResultVisualizer:
    """结果可视化器"""
    
    def __init__(self, style: str = 'seaborn'):
        """
        初始化可视化器
        
        Args:
            style: 图表样式
        """
        plt.style.use(style)
        sns.set_palette("husl")
        
        # 创建PGN输出目录
        self.pgn_dir = os.path.join('logs', 'pgn_output')
        os.makedirs(self.pgn_dir, exist_ok=True)
        
        # 初始化游戏计数器
        self.game_counter = 0
    
    def create_pgn_file(self, moves: List[str], result: str) -> str:
        """
        创建PGN棋谱文件
        
        Args:
            moves: 移动列表 (UCI格式，如 'e2e4')
            result: 游戏结果 ('1-0', '0-1', '1/2-1/2')
            
        Returns:
            str: PGN文件路径
        """
        self.game_counter += 1
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = f'game_{self.game_counter}_{timestamp}.pgn'
        filepath = os.path.join(self.pgn_dir, filename)
        
        # 创建棋盘用于UCI到SAN转换
        board = chess.Board()
        
        # 生成PGN头信息
        pgn_content = [
            '[Event "AlphaZero Self-play"]',
            f'[Site "Chess-AI"]',
            f'[Date "{time.strftime("%Y.%m.%d")}"]',
            f'[Round "{self.game_counter}"]',
            '[White "AlphaZero"]',
            '[Black "AlphaZero"]',
            f'[Result "{result}"]',
            ''
        ]
        
        # 转换移动并添加到PGN
        move_text = ""
        for i, move_uci in enumerate(moves):
            try:
                # 将UCI格式转换为Move对象
                move = chess.Move.from_uci(move_uci)
        
                # 检查移动是否合法
                if move in board.legal_moves:
                    # 获取SAN表示
                    san_move = board.san(move)
                    
                    # 添加到PGN
                    if i % 2 == 0:  # 白方
                        move_number = i // 2 + 1
                        move_text += f"{move_number}. {san_move} "
                    else:  # 黑方
                        move_text += f"{san_move} "
                    
                    # 执行移动
                    board.push(move)
                else:
                    print(f"警告: 非法移动 {move_uci} 在位置 {board.fen()}")
                    continue
            except ValueError as e:
                print(f"无效的移动: {move_uci}, 错误: {e}")
                continue
        
        # 添加结果
        move_text += result
        
        # 合并PGN内容
        pgn_content.append(move_text)
        pgn_content.append("")  # 添加最后的空行
        
        # 保存PGN文件
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write('\n'.join(pgn_content))
        
        print(f"\n游戏PGN已保存到: {filepath}")
        return filepath
    
    def display_game(self, moves: List[str], result: str) -> None:
        """
        显示游戏，生成PGN文件
        
        Args:
            moves: 移动列表 (UCI格式)
            result: 游戏结果
        """
        filepath = self.create_pgn_file(moves, result)
        print(f"\n请使用国际象棋软件打开PGN文件查看对局详情：")
        print(f"文件路径: {os.path.abspath(filepath)}")
        
        # 打印对局基本信息
        print(f"\n对局信息:")
        print(f"- 总步数: {len(moves)}")
        print(f"- 结果: {result}")
        print(f"- 时间: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        
        # 打印最后几步棋
        if len(moves) > 0:
            print("\n最后几步棋:")
            last_moves = moves[-min(10, len(moves)):]
            for i, move in enumerate(last_moves):
                move_num = len(moves) - len(last_moves) + i + 1
                print(f"步骤 {move_num}: {move}")
    
    def plot_training_history(self, 
                             history: Dict[str, List],
                             save_path: Optional[str] = None,
                             figsize: tuple = (15, 10)) -> None:
        """
        绘制训练历史图表
        
        Args:
            history: 训练历史数据
            save_path: 保存路径
            figsize: 图表大小
        """
        plt.close('all') # 关闭所有打开的图形
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        fig.suptitle('AlphaZero 训练历史', fontsize=16)
        
        # 训练损失
        if 'training_loss' in history and history['training_loss']:
            axes[0, 0].plot(history['iterations'], history['training_loss'], 'b-', label='训练损失')
            axes[0, 0].set_title('训练损失')
            axes[0, 0].set_xlabel('迭代次数')
            axes[0, 0].set_ylabel('损失值')
            axes[0, 0].grid(True)
        
        # 胜率
        if 'win_rates' in history and history['win_rates']:
            win_rate_iterations = [history['iterations'][i] for i in range(len(history['win_rates']))]
            axes[0, 1].plot(win_rate_iterations, history['win_rates'], 'g-', label='胜率')
            axes[0, 1].axhline(y=0.55, color='r', linestyle='--', label='目标胜率')
            axes[0, 1].set_title('模型胜率')
            axes[0, 1].set_xlabel('迭代次数')
            axes[0, 1].set_ylabel('胜率')
            axes[0, 1].legend()
            axes[0, 1].grid(True)
        
        # 自我对弈游戏数
        if 'self_play_games' in history and history['self_play_games']:
            axes[1, 0].bar(history['iterations'], history['self_play_games'], alpha=0.7)
            axes[1, 0].set_title('每轮自我对弈游戏数')
            axes[1, 0].set_xlabel('迭代次数')
            axes[1, 0].set_ylabel('游戏数')
            axes[1, 0].grid(True)
        
        # 模型更新
        if 'model_updates' in history and history['model_updates']:
            update_iterations = [history['iterations'][i] for i in range(len(history['model_updates']))]
            updates = [1 if update else 0 for update in history['model_updates']]
            axes[1, 1].bar(update_iterations, updates, alpha=0.7, color='orange')
            axes[1, 1].set_title('模型更新')
            axes[1, 1].set_xlabel('迭代次数')
            axes[1, 1].set_ylabel('是否更新')
            axes[1, 1].grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def plot_self_play_stats(self, 
                            stats: Dict[str, Any],
                            save_path: Optional[str] = None,
                            figsize: tuple = (12, 8)) -> None:
        """
        绘制自我对弈统计图表
        
        Args:
            stats: 自我对弈统计数据
            save_path: 保存路径
            figsize: 图表大小
        """
        plt.close('all') # 关闭所有打开的图形
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        fig.suptitle('自我对弈统计', fontsize=16)
        
        # 胜负分布
        results = [stats['white_wins'], stats['black_wins'], stats['draws']]
        labels = ['白方胜', '黑方胜', '平局']
        colors = ['white', 'black', 'gray']
        
        axes[0, 0].pie(results, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
        axes[0, 0].set_title('胜负分布')
        
        # 胜率条形图
        win_rates = [stats['white_win_rate'], stats['black_win_rate'], stats['draw_rate']]
        axes[0, 1].bar(labels, win_rates, color=colors)
        axes[0, 1].set_title('胜率统计')
        axes[0, 1].set_ylabel('胜率')
        axes[0, 1].set_ylim(0, 1)
        
        # 游戏长度分布（如果有数据）
        if 'game_lengths' in stats:
            axes[1, 0].hist(stats['game_lengths'], bins=20, alpha=0.7, color='skyblue')
            axes[1, 0].set_title('游戏长度分布')
            axes[1, 0].set_xlabel('游戏长度')
            axes[1, 0].set_ylabel('频次')
        
        # 统计信息文本
        info_text = f"""
        总游戏数: {stats['total_games']}
        平均游戏长度: {stats['average_game_length']:.1f}
        白方胜率: {stats['white_win_rate']:.3f}
        黑方胜率: {stats['black_win_rate']:.3f}
        平局率: {stats['draw_rate']:.3f}
        """
        axes[1, 1].text(0.1, 0.5, info_text, transform=axes[1, 1].transAxes, 
                        fontsize=12, verticalalignment='center')
        axes[1, 1].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def plot_model_comparison(self, 
                            comparison_results: Dict[str, Any],
                            save_path: Optional[str] = None,
                            figsize: tuple = (12, 8)) -> None:
        """
        绘制模型比较图表
        
        Args:
            comparison_results: 模型比较结果
            save_path: 保存路径
            figsize: 图表大小
        """
        plt.close('all') # 关闭所有打开的图形
        fig, axes = plt.subplots(1, 2, figsize=figsize)
        fig.suptitle('模型比较', fontsize=16)
        
        # 胜率矩阵热力图
        win_matrix = comparison_results['win_matrix']
        model_names = comparison_results['model_names']
        
        sns.heatmap(win_matrix, annot=True, fmt='.3f', cmap='RdYlBu_r',
                   xticklabels=model_names, yticklabels=model_names,
                   ax=axes[0])
        axes[0].set_title('胜率矩阵')
        axes[0].set_xlabel('对手模型')
        axes[0].set_ylabel('当前模型')
        
        # 模型排名
        scores = comparison_results['scores']
        ranking = comparison_results['ranking']
        ranked_names = [model_names[i] for i in ranking]
        ranked_scores = [scores[i] for i in ranking]
        
        axes[1].barh(ranked_names, ranked_scores, color='skyblue')
        axes[1].set_title('模型排名')
        axes[1].set_xlabel('平均胜率')
        axes[1].set_xlim(0, 1)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def plot_mcts_analysis(self, 
                          mcts_stats: Dict[str, Any],
                          save_path: Optional[str] = None,
                          figsize: tuple = (12, 6)) -> None:
        """
        绘制MCTS分析图表
        
        Args:
            mcts_stats: MCTS统计数据
            save_path: 保存路径
            figsize: 图表大小
        """
        plt.close('all') # 关闭所有打开的图形
        fig, axes = plt.subplots(1, 2, figsize=figsize)
        fig.suptitle('MCTS 分析', fontsize=16)
        
        # 搜索时间分布
        if 'search_times' in mcts_stats:
            axes[0].hist(mcts_stats['search_times'], bins=20, alpha=0.7, color='lightgreen')
            axes[0].set_title('搜索时间分布')
            axes[0].set_xlabel('搜索时间 (秒)')
            axes[0].set_ylabel('频次')
        
        # 树深度分布
        if 'tree_depths' in mcts_stats:
            axes[1].hist(mcts_stats['tree_depths'], bins=20, alpha=0.7, color='lightcoral')
            axes[1].set_title('搜索树深度分布')
            axes[1].set_xlabel('最大深度')
            axes[1].set_ylabel('频次')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def plot_loss_curves(self, 
                        training_history: Dict[str, List],
                        save_path: Optional[str] = None,
                        figsize: tuple = (12, 6)) -> None:
        """
        绘制损失曲线
        
        Args:
            training_history: 训练历史
            save_path: 保存路径
            figsize: 图表大小
        """
        plt.close('all') # 关闭所有打开的图形
        fig, axes = plt.subplots(1, 2, figsize=figsize)
        fig.suptitle('损失曲线', fontsize=16)
        
        # 训练损失
        if 'train_loss' in training_history:
            axes[0].plot(training_history['train_loss'], 'b-', label='训练损失')
            if 'train_policy_loss' in training_history:
                axes[0].plot(training_history['train_policy_loss'], 'r--', label='策略损失')
            if 'train_value_loss' in training_history:
                axes[0].plot(training_history['train_value_loss'], 'g--', label='价值损失')
            axes[0].set_title('训练损失')
            axes[0].set_xlabel('Epoch')
            axes[0].set_ylabel('损失值')
            axes[0].legend()
            axes[0].grid(True)
        
        # 验证损失
        if 'val_loss' in training_history:
            axes[1].plot(training_history['val_loss'], 'b-', label='验证损失')
            if 'val_policy_loss' in training_history:
                axes[1].plot(training_history['val_policy_loss'], 'r--', label='策略损失')
            if 'val_value_loss' in training_history:
                axes[1].plot(training_history['val_value_loss'], 'g--', label='价值损失')
            axes[1].set_title('验证损失')
            axes[1].set_xlabel('Epoch')
            axes[1].set_ylabel('损失值')
            axes[1].legend()
            axes[1].grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def create_training_dashboard(self, 
                                 training_stats: Dict[str, Any],
                                 save_path: Optional[str] = None,
                                 figsize: tuple = (20, 12)) -> None:
        """
        创建训练仪表板
        
        Args:
            training_stats: 训练统计数据
            save_path: 保存路径
            figsize: 图表大小
        """
        plt.close('all') # 关闭所有打开的图形
        fig = plt.figure(figsize=figsize)
        gs = fig.add_gridspec(3, 4, hspace=0.3, wspace=0.3)
        
        # 1. 训练损失
        ax1 = fig.add_subplot(gs[0, :2])
        if 'training_history' in training_stats:
            history = training_stats['training_history']
            if 'training_loss' in history:
                ax1.plot(history['iterations'], history['training_loss'], 'b-')
                ax1.set_title('训练损失')
                ax1.set_xlabel('迭代次数')
                ax1.set_ylabel('损失值')
                ax1.grid(True)
        
        # 2. 胜率趋势
        ax2 = fig.add_subplot(gs[0, 2:])
        if 'training_history' in training_stats and 'win_rates' in training_stats['training_history']:
            win_rates = training_stats['training_history']['win_rates']
            if win_rates:
                ax2.plot(range(len(win_rates)), win_rates, 'g-')
                ax2.axhline(y=0.55, color='r', linestyle='--', label='目标胜率')
                ax2.set_title('胜率趋势')
                ax2.set_xlabel('评估轮次')
                ax2.set_ylabel('胜率')
                ax2.legend()
                ax2.grid(True)
        
        # 3. 自我对弈统计
        ax3 = fig.add_subplot(gs[1, :2])
        if 'self_play_stats' in training_stats:
            stats = training_stats['self_play_stats']
            results = [stats['white_wins'], stats['black_wins'], stats['draws']]
            labels = ['白方胜', '黑方胜', '平局']
            ax3.pie(results, labels=labels, autopct='%1.1f%%', startangle=90)
            ax3.set_title('自我对弈结果分布')
        
        # 4. 数据管理器统计
        ax4 = fig.add_subplot(gs[1, 2:])
        if 'data_manager_stats' in training_stats:
            stats = training_stats['data_manager_stats']
            info_text = f"""
            缓冲区大小: {stats.get('buffer_size', 0)}
            总游戏数: {stats.get('total_games', 0)}
            总样本数: {stats.get('total_samples', 0)}
            平均游戏长度: {stats.get('avg_game_length', 0):.1f}
            """
            ax4.text(0.1, 0.5, info_text, transform=ax4.transAxes, 
                    fontsize=12, verticalalignment='center')
            ax4.set_title('数据统计')
            ax4.axis('off')
        
        # 5. 模型信息
        ax5 = fig.add_subplot(gs[2, :])
        if 'model_info' in training_stats:
            info = training_stats['model_info']
            model_text = f"""
            模型参数总数: {info.get('total_params', 0):,}
            可训练参数: {info.get('trainable_params', 0):,}
            输入通道数: {info.get('input_channels', 0)}
            卷积通道数: {info.get('num_channels', 0)}
            动作空间大小: {info.get('action_size', 0)}
            """
            ax5.text(0.1, 0.5, model_text, transform=ax5.transAxes, 
                    fontsize=12, verticalalignment='center')
            ax5.set_title('模型信息')
            ax5.axis('off')
        
        plt.suptitle('AlphaZero 训练仪表板', fontsize=20)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def __str__(self) -> str:
        """字符串表示"""
        return "ResultVisualizer"
    
    def __repr__(self) -> str:
        """调试表示"""
        return self.__str__() 

"""
训练过程可视化模块
"""

import chess
import chess.svg
import matplotlib.pyplot as plt
import numpy as np
# 移除IPython依赖
# from IPython.display import SVG, display, clear_output
import seaborn as sns
from typing import List, Tuple, Dict, Optional
import time
import os
from datetime import datetime
from ..game.chess_board import ChessBoard


class TrainingVisualizer:
    """训练过程可视化器"""
    
    def __init__(self, config):
        """
        初始化可视化器
        
        Args:
            config: 配置对象
        """
        self.config = config
        self.game_count = 0
        self.move_count = 0
        
        # 创建可视化输出目录
        self.vis_dir = os.path.join(config.LOG_DIR, 'visualizations')
        os.makedirs(self.vis_dir, exist_ok=True)
        
        # 创建当前训练会话目录
        self.session_dir = os.path.join(
            self.vis_dir,
            datetime.now().strftime('%Y%m%d_%H%M%S')
        )
        os.makedirs(self.session_dir, exist_ok=True)
        
        # 初始化matplotlib
        plt.style.use('seaborn')
        
        # 棋子符号映射
        self.piece_symbols = {
            'P': '♙', 'N': '♘', 'B': '♗', 'R': '♖', 'Q': '♕', 'K': '♔',
            'p': '♟', 'n': '♞', 'b': '♝', 'r': '♜', 'q': '♛', 'k': '♚',
            '.': ' '
        }
    
    def _create_board_image(self, board: chess.Board) -> np.ndarray:
        """
        创建棋盘图像
        
        Args:
            board: 棋盘对象
            
        Returns:
            np.ndarray: 棋盘图像数组
        """
        # 创建8x8的图像数组
        board_array = np.zeros((8, 8, 3))
        
        # 设置棋盘格颜色
        for i in range(8):
            for j in range(8):
                if (i + j) % 2 == 0:
                    board_array[i, j] = [0.9, 0.9, 0.8]  # 浅色格
                else:
                    board_array[i, j] = [0.5, 0.5, 0.4]  # 深色格
        
        return board_array
    
    def visualize_board(self,
                       board: ChessBoard,
                       policy: np.ndarray,
                       value: float,
                       move_probs: Dict[str, float],
                       save: bool = True) -> None:
        """
        可视化当前棋局状态
        
        Args:
            board: 棋盘对象
            policy: 策略分布
            value: 局面评估值
            move_probs: 移动概率字典
            save: 是否保存图像
        """
        # 显示进度
        print(f"\r正在生成可视化图像... 游戏 {self.game_count}, 步数 {self.move_count}", end="")
        
        # 创建图形
        fig = plt.figure(figsize=(15, 8))
        
        # 1. 棋盘状态
        ax1 = plt.subplot(121)
        board_image = self._create_board_image(board.board)
        ax1.imshow(board_image)
        
        # 添加棋子
        for i in range(8):
            for j in range(8):
                square = chess.square(j, 7-i)  # chess库使用的是从下到上的坐标
                piece = board.board.piece_at(square)
                if piece:
                    color = 'white' if piece.color else 'black'
                    ax1.text(j, i, self.piece_symbols[piece.symbol()],
                            ha='center', va='center', color=color,
                            fontsize=20)
        
        # 添加坐标标签
        ax1.set_xticks(range(8))
        ax1.set_xticklabels(['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h'])
        ax1.set_yticks(range(8))
        ax1.set_yticklabels(['8', '7', '6', '5', '4', '3', '2', '1'])
        
        # 2. 移动概率分布
        ax2 = plt.subplot(122)
        moves = list(move_probs.keys())
        probs = list(move_probs.values())
        
        # 只显示概率最高的前10个移动
        if len(moves) > 10:
            indices = np.argsort(probs)[-10:]
            moves = [moves[i] for i in indices]
            probs = [probs[i] for i in indices]
        
        y_pos = np.arange(len(moves))
        ax2.barh(y_pos, probs)
        ax2.set_yticks(y_pos)
        ax2.set_yticklabels(moves)
        ax2.invert_yaxis()
        ax2.set_xlabel('概率')
        ax2.set_title(f'局面评估: {value:.2f}')
        
        plt.tight_layout()
        
        if save:
            # 保存图像
            game_dir = os.path.join(self.session_dir, f'game_{self.game_count}')
            os.makedirs(game_dir, exist_ok=True)
            plt.savefig(os.path.join(game_dir, f'move_{self.move_count}.png'))
            self.move_count += 1
        
        plt.close()
    
    def new_game(self) -> None:
        """开始新的游戏"""
        self.game_count += 1
        self.move_count = 0
    
    def plot_training_stats(self,
                          stats: Dict[str, List[float]],
                          save: bool = True) -> None:
        """
        绘制训练统计信息
        
        Args:
            stats: 训练统计数据
            save: 是否保存图像
        """
        plt.close('all') # 关闭所有打开的图形
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # 1. 总损失
        ax = axes[0, 0]
        ax.plot(stats['total_loss'])
        ax.set_title('总损失')
        ax.set_xlabel('迭代次数')
        ax.set_ylabel('损失值')
        
        # 2. 策略损失
        ax = axes[0, 1]
        ax.plot(stats['policy_loss'])
        ax.set_title('策略损失')
        ax.set_xlabel('迭代次数')
        ax.set_ylabel('损失值')
        
        # 3. 价值损失
        ax = axes[1, 0]
        ax.plot(stats['value_loss'])
        ax.set_title('价值损失')
        ax.set_xlabel('迭代次数')
        ax.set_ylabel('损失值')
        
        # 4. 学习率
        ax = axes[1, 1]
        ax.plot(stats['learning_rate'])
        ax.set_title('学习率')
        ax.set_xlabel('迭代次数')
        ax.set_ylabel('学习率')
        
        plt.tight_layout()
        
        if save:
            plt.savefig(os.path.join(self.session_dir, 'training_stats.png'))
        
        plt.close()
    
    def plot_game_stats(self,
                       game_lengths: List[int],
                       win_rates: List[float],
                       save: bool = True) -> None:
        """
        绘制游戏统计信息
        
        Args:
            game_lengths: 游戏长度列表
            win_rates: 胜率列表
            save: 是否保存图像
        """
        plt.close('all') # 关闭所有打开的图形
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # 1. 游戏长度分布
        ax1.hist(game_lengths, bins=20)
        ax1.set_title('游戏长度分布')
        ax1.set_xlabel('步数')
        ax1.set_ylabel('频率')
        
        # 2. 胜率变化
        ax2.plot(win_rates)
        ax2.set_title('胜率变化')
        ax2.set_xlabel('游戏编号')
        ax2.set_ylabel('胜率')
        ax2.set_ylim(0, 1)
        
        plt.tight_layout()
        
        if save:
            plt.savefig(os.path.join(self.session_dir, 'game_stats.png'))
        
        plt.close()
    
    def save_game_pgn(self, game_moves: List[str], result: str) -> None:
        """
        保存游戏PGN记录
        
        Args:
            game_moves: 游戏移动列表
            result: 游戏结果
        """
        game_dir = os.path.join(self.session_dir, f'game_{self.game_count}')
        os.makedirs(game_dir, exist_ok=True)
        
        pgn = f'[Event "Self-play Game {self.game_count}"]\n'
        pgn += f'[Date "{datetime.now().strftime("%Y.%m.%d")}"]\n'
        pgn += f'[Result "{result}"]\n\n'
        
        for i, move in enumerate(game_moves):
            if i % 2 == 0:
                pgn += f'{i//2 + 1}. {move} '
            else:
                pgn += f'{move} '
        
        pgn += result
        
        with open(os.path.join(game_dir, 'game.pgn'), 'w') as f:
            f.write(pgn) 