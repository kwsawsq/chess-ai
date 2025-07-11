"""
结果可视化工具
"""
import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, List, Any, Optional
import seaborn as sns
import pandas as pd


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