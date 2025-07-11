"""
AlphaZero配置模块
定义所有训练和评估参数
"""

import os
from dataclasses import dataclass


@dataclass
class AlphaZeroConfig:
    """AlphaZero配置类"""
    
    def __init__(self):
        # 项目路径
        self.PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        
        # 目录配置
        self.DATA_DIR = os.path.join(self.PROJECT_ROOT, 'data')
        self.MODEL_DIR = os.path.join(self.PROJECT_ROOT, 'models')
        self.LOG_DIR = os.path.join(self.PROJECT_ROOT, 'logs')
        
        # 创建必要的目录
        for dir_path in [self.DATA_DIR, self.MODEL_DIR, self.LOG_DIR]:
            os.makedirs(dir_path, exist_ok=True)
        
        # 游戏参数
        self.BOARD_SIZE = 8  # 棋盘大小
        self.ACTION_SIZE = 4096  # 动作空间大小
        self.IN_CHANNELS = 20  # 输入通道数（12个棋子类型 + 8个额外特征）
        
        # 神经网络参数 - 增加模型容量
        self.NUM_CHANNELS = 512  # 卷积层通道数（增加到512）
        self.NUM_RESIDUAL_BLOCKS = 40  # 残差块数量（增加到40）
        self.DROPOUT_RATE = 0.3  # Dropout比率
        self.BATCH_NORM_MOMENTUM = 0.9  # 批归一化动量
        
        # MCTS参数
        self.NUM_MCTS_SIMS = 800  # 每步MCTS模拟次数（增加到800以提高性能）
        self.C_PUCT = 1.0  # PUCT公式中的探索常数
        self.DIRICHLET_ALPHA = 0.3  # Dirichlet噪声参数
        self.DIRICHLET_EPSILON = 0.25  # Dirichlet噪声权重
        
        # 自我对弈参数
        self.SELF_PLAY_GAMES = 100  # 每次迭代的自我对弈局数
        self.PARALLEL_GAMES = 40  # 并行自我对弈的游戏数（增加到40）
        self.TEMP_THRESHOLD = 30  # 温度阈值
        self.TEMP_INIT = 1.0  # 初始温度
        self.TEMP_FINAL = 0.2  # 最终温度
        self.MAX_GAME_STEPS = 512  # 单局游戏最大步数
        
        # 训练参数 - 优化GPU利用率
        self.BATCH_SIZE = 2048  # 训练批次大小（增加到2048）
        self.NUM_EPOCHS = 20  # 每次迭代的训练轮数
        self.LEARNING_RATE = 0.001  # 学习率
        self.MOMENTUM = 0.9  # 动量
        self.L2_REG = 0.0001  # L2正则化系数
        self.GRAD_CLIP = 5.0  # 梯度裁剪阈值
        self.NUM_ITERATIONS = 100  # 训练迭代次数
        self.EPOCHS_PER_ITERATION = 10  # 每次迭代的训练轮数
        self.MAX_TRAINING_DATA_SIZE = 1000000  # 训练数据最大数量（增加到100万）
        
        # 评估参数
        self.EVAL_GAMES = 40  # 评估对弈局数
        self.EVAL_TEMPERATURE = 0.1  # 评估时的温度参数
        self.WIN_RATE_THRESHOLD = 0.55  # 新模型接受的胜率阈值
        
        # 可视化参数
        self.PLOT_WIN_RATE = True  # 是否绘制胜率曲线
        self.PLOT_LOSS = True  # 是否绘制损失曲线
        self.SAVE_PLOTS = True  # 是否保存图表
        
        # 日志参数
        self.LOG_INTERVAL = 100  # 日志记录间隔（步数）
        self.CHECKPOINT_INTERVAL = 5  # 检查点保存间隔（迭代次数）
        
        # 硬件参数 - 优化数据加载
        self.USE_GPU = True  # 是否使用GPU
        self.NUM_WORKERS = 16  # 数据加载器的工作进程数（增加到16）
        
        # 调试参数
        self.DEBUG_MODE = False  # 是否启用调试模式
        self.PROFILE_MODE = False  # 是否启用性能分析模式 