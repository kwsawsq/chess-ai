"""
配置文件
"""

import os
import torch

class Config:
    def __init__(self):
        # 基础目录
        self.BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.DATA_DIR = os.path.join(self.BASE_DIR, 'data')
        self.MODEL_DIR = os.path.join(self.BASE_DIR, 'models')
        self.LOG_DIR = os.path.join(self.BASE_DIR, 'logs')
        
        # 创建必要的目录
        for dir_path in [self.DATA_DIR, self.MODEL_DIR, self.LOG_DIR]:
            os.makedirs(dir_path, exist_ok=True)
        
        # GPU配置
        self.USE_GPU = torch.cuda.is_available()  # 是否使用GPU
        self.DEVICE = 'cuda' if self.USE_GPU else 'cpu'
        self.GPU_ID = 0  # 使用的GPU ID
        if self.USE_GPU:
            torch.cuda.set_device(self.GPU_ID)
        
        # 神经网络配置
        self.NUM_CHANNELS = 256  # 减小通道数以加快训练
        self.NUM_RESIDUAL_BLOCKS = 20  # 减小残差块数量
        self.DROPOUT_RATE = 0.3
        
        # 神经网络输入输出配置
        self.IN_CHANNELS = 20  # 输入通道数 (12个棋子类型 + 4个重复位置历史 + 4个额外特征)
        self.BOARD_SIZE = 8  # 棋盘大小
        self.ACTION_SIZE = 4096  # 可能的行动数量 (64 * 64)
        self.VALUE_HEAD_HIDDEN = 256  # 价值头隐藏层大小
        self.POLICY_HEAD_HIDDEN = 256  # 策略头隐藏层大小
        
        # MCTS配置
        self.NUM_MCTS_SIMS = 100  # MCTS模拟次数
        self.C_PUCT = 1.0  # PUCT常数
        self.DIRICHLET_ALPHA = 0.3  # 狄利克雷噪声参数alpha
        self.DIRICHLET_EPSILON = 0.25  # 狄利克雷噪声权重
        
        # 训练配置
        self.BATCH_SIZE = 512  # 增大批量大小
        self.NUM_EPOCHS = 20
        self.LEARNING_RATE = 0.001
        self.WEIGHT_DECAY = 1e-4
        self.MAX_GRAD_NORM = 1.0
        
        # 学习率调度
        self.LR_MILESTONES = [100, 200, 300]  # 在这些epoch时降低学习率
        self.LR_GAMMA = 0.1  # 学习率衰减因子
        
        # 训练迭代次数
        self.NUM_ITERATIONS = 1000  # 总训练迭代次数
        
        # 自我对弈配置
        self.NUM_SELF_PLAY_GAMES = 100
        self.PARALLEL_GAMES = 8  # 增加并行游戏数
        self.NUM_WORKERS = 8  # 增加数据加载的工作进程数
        self.TEMP_THRESHOLD = 10
        
        # 评估配置
        self.EVAL_EPISODES = 10
        self.EVAL_WIN_RATE = 0.55
        
        # 数据增强
        self.USE_DATA_AUGMENTATION = True
        
        # 混合精度训练
        self.USE_AMP = True  # 启用自动混合精度
        
        # 缓存设置
        self.REPLAY_BUFFER_SIZE = 100000
        self.MIN_REPLAY_SIZE = 1000
        
        # 保存和加载
        self.SAVE_INTERVAL = 100
        self.CHECKPOINT_INTERVAL = 1000
        
        # 日志设置
        self.TENSORBOARD_LOG_DIR = os.path.join(self.LOG_DIR, 'tensorboard')
        self.ENABLE_LOGGING = True
        
        # 性能优化
        self.PIN_MEMORY = True  # 启用PIN_MEMORY
        self.ASYNC_LOADING = True  # 启用异步数据加载
        self.PREFETCH_FACTOR = 2  # 预取因子
        
        # CUDA优化
        if self.USE_GPU:
            self.CUDA_LAUNCH_BLOCKING = "0"  # 禁用CUDA启动阻塞
            torch.backends.cudnn.benchmark = True  # 启用cuDNN基准测试
            torch.backends.cudnn.deterministic = False  # 关闭确定性模式以提高性能
        
        # 进度显示
        self.SHOW_PROGRESS = True
        self.PROGRESS_INTERVAL = 10  # 每10步显示一次进度

config = Config() 