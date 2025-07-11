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
        
        # 设备配置
        self.DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # 神经网络配置
        self.NUM_CHANNELS = 256  # 减小通道数以加快训练
        self.NUM_RESIDUAL_BLOCKS = 20  # 减小残差块数量
        self.DROPOUT_RATE = 0.3
        
        # MCTS配置
        self.NUM_SIMULATIONS = 100  # 减少模拟次数以加快每步决策
        self.CPUCT = 1.0
        self.DIRICHLET_ALPHA = 0.3
        self.DIRICHLET_EPSILON = 0.25
        
        # 训练配置
        self.BATCH_SIZE = 512  # 增大批量大小
        self.NUM_EPOCHS = 20
        self.LEARNING_RATE = 0.001
        self.WEIGHT_DECAY = 1e-4
        self.MAX_GRAD_NORM = 1.0
        
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
        if torch.cuda.is_available():
            self.CUDA_LAUNCH_BLOCKING = "0"  # 禁用CUDA启动阻塞
            torch.backends.cudnn.benchmark = True  # 启用cuDNN基准测试
            torch.backends.cudnn.deterministic = False  # 关闭确定性模式以提高性能
        
        # 进度显示
        self.SHOW_PROGRESS = True
        self.PROGRESS_INTERVAL = 10  # 每10步显示一次进度

config = Config() 