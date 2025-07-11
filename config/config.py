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
        
        # 神经网络配置 - 最大化模型容量
        self.NUM_CHANNELS = 1024  # 显著增加通道数
        self.NUM_RESIDUAL_BLOCKS = 32  # 增加残差块数量
        self.DROPOUT_RATE = 0.3
        
        # 神经网络输入输出配置
        self.IN_CHANNELS = 20  # 输入通道数 (12个棋子类型 + 4个重复位置历史 + 4个额外特征)
        self.BOARD_SIZE = 8  # 棋盘大小
        self.ACTION_SIZE = 4096  # 可能的行动数量 (64 * 64)
        self.VALUE_HEAD_HIDDEN = 1024  # 增加价值头隐藏层大小
        self.POLICY_HEAD_HIDDEN = 1024  # 增加策略头隐藏层大小
        
        # MCTS配置 - 增加计算强度
        self.NUM_MCTS_SIMS = 1600  # 大幅增加MCTS模拟次数
        self.C_PUCT = 1.0  # PUCT常数
        self.DIRICHLET_ALPHA = 0.3  # 狄利克雷噪声参数alpha
        self.DIRICHLET_EPSILON = 0.25  # 狄利克雷噪声权重
        
        # 训练配置 - 最大化GPU利用
        self.BATCH_SIZE = 2048  # 大幅增加批量大小
        self.NUM_EPOCHS = 20
        self.LEARNING_RATE = 0.002
        self.WEIGHT_DECAY = 1e-4
        self.MAX_GRAD_NORM = 1.0
        
        # 学习率调度
        self.LR_MILESTONES = [100, 200, 300]
        self.LR_GAMMA = 0.1
        
        # 训练迭代次数
        self.NUM_ITERATIONS = 1000
        
        # 自我对弈配置 - 增加并行度
        self.NUM_SELF_PLAY_GAMES = 400  # 大幅增加自我对弈游戏数
        self.PARALLEL_GAMES = 32  # 增加并行游戏数
        self.NUM_WORKERS = 14  # 增加工作进程数（留2个核心给系统）
        self.TEMP_THRESHOLD = 10
        
        # 评估配置
        self.EVAL_EPISODES = 10
        self.EVAL_WIN_RATE = 0.55
        
        # 数据增强
        self.USE_DATA_AUGMENTATION = True
        
        # 混合精度训练
        self.USE_AMP = True  # 启用自动混合精度
        
        # 缓存设置 - 增加缓存以支持更大批量
        self.REPLAY_BUFFER_SIZE = 500000  # 显著增加回放缓冲区大小
        self.MIN_REPLAY_SIZE = 5000
        
        # 保存和加载
        self.SAVE_INTERVAL = 100
        self.CHECKPOINT_INTERVAL = 1000
        
        # 日志设置
        self.TENSORBOARD_LOG_DIR = os.path.join(self.LOG_DIR, 'tensorboard')
        self.ENABLE_LOGGING = True
        
        # 性能优化 - 最大化GPU利用
        self.PIN_MEMORY = True
        self.ASYNC_LOADING = True
        self.PREFETCH_FACTOR = 4
        
        # CUDA优化 - 4090特定优化
        if self.USE_GPU:
            os.environ['CUDA_LAUNCH_BLOCKING'] = '0'  # 禁用CUDA启动阻塞以提高性能
            torch.backends.cudnn.benchmark = True  # 启用cuDNN基准测试
            torch.backends.cudnn.deterministic = False  # 关闭确定性模式以提高性能
            torch.backends.cuda.matmul.allow_tf32 = True  # 启用TF32
            torch.backends.cudnn.allow_tf32 = True  # 启用cuDNN TF32
            # 4090特定优化
            torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = True
            torch.backends.cudnn.enabled = True
        
        # 进度显示
        self.SHOW_PROGRESS = True
        self.PROGRESS_INTERVAL = 10

config = Config() 