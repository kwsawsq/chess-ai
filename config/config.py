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
        self.USE_GPU = torch.cuda.is_available()
        self.DEVICE = 'cuda' if self.USE_GPU else 'cpu'
        self.GPU_ID = 0
        if self.USE_GPU:
            torch.cuda.set_device(self.GPU_ID)
        
        # 神经网络配置 - 保持大模型以充分利用GPU
        self.NUM_CHANNELS = 256
        self.NUM_RESIDUAL_BLOCKS = 8
        self.DROPOUT_RATE = 0.3
        
        # 神经网络输入输出配置
        self.IN_CHANNELS = 20
        self.BOARD_SIZE = 8
        self.ACTION_SIZE = 4096
        self.VALUE_HEAD_HIDDEN = 256
        self.POLICY_HEAD_HIDDEN = 256
        
        # MCTS配置 - 优化搜索深度和评估
        self.NUM_MCTS_SIMS = 800  # 训练时的模拟次数
        self.MCTS_BATCH_SIZE = 16 # MCTS内部评估的批处理大小，以充分利用GPU
        self.NUM_MCTS_SIMS_EVAL = 200  # 降低评估复杂度，加快评估速度
        self.C_PUCT = 1.0
        self.DIRICHLET_ALPHA = 0.3
        self.DIRICHLET_EPSILON = 0.25
        
        # 训练配置 - 优化批处理以加快训练
        self.BATCH_SIZE = 1024  # 增大批次大小，充分利用4090D的显存
        self.NUM_EPOCHS = 5
        self.LEARNING_RATE = 0.0005  # 提高学习率以跳出平台期
        self.WEIGHT_DECAY = 1e-4
        self.MAX_GRAD_NORM = 0.5
        
        # 学习率调度
        self.LR_MILESTONES = [100, 200, 300]
        self.LR_GAMMA = 0.5
        
        # 训练迭代次数
        self.NUM_ITERATIONS = 50  # 先测试50次迭代，验证优化效果
        
        # 自我对弈配置 - 充分利用多核CPU和GPU
        self.NUM_SELF_PLAY_GAMES = 500  # 每轮生成更多的游戏数据
        self.PARALLEL_GAMES = 16  # 增加并行游戏数
        self.NUM_WORKERS = 12  # 使用12个CPU核心，留出4个核心给系统
        self.TEMP_THRESHOLD = 10
        
        # 评估配置
        self.EVAL_EPISODES = 10  # 减少评估对局数以加快速度
        self.EVAL_INTERVAL = 10  # 每10次迭代才评估一次
        self.EVAL_WIN_RATE = 0.55
        
        # 数据增强
        self.USE_DATA_AUGMENTATION = True
        
        # 混合精度训练
        self.USE_AMP = True
        torch.set_float32_matmul_precision('medium')
        
        # 缓存设置
        self.REPLAY_BUFFER_SIZE = 500000
        self.MIN_REPLAY_SIZE = 20000 # 确保有足够多的初始数据再开始训练
        
        # 保存和加载
        self.SAVE_INTERVAL = 10  # 更频繁地保存
        self.CHECKPOINT_INTERVAL = 100
        
        # 日志设置
        self.TENSORBOARD_LOG_DIR = os.path.join(self.LOG_DIR, 'tensorboard')
        self.ENABLE_LOGGING = True
        
        # 性能优化
        self.PIN_MEMORY = True
        self.ASYNC_LOADING = True
        self.PREFETCH_FACTOR = 4
        
        # CUDA优化
        if self.USE_GPU:
            os.environ['CUDA_LAUNCH_BLOCKING'] = '0'
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = True
            torch.backends.cudnn.enabled = True
            torch.cuda.empty_cache()
            torch.jit.enable_onednn_fusion(True)
        
        # 进度显示 - 更频繁地显示进度
        self.SHOW_PROGRESS = True
        self.PROGRESS_INTERVAL = 5  # 每5步显示一次进度

config = Config() 