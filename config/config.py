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
        
        # 神经网络配置 - 极限模型容量
        self.NUM_CHANNELS = 2048  # 极大增加通道数
        self.NUM_RESIDUAL_BLOCKS = 40  # 极大增加残差块数量
        self.DROPOUT_RATE = 0.3
        
        # 神经网络输入输出配置
        self.IN_CHANNELS = 20
        self.BOARD_SIZE = 8
        self.ACTION_SIZE = 4096
        self.VALUE_HEAD_HIDDEN = 2048  # 极大增加隐藏层
        self.POLICY_HEAD_HIDDEN = 2048  # 极大增加隐藏层
        
        # MCTS配置 - 极限计算强度
        self.NUM_MCTS_SIMS = 3200  # 极大增加模拟次数
        self.C_PUCT = 1.0
        self.DIRICHLET_ALPHA = 0.3
        self.DIRICHLET_EPSILON = 0.25
        
        # 训练配置 - 极限GPU利用
        self.BATCH_SIZE = 4096  # 极大增加批量大小
        self.NUM_EPOCHS = 20
        self.LEARNING_RATE = 0.002
        self.WEIGHT_DECAY = 1e-4
        self.MAX_GRAD_NORM = 1.0
        
        # 学习率调度
        self.LR_MILESTONES = [100, 200, 300]
        self.LR_GAMMA = 0.1
        
        # 训练迭代次数
        self.NUM_ITERATIONS = 1000
        
        # 自我对弈配置 - 极限并行
        self.NUM_SELF_PLAY_GAMES = 800  # 极大增加游戏数
        self.PARALLEL_GAMES = 64  # 极大增加并行数
        self.NUM_WORKERS = 14  # 保持工作进程数（防止CPU成为瓶颈）
        self.TEMP_THRESHOLD = 10
        
        # 评估配置
        self.EVAL_EPISODES = 10
        self.EVAL_WIN_RATE = 0.55
        
        # 数据增强
        self.USE_DATA_AUGMENTATION = True
        
        # 混合精度训练 - 优化显存使用
        self.USE_AMP = True
        torch.set_float32_matmul_precision('medium')  # 设置中等精度以平衡性能和显存
        
        # 缓存设置 - 极限缓存
        self.REPLAY_BUFFER_SIZE = 1000000  # 100万样本缓存
        self.MIN_REPLAY_SIZE = 10000
        
        # 保存和加载
        self.SAVE_INTERVAL = 100
        self.CHECKPOINT_INTERVAL = 1000
        
        # 日志设置
        self.TENSORBOARD_LOG_DIR = os.path.join(self.LOG_DIR, 'tensorboard')
        self.ENABLE_LOGGING = True
        
        # 性能优化 - 极限优化
        self.PIN_MEMORY = True
        self.ASYNC_LOADING = True
        self.PREFETCH_FACTOR = 8  # 增加预取因子
        
        # CUDA优化 - 4090极限优化
        if self.USE_GPU:
            os.environ['CUDA_LAUNCH_BLOCKING'] = '0'
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            # 4090特定优化
            torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = True
            torch.backends.cudnn.enabled = True
            # 显存优化
            torch.cuda.empty_cache()  # 清理显存缓存
            # 启用JIT编译优化
            torch.jit.enable_onednn_fusion(True)
        
        # 进度显示
        self.SHOW_PROGRESS = True
        self.PROGRESS_INTERVAL = 10

config = Config() 