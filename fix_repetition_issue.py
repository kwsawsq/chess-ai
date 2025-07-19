#!/usr/bin/env python3
"""
修复AI重复走棋问题的脚本
"""

import os
import sys
import shutil
from pathlib import Path

# 添加项目根目录到Python路径
project_root = str(Path(__file__).parent)
if project_root not in sys.path:
    sys.path.append(project_root)

def backup_current_config():
    """备份当前配置"""
    config_file = "config/config.py"
    backup_file = "config/config_backup.py"
    
    if os.path.exists(config_file):
        shutil.copy2(config_file, backup_file)
        print(f"✅ 已备份配置文件到: {backup_file}")

def fix_config():
    """修复配置文件"""
    config_content = '''"""
配置文件 - 修复重复走棋问题
"""

import os
import torch

class Config:
    def __init__(self):
        # 基础目录
        self.BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        
        # 数据盘路径配置
        self.DATA_DISK_PATH = os.environ.get('CHESS_AI_DATA_PATH', '/root/autodl-tmp')
        
        if not os.path.exists(self.DATA_DISK_PATH):
            print(f"警告: 数据盘路径 {self.DATA_DISK_PATH} 不存在，使用项目目录")
            self.DATA_DISK_PATH = self.BASE_DIR
        
        # 数据目录
        self.DATA_DIR = os.path.join(self.DATA_DISK_PATH, 'chess-ai-data')
        self.MODEL_DIR = os.path.join(self.DATA_DISK_PATH, 'chess-ai-models')
        self.LOG_DIR = os.path.join(self.DATA_DISK_PATH, 'chess-ai-logs')
        
        # 创建必要的目录
        for dir_path in [self.DATA_DIR, self.MODEL_DIR, self.LOG_DIR]:
            os.makedirs(dir_path, exist_ok=True)
            
        print(f"数据目录: {self.DATA_DIR}")
        print(f"模型目录: {self.MODEL_DIR}")
        print(f"日志目录: {self.LOG_DIR}")
        
        # GPU配置
        self.USE_GPU = torch.cuda.is_available()
        self.DEVICE = 'cuda' if self.USE_GPU else 'cpu'
        self.GPU_ID = 0
        if self.USE_GPU:
            torch.cuda.set_device(self.GPU_ID)
        
        # 神经网络配置 - 重新调整以避免过拟合
        self.NUM_CHANNELS = 256  # 减少通道数，降低过拟合风险
        self.NUM_RESIDUAL_BLOCKS = 8  # 减少层数
        self.DROPOUT_RATE = 0.5  # 增加dropout
        
        # 神经网络输入输出配置
        self.IN_CHANNELS = 20
        self.BOARD_SIZE = 8
        self.ACTION_SIZE = 4096
        self.VALUE_HEAD_HIDDEN = 256
        self.POLICY_HEAD_HIDDEN = 256
        
        # MCTS配置 - 增加探索性
        self.NUM_MCTS_SIMS = 200  # 减少模拟次数，增加随机性
        self.MCTS_BATCH_SIZE = 128
        self.NUM_MCTS_SIMS_EVAL = 50
        self.C_PUCT = 4.0  # 大幅增加探索常数
        self.DIRICHLET_ALPHA = 0.5  # 增加噪声
        self.DIRICHLET_EPSILON = 0.4  # 增加噪声比例
        
        # 训练配置 - 防止过拟合
        self.BATCH_SIZE = 1024  # 减小批次大小
        self.NUM_EPOCHS = 3  # 减少epoch数
        self.LEARNING_RATE = 0.001  # 降低学习率
        self.WEIGHT_DECAY = 1e-3  # 增加正则化
        self.MAX_GRAD_NORM = 1.0
        
        # 学习率调度
        self.LR_MILESTONES = [50, 100, 150]
        self.LR_GAMMA = 0.3
        
        # 训练迭代次数
        self.NUM_ITERATIONS = 100
        
        # 自我对弈配置 - 增加多样性
        self.NUM_SELF_PLAY_GAMES = 100  # 减少游戏数，提高质量
        self.PARALLEL_GAMES = 8
        self.NUM_WORKERS = 4  # 减少worker数
        self.TEMP_THRESHOLD = 30  # 大幅延长随机探索阶段
        self.MAX_GAME_LENGTH = 150  # 添加最大游戏长度限制
        self.DRAW_THRESHOLD = 2  # 重复2次就判定和棋
        
        # 评估配置
        self.EVAL_EPISODES = 20  # 增加评估局数
        self.EVAL_INTERVAL = 5  # 更频繁评估
        self.EVAL_WIN_RATE = 0.6
        
        # 数据增强
        self.USE_DATA_AUGMENTATION = True
        
        # 混合精度训练
        self.USE_AMP = True
        torch.set_float32_matmul_precision('medium')
        
        # 缓存设置 - 增加数据多样性
        self.REPLAY_BUFFER_SIZE = 200000  # 减小缓冲区，保持数据新鲜
        self.MIN_REPLAY_SIZE = 10000
        
        # 保存和加载
        self.SAVE_INTERVAL = 5
        self.CHECKPOINT_INTERVAL = 50
        
        # 日志设置
        self.TENSORBOARD_LOG_DIR = os.path.join(self.LOG_DIR, 'tensorboard')
        self.ENABLE_LOGGING = True
        
        # PGN棋谱保存
        self.SAVE_PGN = True
        self.PGN_DIR = os.path.join(self.DATA_DISK_PATH, 'chess-ai-games')
        
        # 性能优化
        self.PIN_MEMORY = True
        self.ASYNC_LOADING = True
        self.PREFETCH_FACTOR = 2
        
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
        
        # 进度显示
        self.SHOW_PROGRESS = True
        self.PROGRESS_INTERVAL = 5
        
        # 反重复走棋机制
        self.ANTI_REPETITION = True  # 启用反重复机制
        self.REPETITION_PENALTY = 0.5  # 重复走棋的惩罚系数
        self.FORCE_PROGRESS = True  # 强制游戏进展

config = Config()
'''
    
    with open("config/config.py", "w", encoding="utf-8") as f:
        f.write(config_content)
    
    print("✅ 已更新配置文件，主要修改：")
    print("  - 增加探索性参数 (C_PUCT=4.0, 更多噪声)")
    print("  - 延长随机探索阶段 (TEMP_THRESHOLD=30)")
    print("  - 添加游戏长度限制 (MAX_GAME_LENGTH=150)")
    print("  - 增加正则化防止过拟合")
    print("  - 减少模型复杂度")

def clear_old_models():
    """清理旧模型，强制重新开始训练"""
    try:
        from config.config import Config
        config = Config()
        
        # 清理模型文件
        model_files = list(Path(config.MODEL_DIR).glob("*.pth"))
        if model_files:
            print(f"🗑️  发现 {len(model_files)} 个旧模型文件")
            response = input("是否删除所有旧模型重新开始训练? (y/n): ")
            if response.lower() in ['y', 'yes', '是']:
                for model_file in model_files:
                    model_file.unlink()
                    print(f"  删除: {model_file.name}")
                print("✅ 已清理所有旧模型")
            else:
                print("⚠️  保留旧模型，但建议重新开始训练")
        
        # 清理训练数据
        data_files = list(Path(config.DATA_DIR).glob("*.npz"))
        if data_files:
            print(f"🗑️  发现 {len(data_files)} 个旧训练数据文件")
            response = input("是否删除旧训练数据? (y/n): ")
            if response.lower() in ['y', 'yes', '是']:
                for data_file in data_files:
                    data_file.unlink()
                    print(f"  删除: {data_file.name}")
                print("✅ 已清理旧训练数据")
                
    except Exception as e:
        print(f"清理过程中出错: {e}")

def main():
    print("=== Chess AI 重复走棋问题修复工具 ===\n")
    
    print("🔍 问题诊断:")
    print("  - AI陷入重复走棋循环")
    print("  - 胜率始终0%，平局率100%")
    print("  - 模型过拟合到安全策略")
    print()
    
    print("🛠️  修复方案:")
    print("  1. 增加MCTS探索性 (C_PUCT, 噪声)")
    print("  2. 延长随机探索阶段")
    print("  3. 添加游戏长度限制")
    print("  4. 增加正则化防止过拟合")
    print("  5. 清理旧模型重新开始")
    print()
    
    response = input("是否执行修复? (y/n): ")
    if response.lower() not in ['y', 'yes', '是']:
        print("修复已取消")
        return
    
    # 执行修复
    backup_current_config()
    fix_config()
    clear_old_models()
    
    print("\n🎉 修复完成！")
    print("\n📋 下一步操作:")
    print("1. 运行训练脚本:")
    print("   python train_with_data_disk.py")
    print("2. 观察日志中的游戏长度变化")
    print("3. 检查胜率是否开始变化")
    print("\n💡 预期改善:")
    print("  - 游戏长度应该变得更多样化")
    print("  - 胜率应该开始出现变化")
    print("  - 减少重复走棋现象")

if __name__ == "__main__":
    main()
