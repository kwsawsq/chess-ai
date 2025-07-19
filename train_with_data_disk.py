#!/usr/bin/env python3
"""
使用数据盘的训练脚本
专门用于解决autodl环境中磁盘空间不足的问题
"""

import os
import sys
import shutil
import logging
from pathlib import Path

# 添加项目根目录到Python路径
project_root = str(Path(__file__).parent)
if project_root not in sys.path:
    sys.path.append(project_root)

def check_disk_space(path):
    """检查磁盘空间"""
    try:
        stat = shutil.disk_usage(path)
        total_gb = stat.total / (1024**3)
        free_gb = stat.free / (1024**3)
        used_gb = (stat.total - stat.free) / (1024**3)
        return total_gb, used_gb, free_gb
    except Exception as e:
        print(f"无法检查路径 {path} 的磁盘空间: {e}")
        return None, None, None

def setup_data_path():
    """设置数据路径"""
    # 检查环境变量
    data_path = os.environ.get('CHESS_AI_DATA_PATH')
    
    if not data_path:
        # 尝试常见的autodl数据盘路径
        possible_paths = [
            '/root/autodl-tmp',
            '/root/autodl-nas', 
            '/autodl-tmp',
            '/data'
        ]
        
        for path in possible_paths:
            if os.path.exists(path) and os.access(path, os.W_OK):
                total, used, free = check_disk_space(path)
                if free and free > 5:  # 至少5GB可用空间
                    data_path = path
                    print(f"自动检测到数据盘: {data_path} (可用空间: {free:.1f}GB)")
                    break
    
    if not data_path:
        print("错误: 未找到合适的数据盘路径")
        print("请运行 python setup_data_disk.py 来设置数据盘路径")
        print("或者设置环境变量: export CHESS_AI_DATA_PATH=/your/data/path")
        sys.exit(1)
    
    # 设置环境变量
    os.environ['CHESS_AI_DATA_PATH'] = data_path
    print(f"使用数据路径: {data_path}")
    
    # 检查可用空间
    total, used, free = check_disk_space(data_path)
    if total:
        print(f"磁盘空间: 总计{total:.1f}GB, 已用{used:.1f}GB, 可用{free:.1f}GB")
        if free < 2:
            print("⚠️  警告: 可用空间不足2GB，训练可能会失败")
            response = input("是否继续? (y/n): ")
            if response.lower() not in ['y', 'yes', '是']:
                sys.exit(1)
    
    return data_path

def clean_old_data():
    """清理旧的训练数据以释放空间"""
    from config.config import Config
    config = Config()

    print("\n检查是否需要清理旧数据...")

    total_freed = 0

    # 检查模型目录
    if os.path.exists(config.MODEL_DIR):
        model_files = list(Path(config.MODEL_DIR).glob("*.pth"))
        if len(model_files) > 5:  # 保留最新的5个模型
            print(f"发现 {len(model_files)} 个模型文件，清理旧模型...")
            model_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
            for old_model in model_files[5:]:
                try:
                    file_size = old_model.stat().st_size
                    old_model.unlink()
                    total_freed += file_size
                    print(f"删除旧模型: {old_model} ({file_size / (1024*1024):.1f}MB)")
                except Exception as e:
                    print(f"删除失败 {old_model}: {e}")

    # 检查数据目录
    if os.path.exists(config.DATA_DIR):
        data_files = list(Path(config.DATA_DIR).glob("*.npz"))
        if len(data_files) > 3:  # 保留最新的3个数据文件
            print(f"发现 {len(data_files)} 个数据文件，清理旧数据...")
            data_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
            for old_data in data_files[3:]:
                try:
                    file_size = old_data.stat().st_size
                    old_data.unlink()
                    total_freed += file_size
                    print(f"删除旧数据: {old_data} ({file_size / (1024*1024):.1f}MB)")
                except Exception as e:
                    print(f"删除失败 {old_data}: {e}")

    # 清理临时文件
    temp_patterns = ['*.tmp', '*.temp', '*~', '.DS_Store']
    for pattern in temp_patterns:
        for temp_file in Path(config.DATA_DIR).rglob(pattern):
            try:
                file_size = temp_file.stat().st_size
                temp_file.unlink()
                total_freed += file_size
            except:
                pass

    if total_freed > 0:
        print(f"总共释放空间: {total_freed / (1024*1024):.1f}MB")
    else:
        print("没有需要清理的文件")

def monitor_training_space(config):
    """训练过程中的空间监控"""
    import threading
    import time

    def space_monitor():
        while True:
            try:
                total, used, free = check_disk_space(config.DATA_DIR)
                if free and free < 2.0:  # 少于2GB时警告
                    print(f"\n⚠️  磁盘空间警告: 可用空间仅剩 {free:.1f}GB")
                    if free < 1.0:  # 少于1GB时强制清理
                        print("🔧 自动清理旧文件...")
                        clean_old_data()

                time.sleep(300)  # 每5分钟检查一次
            except Exception as e:
                print(f"空间监控出错: {e}")
                time.sleep(600)  # 出错时等待10分钟再试

    # 启动后台监控线程
    monitor_thread = threading.Thread(target=space_monitor, daemon=True)
    monitor_thread.start()
    print("✅ 磁盘空间监控已启动")

def main():
    """主函数"""
    print("=== Chess AI 训练 (数据盘版本) ===\n")
    
    # 设置数据路径
    data_path = setup_data_path()
    
    # 清理旧数据
    clean_old_data()
    
    # 导入配置和训练模块
    try:
        from config.config import Config
        from src.training.training_pipeline import TrainingPipeline
        
        # 创建配置
        config = Config()
        
        print(f"\n配置信息:")
        print(f"数据目录: {config.DATA_DIR}")
        print(f"模型目录: {config.MODEL_DIR}")
        print(f"日志目录: {config.LOG_DIR}")
        print(f"训练迭代次数: {config.NUM_ITERATIONS}")
        print(f"每轮自我对弈游戏数: {config.NUM_SELF_PLAY_GAMES}")
        
        # 启动磁盘空间监控
        monitor_training_space(config)

        # 创建训练流水线
        pipeline = TrainingPipeline(config)

        # 检查是否有检查点可以恢复
        start_iteration = 1
        checkpoint_files = list(Path(config.MODEL_DIR).glob("checkpoint_iter_*.pth"))
        if checkpoint_files:
            # 找到最新的检查点
            latest_checkpoint = max(checkpoint_files, key=lambda x: x.stat().st_mtime)
            print(f"\n发现检查点: {latest_checkpoint}")
            response = input("是否从检查点恢复训练? (y/n): ")
            if response.lower() in ['y', 'yes', '是']:
                # 从文件名提取迭代次数
                import re
                match = re.search(r'checkpoint_iter_(\d+)\.pth', str(latest_checkpoint))
                if match:
                    start_iteration = int(match.group(1)) + 1
                    print(f"从第 {start_iteration} 次迭代开始训练")
        
        print(f"\n开始训练...")
        print("按 Ctrl+C 可以安全停止训练并保存检查点")
        
        # 开始训练
        pipeline.train(start_iteration)
        
    except KeyboardInterrupt:
        print("\n\n训练被用户中断")
        print("检查点已自动保存")
    except Exception as e:
        print(f"\n训练出错: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
