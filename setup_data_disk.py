#!/usr/bin/env python3
"""
设置数据盘路径的脚本
用于解决autodl环境中磁盘空间不足的问题
"""

import os
import sys
import shutil
from pathlib import Path

def check_disk_space(path):
    """检查指定路径的磁盘空间"""
    try:
        stat = shutil.disk_usage(path)
        total_gb = stat.total / (1024**3)
        free_gb = stat.free / (1024**3)
        used_gb = (stat.total - stat.free) / (1024**3)
        return total_gb, used_gb, free_gb
    except Exception as e:
        print(f"无法检查路径 {path} 的磁盘空间: {e}")
        return None, None, None

def find_autodl_data_disk():
    """查找autodl的数据盘挂载点"""
    possible_paths = [
        '/root/autodl-tmp',
        '/root/autodl-nas',
        '/autodl-tmp',
        '/autodl-nas',
        '/data',
        '/tmp',
        '/mnt/data'
    ]
    
    print("正在查找可用的数据盘...")
    for path in possible_paths:
        if os.path.exists(path) and os.access(path, os.W_OK):
            total, used, free = check_disk_space(path)
            if total and free > 5:  # 至少5GB可用空间
                print(f"找到数据盘: {path}")
                print(f"  总空间: {total:.1f}GB")
                print(f"  已用空间: {used:.1f}GB")
                print(f"  可用空间: {free:.1f}GB")
                return path
    
    return None

def setup_environment():
    """设置环境变量和目录"""
    print("=== Chess AI 数据盘设置工具 ===\n")
    
    # 检查当前目录的磁盘空间
    current_dir = os.getcwd()
    print(f"当前工作目录: {current_dir}")
    total, used, free = check_disk_space(current_dir)
    if total:
        print(f"当前目录磁盘空间:")
        print(f"  总空间: {total:.1f}GB")
        print(f"  已用空间: {used:.1f}GB")
        print(f"  可用空间: {free:.1f}GB")
        
        if free < 2:
            print("⚠️  当前目录可用空间不足2GB，建议使用数据盘")
        else:
            print("✅ 当前目录空间充足")
    
    print()
    
    # 查找数据盘
    data_disk = find_autodl_data_disk()
    
    if data_disk:
        print(f"\n推荐使用数据盘: {data_disk}")
        
        # 询问用户是否使用推荐的数据盘
        while True:
            choice = input(f"是否使用 {data_disk} 作为数据存储路径? (y/n): ").lower().strip()
            if choice in ['y', 'yes', '是']:
                data_path = data_disk
                break
            elif choice in ['n', 'no', '否']:
                data_path = input("请输入自定义数据盘路径: ").strip()
                if not os.path.exists(data_path):
                    print(f"路径 {data_path} 不存在，请重新输入")
                    continue
                break
            else:
                print("请输入 y 或 n")
    else:
        print("未找到合适的数据盘，请手动指定:")
        data_path = input("请输入数据盘路径: ").strip()
        
        if not os.path.exists(data_path):
            print(f"路径 {data_path} 不存在")
            return False
    
    # 验证路径可写
    if not os.access(data_path, os.W_OK):
        print(f"错误: 路径 {data_path} 不可写")
        return False
    
    # 设置环境变量
    print(f"\n设置数据路径为: {data_path}")
    
    # 创建环境变量设置脚本
    env_script = f"""#!/bin/bash
# Chess AI 数据盘环境变量设置
export CHESS_AI_DATA_PATH="{data_path}"
echo "Chess AI 数据路径已设置为: $CHESS_AI_DATA_PATH"
"""
    
    with open('set_data_path.sh', 'w') as f:
        f.write(env_script)
    
    os.chmod('set_data_path.sh', 0o755)
    
    print("✅ 环境设置完成!")
    print("\n使用方法:")
    print("1. 运行以下命令设置环境变量:")
    print("   source set_data_path.sh")
    print("2. 然后运行训练脚本:")
    print("   python fixed_test_training.py")
    print("\n或者直接运行:")
    print(f"   CHESS_AI_DATA_PATH='{data_path}' python fixed_test_training.py")
    
    return True

if __name__ == "__main__":
    try:
        setup_environment()
    except KeyboardInterrupt:
        print("\n\n设置被用户取消")
        sys.exit(1)
    except Exception as e:
        print(f"\n设置过程中出现错误: {e}")
        sys.exit(1)
