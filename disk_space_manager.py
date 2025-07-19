#!/usr/bin/env python3
"""
磁盘空间管理工具
用于监控和管理Chess AI训练过程中的磁盘空间
"""

import os
import sys
import shutil
import time
import argparse
from pathlib import Path
from datetime import datetime, timedelta

# 添加项目根目录到Python路径
project_root = str(Path(__file__).parent)
if project_root not in sys.path:
    sys.path.append(project_root)

def check_disk_space(path):
    """检查磁盘空间"""
    try:
        stat = shutil.disk_usage(path)
        total_gb = stat.total / (1024**3)
        used_gb = (stat.total - stat.free) / (1024**3)
        free_gb = stat.free / (1024**3)
        return total_gb, used_gb, free_gb
    except Exception as e:
        print(f"无法检查路径 {path} 的磁盘空间: {e}")
        return None, None, None

def get_directory_size(path):
    """获取目录大小"""
    total_size = 0
    try:
        for dirpath, dirnames, filenames in os.walk(path):
            for filename in filenames:
                filepath = os.path.join(dirpath, filename)
                try:
                    total_size += os.path.getsize(filepath)
                except (OSError, FileNotFoundError):
                    pass
    except Exception as e:
        print(f"计算目录大小失败 {path}: {e}")
    return total_size

def format_size(size_bytes):
    """格式化文件大小"""
    if size_bytes < 1024:
        return f"{size_bytes}B"
    elif size_bytes < 1024**2:
        return f"{size_bytes/1024:.1f}KB"
    elif size_bytes < 1024**3:
        return f"{size_bytes/(1024**2):.1f}MB"
    else:
        return f"{size_bytes/(1024**3):.1f}GB"

def analyze_space_usage():
    """分析空间使用情况"""
    try:
        from config.config import Config
        config = Config()
    except ImportError:
        print("无法导入配置，请确保在项目根目录运行此脚本")
        return
    
    print("=== Chess AI 磁盘空间分析 ===\n")
    
    # 检查各个目录的空间使用
    directories = {
        "数据目录": config.DATA_DIR,
        "模型目录": config.MODEL_DIR,
        "日志目录": config.LOG_DIR,
    }
    
    if hasattr(config, 'PGN_DIR'):
        directories["PGN目录"] = config.PGN_DIR
    
    total_project_size = 0
    
    for name, path in directories.items():
        if os.path.exists(path):
            size = get_directory_size(path)
            total_project_size += size
            print(f"{name}: {path}")
            print(f"  大小: {format_size(size)}")
            
            # 列出最大的文件
            files = []
            try:
                for file_path in Path(path).rglob("*"):
                    if file_path.is_file():
                        files.append((file_path, file_path.stat().st_size))
            except Exception as e:
                print(f"  扫描文件失败: {e}")
                continue
                
            if files:
                files.sort(key=lambda x: x[1], reverse=True)
                print(f"  最大的文件:")
                for file_path, size in files[:3]:
                    print(f"    {file_path.name}: {format_size(size)}")
            print()
        else:
            print(f"{name}: {path} (不存在)")
            print()
    
    print(f"项目总大小: {format_size(total_project_size)}")
    
    # 检查磁盘空间
    data_disk = os.path.dirname(config.DATA_DIR)
    total, used, free = check_disk_space(data_disk)
    if total:
        print(f"\n磁盘空间 ({data_disk}):")
        print(f"  总计: {total:.1f}GB")
        print(f"  已用: {used:.1f}GB ({used/total*100:.1f}%)")
        print(f"  可用: {free:.1f}GB ({free/total*100:.1f}%)")
        
        if free < 2:
            print("  ⚠️  警告: 可用空间不足2GB")
        elif free < 5:
            print("  ⚠️  注意: 可用空间不足5GB")
        else:
            print("  ✅ 空间充足")

def cleanup_old_files(dry_run=False):
    """清理旧文件"""
    try:
        from config.config import Config
        config = Config()
    except ImportError:
        print("无法导入配置，请确保在项目根目录运行此脚本")
        return
    
    print("=== 清理旧文件 ===\n")
    
    if dry_run:
        print("🔍 预览模式 - 不会实际删除文件\n")
    
    total_freed = 0
    
    # 清理旧模型文件 (保留最新5个)
    if os.path.exists(config.MODEL_DIR):
        model_files = list(Path(config.MODEL_DIR).glob("*.pth"))
        if len(model_files) > 5:
            print(f"模型文件: 发现 {len(model_files)} 个，保留最新5个")
            model_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
            for old_model in model_files[5:]:
                size = old_model.stat().st_size
                total_freed += size
                print(f"  {'[预览]' if dry_run else '删除'} {old_model.name} ({format_size(size)})")
                if not dry_run:
                    try:
                        old_model.unlink()
                    except Exception as e:
                        print(f"    删除失败: {e}")
        else:
            print("模型文件: 无需清理")
    
    # 清理旧数据文件 (保留最新3个)
    if os.path.exists(config.DATA_DIR):
        data_files = list(Path(config.DATA_DIR).glob("*.npz"))
        if len(data_files) > 3:
            print(f"数据文件: 发现 {len(data_files)} 个，保留最新3个")
            data_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
            for old_data in data_files[3:]:
                size = old_data.stat().st_size
                total_freed += size
                print(f"  {'[预览]' if dry_run else '删除'} {old_data.name} ({format_size(size)})")
                if not dry_run:
                    try:
                        old_data.unlink()
                    except Exception as e:
                        print(f"    删除失败: {e}")
        else:
            print("数据文件: 无需清理")
    
    # 清理旧日志文件 (保留最近7天)
    if os.path.exists(config.LOG_DIR):
        week_ago = time.time() - (7 * 24 * 3600)
        old_logs = []
        for log_file in Path(config.LOG_DIR).rglob("*.log"):
            if log_file.stat().st_mtime < week_ago:
                old_logs.append(log_file)
        
        if old_logs:
            print(f"日志文件: 发现 {len(old_logs)} 个超过7天的日志")
            for log_file in old_logs:
                size = log_file.stat().st_size
                total_freed += size
                print(f"  {'[预览]' if dry_run else '删除'} {log_file.name} ({format_size(size)})")
                if not dry_run:
                    try:
                        log_file.unlink()
                    except Exception as e:
                        print(f"    删除失败: {e}")
        else:
            print("日志文件: 无需清理")
    
    print(f"\n{'预计释放' if dry_run else '总共释放'}空间: {format_size(total_freed)}")

def main():
    parser = argparse.ArgumentParser(description="Chess AI 磁盘空间管理工具")
    parser.add_argument("command", choices=["analyze", "cleanup", "preview"], 
                       help="操作类型: analyze(分析), cleanup(清理), preview(预览清理)")
    
    args = parser.parse_args()
    
    if args.command == "analyze":
        analyze_space_usage()
    elif args.command == "cleanup":
        cleanup_old_files(dry_run=False)
    elif args.command == "preview":
        cleanup_old_files(dry_run=True)

if __name__ == "__main__":
    main()
