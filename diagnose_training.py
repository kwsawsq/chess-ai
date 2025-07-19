#!/usr/bin/env python3
"""
训练状态诊断脚本
"""

import os
import sys
import re
from pathlib import Path
from datetime import datetime

# 添加项目根目录到Python路径
project_root = str(Path(__file__).parent)
if project_root not in sys.path:
    sys.path.append(project_root)

def analyze_log_files():
    """分析日志文件"""
    try:
        from config.config import Config
        config = Config()
        log_dir = Path(config.LOG_DIR)
    except:
        log_dir = Path("logs")
    
    print("=== 训练日志分析 ===\n")
    
    # 查找最新的日志文件
    log_files = list(log_dir.rglob("*.log"))
    if not log_files:
        print("❌ 未找到日志文件")
        return
    
    latest_log = max(log_files, key=lambda x: x.stat().st_mtime)
    print(f"📄 分析日志文件: {latest_log}")
    
    try:
        with open(latest_log, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 分析胜率统计
        win_rates = re.findall(r'胜率: ([\d.]+)%', content)
        if win_rates:
            win_rates = [float(rate) for rate in win_rates]
            print(f"🎯 胜率变化: {win_rates[-5:] if len(win_rates) > 5 else win_rates}")
            
            if all(rate == 0.0 for rate in win_rates[-10:]):
                print("⚠️  问题: 胜率持续为0%")
        
        # 分析游戏长度
        game_lengths = re.findall(r'平均游戏长度: ([\d.]+)', content)
        if game_lengths:
            lengths = [float(length) for length in game_lengths]
            print(f"📏 游戏长度: {lengths[-5:] if len(lengths) > 5 else lengths}")
            
            if all(abs(length - 50.0) < 0.1 for length in lengths[-5:]):
                print("⚠️  问题: 游戏长度固定在50步")
        
        # 分析损失
        policy_losses = re.findall(r'策略损失: ([\d.]+)', content)
        value_losses = re.findall(r'价值损失: ([\d.]+)', content)
        
        if policy_losses and value_losses:
            recent_policy = [float(loss) for loss in policy_losses[-5:]]
            recent_value = [float(loss) for loss in value_losses[-5:]]
            print(f"📉 最近策略损失: {recent_policy}")
            print(f"📉 最近价值损失: {recent_value}")
            
            if all(loss < 0.01 for loss in recent_policy[-3:]):
                print("⚠️  问题: 策略损失过低，可能过拟合")
        
        # 分析重复走棋
        repetition_count = content.count("重复局面")
        if repetition_count > 0:
            print(f"🔄 检测到重复局面: {repetition_count} 次")
        
        # 分析平局率
        draw_matches = re.findall(r'平局率: ([\d.]+)%', content)
        if draw_matches:
            draw_rates = [float(rate) for rate in draw_matches]
            if draw_rates and draw_rates[-1] > 90:
                print(f"⚠️  问题: 平局率过高 {draw_rates[-1]}%")
        
    except Exception as e:
        print(f"❌ 分析日志文件失败: {e}")

def check_model_files():
    """检查模型文件"""
    try:
        from config.config import Config
        config = Config()
        model_dir = Path(config.MODEL_DIR)
    except:
        model_dir = Path("models")
    
    print("\n=== 模型文件检查 ===\n")
    
    if not model_dir.exists():
        print("❌ 模型目录不存在")
        return
    
    # 检查模型文件
    model_files = list(model_dir.glob("*.pth"))
    if model_files:
        print(f"📁 发现 {len(model_files)} 个模型文件:")
        for model_file in sorted(model_files, key=lambda x: x.stat().st_mtime, reverse=True)[:5]:
            size_mb = model_file.stat().st_size / (1024 * 1024)
            mtime = datetime.fromtimestamp(model_file.stat().st_mtime)
            print(f"  {model_file.name} ({size_mb:.1f}MB, {mtime.strftime('%Y-%m-%d %H:%M')})")
    else:
        print("❌ 未找到模型文件")
    
    # 检查检查点
    checkpoint_dir = model_dir / "checkpoints"
    if checkpoint_dir.exists():
        checkpoints = list(checkpoint_dir.glob("*.pth"))
        if checkpoints:
            latest_checkpoint = max(checkpoints, key=lambda x: x.stat().st_mtime)
            print(f"📋 最新检查点: {latest_checkpoint.name}")

def check_training_data():
    """检查训练数据"""
    try:
        from config.config import Config
        config = Config()
        data_dir = Path(config.DATA_DIR)
    except:
        data_dir = Path("data")
    
    print("\n=== 训练数据检查 ===\n")
    
    if not data_dir.exists():
        print("❌ 数据目录不存在")
        return
    
    # 检查数据文件
    data_files = list(data_dir.glob("*.npz"))
    if data_files:
        total_size = sum(f.stat().st_size for f in data_files)
        print(f"📊 发现 {len(data_files)} 个数据文件，总大小: {total_size / (1024**3):.2f}GB")
        
        # 检查最新数据文件
        latest_data = max(data_files, key=lambda x: x.stat().st_mtime)
        print(f"📈 最新数据: {latest_data.name}")
    else:
        print("❌ 未找到训练数据文件")

def provide_recommendations():
    """提供建议"""
    print("\n=== 问题诊断和建议 ===\n")
    
    print("🔍 当前问题:")
    print("  1. AI陷入重复走棋循环")
    print("  2. 胜率始终0%，平局率100%")
    print("  3. 游戏长度固定，缺乏多样性")
    print("  4. 模型可能过拟合到安全策略")
    
    print("\n💡 解决建议:")
    print("  1. 运行修复脚本:")
    print("     python fix_repetition_issue.py")
    print("  2. 增加MCTS探索性参数")
    print("  3. 延长随机探索阶段")
    print("  4. 添加反重复走棋机制")
    print("  5. 考虑重新开始训练")
    
    print("\n⚡ 立即行动:")
    print("  - 停止当前训练")
    print("  - 备份重要模型")
    print("  - 运行修复脚本")
    print("  - 重新开始训练")

def main():
    print("=== Chess AI 训练状态诊断 ===\n")
    
    analyze_log_files()
    check_model_files()
    check_training_data()
    provide_recommendations()
    
    print("\n" + "="*50)
    print("诊断完成！请根据建议采取相应行动。")

if __name__ == "__main__":
    main()
