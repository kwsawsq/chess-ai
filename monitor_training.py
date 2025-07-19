#!/usr/bin/env python
"""
训练性能监控脚本
用于分析GPU利用率和训练瓶颈
"""

import time
import torch
import psutil
import numpy as np
import sys
import threading
from datetime import datetime

def monitor_system():
    """监控系统资源使用情况"""
    print("=== 系统性能监控 ===")
    print(f"时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # CPU信息
    cpu_percent = psutil.cpu_percent(interval=1)
    cpu_count = psutil.cpu_count()
    print(f"CPU: {cpu_percent:.1f}% ({cpu_count} cores)")
    
    # 内存信息
    memory = psutil.virtual_memory()
    print(f"内存: {memory.percent:.1f}% ({memory.available/1024**3:.1f}GB 可用)")
    
    # GPU信息
    if torch.cuda.is_available():
        device = torch.cuda.current_device()
        gpu_name = torch.cuda.get_device_name(device)
        
        # GPU内存
        memory_allocated = torch.cuda.memory_allocated(device)
        memory_reserved = torch.cuda.memory_reserved(device)
        memory_total = torch.cuda.get_device_properties(device).total_memory
        
        print(f"GPU: {gpu_name}")
        print(f"GPU内存已分配: {memory_allocated/1024**3:.2f}GB")
        print(f"GPU内存已保留: {memory_reserved/1024**3:.2f}GB")
        print(f"GPU内存总计: {memory_total/1024**3:.2f}GB")
        print(f"GPU内存使用率: {memory_allocated/memory_total*100:.1f}%")
        
        # GPU利用率（需要nvidia-ml-py库）
        try:
            import pynvml
            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(device)
            util = pynvml.nvmlDeviceGetUtilizationRates(handle)
            print(f"GPU计算利用率: {util.gpu}%")
            print(f"GPU内存利用率: {util.memory}%")
        except ImportError:
            print("GPU利用率: 需要安装pynvml库")
    else:
        print("GPU: 不可用")
    
    print("-" * 50)

def benchmark_model_inference():
    """测试模型推理性能"""
    print("=== 模型推理性能测试 ===")

    # 使用相对路径而不是硬编码的绝对路径
    current_dir = os.path.dirname(os.path.abspath(__file__))
    sys.path.append(current_dir)
    from config.config import config
    from src.neural_network.model import AlphaZeroNet
    
    # 创建模型
    model = AlphaZeroNet(config)
    model.to(config.DEVICE)
    model.eval()
    
    # 测试不同批处理大小的性能
    batch_sizes = [1, 4, 8, 16, 32, 64]
    
    for batch_size in batch_sizes:
        # 创建测试数据
        test_input = torch.randn(batch_size, 20, 8, 8, device=config.DEVICE)
        
        # 预热
        with torch.no_grad():
            for _ in range(10):
                _ = model(test_input)
        
        # 计时
        torch.cuda.synchronize()  # 等待GPU操作完成
        start_time = time.time()
        
        num_iterations = 100
        with torch.no_grad():
            for _ in range(num_iterations):
                _ = model(test_input)
        
        torch.cuda.synchronize()
        end_time = time.time()
        
        avg_time = (end_time - start_time) / num_iterations
        throughput = batch_size / avg_time
        
        print(f"批处理大小 {batch_size:2d}: {avg_time*1000:.2f}ms/batch, {throughput:.1f} samples/s")
    
    print("-" * 50)

def benchmark_mcts_performance():
    """测试MCTS性能"""
    print("=== MCTS性能测试 ===")

    # 使用相对路径而不是硬编码的绝对路径
    current_dir = os.path.dirname(os.path.abspath(__file__))
    sys.path.append(current_dir)
    from config.config import config
    from src.neural_network.model import AlphaZeroNet
    from src.mcts.mcts import MCTS
    from src.game.chess_board import ChessBoard
    
    # 创建模型和MCTS
    model = AlphaZeroNet(config)
    model.to(config.DEVICE)
    model.eval()
    
    mcts = MCTS(model, config)
    board = ChessBoard()
    
    print(f"MCTS配置: {config.NUM_MCTS_SIMS} 模拟, 批处理大小 {config.MCTS_BATCH_SIZE}")
    
    # 计时MCTS搜索
    start_time = time.time()
    policy, value = mcts.search(board, add_noise=False)
    end_time = time.time()
    
    search_time = end_time - start_time
    simulations_per_second = config.NUM_MCTS_SIMS / search_time
    
    print(f"MCTS搜索时间: {search_time:.2f}s")
    print(f"模拟速度: {simulations_per_second:.1f} sims/s")
    print(f"每次模拟平均时间: {search_time/config.NUM_MCTS_SIMS*1000:.2f}ms")
    
    print("-" * 50)

def continuous_monitor(interval=10):
    """持续监控系统资源"""
    print("开始持续监控... (Ctrl+C停止)")
    try:
        while True:
            monitor_system()
            time.sleep(interval)
    except KeyboardInterrupt:
        print("监控停止。")

def main():
    """主函数"""
    print("AlphaZero训练性能分析工具")
    print("=" * 50)
    
    # 1. 系统资源监控
    monitor_system()
    
    # 2. 模型推理性能测试
    benchmark_model_inference()
    
    # 3. MCTS性能测试
    benchmark_mcts_performance()
    
    # 4. 询问是否继续监控
    response = input("是否开始持续监控? (y/n): ")
    if response.lower() == 'y':
        continuous_monitor()

if __name__ == "__main__":
    main() 