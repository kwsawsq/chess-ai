#!/bin/bash

echo "=== AutoDL 训练启动脚本 ==="

# 检查GPU状态
echo "检查GPU状态..."
nvidia-smi

# 安装依赖
echo "安装依赖..."
pip install -r requirements_autodl.txt

# 创建必要的目录
echo "创建目录..."
mkdir -p data models logs models/checkpoints

# 检查系统资源
echo "检查系统资源..."
free -h
df -h

# 启动训练（使用改进的启动脚本）
echo "启动训练..."
python start_training.py