#!/bin/bash

# 安装依赖
pip install -r requirements_autodl.txt

# 创建必要的目录
mkdir -p data models logs

# 启动训练
python -m src.main train 