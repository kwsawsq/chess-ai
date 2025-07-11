# AlphaZero 国际象棋AI

这是一个完整的AlphaZero算法实现，用于训练国际象棋AI。

## 项目结构

```
chess-ai/
├── src/                  # 源代码
│   ├── game/            # 游戏引擎
│   ├── neural_network/  # 神经网络
│   ├── mcts/            # 蒙特卡洛树搜索
│   ├── self_play/       # 自我对弈
│   ├── training/        # 训练模块
│   └── evaluation/      # 评估模块
├── config/              # 配置文件
├── models/              # 保存的模型
├── data/               # 训练数据
├── logs/               # 日志文件
└── tests/              # 测试文件
```

## 安装依赖

```bash
pip install -r requirements.txt
```

## 使用方法

### 开始训练
```bash
python main.py --mode train
```

### 人机对弈
```bash
python main.py --mode play
```

### 评估模型
```bash
python main.py --mode evaluate
```

## 算法概述

AlphaZero算法结合了深度神经网络和蒙特卡洛树搜索：

1. **神经网络**：输入棋盘状态，输出每个动作的概率分布和位置评估
2. **MCTS**：使用神经网络指导搜索，提高搜索效率
3. **自我对弈**：AI与自己对弈生成训练数据
4. **训练**：使用自我对弈数据训练神经网络

## 特性

- 完整的AlphaZero算法实现
- 支持国际象棋完整规则
- 可视化训练过程
- 模型评估和比较
- 分布式训练支持
- 人机对弈界面 