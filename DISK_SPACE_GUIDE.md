# Chess AI 磁盘空间管理指南

本指南帮助您解决在AutoDL等环境中训练Chess AI时遇到的磁盘空间不足问题。

## 🚨 问题症状

如果您遇到以下错误，说明磁盘空间不足：
```
OSError: [Errno 28] No space left on device
```

## 🛠️ 解决方案

### 方案1: 自动设置数据盘 (推荐)

1. **运行设置脚本**
   ```bash
   python setup_data_disk.py
   ```
   
2. **设置环境变量**
   ```bash
   source set_data_path.sh
   ```
   
3. **开始训练**
   ```bash
   python train_with_data_disk.py
   ```

### 方案2: 手动指定数据盘

```bash
# 设置环境变量指向您的数据盘
export CHESS_AI_DATA_PATH=/root/autodl-tmp

# 运行训练
python fixed_test_training.py
```

### 方案3: 一行命令

```bash
CHESS_AI_DATA_PATH=/root/autodl-tmp python train_with_data_disk.py
```

## 📊 磁盘空间管理工具

### 分析空间使用情况
```bash
python disk_space_manager.py analyze
```
输出示例：
```
=== Chess AI 磁盘空间分析 ===

数据目录: /root/autodl-tmp/chess-ai-data
  大小: 2.3GB
  最大的文件:
    final_training_data.npz: 1.8GB
    game_data_1234567890_0.pkl: 256.7MB
    training_batch_50.npz: 128.3MB

模型目录: /root/autodl-tmp/chess-ai-models
  大小: 456.7MB
  最大的文件:
    final_model.pth: 123.4MB
    checkpoint_iter_50.pth: 123.4MB
    best_model.pth: 123.4MB

项目总大小: 2.8GB

磁盘空间 (/root/autodl-tmp):
  总计: 50.0GB
  已用: 15.2GB (30.4%)
  可用: 34.8GB (69.6%)
  ✅ 空间充足
```

### 预览清理操作
```bash
python disk_space_manager.py preview
```

### 执行清理
```bash
python disk_space_manager.py cleanup
```

## 🔧 自动化功能

### 训练过程中的自动管理

使用 `train_with_data_disk.py` 脚本时，会自动：

1. **磁盘空间检查**: 每次迭代前检查可用空间
2. **自动清理**: 空间不足时自动删除旧文件
3. **后台监控**: 每5分钟检查一次磁盘空间
4. **智能保存**: 根据可用空间决定是否保存大文件

### 清理策略

- **模型文件**: 保留最新5个 `.pth` 文件
- **数据文件**: 保留最新3个 `.npz` 文件  
- **日志文件**: 保留最近7天的日志
- **临时文件**: 删除 `.tmp`, `.temp`, `*~` 等临时文件

## 📁 目录结构

使用数据盘后，文件将保存在：

```
/root/autodl-tmp/  (或您指定的数据盘路径)
├── chess-ai-data/          # 训练数据
│   ├── final_training_data.npz
│   └── game_data_*.pkl
├── chess-ai-models/        # 模型文件
│   ├── final_model.pth
│   ├── best_model.pth
│   └── checkpoint_iter_*.pth
├── chess-ai-logs/          # 日志文件
│   └── training/
└── chess-ai-games/         # PGN棋谱文件
```

## ⚙️ 配置选项

在 `config/config.py` 中，您可以调整：

```python
# 磁盘空间管理配置
self.min_free_space_gb = 1.0      # 最小保留空间(GB)
self.max_model_files = 5          # 最多保留的模型文件数
self.max_data_files = 3           # 最多保留的数据文件数
```

## 🚀 最佳实践

1. **定期监控**: 使用 `disk_space_manager.py analyze` 定期检查空间使用
2. **及时清理**: 训练完成后运行 `disk_space_manager.py cleanup`
3. **合理配置**: 根据您的磁盘大小调整保留文件数量
4. **备份重要模型**: 将最佳模型复制到安全位置

## 🔍 故障排除

### 问题1: 找不到数据盘
```bash
# 手动检查可用挂载点
df -h
mount | grep -E "(autodl|tmp|data)"

# 手动指定路径
export CHESS_AI_DATA_PATH=/your/data/path
```

### 问题2: 权限不足
```bash
# 检查目录权限
ls -la /root/autodl-tmp

# 创建目录
mkdir -p /root/autodl-tmp/chess-ai-data
```

### 问题3: 仍然空间不足
```bash
# 强制清理
python disk_space_manager.py cleanup

# 减少保留文件数量
# 编辑 config/config.py，减少 max_model_files 和 max_data_files
```

## 📞 获取帮助

如果仍然遇到问题，请：

1. 运行 `python disk_space_manager.py analyze` 获取详细信息
2. 检查日志文件中的错误信息
3. 确认数据盘路径和权限设置正确

---

通过以上方法，您应该能够成功解决磁盘空间不足的问题，并顺利进行Chess AI的训练。
