@echo off
echo === Git 上传脚本 ===

echo 1. 检查Git状态...
git status

echo.
echo 2. 添加所有更改...
git add .

echo.
echo 3. 提交更改...
git commit -m "修复硬编码路径问题，优化训练配置和错误处理

- 修复 fixed_test_training.py, monitor_training.py, test_training.py 中的硬编码 /root 路径
- 还原训练参数配置到原始值
- 添加改进的启动脚本 start_training.py，包含更好的错误处理和资源监控
- 添加配置测试脚本 test_config.py
- 更新 AutoDL 运行脚本
- 确保所有文件保存在项目目录而不是 /root 目录"

echo.
echo 4. 推送到GitHub...
git push origin master

echo.
echo === 上传完成 ===
pause
