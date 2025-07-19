#!/usr/bin/env python3
"""
Git上传脚本 - 自动提交和推送更改
"""

import subprocess
import sys
import os

def run_command(cmd, cwd=None):
    """运行命令并返回结果"""
    try:
        result = subprocess.run(
            cmd, 
            shell=True, 
            capture_output=True, 
            text=True, 
            cwd=cwd or os.getcwd()
        )
        return result.returncode == 0, result.stdout, result.stderr
    except Exception as e:
        return False, "", str(e)

def main():
    """主函数"""
    print("=== Git 上传脚本 ===")
    
    # 检查Git状态
    print("1. 检查Git状态...")
    success, stdout, stderr = run_command("git status --porcelain")
    if not success:
        print(f"错误: 无法检查Git状态: {stderr}")
        return 1
    
    if not stdout.strip():
        print("没有需要提交的更改")
        return 0
    
    print("发现以下更改:")
    print(stdout)
    
    # 添加所有更改
    print("\n2. 添加所有更改...")
    success, stdout, stderr = run_command("git add .")
    if not success:
        print(f"错误: 无法添加文件: {stderr}")
        return 1
    print("✓ 文件添加成功")
    
    # 提交更改
    commit_message = "修复硬编码路径问题，优化训练配置和错误处理"
    print(f"\n3. 提交更改: {commit_message}")
    success, stdout, stderr = run_command(f'git commit -m "{commit_message}"')
    if not success:
        if "nothing to commit" in stderr:
            print("没有需要提交的更改")
            return 0
        print(f"错误: 无法提交更改: {stderr}")
        return 1
    print("✓ 提交成功")
    
    # 推送到远程仓库
    print("\n4. 推送到GitHub...")
    success, stdout, stderr = run_command("git push origin master")
    if not success:
        print(f"错误: 无法推送到远程仓库: {stderr}")
        print("可能需要先设置远程仓库或解决冲突")
        return 1
    print("✓ 推送成功")
    
    print("\n=== 上传完成 ===")
    print("所有更改已成功上传到GitHub!")
    return 0

if __name__ == "__main__":
    exit(main())
