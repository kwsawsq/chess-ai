"""
自我对弈工作进程模块
将工作函数分离以避免多进程死锁
"""

import torch
import numpy as np
from typing import List, Tuple, Optional, Dict, Any
import sys
import os
import logging

# 全局变量
g_model: Optional[Any] = None
g_config: Optional[Any] = None

def init_worker_self_play(model_state: Dict, config: Any):
    """
    初始化自我对弈工作进程。
    """
    global g_model, g_config
    
    try:
        g_config = config
        
        # 为每个工作进程设置单独的日志文件
        worker_log_dir = os.path.join(g_config.LOG_DIR, 'workers')
        os.makedirs(worker_log_dir, exist_ok=True)
        worker_log_file = os.path.join(worker_log_dir, f'worker_{os.getpid()}.log')
        
        # 配置日志
        logging.basicConfig(
            level=logging.DEBUG,
            format='%(asctime)s - %(process)d - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(worker_log_file),
                logging.StreamHandler(sys.stdout) # 将日志也输出到标准输出
            ],
            force=True # 覆盖任何现有的日志配置
        )
        
        logger = logging.getLogger(__name__)

        logger.info(f"工作进程 {os.getpid()} 开始初始化...")
        
        # 禁用梯度计算以进行推理优化
        torch.set_grad_enabled(False)
        
        # 动态导入模块以避免循环导入问题
        from src.game import ChessGame
        from src.neural_network import AlphaZeroNet
        from src.mcts import MCTS
        
        # 在新进程中重新创建和加载模型
        g_model = AlphaZeroNet(g_config)
        g_model.load_state_dict(model_state)
        g_model.to(g_config.DEVICE)
        g_model.eval()
        
        logger.info(f"工作进程 {os.getpid()} 初始化成功，设备: {g_config.DEVICE}")
        
    except Exception as e:
        logger.error(f"工作进程 {os.getpid()} 初始化失败: {e}", exc_info=True)
        raise

def play_one_game_worker() -> Optional[List[Tuple[np.ndarray, np.ndarray, float]]]:
    """
    在工作进程中进行一局自我对弈。
    """
    global g_model, g_config
    logger = logging.getLogger(__name__)
    worker_id = os.getpid()
    
    try:
        logger.info(f"[Worker {worker_id}] 开始一局新的对弈。")
        # 动态导入模块
        from src.game import ChessGame
        from src.mcts import MCTS
        
        game = ChessGame(g_config)
        mcts = MCTS(g_model, g_config)
        
        states_hist, policies_hist, values_hist = [], [], []
        move_count = 0

        while not game.is_over():
            move_count += 1
            logger.debug(f"[Worker {worker_id}] 对弈进行中，第 {move_count} 步...")
            state = game.get_state()
            policy, value = mcts.search(game.board)
            
            states_hist.append(state)
            policies_hist.append(policy)
            values_hist.append(value)
            
            # 根据温度参数选择确定性或随机性走法
            temp = 1 if len(game.board.board.move_stack) <= g_config.TEMP_THRESHOLD else 0
            move = game.select_move(policy, deterministic=(temp == 0))
            game.make_move(move)

        if not game.is_over():
            logger.warning(f"[Worker {worker_id}] 对弈异常结束，未分出胜负。")
            return None

        game_result = game.get_result()
        logger.info(f"[Worker {worker_id}] 对弈结束，共 {move_count} 步，结果: {game_result}")
        # 根据最终结果调整价值
        final_values = []
        for i in range(len(values_hist)):
            # 价值是从当前玩家的角度出发的，所以需要交替
            value = game_result if i % 2 == 0 else -game_result
            final_values.append(value)

        return list(zip(states_hist, policies_hist, final_values))
        
    except Exception as e:
        logger.error(f"[Worker {worker_id}] 对弈失败: {e}", exc_info=True)
        return None 