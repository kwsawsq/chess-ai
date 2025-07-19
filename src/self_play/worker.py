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
from datetime import datetime

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
            # logger.debug(f"[Worker {worker_id}] 对弈进行中，第 {move_count} 步...")
            
            # 使用MCTS进行搜索
            policy, value = mcts.search(game.board, add_noise=True)
            
            states_hist.append(game.get_state())
            policies_hist.append(policy)
            values_hist.append(value)
            
            # 根据温度参数选择确定性或随机性走法
            temp = 1 if len(game.board.board.move_stack) <= g_config.TEMP_THRESHOLD else 0
            move = game.select_move(policy, deterministic=(temp == 0))
            game.make_move(move)
            
            # 添加调试信息
            if move_count <= 5 or move_count % 10 == 0:
                logger.debug(f"[Worker {worker_id}] 第{move_count}步: {move}, 价值: {value:.3f}, 温度: {temp}")

        if not game.is_over():
            logger.warning(f"[Worker {worker_id}] 对弈异常结束，未分出胜负。")
            return None

        game_result = game.get_result()
        logger.info(f"[Worker {worker_id}] 对弈结束，共 {move_count} 步，结果: {game_result}")
        
        # 添加更详细的结果信息
        if game_result == 0:
            termination_reason = game.board.get_game_termination_reason()
            logger.info(f"[Worker {worker_id}] 和棋原因: {termination_reason}")
        else:
            logger.info(f"[Worker {worker_id}] 获胜方: {'白方' if game_result == 1 else '黑方'}")

        # 如果配置允许，保存PGN文件
        if g_config.SAVE_PGN:
            try:
                # 确保PGN目录存在
                os.makedirs(g_config.PGN_DIR, exist_ok=True)
                
                # 创建唯一文件名
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                pgn_filename = f"game_{timestamp}_worker_{worker_id}.pgn"
                pgn_filepath = os.path.join(g_config.PGN_DIR, pgn_filename)
                
                # 导出并保存PGN
                pgn_data = game.board.export_pgn()
                with open(pgn_filepath, "w", encoding="utf-8") as f:
                    f.write(pgn_data)
                logger.debug(f"[Worker {worker_id}] PGN棋谱已保存至 {pgn_filepath}")
                
            except Exception as e:
                logger.error(f"[Worker {worker_id}] 保存PGN文件时出错: {e}", exc_info=True)

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