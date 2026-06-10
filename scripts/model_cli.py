import time
import numpy as np
import threading
import logging
from typing import Any, Dict, Optional
import queue
import sys
import os

sys.path.append("/root/foresee/FastWAM/scripts")

from network_utils import WebsocketClientPolicy

logger = logging.getLogger(__name__)


# ===================== 模型监控器 =====================
class ModelMonitor:
    """
    模型监控器：直接通过 WebSocket 调用模型推理
    - 接收观测数据（obs），在推理线程中调用 WebsocketClientPolicy 进行预测
    - 预测完成后将动作放入 action 队列
    - 线程安全的消息队列和状态管理
    """

    def __init__(
        self,
        host: str = "0.0.0.0",
        port: int = 8765,
        api_key: Optional[str] = None,
        allow_reconnect: bool = True,
        obs_queue_size: int = 10,
        action_queue_size: int = 10,
    ):
        """
        初始化模型监控器
        :param host: WebSocket 模型服务地址
        :param port: WebSocket 模型服务端口
        :param api_key: 可选的 API Key
        :param allow_reconnect: 是否允许断线重连
        :param obs_queue_size: 观测队列最大长度
        :param action_queue_size: 动作队列最大长度
        """
        self.host = host
        self.port = port

        # 状态控制
        self.running = False

        # WebSocket 推理客户端
        self.policy = WebsocketClientPolicy(
            host=host, port=port, api_key=api_key, allow_reconnect=allow_reconnect
        )

        # 观测消息队列（线程安全）
        self.obs_queue: queue.Queue = queue.Queue(maxsize=obs_queue_size)

        # 动作结果队列（线程安全），预测完成后放入
        self.action_queue: queue.Queue = queue.Queue(maxsize=action_queue_size)

        # 推理工作线程
        self.infer_thread: Optional[threading.Thread] = None

    def start(self):
        """启动推理线程"""
        if self.running:
            logger.warning("ModelMonitor 已启动，无需重复启动")
            return

        self.running = True

        # 启动推理线程
        self.infer_thread = threading.Thread(target=self._infer_loop, daemon=True)
        self.infer_thread.start()

        logger.info(f"ModelMonitor 已启动，连接模型 ws://{self.host}:{self.port}")

    def stop(self):
        """停止推理线程并清理资源"""
        self.running = False

        if self.infer_thread:
            self.infer_thread.join(timeout=5.0)

        logger.info("ModelMonitor 已停止")

    def _infer_loop(self):
        """推理循环（独立线程）：从 obs_queue 取观测，调用 policy 推理，结果放入 action_queue"""
        while self.running:
            try:
                # 从观测队列取数据，超时后继续循环检查 running
                try:
                    obs_data = self.obs_queue.get(timeout=0.05)
                except queue.Empty:
                    continue

                # 调用 WebSocket 推理
                start_t = time.time()
                infer_result = self.policy.act(obs_data)
                elapsed = time.time() - start_t

                action = (infer_result["action"])
                action_data = {"action": action}

                # 如果返回了 success_proba，也保留
                if "success_proba" in infer_result:
                    action_data["success_proba"] = (infer_result["success_proba"])
                if "keypoint" in infer_result:
                    action_data["keypoint"] = infer_result["keypoint"]

                action_data["timestamp"] = time.time()

                logger.info(f"推理完成，耗时 {elapsed:.3f}s，action shape: {action.shape}")

                # 放入动作队列，队列满时丢弃最旧的
                if self.action_queue.full():
                    try:
                        self.action_queue.get_nowait()
                    except queue.Empty:
                        pass
                self.action_queue.put(action_data)

            except Exception as e:
                logger.error(f"推理失败: {e}")
                time.sleep(0.01)

    def send_observation(self, obs_data: Dict[str, Any]):
        """
        发送观测数据进行推理（线程安全）
        :param obs_data: 观测数据字典
        """
        if not self.running:
            logger.warning("ModelMonitor 未启动，丢弃观测数据")
            return

        # 如果队列满，丢弃旧的观测，保留最新
        if self.obs_queue.full():
            try:
                self.obs_queue.get_nowait()
            except queue.Empty:
                pass

        try:
            self.obs_queue.put_nowait(obs_data)
        except queue.Full:
            logger.warning("观测队列已满，丢弃当前观测")
        except Exception as e:
            logger.error(f"添加观测到队列失败: {e}")

    def get_latest_action(self) -> Optional[Dict[str, Any]]:
        """
        获取最新的模型动作（线程安全）
        会清空队列中的旧动作，只返回最新的一个
        :return: 动作字典 {"action": np.ndarray, "timestamp": float, ...} 或 None
        """
        latest = None
        while not self.action_queue.empty():
            try:
                latest = self.action_queue.get_nowait()
            except queue.Empty:
                break
        return latest

    def get_all_actions(self) -> list:
        """
        获取动作队列中的所有动作
        :return: 动作字典列表
        """
        actions = []
        while not self.action_queue.empty():
            try:
                actions.append(self.action_queue.get_nowait())
            except queue.Empty:
                break
        return actions

    def reset(self):
        """重置模型（如需要）"""
        try:
            self.policy.reset()
            logger.info("模型已重置")
        except Exception as e:
            logger.error(f"模型重置失败: {e}")
