import time
import os 
import pickle
import sys
import uuid
import numpy as np
from pathlib import Path

sys.path.append("/root/foresee/FastWAM/scripts")

 
from model_cli import ModelMonitor


def test_model_infer():
    model_monitor = ModelMonitor()
    model_monitor.start()

    frame = {
        "observation.images.head_right": np.random.rand(1, 3, 256, 256).astype(np.float32),
        "observation.images.head_left": np.random.rand(1, 3, 256, 256).astype(np.float32),
        "observation.images.hand_right": np.random.rand(1, 3, 256, 256).astype(np.float32),
        "observation.images.hand_left": np.random.rand(1, 3, 256, 256).astype(np.float32),
        "observation.images.orbbec": np.random.rand(1, 3, 256, 256).astype(np.float32),
        "observation.state": np.random.rand(1, 28).astype(np.float32),
        "idx": 1,
        "task": "task_grab",
    }

    # 将 frame 序列化到文件，方便后续复现
    frame_dump_path = os.path.join(os.path.dirname(__file__), "test_frame.pkl")
    with open(frame_dump_path, "wb") as f:
        pickle.dump(frame, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"Frame dumped to {frame_dump_path}")

    model_monitor.send_observation(frame)
    time.sleep(5)
    latest_action = model_monitor.get_latest_action()
    print(latest_action)

    # 将 action 结果序列化到文件，方便后续复现
    action_dump_path = os.path.join(os.path.dirname(__file__), f"test_action_{uuid.uuid4().hex[:8]}.pkl")
    with open(action_dump_path, "wb") as f:
        pickle.dump(latest_action, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"Action dumped to {action_dump_path}")
 
if __name__ == "__main__":
    test_model_infer()          # 随机数据推理