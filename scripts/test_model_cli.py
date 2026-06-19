import time
import os 
import pickle
import sys
import uuid
import numpy as np
from pathlib import Path

sys.path.append("/root/foresee/FastWAM/scripts")

 
from model_cli import ModelMonitor


def test_model_infer_random_data():
    host = "127.0.0.1"
    port = 6006
    # https://u673024-ae2e-cd2077e9.westc.seetacloud.com:8443
    # host = "connect.westc.seetacloud.com"
    # port = 15245
    model_monitor = ModelMonitor(host=host, port=port)
    model_monitor.start()

    frame = {
        "observation.images.head_right": np.random.rand(1, 3, 256, 256).astype(np.float32),
        "observation.images.head_left": np.random.rand(1, 3, 256, 256).astype(np.float32),
        "observation.images.hand_right": np.random.rand(1, 3, 256, 256).astype(np.float32),
        "observation.images.hand_left": np.random.rand(1, 3, 256, 256).astype(np.float32),
        "observation.state": np.random.rand(1, 28).astype(np.float32),
        "idx": 1,
        "task": "right_pangceban",
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
    
    model_monitor.reset()
    

def test_read_dataset_and_infer():
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from tqdm import tqdm

    dataset_path = "/root/autodl-fs/datasets/test/pangceban_left_right_data_20260602_658_delta_action"
    dataset = LeRobotDataset(repo_id="replay", root=dataset_path, video_backend="pyav", episodes=[15])
    DURATION_TIME = 300
    EVAL_TIME = 10
    n = min(DURATION_TIME, len(dataset))
    
    host = "127.0.0.1"
    port = 6006
    # host = "connect.westc.seetacloud.com"
    # port = 15245
    model_monitor = ModelMonitor(host=host, port=port)
    model_monitor.start()
    

    for i in tqdm(range(n)):
        if i % EVAL_TIME != 0:
            continue
        temp = dataset[i]
        print(f'eval step {i}')
        frame = {
            "observation.images.head_right": np.array(temp["observation.images.head_right"]),
            "observation.images.head_left": np.array(temp["observation.images.head_left"]),
            "observation.images.hand_right": np.array(temp["observation.images.hand_right"]),
            "observation.images.hand_left": np.array(temp["observation.images.hand_left"]),
            "observation.state": np.array(temp["observation.state"]),
            "control": np.array(temp["control"]),
            "idx": i,
            "task": "right_pangceban",
        }
        model_monitor.send_observation(frame)
        time.sleep(3)
    
    model_monitor.reset()
    ...
 
def test_model_infer_libero_image():
    """从本地图片读取，测试 LIBERO 任务推理。"""
    from PIL import Image
    # host = "127.0.0.1"
    # port = 6006
    # https://u673024-ae2e-cd2077e9.westc.seetacloud.com:8443
    
    host = "u673024-ae2e-cd2077e9.westc.seetacloud.com"
    port = 6006
    model_monitor = ModelMonitor(host=host, port=port)
    model_monitor.start()

    # 读取图片并转为 (1, 3, H, W) float32
    img_path = "/root/foresee/FastWAM/debug_images/image.png"
    wrist_path = "/root/foresee/FastWAM/debug_images/wrist_image.png"

    img = np.array(Image.open(img_path).convert("RGB"), dtype=np.float32)  # (H, W, 3)
    img = np.transpose(img, (2, 0, 1))[None] / 255.0  # (1, 3, H, W)

    wrist_img = np.array(Image.open(wrist_path).convert("RGB"), dtype=np.float32)  # (H, W, 3)
    wrist_img = np.transpose(wrist_img, (2, 0, 1))[None] / 255.0 # (1, 3, H, W)

    frame = {
        "observation.images.image": img,
        "observation.images.wrist_image": wrist_img,
        "observation.state": np.array([[-2.1681985e-01 , 9.1387788e-03 , 1.1711580e+00 , 3.1401653e+00, 1.6791669e-03, -8.8790134e-02, 3.9506316e-02, -3.9507169e-02]]),
        "idx": 1,
        "task": "open the middle drawer of the cabinet",
    }

    # 将 frame 序列化到文件，方便后续复现
    frame_dump_path = os.path.join(os.path.dirname(__file__), "test_frame_libero.pkl")
    with open(frame_dump_path, "wb") as f:
        pickle.dump(frame, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"Frame dumped to {frame_dump_path}")

    model_monitor.send_observation(frame)
    time.sleep(5)
    latest_action = model_monitor.get_latest_action()
    print(latest_action)

    # 将 action 结果序列化到文件
    action_dump_path = os.path.join(os.path.dirname(__file__), f"test_action_libero_{uuid.uuid4().hex[:8]}.pkl")
    with open(action_dump_path, "wb") as f:
        pickle.dump(latest_action, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"Action dumped to {action_dump_path}")


if __name__ == "__main__":
    test_read_dataset_and_infer() # 从数据集里面读取数据并推理
    # test_model_infer_random_data()  # 随机数据推理
    # test_model_infer_libero_image()  # 本地图片推理