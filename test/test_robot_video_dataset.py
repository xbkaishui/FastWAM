"""
简单测试脚本：验证 RobotVideoDataset 数据加载内容。
用法:
    cd /root/foresee/FastWAM
    python test/test_robot_video_dataset.py
"""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
os.chdir(os.path.join(os.path.dirname(__file__), ".."))

import torch
import torchvision
from omegaconf import OmegaConf
from fastwam.datasets.lerobot.robot_video_dataset import RobotVideoDataset


def load_config():
    """加载 libero_2cam 数据配置，并包裹在 data 层级下以解析 OmegaConf 插值"""
    config_path = "configs/data/libero_2cam.yaml"
    config_path = "configs/data/pangceban_3cam.yaml"
    data_cfg = OmegaConf.load(config_path)
    # processor 中有 ${data.train.shape_meta} 等插值，需要从根节点能解析到
    full_cfg = OmegaConf.create({"data": data_cfg})
    return full_cfg


def test_dataset_basic():
    """测试基本的数据集加载和单样本获取"""
    cfg = load_config()
    train_cfg = cfg.data.train

    print("=" * 60)
    print("创建 RobotVideoDataset (训练集) ...")
    print("=" * 60)
    dataset_dirs = train_cfg.dataset_dirs
    dataset_dirs = [train_cfg.dataset_dirs[0], train_cfg.dataset_dirs[1]]
    dataset = RobotVideoDataset(
        dataset_dirs=dataset_dirs,
        shape_meta=train_cfg.shape_meta,
        num_frames=train_cfg.num_frames,
        video_size=train_cfg.video_size,
        camera_key=train_cfg.camera_key,
        processor=train_cfg.processor,
        text_embedding_cache_dir=train_cfg.text_embedding_cache_dir,
        context_len=train_cfg.context_len,
        val_set_proportion=train_cfg.val_set_proportion,
        is_training_set=True,
        global_sample_stride=train_cfg.global_sample_stride,
        action_video_freq_ratio=train_cfg.action_video_freq_ratio,
        skip_padding_as_possible=train_cfg.get("skip_padding_as_possible", False),
        concat_multi_camera=train_cfg.get("concat_multi_camera", "horizontal"),
    )

    print(f"\n数据集大小: {len(dataset)}")
    print(f"num_frames: {train_cfg.num_frames}")
    print(f"action_video_freq_ratio: {train_cfg.action_video_freq_ratio}")
    print(f"video_size: {train_cfg.video_size}")
    print(f"video_sample_indices: {dataset.video_sample_indices}")

    # 获取第一个样本
    print("\n" + "=" * 60)
    print("获取第 0 个样本 ...")
    print("=" * 60)
    sample = dataset[0]

    print(f"\n样本包含的 keys: {list(sample.keys())}")
    print("-" * 40)

    # 打印每个字段的 shape/type/范围
    for key, val in sample.items():
        if isinstance(val, torch.Tensor):
            print(f"  {key:20s} | shape={str(list(val.shape)):20s} | dtype={val.dtype} | range=[{val.min().item():.4f}, {val.max().item():.4f}]")
        elif isinstance(val, str):
            print(f"  {key:20s} | type=str | value={val[:80]}{'...' if len(val) > 80 else ''}")
        else:
            print(f"  {key:20s} | type={type(val).__name__} | value={val}")

    # 验证关键维度关系
    print("\n" + "=" * 60)
    print("验证维度关系")
    print("=" * 60)

    video = sample["video"]
    action = sample["action"]
    proprio = sample["proprio"]
    context = sample["context"]
    context_mask = sample["context_mask"]

    C, T_video, H, W = video.shape
    T_action, action_dim = action.shape
    T_proprio, proprio_dim = proprio.shape

    print(f"  video:   [C={C}, T_video={T_video}, H={H}, W={W}]")
    print(f"  action:  [T_action={T_action}, action_dim={action_dim}]")
    print(f"  proprio: [T_proprio={T_proprio}, proprio_dim={proprio_dim}]")
    print(f"  context: {list(context.shape)}, context_mask: {list(context_mask.shape)}")

    # 检查约束
    assert T_action == train_cfg.num_frames - 1, \
        f"T_action({T_action}) != num_frames-1({train_cfg.num_frames - 1})"
    assert T_action % (T_video - 1) == 0, \
        f"T_action({T_action}) 不能被 T_video-1({T_video - 1}) 整除"
    ratio = T_action // (T_video - 1)
    assert ratio == train_cfg.action_video_freq_ratio, \
        f"推算 ratio={ratio} != action_video_freq_ratio={train_cfg.action_video_freq_ratio}"
    assert T_proprio == T_action, \
        f"T_proprio({T_proprio}) != T_action({T_action})"

    print(f"\n  ✓ T_action = num_frames - 1 = {T_action}")
    print(f"  ✓ T_action / (T_video - 1) = {T_action} / {T_video - 1} = {ratio} == action_video_freq_ratio")
    print(f"  ✓ T_proprio == T_action = {T_proprio}")
    print(f"  ✓ video range in [-1, 1]: [{video.min().item():.4f}, {video.max().item():.4f}]")

    # 保存视频第一帧图片用于可视化验证
    print("\n" + "=" * 60)
    print("保存视频第一帧图片进行可视化验证")
    print("=" * 60)
    save_dir = "test/visualize"
    os.makedirs(save_dir, exist_ok=True)
    # video shape: [C, T_video, H, W], 取第一帧 -> [C, H, W]
    first_frame = video[:, 0, :, :]  # [C, H, W]
    # 从 [-1, 1] 反归一化到 [0, 1]
    first_frame = (first_frame + 1.0) / 2.0
    first_frame = first_frame.clamp(0, 1)
    save_path = os.path.join(save_dir, "sample0_first_frame.png")
    torchvision.utils.save_image(first_frame, save_path)
    print(f"  已保存第一帧图片到: {save_path}")

    print("\n" + "=" * 60)
    print("测试多个样本 (随机采样 5 个)")
    print("=" * 60)
    import random
    indices = random.sample(range(len(dataset)), min(5, len(dataset)))
    for i, idx in enumerate(indices):
        s = dataset[idx]
        v = s["video"]
        a = s["action"]
        prompt = s["prompt"]
        print(f"  样本 {idx:6d} | video={list(v.shape)} | action={list(a.shape)} | prompt={prompt[:60]}...")

    print("\n" + "=" * 60)
    print("所有测试通过!")
    print("=" * 60)


if __name__ == "__main__":
    test_dataset_basic()
