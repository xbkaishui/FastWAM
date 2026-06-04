"""
FastWAM model inference debug script.

Uses random data to test the model's forward/inference execution flow.
No LIBERO simulation environment required.

Usage:
    cd /root/foresee/FastWAM
    python experiments/libero/test_fastwam_inference.py
"""

import logging
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

# Add project root to sys.path
project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

from fastwam.datasets.lerobot.processors.fastwam_processor import FastWAMProcessor
from fastwam.datasets.lerobot.utils.normalizer import load_dataset_stats_from_json
from fastwam.datasets.lerobot.robot_video_dataset import DEFAULT_PROMPT

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# Register OmegaConf resolvers used by configs
OmegaConf.register_new_resolver("eval", eval, replace=True)
OmegaConf.register_new_resolver("max", lambda x: max(x), replace=True)
OmegaConf.register_new_resolver("split", lambda s, idx: s.split("/")[int(idx)], replace=True)

# ===== Configuration =====
CKPT_PATH = "/root/autodl-fs/ckpts/models/fastwam/libero_uncond_2cam224.pt"
DATASET_STATS_PATH = "/root/autodl-fs/ckpts/models/fastwam/libero_uncond_2cam224_dataset_stats.json"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"DEVICE: {DEVICE}")
MIXED_PRECISION = "bf16"

# Image dimensions from config: video_size=[224, 448] (H, W) for 2-camera horizontal concat
INPUT_H = 224
INPUT_W = 448  # 224 * 2 cameras concatenated horizontally

# Action config
ACTION_DIM = 7
PROPRIO_DIM = 8
ACTION_HORIZON = 32  # num_frames - 1 = 33 - 1

# Inference params
NUM_INFERENCE_STEPS = 20
TASK_DESCRIPTION = "pick up the red block and place it on the shelf"


def get_model_dtype(mixed_precision: str) -> torch.dtype:
    if mixed_precision == "bf16":
        return torch.bfloat16
    elif mixed_precision == "fp16":
        return torch.float16
    return torch.float32


def create_random_image(height: int, width: int, device: str, dtype: torch.dtype) -> torch.Tensor:
    """Create a random image tensor in the range [-1, 1] with shape [1, 3, H, W]."""
    image = torch.randn(1, 3, height, width, device=device, dtype=dtype)
    image = image.clamp(-1.0, 1.0)
    return image


def create_random_proprio(proprio_dim: int, processor: FastWAMProcessor) -> torch.Tensor:
    """Create a random proprioceptive state and normalize it using the processor."""
    # Create a random state in reasonable range
    raw_state = np.random.uniform(-0.5, 0.5, size=(proprio_dim,)).astype(np.float32)

    # Normalize using the processor's normalizer (same as eval_libero_single.py)
    state_meta = processor.shape_meta["state"]
    state_key = state_meta[0]["key"]
    state_batch = {"state": {state_key: torch.as_tensor(raw_state, dtype=torch.float32).unsqueeze(0)}}

    # Apply action_state_transform if needed (processor.action_state_transform)
    # For this test, we go through the normalizer directly
    state_batch = processor.normalizer.forward(state_batch)
    return state_batch["state"][state_key]


def load_model_with_hydra() -> tuple:
    """Load model using Hydra config composition (same as eval_libero_single.py)."""
    config_path = str(project_root / "configs")

    with initialize_config_dir(config_dir=config_path, version_base="1.3"):
        cfg = compose(
            config_name="sim_libero.yaml",
            overrides=[
                "task=libero_uncond_2cam224_1e-4",
                f"ckpt={CKPT_PATH}",
                "model.load_text_encoder=true",
                "model.skip_dit_load_from_pretrain=true",
            ],
        )

    model_dtype = get_model_dtype(MIXED_PRECISION)
    logger.info("Instantiating model on device=%s dtype=%s...", DEVICE, model_dtype)
    t0 = time.time()
    model = instantiate(cfg.model, model_dtype=model_dtype, device=DEVICE)
    logger.info("Model instantiated in %.2fs", time.time() - t0)

    logger.info("Loading checkpoint: %s", CKPT_PATH)
    t0 = time.time()
    model.load_checkpoint(CKPT_PATH)
    logger.info("Checkpoint loaded in %.2fs", time.time() - t0)
    model = model.to(dtype=torch.bfloat16, device=DEVICE).eval()
    logger.info("Model cast to bf16 on %s", DEVICE)

    # Load processor and dataset stats
    logger.info("Loading dataset stats: %s", DATASET_STATS_PATH)
    dataset_stats = load_dataset_stats_from_json(DATASET_STATS_PATH)
    processor: FastWAMProcessor = instantiate(cfg.data.train.processor).eval()
    processor.set_normalizer_from_stats(dataset_stats)

    return model, processor, cfg


def test_infer_action(model, processor: FastWAMProcessor, cfg: DictConfig):
    """Test the infer_action method with random inputs."""
    logger.info("=" * 60)
    logger.info("TEST: infer_action with random data")
    logger.info("=" * 60)

    model_dtype = model.torch_dtype

    # Stage 1: Create random image input [1, 3, H, W] in [-1, 1]
    t_stage = time.time()
    input_image = create_random_image(INPUT_H, INPUT_W, device=DEVICE, dtype=model_dtype)
    logger.info("[Stage 1] Image preparation: %.4fs | shape: %s dtype: %s",
                time.time() - t_stage, tuple(input_image.shape), input_image.dtype)

    # Stage 2: Create random proprio
    t_stage = time.time()
    proprio = create_random_proprio(PROPRIO_DIM, processor)
    logger.info("[Stage 2] Proprio preparation: %.4fs | shape: %s",
                time.time() - t_stage, tuple(proprio.shape))

    # Stage 3: Format the prompt
    t_stage = time.time()
    prompt = DEFAULT_PROMPT.format(task=TASK_DESCRIPTION)
    logger.info("[Stage 3] Prompt formatting: %.4fs | prompt: %s",
                time.time() - t_stage, prompt)

    # Stage 4: Run inference
    logger.info("Running infer_action (num_inference_steps=%d, action_horizon=%d)...", 
                NUM_INFERENCE_STEPS, ACTION_HORIZON)
    t0 = time.time()
    with torch.no_grad():
        result = model.infer_action(
            prompt=prompt,
            input_image=input_image,
            action_horizon=ACTION_HORIZON,
            proprio=proprio,
            num_inference_steps=NUM_INFERENCE_STEPS,
            seed=42,
            rand_device="cpu",
            tiled=False,
        )
    elapsed = time.time() - t0
    logger.info("[Stage 4] Model inference (infer_action): %.4fs", elapsed)
    
    # run perf count with proper CUDA sync
    run_cnt = 10
    for i in range(run_cnt):
        torch.cuda.synchronize()
        t0 = time.time()
        with torch.no_grad():
            result = model.infer_action(
                prompt=prompt,
                input_image=input_image,
                action_horizon=ACTION_HORIZON,
                proprio=proprio,
                num_inference_steps=NUM_INFERENCE_STEPS,
                seed=42,
                rand_device="cpu",
                tiled=False,
            )
        torch.cuda.synchronize()
        elapsed = time.time() - t0
        logger.info("[Stage 4] Model inference (infer_action) - run %d: %.4fs", i+1, elapsed)

    # Validate output
    action = result["action"]
    logger.info("Output action shape: %s dtype: %s", tuple(action.shape), action.dtype)
    logger.info("Action stats: min=%.4f max=%.4f mean=%.4f std=%.4f",
                action.min().item(), action.max().item(), 
                action.mean().item(), action.std().item())

    assert action.shape == (ACTION_HORIZON, ACTION_DIM), \
        f"Expected action shape ({ACTION_HORIZON}, {ACTION_DIM}), got {tuple(action.shape)}"
    assert action.dtype == torch.float32, f"Expected float32 output, got {action.dtype}"
    assert not torch.isnan(action).any(), "Action contains NaN values!"
    assert not torch.isinf(action).any(), "Action contains Inf values!"

    logger.info("PASSED: infer_action output is valid")
    return action


def test_infer_action_with_context(model, processor: FastWAMProcessor, cfg: DictConfig):
    """Test infer_action with pre-encoded context (bypass text encoder)."""
    logger.info("=" * 60)
    logger.info("TEST: infer_action with pre-encoded context")
    logger.info("=" * 60)

    model_dtype = model.torch_dtype

    # Stage 1: Image preparation
    t_stage = time.time()
    input_image = create_random_image(INPUT_H, INPUT_W, device=DEVICE, dtype=model_dtype)
    logger.info("[Stage 1] Image preparation: %.4fs", time.time() - t_stage)

    # Stage 2: Encode prompt to get context/context_mask
    prompt = DEFAULT_PROMPT.format(task=TASK_DESCRIPTION)
    logger.info("Pre-encoding prompt...")
    t_stage = time.time()
    with torch.no_grad():
        context, context_mask = model.encode_prompt(prompt)
    logger.info("[Stage 2] Prompt encoding (encode_prompt): %.4fs | context shape: %s, mask shape: %s",
                time.time() - t_stage, tuple(context.shape), tuple(context_mask.shape))

    # Stage 3: Proprio preparation
    t_stage = time.time()
    proprio = create_random_proprio(PROPRIO_DIM, processor)
    logger.info("[Stage 3] Proprio preparation: %.4fs", time.time() - t_stage)

    # Stage 4: Run inference with pre-encoded context
    logger.info("Running infer_action with context...")
    t0 = time.time()
    with torch.no_grad():
        result = model.infer_action(
            prompt=None,
            input_image=input_image,
            action_horizon=ACTION_HORIZON,
            proprio=proprio,
            context=context,
            context_mask=context_mask,
            num_inference_steps=NUM_INFERENCE_STEPS,
            seed=42,
            rand_device="cpu",
        )
    elapsed = time.time() - t0
    logger.info("[Stage 4] Model inference (with pre-encoded context): %.4fs", elapsed)

    action = result["action"]
    logger.info("Output action shape: %s", tuple(action.shape))
    assert action.shape == (ACTION_HORIZON, ACTION_DIM)
    assert not torch.isnan(action).any(), "Action contains NaN!"

    logger.info("PASSED: infer_action with context is valid")
    return action


def test_denormalize_action(action: torch.Tensor, processor: FastWAMProcessor):
    """Test action denormalization."""
    logger.info("=" * 60)
    logger.info("TEST: action denormalization")
    logger.info("=" * 60)

    # Stage 1: Normalizer lookup
    t_stage = time.time()
    action_meta = processor.shape_meta["action"]
    action_key = action_meta[0]["key"]
    normalizer = processor.normalizer.normalizers["action"][action_key]
    logger.info("[Stage 1] Normalizer lookup: %.4fs", time.time() - t_stage)

    # Stage 2: Denormalization
    t_stage = time.time()
    action_3d = action.unsqueeze(0).to(dtype=torch.float32, device="cpu")
    denorm = normalizer.backward(action_3d)
    denorm_np = denorm.numpy()
    logger.info("[Stage 2] Action denormalization: %.4fs", time.time() - t_stage)

    logger.info("Denormalized action shape: %s", denorm_np.shape)
    logger.info("Denorm stats: min=%.4f max=%.4f mean=%.4f",
                denorm_np.min(), denorm_np.max(), denorm_np.mean())
    assert not np.isnan(denorm_np).any(), "Denormalized action contains NaN!"

    logger.info("PASSED: denormalization works correctly")
    return denorm_np


def test_multiple_inference_steps(model, processor: FastWAMProcessor, cfg: DictConfig):
    """Test inference with different num_inference_steps to verify scheduler."""
    logger.info("=" * 60)
    logger.info("TEST: multiple inference step counts")
    logger.info("=" * 60)

    model_dtype = model.torch_dtype
    input_image = create_random_image(INPUT_H, INPUT_W, device=DEVICE, dtype=model_dtype)
    proprio = create_random_proprio(PROPRIO_DIM, processor)
    prompt = DEFAULT_PROMPT.format(task=TASK_DESCRIPTION)

    for i, steps in enumerate([5, 10, 20], 1):
        logger.info("[Run %d] num_inference_steps=%d", i, steps)
        t0 = time.time()
        with torch.no_grad():
            result = model.infer_action(
                prompt=prompt,
                input_image=input_image,
                action_horizon=ACTION_HORIZON,
                proprio=proprio,
                num_inference_steps=steps,
                seed=42,
                rand_device="cpu",
            )
        elapsed = time.time() - t0
        action = result["action"]
        logger.info("  [Run %d] steps=%d -> action shape=%s, inference_time=%.4fs, mean=%.4f",
                    i, steps, tuple(action.shape), elapsed, action.mean().item())
        assert not torch.isnan(action).any()

    logger.info("PASSED: all inference step counts work correctly")


def main():
    logger.info("FastWAM Inference Debug Test")
    logger.info("Checkpoint: %s", CKPT_PATH)
    logger.info("Device: %s", DEVICE)
    logger.info("=" * 60)

    # Verify files exist
    assert Path(CKPT_PATH).exists(), f"Checkpoint not found: {CKPT_PATH}"
    assert Path(DATASET_STATS_PATH).exists(), f"Dataset stats not found: {DATASET_STATS_PATH}"

    # Load model
    t_total = time.time()
    t_phase = time.time()
    model, processor, cfg = load_model_with_hydra()
    logger.info("[Phase] Model loading total: %.2fs", time.time() - t_phase)

    # Print model info
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info("Model total params: %.2fM", total_params / 1e6)
    logger.info("Model trainable params: %.2fM", trainable_params / 1e6)

    # Run tests
    t_phase = time.time()
    action = test_infer_action(model, processor, cfg)
    logger.info("[Phase] test_infer_action total: %.2fs", time.time() - t_phase)

     # Run tests
    # t_phase = time.time()
    # action = test_infer_action(model, processor, cfg)
    # logger.info("[Phase] test_infer_action2 total: %.2fs", time.time() - t_phase)


    # t_phase = time.time()
    # test_denormalize_action(action, processor)
    # logger.info("[Phase] test_denormalize_action total: %.2fs", time.time() - t_phase)

    # t_phase = time.time()
    # test_infer_action_with_context(model, processor, cfg)
    # logger.info("[Phase] test_infer_action_with_context total: %.2fs", time.time() - t_phase)

    # t_phase = time.time()
    # test_multiple_inference_steps(model, processor, cfg)
    # logger.info("[Phase] test_multiple_inference_steps total: %.2fs", time.time() - t_phase)

    # logger.info("=" * 60)
    # logger.info("ALL TESTS PASSED (total elapsed: %.2fs)", time.time() - t_total)
    # logger.info("=" * 60)


if __name__ == "__main__":
    main()
