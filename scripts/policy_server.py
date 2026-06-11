"""
Generic FastWAM Policy Server.

Starts a WebSocket-based VLA server that loads a FastWAM model from a Hydra
config file, performs preprocessing + inference + postprocessing, and returns
predicted actions to the client.

Usage:
    python scripts/policy_server.py \
        --config sim_libero.yaml \
        --ckpt /path/to/checkpoint.pt \
        --dataset_stats_path /path/to/dataset_stats.json \
        --port 8765

    python scripts/policy_server.py \
        --config sim_robotwin.yaml \
        --ckpt /path/to/checkpoint.pt \
        --dataset_stats_path /path/to/dataset_stats.json \
        --task robotwin_uncond_3cam_384_1e-4 \
        --port 8765

Client observation format (msgpack dict):
    {
        "images": {
            "camera_0": np.ndarray [H, W, 3] uint8,
            "camera_1": np.ndarray [H, W, 3] uint8,  # optional
            ...
        },
        "state": np.ndarray [D] float32,  # proprioceptive state
        "instruction": str,  # task instruction text
    }

Server response format (msgpack dict):
    {
        "action": np.ndarray [replan_steps, action_dim] float32,
        "server_timing": {"infer_ms": float},
    }

Special messages:
    {"reset": True}  -> resets the policy internal state
"""

import argparse
import inspect
import logging
import os
import sys
import time
from collections import deque
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch
from PIL import Image

# add path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))
if str(PROJECT_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

from fastwam.datasets.lerobot.processors.fastwam_processor import FastWAMProcessor
from fastwam.datasets.lerobot.robot_video_dataset import DEFAULT_PROMPT
from fastwam.datasets.lerobot.utils.normalizer import load_dataset_stats_from_json
from fastwam.datasets.dataset_utils import ResizeSmallestSideAspectPreserving, CenterCrop, Normalize
from network_utils import WebsocketPolicyServer

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger(__name__)


# ============================================================================
# Utility functions
# ============================================================================


def _is_none_like(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        return value.strip().lower() in {"", "none", "null"}
    return False


def _normalize_mixed_precision(mixed_precision: str) -> str:
    key = str(mixed_precision).strip().lower()
    if key not in {"no", "fp16", "bf16"}:
        raise ValueError(
            f"Unsupported mixed_precision: {mixed_precision}. "
            "Expected one of: ['no', 'fp16', 'bf16']."
        )
    return key


def _mixed_precision_to_model_dtype(mixed_precision: str) -> torch.dtype:
    precision = _normalize_mixed_precision(mixed_precision)
    if precision == "no":
        return torch.float32
    if precision == "fp16":
        return torch.float16
    return torch.bfloat16


def _resize_rgb(image: np.ndarray, width: int, height: int) -> np.ndarray:
    """Resize an RGB image (H, W, 3) to (height, width, 3) using bilinear."""
    pil_image = Image.fromarray(image.astype(np.uint8), mode="RGB")
    resized = pil_image.resize((width, height), resample=Image.BILINEAR)
    return np.asarray(resized, dtype=np.uint8)


def _center_crop_resize(image: np.ndarray, width: int, height: int) -> np.ndarray:
    """Center-crop and resize to target dimensions."""
    pil_image = Image.fromarray(image)
    src_w, src_h = pil_image.size
    scale = max(width / src_w, height / src_h)
    resized = pil_image.resize((round(src_w * scale), round(src_h * scale)), resample=Image.BILINEAR)
    rw, rh = resized.size
    left = max((rw - width) // 2, 0)
    top = max((rh - height) // 2, 0)
    cropped = resized.crop((left, top, left + width, top + height))
    return np.asarray(cropped, dtype=np.uint8)


def _compose_cfg(config_name: str, task_override: Optional[str] = None) -> DictConfig:
    """Compose Hydra config from configs/ directory."""
    configs_root = (PROJECT_ROOT / "configs").resolve()
    overrides = []
    if task_override is not None:
        overrides.append(f"task={task_override}")

    if GlobalHydra.instance().is_initialized():
        GlobalHydra.instance().clear()
    print(f'configs_root {configs_root}')
    with initialize_config_dir(version_base="1.3", config_dir=str(configs_root)):
        cfg = compose(config_name=config_name, overrides=overrides)
    return cfg


def _resolve_dataset_stats_path(
    explicit_path: Optional[str],
    ckpt_path: Optional[Path] = None,
) -> Path:
    """Locate dataset_stats.json from explicit path or checkpoint parent dirs."""
    candidates: list[Path] = []

    if not _is_none_like(explicit_path):
        candidates.append(Path(str(explicit_path)).expanduser().resolve())

    if ckpt_path is not None:
        for parent in list(ckpt_path.parents)[:4]:
            candidates.append((parent / "dataset_stats.json").resolve())

    seen: set[Path] = set()
    for path in candidates:
        resolved = path.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        if resolved.exists():
            return resolved

    raise FileNotFoundError(
        "Failed to locate dataset_stats.json. "
        "Please pass --dataset_stats_path=/path/to/dataset_stats.json."
    )


# ============================================================================
# FastWAM Generic Policy
# ============================================================================


class FastWAMPolicy:
    """Generic FastWAM policy wrapper for WebSocket server.

    Handles:
      - Model loading and initialization
      - Image preprocessing (multi-camera concat)
      - State normalization
      - Action inference via model.infer_action()
      - Action denormalization
      - Replan action queue management
    """

    def __init__(
        self,
        cfg: DictConfig,
        checkpoint_path: str,
        dataset_stats_path: Path,
        device: str = "cuda",
        port: int = 8765,
        save_debug_images: bool = False,
    ) -> None:
        self.cfg = cfg
        self.device = device
        self.save_debug_images = save_debug_images

        # --- Model dtype ---
        mixed_precision = str(cfg.get("mixed_precision", "bf16"))
        self.model_dtype = _mixed_precision_to_model_dtype(mixed_precision)

        # --- Instantiate model ---
        model_cfg = OmegaConf.create(OmegaConf.to_container(cfg.model, resolve=True))
        model_cfg.load_text_encoder = True
        logger.info("Instantiating model...")
        self.model = instantiate(model_cfg, model_dtype=self.model_dtype, device=device)
        self.model.load_checkpoint(checkpoint_path)
        self.model = self.model.to(device).eval()
        logger.info("Model loaded from: %s", checkpoint_path)

        # --- Instantiate processor ---
        self.processor: FastWAMProcessor = instantiate(cfg.data.train.processor).eval()
        dataset_stats = load_dataset_stats_from_json(str(dataset_stats_path))
        self.processor.set_normalizer_from_stats(dataset_stats)
        logger.info("Processor initialized with stats: %s", dataset_stats_path)

        # --- Inference parameters from config ---
        eval_cfg = cfg.get("EVALUATION", {})
        action_horizon_cfg = eval_cfg.get("action_horizon", None)
        if _is_none_like(action_horizon_cfg):
            self.action_horizon = int(cfg.data.train.num_frames) - 1
        else:
            self.action_horizon = int(action_horizon_cfg)

        replan_cfg = eval_cfg.get("replan_steps", None)
        self.replan_steps = int(replan_cfg) if replan_cfg is not None else self.action_horizon
        self.replan_steps = max(1, min(self.replan_steps, self.action_horizon))

        num_inf_steps_cfg = eval_cfg.get("num_inference_steps", None)
        if _is_none_like(num_inf_steps_cfg):
            self.num_inference_steps = int(cfg.get("eval_num_inference_steps", 20))
        else:
            self.num_inference_steps = int(num_inf_steps_cfg)

        self.sigma_shift = (
            None if _is_none_like(eval_cfg.get("sigma_shift")) else float(eval_cfg.get("sigma_shift"))
        )
        self.text_cfg_scale = float(eval_cfg.get("text_cfg_scale", 1.0))
        self.negative_prompt = str(eval_cfg.get("negative_prompt", ""))
        self.rand_device = str(eval_cfg.get("rand_device", "cpu"))
        self.tiled = bool(eval_cfg.get("tiled", False))
        self.seed = None if _is_none_like(cfg.get("seed")) else int(cfg.seed)

        # --- Image layout ---
        video_size = cfg.data.train.get("video_size", [224, 224])
        self.input_h = int(video_size[0])
        self.input_w = int(video_size[1])
        self.concat_multi_camera = cfg.data.train.get("concat_multi_camera", "horizontal")

        # --- Image transforms (same as training: resize -> crop -> normalize) ---
        self._resize_transform = ResizeSmallestSideAspectPreserving(
            args={"img_w": self.input_w, "img_h": self.input_h},
        )
        self._crop_transform = CenterCrop(
            args={"img_w": self.input_w, "img_h": self.input_h},
        )
        self._normalize_transform = Normalize(
            args={"mean": 0.5, "std": 0.5},
        )

        # --- Num video frames (needed for joint/fastwam_joint models) ---
        action_video_freq_ratio = int(cfg.data.train.get("action_video_freq_ratio", 1))
        num_frames = int(cfg.data.train.num_frames)
        self._num_video_frames = (num_frames - 1) // action_video_freq_ratio + 1

        self._step_count = 0

        logger.info(
            "Policy ready | action_horizon=%d | replan_steps=%d | "
            "num_inference_steps=%d | image_size=(%d,%d) | dtype=%s",
            self.action_horizon,
            self.replan_steps,
            self.num_inference_steps,
            self.input_h,
            self.input_w,
            self.model_dtype,
        )

    # ------------------------------------------------------------------
    # Preprocessing
    # ------------------------------------------------------------------

    def _build_image_tensor(self, images: Dict[str, Any]) -> torch.Tensor:
        """Build image tensor from observation images dict.

        Supports multiple camera layouts (controlled by concat_multi_camera config):
        - Single camera: resize to per-camera meta shape
        - Two cameras (horizontal/vertical): resize each then concat
        - Three cameras "robotwin": head[256,320] + left[128,160]/right[128,160] -> [384,320]
        - Three cameras "lerobot": top[224,224] + left[112,112]/right[112,112] -> [336,224]
        - Generic horizontal/vertical: resize each per meta then concat

        Args:
            images: Dict mapping camera names to image data. Supported formats:
                - np.ndarray [H, W, 3] uint8 (HWC RGB)
                - np.ndarray [3, H, W] (CHW)
                - np.ndarray [1, 3, H, W] (batch CHW)
                - torch.Tensor [3, H, W] or [1, 3, H, W]

        Returns:
            Image tensor [1, 3, H, W] in range [-1, 1].
        """
        image_meta = self.processor.shape_meta["images"]
        num_cameras = self.processor.num_output_cameras
        logger.info("Building image tensor from %d cameras", num_cameras)

        def _meta_to_hw(meta: dict) -> tuple[int, int]:
            shape = meta["shape"]
            return int(shape[1]), int(shape[2])

        def _to_hwc_numpy(img: Any) -> np.ndarray:
            """Convert various image formats to [H, W, 3] uint8 numpy array."""
            if isinstance(img, torch.Tensor):
                img = img.detach().cpu().numpy()
            if isinstance(img, np.ndarray):
                if img.ndim == 4:
                    # [1, 3, H, W] -> [3, H, W]
                    img = img[0]
                if img.ndim == 3 and img.shape[0] in (1, 3):
                    # [C, H, W] -> [H, W, C]
                    img = img.transpose(1, 2, 0)
                # Now should be [H, W, 3]
                if img.ndim == 3 and img.shape[2] in (1, 3):
                    if img.dtype in (np.float32, np.float64):
                        # Assume range [0, 1] or [-1, 1]
                        if img.min() < 0:
                            # [-1, 1] -> [0, 255]
                            img = ((img + 1.0) * 127.5).clip(0, 255).astype(np.uint8)
                        else:
                            # [0, 1] -> [0, 255]
                            img = (img * 255.0).clip(0, 255).astype(np.uint8)
                    else:
                        img = img.astype(np.uint8)
                    return img
            raise ValueError(
                f"Unsupported image format: type={type(img)}, "
                f"shape={getattr(img, 'shape', '?')}, dtype={getattr(img, 'dtype', '?')}"
            )

        # Convert all images to [H, W, 3] uint8
        images_hwc: Dict[str, np.ndarray] = {k: _to_hwc_numpy(v) for k, v in images.items()}

        # Use the order defined in shape_meta["images"] (same as training)
        camera_keys = [meta["key"] for meta in image_meta]
        # Validate that all required camera keys are present
        missing = [k for k in camera_keys if k not in images_hwc]
        if missing:
            # Try fuzzy match: incoming keys might be suffixes of meta keys or vice versa
            available = list(images_hwc.keys())
            logger.warning(
                "Camera key mismatch. Expected (from shape_meta): %s, got: %s. "
                "Falling back to available keys in shape_meta order.",
                camera_keys, available,
            )
            # Match by checking if available key ends with meta key or meta key ends with available key
            matched_keys = []
            for meta_key in camera_keys:
                found = None
                for avail_key in available:
                    if avail_key == meta_key or avail_key.endswith(meta_key) or meta_key.endswith(avail_key):
                        found = avail_key
                        break
                if found:
                    matched_keys.append(found)
            camera_keys = matched_keys if matched_keys else available[:num_cameras]
        print(f'Using camera keys: {camera_keys}')
        if num_cameras == 1:
            # Single camera
            cam_key = camera_keys[0]
            h, w = _meta_to_hw(image_meta[0])
            rgb = _center_crop_resize(images_hwc[cam_key], width=w, height=h)

        elif num_cameras == 2:
            # Two cameras with concat
            primary_h, primary_w = _meta_to_hw(image_meta[0])
            secondary_h, secondary_w = _meta_to_hw(image_meta[1])

            primary = _center_crop_resize(images_hwc[camera_keys[0]], width=primary_w, height=primary_h)
            secondary = _center_crop_resize(images_hwc[camera_keys[1]], width=secondary_w, height=secondary_h)

            if self.concat_multi_camera == "horizontal":
                rgb = np.concatenate([primary, secondary], axis=1)
            elif self.concat_multi_camera == "vertical":
                rgb = np.concatenate([primary, secondary], axis=0)
            else:
                rgb = np.concatenate([primary, secondary], axis=1)

        elif num_cameras == 3 and self.concat_multi_camera == "robotwin":
            # RoboTwin style: head [256,320] + left [128,160] / right [128,160] -> [384,320]
            head = _resize_rgb(images_hwc[camera_keys[0]], width=320, height=256)
            left = _resize_rgb(images_hwc[camera_keys[1]], width=160, height=128)
            right = _resize_rgb(images_hwc[camera_keys[2]], width=160, height=128)

            bottom = np.concatenate([left, right], axis=1)  # [128, 320, 3]
            rgb = np.concatenate([head, bottom], axis=0)  # [384, 320, 3]

        elif num_cameras == 3 and self.concat_multi_camera == "lerobot":
            # LeRobot style: top [224,224] + left [112,112] / right [112,112] -> [336,224]
            cam_top = _resize_rgb(images_hwc[camera_keys[0]], width=224, height=224)
            cam_left = _resize_rgb(images_hwc[camera_keys[1]], width=112, height=112)
            cam_right = _resize_rgb(images_hwc[camera_keys[2]], width=112, height=112)

            bottom = np.concatenate([cam_left, cam_right], axis=1)  # [112, 224, 3]
            rgb = np.concatenate([cam_top, bottom], axis=0)  # [336, 224, 3]

        elif num_cameras > 1 and self.concat_multi_camera == "horizontal":
            # Generic horizontal concat
            frames = []
            for i, cam_key in enumerate(camera_keys[:num_cameras]):
                meta_idx = min(i, len(image_meta) - 1)
                h, w = _meta_to_hw(image_meta[meta_idx])
                frames.append(_center_crop_resize(images_hwc[cam_key], width=w, height=h))
            rgb = np.concatenate(frames, axis=1)

        elif num_cameras > 1 and self.concat_multi_camera == "vertical":
            # Generic vertical concat
            frames = []
            for i, cam_key in enumerate(camera_keys[:num_cameras]):
                meta_idx = min(i, len(image_meta) - 1)
                h, w = _meta_to_hw(image_meta[meta_idx])
                frames.append(_center_crop_resize(images_hwc[cam_key], width=w, height=h))
            rgb = np.concatenate(frames, axis=0)

        else:
            # Fallback: concat all cameras horizontally
            frames = []
            for i, cam_key in enumerate(camera_keys[:num_cameras]):
                meta_idx = min(i, len(image_meta) - 1)
                h, w = _meta_to_hw(image_meta[meta_idx])
                frames.append(_center_crop_resize(images_hwc[cam_key], width=w, height=h))
            rgb = np.concatenate(frames, axis=1)

        # Convert to tensor [3, H, W] uint8
        image_tensor = torch.from_numpy(rgb.copy()).permute(2, 0, 1)  # [3, H, W]

        # Final resize + center crop + normalize, same as training pipeline
        # (robot_video_dataset.py: resize_transform -> crop_transform -> normalize_transform)
        image_tensor = self._resize_transform(image_tensor)
        image_tensor = self._crop_transform(image_tensor)
        # Save to local path for debugging
        if self.save_debug_images:
            debug_dir = PROJECT_ROOT / "debug_images"
            debug_dir.mkdir(exist_ok=True)
            save_img = image_tensor.clone()
            if save_img.dtype != torch.uint8:
                save_img = save_img.clamp(0, 255).to(torch.uint8)
            save_path = debug_dir / f"input_step_{self._step_count:04d}.png"
            Image.fromarray(save_img.permute(1, 2, 0).numpy()).save(save_path)
            logger.info("Saved debug image to %s", save_path)

        image_tensor = self._normalize_transform(image_tensor)  # [3, H, W] range [-1, 1]

        # Move to device with target dtype, add batch dim
        image_tensor = image_tensor.unsqueeze(0).to(device=self.device, dtype=self.model_dtype)
        return image_tensor

    def _normalize_state(self, state: np.ndarray) -> torch.Tensor:
        """Normalize proprioceptive state vector using processor normalizer.

        Args:
            state: Raw state vector [D] float32.

        Returns:
            Normalized state tensor [1, D] or [D].
        """
        state_meta = self.processor.shape_meta["state"]
        if len(state_meta) != 1:
            raise ValueError("Expected exactly one merged state key in shape_meta['state'].")
        state_key = state_meta[0]["key"]

        state_batch = {"state": {state_key: torch.as_tensor(state, dtype=torch.float32).unsqueeze(0)}}
        state_batch = self.processor.action_state_transform(state_batch)
        state_batch = self.processor.normalizer.forward(state_batch)
        return state_batch["state"][state_key]

    def _denormalize_action(self, action: torch.Tensor) -> np.ndarray:
        """Denormalize predicted action tensor.

        Args:
            action: Normalized action tensor [T, D] or [B, T, D].

        Returns:
            Denormalized action numpy array [T, D].
        """
        if action.ndim == 2:
            action = action.unsqueeze(0)
        if action.ndim != 3:
            raise ValueError(f"Expected action tensor [B, T, D], got {tuple(action.shape)}")
        if self.model.use_custom_action:
            action = action.to(dtype=torch.float32, device="cpu")
            denorm = self.model.custom_action_norm.backward(action)
            return denorm.numpy()[0]
        action_meta = self.processor.shape_meta["action"]
        if len(action_meta) != 1:
            raise ValueError("Expected exactly one merged action key in shape_meta['action'].")

        action_key = action_meta[0]["key"]
        normalizer = self.processor.normalizer.normalizers["action"][action_key]
        action = action.to(dtype=torch.float32, device="cpu")
        denorm = normalizer.backward(action)
        return denorm.numpy()[0]  # [T, D]

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def _infer_action_chunk(
        self,
        images: Dict[str, np.ndarray],
        state: np.ndarray,
        instruction: str,
    ) -> np.ndarray:
        """Run full inference pipeline: preprocess -> model -> postprocess.

        Args:
            images: Camera images dict {name: [H, W, 3] uint8}.
            state: Proprioceptive state [D] float32.
            instruction: Task instruction text.

        Returns:
            Action chunk [action_horizon, action_dim] float32.
        """
        # Build input image tensor
        image_tensor = self._build_image_tensor(images)

        # Normalize state
        proprio = self._normalize_state(state)
        
        if proprio.ndim == 3:
            proprio = proprio.squeeze(0)

        # Format prompt
        prompt = DEFAULT_PROMPT.format(task=instruction)

        # Build inference kwargs
        infer_kwargs: Dict[str, Any] = {
            "prompt": prompt,
            "input_image": image_tensor,
            "action_horizon": self.action_horizon,
            "proprio": proprio,
            "negative_prompt": self.negative_prompt,
            "text_cfg_scale": self.text_cfg_scale,
            "num_inference_steps": self.num_inference_steps,
            "sigma_shift": self.sigma_shift,
            "seed": self.seed,
            "rand_device": self.rand_device,
            "tiled": self.tiled,
        }

        # Check if model.infer_action accepts num_video_frames (for joint models)
        if "num_video_frames" in inspect.signature(self.model.infer_action).parameters:
            infer_kwargs["num_video_frames"] = self._num_video_frames

        # Run inference
        with torch.no_grad():
            pred = self.model.infer_action(**infer_kwargs)

        # Denormalize action
        action_tensor = pred["action"]  # [T, D]
        action_chunk = self._denormalize_action(action_tensor)  # [T, D]
        return action_chunk

    # ------------------------------------------------------------------
    # WebSocket interface
    # ------------------------------------------------------------------

    def act(self, obs: Dict[str, Any]) -> Dict[str, Any]:
        """Process observation and return action(s).

        Called by WebsocketPolicyServer on each client request.
        Runs inference directly on every call and returns the predicted action chunk.

        Args:
            obs: Observation dict from client with keys:
                - "images": dict of camera images [H, W, 3] uint8
                - "state": np.ndarray [D] float32
                - "instruction": str

        Returns:
            Dict with "action" (torch.Tensor) and "success_proba" (torch.Tensor).
        """
        t0 = time.perf_counter()
        logger.info("Received observation from client.")

        # Extract images: keys containing "observation.images" are camera images
        images = {}
        for k, v in obs.items():
            if "observation.images" in k:
                # Use the last part of the key as camera name (e.g. "head_right")
                cam_name = k.split(".")[-1]
                images[cam_name] = v

        state = obs.get("observation.state")
        instruction = obs.get("task", "")

        if state is None:
            raise ValueError(
                "Observation must contain 'state' (proprioceptive state vector)."
            )
        if len(images) == 0:
            raise ValueError(
                "Observation must contain 'images' dict with at least one camera."
            )

        # state = np.asarray(state, dtype=np.float32)
        action_chunk = self._infer_action_chunk(images, state, instruction)

        self._step_count += 1
        elapsed_ms = (time.perf_counter() - t0) * 1000
        logger.info("Inference step %d took %.1f ms", self._step_count, elapsed_ms)

        # Return full action chunk [T, D]
        return {
            "action": torch.from_numpy(action_chunk.astype(np.float32)),
            "success_proba": torch.tensor([1.0]),
        }

    def reset(self) -> None:
        """Reset policy internal state."""
        self._step_count = 0
        logger.info("Policy reset.")


# ============================================================================
# Server entrypoint
# ============================================================================


def deploy(
    config_name: str,
    checkpoint_path: str,
    dataset_stats_path: str,
    task_override: Optional[str] = None,
    device: str = "cuda",
    port: int = 8765,
    host: str = "0.0.0.0",
    save_debug_images: bool = False,
) -> None:
    """Deploy the FastWAM policy as a WebSocket server.

    Args:
        config_name: Hydra config file name (e.g. 'sim_libero.yaml').
        checkpoint_path: Path to model checkpoint.
        dataset_stats_path: Path to dataset_stats.json.
        task_override: Optional Hydra task override (e.g. 'libero_uncond_2cam224_1e-4').
        device: Torch device.
        port: WebSocket server port.
        host: WebSocket server host.
    """
    # Compose config
    cfg = _compose_cfg(config_name, task_override=task_override)
    # Resolve device
    if device.startswith("cuda") and not torch.cuda.is_available():
        logger.warning("CUDA unavailable, falling back to CPU.")
        device = "cpu"

    # Resolve dataset stats
    ckpt_path = Path(checkpoint_path).expanduser().resolve()
    stats_path = _resolve_dataset_stats_path(
        explicit_path=dataset_stats_path,
        ckpt_path=ckpt_path,
    )

    # Create policy
    policy = FastWAMPolicy(
        cfg=cfg,
        checkpoint_path=str(ckpt_path),
        dataset_stats_path=stats_path,
        device=device,
        save_debug_images=save_debug_images,
    )

    # Start server
    logger.info("Starting WebSocket policy server on %s:%d", host, port)
    server = WebsocketPolicyServer(
        policy=policy,
        host=host,
        port=port,
        metadata={
            "model_family": "fastwam",
            "action_horizon": policy.action_horizon,
            "replan_steps": policy.replan_steps,
            "num_inference_steps": policy.num_inference_steps,
        },
    )
    server.serve_forever()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="FastWAM Generic Policy Server",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--config",
        type=str,
        default="sim_pangceban.yaml",
        help="Hydra config file name under configs/ (e.g. sim_libero.yaml, sim_robotwin.yaml)",
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        default="/root/autodl-fs/ckpts/fast_wam/runs/pangceban_uncond_2cam224_1e-4/2026-06-10_21-00-51/checkpoints/weights/step_006000.pt",
        help="Path to model checkpoint (.pt or directory)",
    )
    parser.add_argument(
        "--dataset_stats_path",
        type=str,
        default="/root/autodl-fs/ckpts/fast_wam/runs/pangceban_uncond_2cam224_1e-4/2026-06-10_21-00-51/dataset_stats.json",
        help="Path to dataset_stats.json. If not provided, will search near checkpoint.",
    )
    parser.add_argument(
        "--task",
        type=str,
        default="pangceban_uncond_2cam224_1e-4",
        help="Hydra task override (e.g. libero_uncond_2cam224_1e-4, robotwin_uncond_3cam_384_1e-4)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Torch device (default: cuda)",
    )
    parser.add_argument(
        "--gpu_id",
        type=int,
        default=None,
        help="GPU ID to use (sets CUDA_VISIBLE_DEVICES)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8765,
        help="WebSocket server port (default: 8765)",
    )
    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="WebSocket server host (default: 0.0.0.0)",
    )
    parser.add_argument(
        "--save_debug_images",
        action="store_true",
        default=False,
        help="Save preprocessed input images to debug_images/ for debugging",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # Set GPU visibility
    if args.gpu_id is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)

    deploy(
        config_name=args.config,
        checkpoint_path=args.ckpt,
        dataset_stats_path=args.dataset_stats_path,
        task_override=args.task,
        device=args.device,
        port=args.port,
        host=args.host,
        save_debug_images=args.save_debug_images,
    )
