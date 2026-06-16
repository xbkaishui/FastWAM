"""
测试脚本：分析 Action DiT 在推理时的注意力分布。

目标：观察 action expert 在做 attention 时主要关注哪些 token 类型：
  1. Mixed Attention (MoT): action Q → [video_KV_cache | action_KV]
  2. Cross-Attention (post_block): action tokens → context (文本 + state/proprio)

用法:
    cd /root/foresee/FastWAM
    python test/test_action_attention.py

模型配置与 start_policy_server.sh 一致（sim_libero.yaml + libero_uncond_2cam224_1e-4）。
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
os.chdir(os.path.join(os.path.dirname(__file__), ".."))

import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from einops import rearrange
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from PIL import Image

from fastwam.datasets.lerobot.processors.fastwam_processor import FastWAMProcessor
from fastwam.datasets.lerobot.robot_video_dataset import DEFAULT_PROMPT
from fastwam.datasets.lerobot.utils.normalizer import load_dataset_stats_from_json

# ============================================================================
# 配置参数 (与 start_policy_server.sh 一致)
# ============================================================================

CONFIG_NAME = "sim_libero.yaml"
TASK_OVERRIDE = "libero_uncond_2cam224_1e-4"
CKPT_PATH = "/root/autodl-fs/ckpts/fast_wam/runs/libero_uncond_2cam224_1e-4/2026-06-04_13-50-48/checkpoints/weights/step_028930.pt"
DATASET_STATS_PATH = "/root/autodl-fs/ckpts/fast_wam/runs/libero_uncond_2cam224_1e-4/2026-06-04_13-50-48/dataset_stats.json"

DEVICE = "cuda"
NUM_INFERENCE_STEPS = 10  # 减少步数加速测试
SEED = 42

input_image_path = "/root/foresee/FastWAM/debug_images/model_input.png"
dummy_state = np.array([-0.2325, -0.0571,  0.5881,  0.5704,  0.0101,  0.0864,  0.8822, -0.8842]).astype(np.float32)
instruction = "open the middle drawer of the cabinet"

OUTPUT_DIR = Path("test/visualize/attention_analysis")


# ============================================================================
# 注意力权重捕获工具
# ============================================================================


class AttentionCapturer:
    """
    通过 hook 捕获 MoT mixed attention 和 cross-attention 的注意力权重。
    
    捕获位置：
    - mixed_attention: action Q attend to [video_K | action_K]
    - cross_attention: action tokens attend to context (text + state)
    """

    def __init__(self, model, capture_layers: Optional[List[int]] = None):
        """
        Args:
            model: FastWAM 模型实例
            capture_layers: 要捕获的层索引列表，None=全部层
        """
        self.model = model
        self.num_layers = model.mot.num_layers
        self.num_heads = model.mot.num_heads
        self.capture_layers = capture_layers or list(range(self.num_layers))

        # 存储每一步去噪的注意力权重
        self.mixed_attn_weights: List[Dict[int, torch.Tensor]] = []  # per step, per layer
        self.cross_attn_weights: List[Dict[int, torch.Tensor]] = []  # per step, per layer

        self._hooks = []
        self._current_step_mixed: Dict[int, torch.Tensor] = {}
        self._current_step_cross: Dict[int, torch.Tensor] = {}
        self._capturing = False
        self._current_layer_idx = -1

    def _compute_attention_weights(
        self, q: torch.Tensor, k: torch.Tensor, num_heads: int, mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """手动计算 attention weights (不执行 value 加权)。
        
        Args:
            q: [B, Sq, H*Dh]
            k: [B, Sk, H*Dh]
            num_heads: 注意力头数
            mask: [Sq, Sk] bool mask (True=可以 attend)
            
        Returns:
            attention_weights: [B, H, Sq, Sk] softmax 后的权重
        """
        head_dim = q.shape[-1] // num_heads
        q = rearrange(q, "b s (n d) -> b n s d", n=num_heads)
        k = rearrange(k, "b s (n d) -> b n s d", n=num_heads)

        # Scaled dot-product
        scale = head_dim ** -0.5
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * scale  # [B, H, Sq, Sk]

        # Apply mask
        if mask is not None:
            if mask.ndim == 2:
                mask = mask.unsqueeze(0).unsqueeze(0)  # [1, 1, Sq, Sk]
            attn_scores = attn_scores.masked_fill(~mask.bool(), float("-inf"))

        attn_weights = F.softmax(attn_scores, dim=-1)
        return attn_weights.detach().cpu().float()

    def install_hooks(self):
        """安装 hooks 来截获 forward_action_with_video_cache 中的 Q/K/V。
        
        由于 MoT 使用自定义的 _mixed_attention 方法而非 nn.Module 的子模块调用，
        我们通过 monkey-patch MoT 的 forward_action_with_video_cache 来捕获注意力。
        """
        mot = self.model.mot
        original_forward = mot.forward_action_with_video_cache
        capturer = self

        def hooked_forward(
            action_tokens: torch.Tensor,
            action_freqs: torch.Tensor,
            action_t_mod: torch.Tensor,
            action_context_payload: Optional[dict],
            video_kv_cache: list,
            attention_mask: torch.Tensor,
            video_seq_len: int,
        ) -> torch.Tensor:
            """与原始方法相同，但额外捕获注意力权重。"""
            if not capturer._capturing:
                return original_forward(
                    action_tokens, action_freqs, action_t_mod,
                    action_context_payload, video_kv_cache,
                    attention_mask, video_seq_len,
                )

            action_seq_len = int(action_tokens.shape[1])
            total_seq_len = int(video_seq_len) + action_seq_len
            action_attention_mask = attention_mask[video_seq_len:total_seq_len, :total_seq_len]

            expert = mot.mixtures["action"]
            x = action_tokens

            for layer_idx in range(mot.num_layers):
                block = expert.blocks[layer_idx]
                (
                    q_action, k_action, v_action,
                    residual_x, gate_msa, shift_mlp, scale_mlp, gate_mlp,
                    use_gradient_checkpointing,
                ) = mot._build_expert_attention_io(
                    expert=expert, block=block, x=x, freqs=action_freqs, t_mod=action_t_mod,
                )

                layer_cache = video_kv_cache[layer_idx]
                k_video = layer_cache["k"]
                v_video = layer_cache["v"]
                k_cat = torch.cat([k_video, k_action], dim=1)
                v_cat = torch.cat([v_video, v_action], dim=1)

                # === 捕获 mixed attention 权重 ===
                if layer_idx in capturer.capture_layers:
                    with torch.no_grad():
                        attn_w = capturer._compute_attention_weights(
                            q=q_action, k=k_cat, num_heads=mot.num_heads,
                            mask=action_attention_mask,
                        )
                        capturer._current_step_mixed[layer_idx] = attn_w

                # 执行原始的 mixed attention
                mixed = mot._mixed_attention(
                    q_cat=q_action, k_cat=k_cat, v_cat=v_cat,
                    attention_mask=action_attention_mask,
                )

                # post block (包含 cross-attention)
                x = mot._apply_post_with_optional_checkpoint(
                    block=block,
                    residual_x=residual_x,
                    gate_msa=gate_msa,
                    shift_mlp=shift_mlp,
                    scale_mlp=scale_mlp,
                    gate_mlp=gate_mlp,
                    use_gradient_checkpointing=False,
                    mixed_slice=mixed,
                    context_payload=action_context_payload,
                )

                # === 捕获 cross-attention 权重 ===
                if layer_idx in capturer.capture_layers and action_context_payload is not None:
                    context = action_context_payload.get("context")
                    ctx_mask = action_context_payload.get("mask")
                    if context is not None:
                        with torch.no_grad():
                            # Cross-attention uses block's cross_attn module
                            norm_x = block.norm3(x)
                            q_cross = block.cross_attn.norm_q(block.cross_attn.q(norm_x))
                            k_cross = block.cross_attn.norm_k(block.cross_attn.k(context))
                            # cross attn 没有 RoPE
                            cross_mask = None
                            if ctx_mask is not None:
                                if ctx_mask.dim() == 3:
                                    cross_mask = ctx_mask  # [B, Sq, Sk]
                                    cross_mask = cross_mask.unsqueeze(1)  # [B, 1, Sq, Sk]
                            cross_w = capturer._compute_attention_weights(
                                q=q_cross, k=k_cross,
                                num_heads=block.cross_attn.num_heads,
                                mask=cross_mask,
                            )
                            capturer._current_step_cross[layer_idx] = cross_w

            return x

        mot.forward_action_with_video_cache = hooked_forward
        print(f"[AttentionCapturer] Hooks installed for layers: {self.capture_layers}")

    def start_capture(self):
        """开始捕获（在 infer_action 调用前）。"""
        self._capturing = True
        self.mixed_attn_weights = []
        self.cross_attn_weights = []

    def on_step_end(self):
        """每个去噪步骤结束后调用，保存当前步的权重。"""
        if self._current_step_mixed:
            self.mixed_attn_weights.append(dict(self._current_step_mixed))
            self._current_step_mixed = {}
        if self._current_step_cross:
            self.cross_attn_weights.append(dict(self._current_step_cross))
            self._current_step_cross = {}

    def stop_capture(self):
        """停止捕获。"""
        self._capturing = False

    def patch_infer_action(self):
        """Patch infer_action 以在每个去噪步后自动调用 on_step_end。"""
        model = self.model
        capturer = self
        original_infer = model.infer_action

        @torch.no_grad()
        def patched_infer_action(**kwargs):
            """与原始 infer_action 相同逻辑，但在每步后触发 on_step_end。"""
            model.eval()
            # 简化调用：直接使用原始逻辑但无法精确 hook 每步...
            # 所以我们在 hooked forward_action_with_video_cache 里自动触发
            # 但由于每步都会调用一次 forward_action_with_video_cache，
            # 我们在那里捕获后需要一个机制来分割步骤。
            # 最简单的方案：patch scheduler step loop
            pass

        # 更好的方案：直接在 hooked forward 里每次调用自动 append
        # 修改 hooked_forward 使其在结束后自动触发 on_step_end
        mot = model.mot
        _prev_hooked = mot.forward_action_with_video_cache

        def auto_step_forward(*args, **kwargs):
            result = _prev_hooked(*args, **kwargs)
            if capturer._capturing:
                capturer.on_step_end()
            return result

        mot.forward_action_with_video_cache = auto_step_forward


# ============================================================================
# 模型加载（与 policy_server.py 相同逻辑）
# ============================================================================


def compose_cfg(config_name: str, task_override: Optional[str] = None) -> DictConfig:
    configs_root = (Path(__file__).resolve().parents[1] / "configs").resolve()
    overrides = []
    if task_override:
        overrides.append(f"task={task_override}")
    if GlobalHydra.instance().is_initialized():
        GlobalHydra.instance().clear()
    with initialize_config_dir(version_base="1.3", config_dir=str(configs_root)):
        cfg = compose(config_name=config_name, overrides=overrides)
    return cfg


def load_model(cfg: DictConfig, ckpt_path: str, device: str = "cuda"):
    """加载模型（与 policy_server.py 中 FastWAMPolicy.__init__ 一致）。"""
    model_cfg = OmegaConf.create(OmegaConf.to_container(cfg.model, resolve=True))
    model_cfg.load_text_encoder = True
    print(f"[load_model] Instantiating model...")
    model = instantiate(model_cfg, model_dtype=torch.bfloat16, device=device)
    model.load_checkpoint(ckpt_path)
    model = model.to(device).eval()
    print(f"[load_model] Model loaded from: {ckpt_path}")
    return model


# ============================================================================
# 可视化
# ============================================================================


def visualize_mixed_attention(
    capturer: AttentionCapturer,
    video_seq_len: int,
    action_seq_len: int,
    save_dir: Path,
):
    """可视化 mixed attention 中 action 对 video vs action tokens 的注意力分配。
    
    分析维度：
    - 按层 (layer): 不同层关注 video 和 action 的比例
    - 按去噪步骤 (step): 注意力分布随去噪进程的变化
    - 按 token 位置: action 中第几个 token 最关注 video
    """
    save_dir.mkdir(parents=True, exist_ok=True)
    num_steps = len(capturer.mixed_attn_weights)
    if num_steps == 0:
        print("[WARNING] No mixed attention weights captured!")
        return

    layers = sorted(capturer.mixed_attn_weights[0].keys())
    print(f"\n{'='*60}")
    print(f"Mixed Attention 分析")
    print(f"{'='*60}")
    print(f"  去噪步数: {num_steps}")
    print(f"  捕获层数: {len(layers)}")
    print(f"  video_seq_len: {video_seq_len}, action_seq_len: {action_seq_len}")
    print(f"  K 序列 = [video({video_seq_len}) | action({action_seq_len})]")

    # === 1. 每层 video vs action 的平均注意力占比 ===
    video_ratios_per_layer = []  # [layer] -> 平均 video 注意力比例
    action_ratios_per_layer = []

    for layer_idx in layers:
        video_ratios_step = []
        for step in range(num_steps):
            if layer_idx not in capturer.mixed_attn_weights[step]:
                continue
            attn_w = capturer.mixed_attn_weights[step][layer_idx]  # [B, H, Sa, Sv+Sa]
            # 分割 video 和 action 部分
            attn_to_video = attn_w[:, :, :, :video_seq_len]  # [B, H, Sa, Sv]
            attn_to_action = attn_w[:, :, :, video_seq_len:]  # [B, H, Sa, Sa]

            # 对所有 head 和 query 取平均
            video_ratio = attn_to_video.sum(dim=-1).mean().item()  # 平均每个 query 分配给 video 的权重
            video_ratios_step.append(video_ratio)

        avg_video_ratio = np.mean(video_ratios_step) if video_ratios_step else 0
        video_ratios_per_layer.append(avg_video_ratio)
        action_ratios_per_layer.append(1.0 - avg_video_ratio)

    # 打印汇总
    print(f"\n  --- 各层 Video vs Action 注意力占比（所有步骤平均） ---")
    print(f"  {'Layer':<8} {'Video%':>10} {'Action%':>10}")
    print(f"  {'-'*30}")
    for i, layer_idx in enumerate(layers):
        print(f"  {layer_idx:<8} {video_ratios_per_layer[i]*100:>9.2f}% {action_ratios_per_layer[i]*100:>9.2f}%")

    # 绘图 1: 按层的 video/action 注意力比例
    fig, ax = plt.subplots(figsize=(12, 5))
    x_pos = np.arange(len(layers))
    width = 0.35
    ax.bar(x_pos - width / 2, [r * 100 for r in video_ratios_per_layer], width, label="Video KV Cache", color="steelblue")
    ax.bar(x_pos + width / 2, [r * 100 for r in action_ratios_per_layer], width, label="Action Self", color="coral")
    ax.set_xlabel("Layer Index")
    ax.set_ylabel("Attention Weight (%)")
    ax.set_title("Mixed Attention: Video vs Action Token Attention per Layer")
    ax.set_xticks(x_pos)
    ax.set_xticklabels([str(l) for l in layers])
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_dir / "mixed_attn_video_vs_action_per_layer.png", dpi=150)
    plt.close(fig)
    print(f"\n  [Saved] {save_dir / 'mixed_attn_video_vs_action_per_layer.png'}")

    # === 2. 按去噪步骤的变化趋势 ===
    # 选几个代表性层
    sample_layers = [layers[0], layers[len(layers) // 2], layers[-1]]
    fig, ax = plt.subplots(figsize=(10, 5))
    for layer_idx in sample_layers:
        step_ratios = []
        for step in range(num_steps):
            if layer_idx in capturer.mixed_attn_weights[step]:
                attn_w = capturer.mixed_attn_weights[step][layer_idx]
                attn_to_video = attn_w[:, :, :, :video_seq_len]
                step_ratios.append(attn_to_video.sum(dim=-1).mean().item() * 100)
        if step_ratios:
            ax.plot(range(len(step_ratios)), step_ratios, marker="o", markersize=3, label=f"Layer {layer_idx}")
    ax.set_xlabel("Denoising Step")
    ax.set_ylabel("Video Attention (%)")
    ax.set_title("Video Attention Ratio across Denoising Steps")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_dir / "mixed_attn_video_ratio_per_step.png", dpi=150)
    plt.close(fig)
    print(f"  [Saved] {save_dir / 'mixed_attn_video_ratio_per_step.png'}")

    # === 3. 注意力热力图 (最后一步，选择中间层) ===
    mid_layer = layers[len(layers) // 2]
    if capturer.mixed_attn_weights and mid_layer in capturer.mixed_attn_weights[-1]:
        attn_w = capturer.mixed_attn_weights[-1][mid_layer]  # [B, H, Sa, Sv+Sa]
        # 取第一个 batch，所有 head 平均
        attn_map = attn_w[0].mean(dim=0).numpy()  # [Sa, Sv+Sa]

        fig, ax = plt.subplots(figsize=(14, 6))
        im = ax.imshow(attn_map, aspect="auto", cmap="viridis")
        ax.axvline(x=video_seq_len - 0.5, color="red", linewidth=2, linestyle="--", label="Video|Action boundary")
        ax.set_xlabel("Key Position (left=Video KV, right=Action KV)")
        ax.set_ylabel("Action Query Position")
        ax.set_title(f"Mixed Attention Heatmap (Layer {mid_layer}, Last Step, Head-averaged)")
        ax.legend(loc="upper right")
        fig.colorbar(im, ax=ax)
        fig.tight_layout()
        fig.savefig(save_dir / "mixed_attn_heatmap.png", dpi=150)
        plt.close(fig)
        print(f"  [Saved] {save_dir / 'mixed_attn_heatmap.png'}")


def visualize_cross_attention(
    capturer: AttentionCapturer,
    context_len: int,
    has_proprio: bool,
    save_dir: Path,
):
    """可视化 cross-attention 中 action 对 text vs state(proprio) tokens 的注意力。
    
    Context 结构: [text_tokens(L) | proprio_token(1)]
    """
    save_dir.mkdir(parents=True, exist_ok=True)
    num_steps = len(capturer.cross_attn_weights)
    if num_steps == 0:
        print("\n[WARNING] No cross-attention weights captured!")
        return

    layers = sorted(capturer.cross_attn_weights[0].keys())
    print(f"\n{'='*60}")
    print(f"Cross-Attention 分析")
    print(f"{'='*60}")
    print(f"  去噪步数: {num_steps}")
    print(f"  Context 结构: [text({context_len - (1 if has_proprio else 0)}) | {'proprio(1)' if has_proprio else ''}]")

    text_end = context_len - (1 if has_proprio else 0)

    # 各层 text vs proprio 注意力占比
    text_ratios = []
    proprio_ratios = []

    for layer_idx in layers:
        text_r_steps = []
        for step in range(num_steps):
            if layer_idx not in capturer.cross_attn_weights[step]:
                continue
            attn_w = capturer.cross_attn_weights[step][layer_idx]  # [B, H, Sa, Ctx]
            attn_to_text = attn_w[:, :, :, :text_end]
            text_ratio = attn_to_text.sum(dim=-1).mean().item()
            text_r_steps.append(text_ratio)

        avg_text = np.mean(text_r_steps) if text_r_steps else 0
        text_ratios.append(avg_text)
        proprio_ratios.append(1.0 - avg_text)

    print(f"\n  --- 各层 Text vs State(Proprio) 注意力占比 ---")
    print(f"  {'Layer':<8} {'Text%':>10} {'State%':>10}")
    print(f"  {'-'*30}")
    for i, layer_idx in enumerate(layers):
        print(f"  {layer_idx:<8} {text_ratios[i]*100:>9.2f}% {proprio_ratios[i]*100:>9.2f}%")

    # 绘图
    fig, ax = plt.subplots(figsize=(12, 5))
    x_pos = np.arange(len(layers))
    width = 0.35
    ax.bar(x_pos - width / 2, [r * 100 for r in text_ratios], width, label="Text Tokens", color="green")
    ax.bar(x_pos + width / 2, [r * 100 for r in proprio_ratios], width, label="State (Proprio) Token", color="orange")
    ax.set_xlabel("Layer Index")
    ax.set_ylabel("Attention Weight (%)")
    ax.set_title("Cross-Attention: Text vs State/Proprio Token per Layer")
    ax.set_xticks(x_pos)
    ax.set_xticklabels([str(l) for l in layers])
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_dir / "cross_attn_text_vs_state_per_layer.png", dpi=150)
    plt.close(fig)
    print(f"\n  [Saved] {save_dir / 'cross_attn_text_vs_state_per_layer.png'}")

    # 热力图
    mid_layer = layers[len(layers) // 2]
    if capturer.cross_attn_weights and mid_layer in capturer.cross_attn_weights[-1]:
        attn_w = capturer.cross_attn_weights[-1][mid_layer]
        attn_map = attn_w[0].mean(dim=0).numpy()  # [Sa, Ctx]

        fig, ax = plt.subplots(figsize=(12, 5))
        im = ax.imshow(attn_map, aspect="auto", cmap="magma")
        if has_proprio:
            ax.axvline(x=text_end - 0.5, color="cyan", linewidth=2, linestyle="--", label="Text|Proprio boundary")
        ax.set_xlabel("Context Position (left=Text, right=State/Proprio)")
        ax.set_ylabel("Action Query Position")
        ax.set_title(f"Cross-Attention Heatmap (Layer {mid_layer}, Last Step, Head-averaged)")
        ax.legend(loc="upper right")
        fig.colorbar(im, ax=ax)
        fig.tight_layout()
        fig.savefig(save_dir / "cross_attn_heatmap.png", dpi=150)
        plt.close(fig)
        print(f"  [Saved] {save_dir / 'cross_attn_heatmap.png'}")


def print_summary(
    capturer: AttentionCapturer,
    video_seq_len: int,
    action_seq_len: int,
    context_len: int,
    has_proprio: bool,
):
    """打印总结分析。"""
    print(f"\n{'='*60}")
    print(f"总结：Action DiT 注意力分布")
    print(f"{'='*60}")

    num_steps = len(capturer.mixed_attn_weights)
    layers = sorted(capturer.mixed_attn_weights[0].keys()) if num_steps > 0 else []

    # 计算总体平均
    all_video_ratios = []
    for step in range(num_steps):
        for layer_idx in layers:
            if layer_idx in capturer.mixed_attn_weights[step]:
                attn_w = capturer.mixed_attn_weights[step][layer_idx]
                attn_to_video = attn_w[:, :, :, :video_seq_len]
                all_video_ratios.append(attn_to_video.sum(dim=-1).mean().item())

    avg_video = np.mean(all_video_ratios) if all_video_ratios else 0

    text_end = context_len - (1 if has_proprio else 0)
    all_text_ratios = []
    num_cross_steps = len(capturer.cross_attn_weights)
    cross_layers = sorted(capturer.cross_attn_weights[0].keys()) if num_cross_steps > 0 else []
    for step in range(num_cross_steps):
        for layer_idx in cross_layers:
            if layer_idx in capturer.cross_attn_weights[step]:
                attn_w = capturer.cross_attn_weights[step][layer_idx]
                attn_to_text = attn_w[:, :, :, :text_end]
                all_text_ratios.append(attn_to_text.sum(dim=-1).mean().item())

    avg_text = np.mean(all_text_ratios) if all_text_ratios else 0

    print(f"""
  ┌─────────────────────────────────────────────────────┐
  │  Mixed Attention (Self-Attention in MoT)            │
  │  ─────────────────────────────────────────────────  │
  │  Action Q → Video KV Cache:  {avg_video*100:6.2f}%               │
  │  Action Q → Action Self KV:  {(1-avg_video)*100:6.2f}%               │
  │                                                     │
  │  Cross-Attention (Post Block)                       │
  │  ─────────────────────────────────────────────────  │
  │  Action → Text Tokens:       {avg_text*100:6.2f}%               │
  │  Action → State/Proprio:     {(1-avg_text)*100:6.2f}%               │
  └─────────────────────────────────────────────────────┘
    """)


# ============================================================================
# 主测试函数
# ============================================================================


def main():
    print("=" * 60)
    print("  Action DiT Attention 分析测试")
    print("=" * 60)

    # --- 检查文件是否存在 ---
    if not Path(CKPT_PATH).exists():
        print(f"[ERROR] Checkpoint 不存在: {CKPT_PATH}")
        print("请修改脚本中的 CKPT_PATH 变量指向正确的模型权重。")
        return
    if not Path(DATASET_STATS_PATH).exists():
        print(f"[ERROR] Dataset stats 不存在: {DATASET_STATS_PATH}")
        return

    # --- 加载配置 ---
    print("\n[1/5] 加载配置...")
    cfg = compose_cfg(CONFIG_NAME, task_override=TASK_OVERRIDE)

    # --- 加载模型 ---
    print("\n[2/5] 加载模型...")
    model = load_model(cfg, CKPT_PATH, device=DEVICE)

    # --- 加载 processor ---
    print("\n[3/5] 加载 Processor...")
    processor: FastWAMProcessor = instantiate(cfg.data.train.processor).eval()
    dataset_stats = load_dataset_stats_from_json(DATASET_STATS_PATH)
    processor.set_normalizer_from_stats(dataset_stats)

    # --- 准备输入数据 ---
    print("\n[4/5] 准备测试输入...")
    # 读取真实的模型输入图片 TODO mock input
    
    
    print(f"  读取输入图片: {input_image_path}")
    pil_img = Image.open(input_image_path).convert("RGB")
    dummy_image = np.asarray(pil_img, dtype=np.uint8)  # [H, W, 3]
    input_h, input_w = dummy_image.shape[0], dummy_image.shape[1]
    print(f"  图片尺寸: H={input_h}, W={input_w}")
    
    # 转为 [-1, 1] tensor
    image_tensor = torch.from_numpy(dummy_image).permute(2, 0, 1).float() / 127.5 - 1.0
    image_tensor = image_tensor.unsqueeze(0).to(device=DEVICE, dtype=torch.bfloat16)

    # State (proprio)
    proprio_dim = int(cfg.data.train.processor.proprio_output_dim)
    
    # 通过 processor 归一化 state
    state_meta = processor.shape_meta["state"]
    state_key = state_meta[0]["key"]
    state_batch = {"state": {state_key: torch.as_tensor(dummy_state, dtype=torch.float32).unsqueeze(0)}}
    state_batch = processor.action_state_transform(state_batch)
    state_batch = processor.normalizer.forward(state_batch)
    proprio = state_batch["state"][state_key]
    if proprio.ndim == 3:
        proprio = proprio.squeeze(0)

    # Prompt
    prompt = DEFAULT_PROMPT.format(task=instruction)

    # Action horizon
    action_horizon = int(cfg.data.train.num_frames) - 1  # 32

    print(f"  image_tensor shape: {image_tensor.shape}")
    print(f"  proprio shape: {proprio.shape}")
    print(f"  action_horizon: {action_horizon}")
    print(f"  prompt: {prompt}")

    # --- 安装 hooks 并推理 ---
    print("\n[5/5] 运行推理并捕获注意力权重...")
    
    # 只捕获部分层（每5层一个，减少内存）
    capture_layers = list(range(0, model.mot.num_layers, 1))  # [0, 5, 10, 15, 20, 25]
    # TODO enable later
    # capturer = AttentionCapturer(model, capture_layers=capture_layers)
    # capturer.install_hooks()
    # capturer.patch_infer_action()
    # capturer.start_capture()

    t0 = time.perf_counter()
    with torch.no_grad():
        result = model.infer_action(
            prompt=prompt,
            input_image=image_tensor,
            action_horizon=action_horizon,
            proprio=proprio,
            num_inference_steps=NUM_INFERENCE_STEPS,
            seed=SEED,
            rand_device="cpu",
            tiled=False,
        )
    capturer.stop_capture()
    elapsed = time.perf_counter() - t0
    print(f"  推理耗时: {elapsed:.2f}s")
    print(f"  预测 action shape: {result['action'].shape}")
    print(f"  捕获到 {len(capturer.mixed_attn_weights)} 步 mixed attention 权重")
    print(f"  捕获到 {len(capturer.cross_attn_weights)} 步 cross attention 权重")

    # --- 获取序列长度信息 ---
    # video_seq_len: 首帧 VAE latent 的 token 数
    # 对于 224x448 图像, VAE 下采样 16x, latent = [1, 48, 1, 28, 56]
    # patch_size = [1, 2, 2], tokens_per_frame = (14/2)*(28/2) = 7 * 14 = 98
    # 首帧 latent frames = 1, 所以 video_seq_len = 98
    vae_h = input_h // 16  # 28
    vae_w = input_w // 16  # 56
    patch_h, patch_w = 2, 2  # from video_dit_config
    video_seq_len = (vae_h // patch_h) * (vae_w // patch_w)  # 7 * 14 = 98
    action_seq_len = action_horizon  # 32

    # context_len: text(128) + proprio(1) = 129 (after text_embedding in action_dit)
    # 但 context 在进入 action_dit 的 cross-attn 之前已经通过 text_embedding 映射
    # 实际 context_len = text_seq_len + (1 if proprio else 0)
    # text_seq_len 来自 tokenizer: context_len=128
    text_seq_len = int(cfg.data.train.context_len)  # 128
    has_proprio = proprio_dim is not None and proprio_dim > 0
    context_total_len = text_seq_len + (1 if has_proprio else 0)

    # --- 可视化 ---
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    visualize_mixed_attention(
        capturer=capturer,
        video_seq_len=video_seq_len,
        action_seq_len=action_seq_len,
        save_dir=OUTPUT_DIR,
    )

    visualize_cross_attention(
        capturer=capturer,
        context_len=context_total_len,
        has_proprio=has_proprio,
        save_dir=OUTPUT_DIR,
    )

    print_summary(
        capturer=capturer,
        video_seq_len=video_seq_len,
        action_seq_len=action_seq_len,
        context_len=context_total_len,
        has_proprio=has_proprio,
    )

    print(f"\n所有可视化结果保存在: {OUTPUT_DIR.resolve()}")
    print("完成!")


if __name__ == "__main__":
    main()
