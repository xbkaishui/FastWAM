"""
Benchmark: SDPA with boolean mask (math backend) vs flex_attention
Tests both training (large seq, mask has False) and inference (small seq, mask all True) scenarios.
"""
import torch
import torch.nn.functional as F
from einops import rearrange
import time


def build_mot_attention_mask(
    video_seq_len: int,
    action_seq_len: int,
    video_tokens_per_frame: int,
    video_mask_mode: str = "first_frame_causal",
    device: torch.device = torch.device("cuda"),
) -> torch.Tensor:
    """Reproduce the MoT attention mask from FastWAM."""
    total_seq_len = video_seq_len + action_seq_len
    mask = torch.zeros((total_seq_len, total_seq_len), dtype=torch.bool, device=device)

    # video -> video
    if video_mask_mode == "bidirectional":
        mask[:video_seq_len, :video_seq_len] = True
    elif video_mask_mode == "first_frame_causal":
        video_mask = torch.ones((video_seq_len, video_seq_len), dtype=torch.bool, device=device)
        first_frame_tokens = min(video_tokens_per_frame, video_seq_len)
        video_mask[:first_frame_tokens, first_frame_tokens:] = False
        mask[:video_seq_len, :video_seq_len] = video_mask
    elif video_mask_mode == "per_frame_causal":
        num_frames = video_seq_len // video_tokens_per_frame
        frame_causal = torch.tril(torch.ones((num_frames, num_frames), dtype=torch.bool, device=device))
        video_mask = frame_causal.repeat_interleave(video_tokens_per_frame, dim=0).repeat_interleave(
            video_tokens_per_frame, dim=1
        )
        mask[:video_seq_len, :video_seq_len] = video_mask

    # action -> action
    mask[video_seq_len:, video_seq_len:] = True
    # action -> first-frame video only
    first_frame_tokens = min(video_tokens_per_frame, video_seq_len)
    mask[video_seq_len:, :first_frame_tokens] = True
    return mask


def sdpa_with_mask(q, k, v, num_heads, attn_mask):
    """Current implementation: SDPA with explicit mask (falls back to math backend)."""
    q = rearrange(q, "b s (n d) -> b n s d", n=num_heads)
    k = rearrange(k, "b s (n d) -> b n s d", n=num_heads)
    v = rearrange(v, "b s (n d) -> b n s d", n=num_heads)
    x = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
    x = rearrange(x, "b n s d -> b s (n d)", n=num_heads)
    return x


def sdpa_no_mask(q, k, v, num_heads):
    """Optimized: SDPA without mask (uses Flash Attention kernel)."""
    q = rearrange(q, "b s (n d) -> b n s d", n=num_heads)
    k = rearrange(k, "b s (n d) -> b n s d", n=num_heads)
    v = rearrange(v, "b s (n d) -> b n s d", n=num_heads)
    x = F.scaled_dot_product_attention(q, k, v)
    x = rearrange(x, "b n s d -> b s (n d)", n=num_heads)
    return x


def flex_attention_impl(q, k, v, num_heads, block_mask):
    """flex_attention with compiled block mask."""
    from torch.nn.attention.flex_attention import flex_attention
    q = rearrange(q, "b s (n d) -> b n s d", n=num_heads)
    k = rearrange(k, "b s (n d) -> b n s d", n=num_heads)
    v = rearrange(v, "b s (n d) -> b n s d", n=num_heads)
    x = flex_attention(q, k, v, block_mask=block_mask)
    x = rearrange(x, "b n s d -> b s (n d)", n=num_heads)
    return x


def flex_attention_compiled_impl(q, k, v, num_heads, block_mask):
    """flex_attention compiled with torch.compile."""
    from torch.nn.attention.flex_attention import flex_attention
    q = rearrange(q, "b s (n d) -> b n s d", n=num_heads)
    k = rearrange(k, "b s (n d) -> b n s d", n=num_heads)
    v = rearrange(v, "b s (n d) -> b n s d", n=num_heads)
    compiled_flex = torch.compile(flex_attention)
    x = compiled_flex(q, k, v, block_mask=block_mask)
    x = rearrange(x, "b n s d -> b s (n d)", n=num_heads)
    return x


def benchmark_fn(fn, warmup=10, repeats=50):
    """Benchmark with CUDA events for precise timing."""
    # Warmup
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    start_events = [torch.cuda.Event(enable_timing=True) for _ in range(repeats)]
    end_events = [torch.cuda.Event(enable_timing=True) for _ in range(repeats)]

    for i in range(repeats):
        start_events[i].record()
        fn()
        end_events[i].record()

    torch.cuda.synchronize()
    times = [s.elapsed_time(e) for s, e in zip(start_events, end_events)]
    avg_ms = sum(times) / len(times)
    min_ms = min(times)
    max_ms = max(times)
    return avg_ms, min_ms, max_ms


def test_inference_scenario():
    """
    Inference scenario (infer_action with KV cache):
    q=(1, 32, 3072), k=(1, 102, 3072), v=(1, 102, 3072)
    num_heads=24, attn_mask=(32, 102) — ALL TRUE
    """
    print("\n" + "=" * 80)
    print("SCENARIO 1: Inference (infer_action with KV cache)")
    print("  q=(1, 32, 3072), k/v=(1, 102, 3072), heads=24, head_dim=128")
    print("  attn_mask=(32, 102) ALL TRUE")
    print("=" * 80)

    device = torch.device("cuda")
    dtype = torch.bfloat16
    B, S_q, S_kv, D, num_heads = 1, 32, 102, 3072, 24

    q = torch.randn(B, S_q, D, device=device, dtype=dtype)
    k = torch.randn(B, S_kv, D, device=device, dtype=dtype)
    v = torch.randn(B, S_kv, D, device=device, dtype=dtype)
    attn_mask = torch.ones(S_q, S_kv, dtype=torch.bool, device=device)

    # 1. Current: SDPA with mask
    avg, mn, mx = benchmark_fn(lambda: sdpa_with_mask(q, k, v, num_heads, attn_mask))
    print(f"\n  [SDPA + bool mask]     avg={avg:.3f}ms  min={mn:.3f}ms  max={mx:.3f}ms")

    # 2. Optimized: SDPA without mask (since mask is all True)
    avg2, mn2, mx2 = benchmark_fn(lambda: sdpa_no_mask(q, k, v, num_heads))
    print(f"  [SDPA no mask]         avg={avg2:.3f}ms  min={mn2:.3f}ms  max={mx2:.3f}ms")
    print(f"  → Speedup (no mask):   {avg / avg2:.2f}x")

    # 3. flex_attention
    try:
        from torch.nn.attention.flex_attention import flex_attention, create_block_mask

        def mask_fn(b, h, q_idx, kv_idx):
            return torch.ones((), dtype=torch.bool, device=device)

        block_mask = create_block_mask(mask_fn, B=B, H=None, Q_LEN=S_q, KV_LEN=S_kv, device=device)
        avg3, mn3, mx3 = benchmark_fn(lambda: flex_attention_impl(q, k, v, num_heads, block_mask))
        print(f"  [flex_attention]       avg={avg3:.3f}ms  min={mn3:.3f}ms  max={mx3:.3f}ms")
        print(f"  → Speedup (flex):      {avg / avg3:.2f}x")
    except Exception as e:
        print(f"  [flex_attention]       SKIPPED: {e}")

    # Correctness check
    out_mask = sdpa_with_mask(q, k, v, num_heads, attn_mask)
    out_no_mask = sdpa_no_mask(q, k, v, num_heads)
    diff = (out_mask - out_no_mask).abs().max().item()
    print(f"\n  Correctness (mask vs no_mask): max_abs_diff = {diff:.6e}")


def test_training_scenario():
    """
    Training scenario (full MoT mixed attention):
    Typical: video=5 frames at 224x224 → latent 5 frames at 14x14 = 980 video tokens
             action_horizon=32 action tokens
    Total seq = 1012, mask has False regions
    """
    print("\n" + "=" * 80)
    print("SCENARIO 2: Training (full MoT mixed attention)")
    print("  5 latent frames × 14×14 = 980 video tokens + 32 action tokens = 1012 total")
    print("  num_heads=24, head_dim=128, hidden_dim=3072")
    print("  attn_mask=(1012, 1012) with False regions (first_frame_causal)")
    print("=" * 80)

    device = torch.device("cuda")
    dtype = torch.bfloat16

    video_tokens_per_frame = 14 * 14  # 196
    num_video_frames = 5
    video_seq_len = num_video_frames * video_tokens_per_frame  # 980
    action_seq_len = 32
    total_seq = video_seq_len + action_seq_len  # 1012
    D = 3072
    num_heads = 24
    B = 1

    q = torch.randn(B, total_seq, D, device=device, dtype=dtype)
    k = torch.randn(B, total_seq, D, device=device, dtype=dtype)
    v = torch.randn(B, total_seq, D, device=device, dtype=dtype)

    attn_mask = build_mot_attention_mask(
        video_seq_len=video_seq_len,
        action_seq_len=action_seq_len,
        video_tokens_per_frame=video_tokens_per_frame,
        video_mask_mode="first_frame_causal",
        device=device,
    )
    true_ratio = attn_mask.float().mean().item()
    print(f"  Mask True ratio: {true_ratio:.2%}")

    # 1. Current: SDPA with mask
    avg, mn, mx = benchmark_fn(lambda: sdpa_with_mask(q, k, v, num_heads, attn_mask))
    print(f"\n  [SDPA + bool mask]     avg={avg:.3f}ms  min={mn:.3f}ms  max={mx:.3f}ms")

    # 2. SDPA no mask (for reference, NOT correct but shows upper bound)
    avg2, mn2, mx2 = benchmark_fn(lambda: sdpa_no_mask(q, k, v, num_heads))
    print(f"  [SDPA no mask (ref)]   avg={avg2:.3f}ms  min={mn2:.3f}ms  max={mx2:.3f}ms")
    print(f"  → Upper bound speedup: {avg / avg2:.2f}x (if we could remove mask)")

    # 3. flex_attention
    try:
        from torch.nn.attention.flex_attention import flex_attention, create_block_mask

        # Capture mask structure into a function
        _video_seq_len = video_seq_len
        _video_tokens_per_frame = video_tokens_per_frame
        _total_seq = total_seq

        def mot_mask_fn(b, h, q_idx, kv_idx):
            # video -> video: first_frame_causal
            is_q_video = q_idx < _video_seq_len
            is_kv_video = kv_idx < _video_seq_len
            is_q_action = q_idx >= _video_seq_len
            is_kv_action = kv_idx >= _video_seq_len

            # video->video: first_frame_causal (first frame can't see later frames)
            q_is_first_frame = q_idx < _video_tokens_per_frame
            kv_is_later_frame = kv_idx >= _video_tokens_per_frame
            video_to_video = is_q_video & is_kv_video & ~(q_is_first_frame & kv_is_later_frame)

            # action->action: full attention
            action_to_action = is_q_action & is_kv_action

            # action->video first frame only
            kv_is_first_frame = kv_idx < _video_tokens_per_frame
            action_to_video_first = is_q_action & is_kv_video & kv_is_first_frame

            # video->action: NOT allowed (zeros in original mask)
            return video_to_video | action_to_action | action_to_video_first

        block_mask = create_block_mask(mot_mask_fn, B=B, H=None, Q_LEN=total_seq, KV_LEN=total_seq, device=device)
        avg3, mn3, mx3 = benchmark_fn(lambda: flex_attention_impl(q, k, v, num_heads, block_mask))
        print(f"  [flex_attention]       avg={avg3:.3f}ms  min={mn3:.3f}ms  max={mx3:.3f}ms")
        print(f"  → Speedup (flex):      {avg / avg3:.2f}x")

        # Correctness check
        out_sdpa = sdpa_with_mask(q, k, v, num_heads, attn_mask)
        out_flex = flex_attention_impl(q, k, v, num_heads, block_mask)
        diff = (out_sdpa - out_flex).abs().max().item()
        print(f"\n  Correctness (sdpa vs flex): max_abs_diff = {diff:.6e}")
    except Exception as e:
        print(f"  [flex_attention]       SKIPPED: {e}")
        import traceback
        traceback.print_exc()


def test_training_scenario_large():
    """
    Larger training scenario:
    video=5 frames at 480p → latent 5 frames at 30x30 = 4500 video tokens
    + 32 action tokens = 4532 total
    """
    print("\n" + "=" * 80)
    print("SCENARIO 3: Training LARGE (higher resolution)")
    print("  5 latent frames × 30×30 = 4500 video tokens + 32 action tokens = 4532 total")
    print("  num_heads=24, head_dim=128, hidden_dim=3072")
    print("  attn_mask=(4532, 4532) with False regions (first_frame_causal)")
    print("=" * 80)

    device = torch.device("cuda")
    dtype = torch.bfloat16

    video_tokens_per_frame = 30 * 30  # 900
    num_video_frames = 5
    video_seq_len = num_video_frames * video_tokens_per_frame  # 4500
    action_seq_len = 32
    total_seq = video_seq_len + action_seq_len  # 4532
    D = 3072
    num_heads = 24
    B = 1

    q = torch.randn(B, total_seq, D, device=device, dtype=dtype)
    k = torch.randn(B, total_seq, D, device=device, dtype=dtype)
    v = torch.randn(B, total_seq, D, device=device, dtype=dtype)

    attn_mask = build_mot_attention_mask(
        video_seq_len=video_seq_len,
        action_seq_len=action_seq_len,
        video_tokens_per_frame=video_tokens_per_frame,
        video_mask_mode="first_frame_causal",
        device=device,
    )
    true_ratio = attn_mask.float().mean().item()
    print(f"  Mask True ratio: {true_ratio:.2%}")

    # 1. Current: SDPA with mask
    avg, mn, mx = benchmark_fn(lambda: sdpa_with_mask(q, k, v, num_heads, attn_mask))
    print(f"\n  [SDPA + bool mask]     avg={avg:.3f}ms  min={mn:.3f}ms  max={mx:.3f}ms")

    # 2. SDPA no mask (upper bound reference)
    avg2, mn2, mx2 = benchmark_fn(lambda: sdpa_no_mask(q, k, v, num_heads))
    print(f"  [SDPA no mask (ref)]   avg={avg2:.3f}ms  min={mn2:.3f}ms  max={mx2:.3f}ms")
    print(f"  → Upper bound speedup: {avg / avg2:.2f}x")

    # 3. flex_attention
    try:
        from torch.nn.attention.flex_attention import flex_attention, create_block_mask

        _video_seq_len = video_seq_len
        _video_tokens_per_frame = video_tokens_per_frame

        def mot_mask_fn(b, h, q_idx, kv_idx):
            is_q_video = q_idx < _video_seq_len
            is_kv_video = kv_idx < _video_seq_len
            is_q_action = q_idx >= _video_seq_len
            is_kv_action = kv_idx >= _video_seq_len

            q_is_first_frame = q_idx < _video_tokens_per_frame
            kv_is_later_frame = kv_idx >= _video_tokens_per_frame
            video_to_video = is_q_video & is_kv_video & ~(q_is_first_frame & kv_is_later_frame)

            action_to_action = is_q_action & is_kv_action

            kv_is_first_frame = kv_idx < _video_tokens_per_frame
            action_to_video_first = is_q_action & is_kv_video & kv_is_first_frame

            return video_to_video | action_to_action | action_to_video_first

        block_mask = create_block_mask(mot_mask_fn, B=B, H=None, Q_LEN=total_seq, KV_LEN=total_seq, device=device)
        avg3, mn3, mx3 = benchmark_fn(lambda: flex_attention_impl(q, k, v, num_heads, block_mask))
        print(f"  [flex_attention]       avg={avg3:.3f}ms  min={mn3:.3f}ms  max={mx3:.3f}ms")
        print(f"  → Speedup (flex):      {avg / avg3:.2f}x")

        # Compiled flex_attention
        compiled_flex_fn = torch.compile(flex_attention)
        q4 = rearrange(q, "b s (n d) -> b n s d", n=num_heads)
        k4 = rearrange(k, "b s (n d) -> b n s d", n=num_heads)
        v4 = rearrange(v, "b s (n d) -> b n s d", n=num_heads)
        # Warmup compilation
        for _ in range(3):
            _ = compiled_flex_fn(q4, k4, v4, block_mask=block_mask)
        torch.cuda.synchronize()
        avg4, mn4, mx4 = benchmark_fn(lambda: compiled_flex_fn(q4, k4, v4, block_mask=block_mask))
        print(f"  [flex_attn compiled]   avg={avg4:.3f}ms  min={mn4:.3f}ms  max={mx4:.3f}ms")
        print(f"  → Speedup (compiled):  {avg / avg4:.2f}x")

        # Correctness
        out_sdpa = sdpa_with_mask(q, k, v, num_heads, attn_mask)
        out_flex = flex_attention_impl(q, k, v, num_heads, block_mask)
        diff = (out_sdpa - out_flex).abs().max().item()
        print(f"\n  Correctness (sdpa vs flex): max_abs_diff = {diff:.6e}")
    except Exception as e:
        print(f"  [flex_attention]       SKIPPED: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    print("FlexAttention vs SDPA Benchmark for FastWAM MoT")
    print(f"PyTorch {torch.__version__}, CUDA {torch.version.cuda}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")

    test_inference_scenario()
    # test_training_scenario()
    # test_training_scenario_large()

    print("\n" + "=" * 80)
    print("DONE")
    print("=" * 80)
