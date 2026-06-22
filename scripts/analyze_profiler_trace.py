#!/usr/bin/env python3
"""Analyze PyTorch Profiler trace JSON to diagnose training performance bottlenecks.

Usage:
    python scripts/analyze_profiler_trace.py <trace_json_path>

Example:
    python scripts/analyze_profiler_trace.py \
        /root/autodl-fs/ckpts/fast_wam/runs/pangceban_uncond_2cam224_1e-4/profiler_logs/autodl-container-*.pt.trace.json
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path


def load_trace(path: str) -> list:
    print(f"Loading trace: {path}")
    with open(path, "r") as f:
        data = json.load(f)
    events = data.get("traceEvents", [])
    print(f"Total events: {len(events)}")
    return events


def find_phase_events(events: list, phase_names: list[str]) -> dict[str, list[dict]]:
    """Find user-annotation events by name (forward, backward, optimizer_step, data_loading)."""
    results = defaultdict(list)
    for e in events:
        name = e.get("name", "")
        dur = e.get("dur", 0)
        if name in phase_names and dur > 0:
            results[name].append(e)
    return results


def analyze_phase_timing(phase_events: dict[str, list[dict]]):
    """Print timing summary for each training phase."""
    print("\n" + "=" * 80)
    print("PHASE TIMING SUMMARY")
    print("=" * 80)
    print(f"{'Phase':<20} {'Count':>6} {'Mean (ms)':>12} {'Min (ms)':>12} {'Max (ms)':>12}")
    print("-" * 80)

    for name in ["data_loading", "forward", "backward", "optimizer_step"]:
        evts = phase_events.get(name, [])
        if not evts:
            continue
        durs = [e["dur"] / 1000.0 for e in evts]
        mean_dur = sum(durs) / len(durs)
        min_dur = min(durs)
        max_dur = max(durs)
        print(f"{name:<20} {len(durs):>6} {mean_dur:>12.2f} {min_dur:>12.2f} {max_dur:>12.2f}")

    # Print ratio
    fwd_evts = phase_events.get("forward", [])
    bwd_evts = phase_events.get("backward", [])
    if fwd_evts and bwd_evts:
        fwd_mean = sum(e["dur"] for e in fwd_evts) / len(fwd_evts) / 1000.0
        bwd_mean = sum(e["dur"] for e in bwd_evts) / len(bwd_evts) / 1000.0
        print(f"\n  backward/forward ratio: {bwd_mean / fwd_mean:.2f}x")


def analyze_nccl_in_phase(events: list, phase_event: dict, phase_name: str):
    """Analyze NCCL communication events within a specific phase."""
    ts_start = phase_event["ts"]
    ts_end = ts_start + phase_event["dur"]

    nccl_by_type = defaultdict(lambda: {"count": 0, "total_dur": 0})

    for e in events:
        ts = e.get("ts", 0)
        dur = e.get("dur", 0)
        name = e.get("name", "")
        if ts >= ts_start and ts <= ts_end and dur > 0:
            if any(kw in name.lower() for kw in ["nccl", "allreduce", "all_reduce", "reduce_scatter", "all_gather"]):
                key = name.split("(")[0].strip()
                nccl_by_type[key]["count"] += 1
                nccl_by_type[key]["total_dur"] += dur

    total_nccl_dur = sum(v["total_dur"] for v in nccl_by_type.values())
    phase_dur = phase_event["dur"]

    print(f"\n{'─' * 80}")
    print(f"NCCL COMMUNICATION in [{phase_name}] (wall time: {phase_dur / 1000:.2f} ms)")
    print(f"{'─' * 80}")

    if not nccl_by_type:
        print("  (no NCCL events found)")
        return total_nccl_dur

    print(f"  Total NCCL time: {total_nccl_dur / 1000:.2f} ms ({total_nccl_dur / phase_dur * 100:.1f}% of phase)")
    print(f"  {'Operation':<65} {'Count':>6} {'Total (ms)':>12}")
    print(f"  {'-' * 83}")

    for name, info in sorted(nccl_by_type.items(), key=lambda x: -x[1]["total_dur"]):
        print(f"  {name:<65} {info['count']:>6} {info['total_dur'] / 1000:>12.2f}")

    return total_nccl_dur


def analyze_top_kernels(events: list, phase_event: dict, phase_name: str, top_n: int = 15):
    """Analyze top GPU kernels within a phase."""
    ts_start = phase_event["ts"]
    ts_end = ts_start + phase_event["dur"]

    kernels = defaultdict(lambda: {"count": 0, "total_dur": 0})
    nccl_dur = 0
    compute_dur = 0

    for e in events:
        ts = e.get("ts", 0)
        dur = e.get("dur", 0)
        cat = e.get("cat", "")
        name = e.get("name", "")
        if ts >= ts_start and ts <= ts_end and dur > 0 and cat == "kernel":
            key = name[:90]
            kernels[key]["count"] += 1
            kernels[key]["total_dur"] += dur
            if "nccl" in name.lower():
                nccl_dur += dur
            else:
                compute_dur += dur

    phase_dur = phase_event["dur"]
    print(f"\n{'─' * 80}")
    print(f"GPU KERNEL BREAKDOWN in [{phase_name}] (wall time: {phase_dur / 1000:.2f} ms)")
    print(f"{'─' * 80}")
    print(f"  Pure compute GPU time: {compute_dur / 1000:.2f} ms")
    print(f"  NCCL kernel GPU time:  {nccl_dur / 1000:.2f} ms")
    print(f"  Total kernel GPU time: {(compute_dur + nccl_dur) / 1000:.2f} ms")
    print(f"\n  {'Kernel':<75} {'Count':>6} {'Total (ms)':>12}")
    print(f"  {'-' * 93}")

    for name, info in sorted(kernels.items(), key=lambda x: -x[1]["total_dur"])[:top_n]:
        print(f"  {name:<75} {info['count']:>6} {info['total_dur'] / 1000:>12.2f}")

    return compute_dur, nccl_dur


def print_optimization_summary(
    fwd_dur: float,
    bwd_dur: float,
    opt_dur: float,
    bwd_compute: float,
    bwd_nccl: float,
    opt_nccl: float,
):
    """Print optimization recommendations."""
    print("\n" + "=" * 80)
    print("OPTIMIZATION SUMMARY")
    print("=" * 80)

    total_step = fwd_dur + bwd_dur + opt_dur
    print(f"\n  Total step time: {total_step:.2f} ms")
    print(f"    forward:        {fwd_dur:>8.2f} ms ({fwd_dur / total_step * 100:>5.1f}%)")
    print(f"    backward:       {bwd_dur:>8.2f} ms ({bwd_dur / total_step * 100:>5.1f}%)")
    print(f"      ├─ compute:   {bwd_compute:>8.2f} ms")
    print(f"      └─ NCCL:      {bwd_nccl:>8.2f} ms ({bwd_nccl / bwd_dur * 100:.1f}% of backward)")
    print(f"    optimizer_step: {opt_dur:>8.2f} ms ({opt_dur / total_step * 100:>5.1f}%)")
    print(f"      └─ NCCL:      {opt_nccl:>8.2f} ms ({opt_nccl / opt_dur * 100:.1f}% of opt_step)")

    print("\n  Bottleneck Analysis:")
    if bwd_nccl / bwd_dur > 0.4:
        print(f"    [!] backward 中 {bwd_nccl / bwd_dur * 100:.0f}% 时间花在 NCCL 通信上")
        print(f"        → 建议: 开启 overlap_comm=true + contiguous_gradients=true")
    if opt_nccl / opt_dur > 0.5:
        print(f"    [!] optimizer_step 中 {opt_nccl / opt_dur * 100:.0f}% 时间花在 NCCL 通信上")
        print(f"        → 主要来自 clip_grad_norm_ 的 all-gather 操作")
        print(f"        → 建议: 考虑 local grad clipping 或增大 reduce_bucket_size")
    if bwd_dur / fwd_dur > 2.0:
        print(f"    [!] backward/forward = {bwd_dur / fwd_dur:.1f}x (理想值 ~2x for compute-only)")
        if bwd_nccl / bwd_dur > 0.4:
            print(f"        → 通信开销是主因，开启 overlap 后预期降至 ~{bwd_compute + bwd_compute * 0.3:.0f} ms")


def main():
    parser = argparse.ArgumentParser(description="Analyze PyTorch Profiler trace for training performance")
    parser.add_argument("trace_path", type=str, help="Path to .pt.trace.json file")
    parser.add_argument("--top-kernels", type=int, default=15, help="Number of top kernels to show per phase")
    parser.add_argument("--step-index", type=int, default=0, help="Which profiler step to analyze (0=first)")
    args = parser.parse_args()

    trace_path = args.trace_path
    if "*" in trace_path:
        import glob
        matches = glob.glob(trace_path)
        if not matches:
            print(f"Error: No files matching pattern: {trace_path}", file=sys.stderr)
            sys.exit(1)
        trace_path = matches[0]

    if not Path(trace_path).exists():
        print(f"Error: File not found: {trace_path}", file=sys.stderr)
        sys.exit(1)

    events = load_trace(trace_path)

    # Find phase events
    phase_names = ["forward", "backward", "optimizer_step", "data_loading"]
    phase_events = find_phase_events(events, phase_names)

    if not phase_events:
        print("Error: No training phase events found (forward/backward/optimizer_step).", file=sys.stderr)
        print("Make sure the trace was captured with record_function annotations.", file=sys.stderr)
        sys.exit(1)

    # Print phase timing overview
    analyze_phase_timing(phase_events)

    # Analyze specific step
    step_idx = args.step_index
    fwd_evt = phase_events.get("forward", [None])[step_idx] if len(phase_events.get("forward", [])) > step_idx else None
    bwd_evt = phase_events.get("backward", [None])[step_idx] if len(phase_events.get("backward", [])) > step_idx else None
    opt_evt = phase_events.get("optimizer_step", [None])[step_idx] if len(phase_events.get("optimizer_step", [])) > step_idx else None

    print(f"\n\n{'=' * 80}")
    print(f"DETAILED ANALYSIS FOR STEP #{step_idx}")
    print(f"{'=' * 80}")

    bwd_nccl_total = 0
    opt_nccl_total = 0
    bwd_compute = 0
    bwd_nccl_kernel = 0

    if fwd_evt:
        analyze_top_kernels(events, fwd_evt, "forward", top_n=args.top_kernels)

    if bwd_evt:
        bwd_nccl_total = analyze_nccl_in_phase(events, bwd_evt, "backward")
        bwd_compute, bwd_nccl_kernel = analyze_top_kernels(events, bwd_evt, "backward", top_n=args.top_kernels)

    if opt_evt:
        opt_nccl_total = analyze_nccl_in_phase(events, opt_evt, "optimizer_step")
        analyze_top_kernels(events, opt_evt, "optimizer_step", top_n=args.top_kernels)

    # Final summary
    if fwd_evt and bwd_evt and opt_evt:
        print_optimization_summary(
            fwd_dur=fwd_evt["dur"] / 1000.0,
            bwd_dur=bwd_evt["dur"] / 1000.0,
            opt_dur=opt_evt["dur"] / 1000.0,
            bwd_compute=bwd_compute / 1000.0,
            bwd_nccl=bwd_nccl_kernel / 1000.0,
            opt_nccl=opt_nccl_total / 1000.0,
        )


if __name__ == "__main__":
    main()
