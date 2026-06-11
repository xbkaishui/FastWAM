#!/bin/bash
# FastWAM Policy Server 启动脚本
# 用法: bash start_policy_server.sh [OPTIONS]

# 默认参数
CONFIG="${CONFIG:-sim_pangceban.yaml}"
CKPT="${CKPT:-/root/autodl-fs/ckpts/fast_wam/runs/pangceban_uncond_2cam224_1e-4/2026-06-11_08-24-59/checkpoints/weights/step_012000.pt}"
DATASET_STATS="${DATASET_STATS:-/root/autodl-fs/ckpts/fast_wam/runs/pangceban_uncond_2cam224_1e-4/2026-06-11_08-24-59/dataset_stats.json}"
TASK="${TASK:-pangceban_uncond_2cam224_1e-4}"
DEVICE="${DEVICE:-cuda}"
GPU_ID="${GPU_ID:-0}"
PORT="${PORT:-6006}"
HOST="${HOST:-0.0.0.0}"

echo "============================================"
echo "  FastWAM Policy Server"
echo "============================================"
echo "Config:          ${CONFIG}"
echo "Checkpoint:      ${CKPT}"
echo "Dataset Stats:   ${DATASET_STATS}"
echo "Task:            ${TASK}"
echo "Device:          ${DEVICE}"
echo "GPU ID:          ${GPU_ID}"
echo "Host:Port:       ${HOST}:${PORT}"
echo "============================================"

python scripts/policy_server.py \
    --config "${CONFIG}" \
    --ckpt "${CKPT}" \
    --dataset_stats_path "${DATASET_STATS}" \
    --task "${TASK}" \
    --device "${DEVICE}" \
    --gpu_id "${GPU_ID}" \
    --port "${PORT}" \
    --host "${HOST}" \
    --save_debug_images
