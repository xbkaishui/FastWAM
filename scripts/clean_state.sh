#!/bin/bash
# 定时清理训练 state 目录，仅保留最新的 N 个 checkpoint
# 用法: bash scripts/clean_state.sh [state_dir] [keep_count] [interval_seconds]

STATE_DIR="${1:-/root/autodl-fs/ckpts/fast_wam/runs/libero_joint_2cam224_1e-4/2026-06-05_21-50-14/checkpoints/state}"
KEEP_COUNT="${2:-2}"
INTERVAL="${3:-300}"  # 默认每 5 分钟检查一次

echo "[clean_state] 监控目录: ${STATE_DIR}"
echo "[clean_state] 保留最新: ${KEEP_COUNT} 个"
echo "[clean_state] 检查间隔: ${INTERVAL} 秒"
echo "---"

while true; do
    if [ ! -d "${STATE_DIR}" ]; then
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] 目录不存在，等待中..."
        sleep "${INTERVAL}"
        continue
    fi

    # 列出所有 step_* 子目录，按名称排序（名称即时间顺序）
    dirs=($(ls -d "${STATE_DIR}"/step_* 2>/dev/null | sort))
    total=${#dirs[@]}

    if [ "${total}" -le "${KEEP_COUNT}" ]; then
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] 当前 ${total} 个 state，无需清理"
    else
        delete_count=$((total - KEEP_COUNT))
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] 当前 ${total} 个 state，将删除最旧的 ${delete_count} 个"
        for ((i=0; i<delete_count; i++)); do
            target="${dirs[$i]}"
            echo "  删除: ${target}"
            rm -rf "${target}"
        done
        echo "  清理完成，剩余 ${KEEP_COUNT} 个"
    fi

    sleep "${INTERVAL}"
done

