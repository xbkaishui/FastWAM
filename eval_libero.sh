python experiments/libero/run_libero_manager.py \
  task=libero_uncond_2cam224_1e-4 \
  ckpt=/root/autodl-fs/ckpts/fast_wam/runs/libero_uncond_2cam224_1e-4/2026-06-04_13-50-48/checkpoints/weights/step_020000.pt \
  EVALUATION.dataset_stats_path=/root/autodl-fs/ckpts/fast_wam/runs/libero_uncond_2cam224_1e-4/2026-06-04_13-50-48/dataset_stats.json \
  MULTIRUN.num_gpus=1