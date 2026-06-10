python experiments/libero/run_libero_manager.py \
  task=libero_joint_2cam224_1e-4 \
  ckpt=/root/autodl-fs/ckpts/fast_wam/runs/libero_joint_2cam224_1e-4/2026-06-06_08-33-10/checkpoints/weights/step_028930.pt \
  EVALUATION.dataset_stats_path=/root/autodl-fs/ckpts/fast_wam/runs/libero_joint_2cam224_1e-4/2026-06-06_08-33-10/dataset_stats.json \
  MULTIRUN.num_gpus=1