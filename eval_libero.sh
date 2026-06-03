python experiments/libero/run_libero_manager.py \
  task=libero_uncond_2cam224_1e-4 \
  ckpt=/root/autodl-fs/ckpts/models/fastwam/libero_uncond_2cam224.pt \
  EVALUATION.dataset_stats_path=/root/autodl-fs/ckpts/models/fastwam/libero_uncond_2cam224_dataset_stats.json \
  MULTIRUN.num_gpus=1