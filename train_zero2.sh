export TORCHINDUCTOR_MIX_ORDER_REDUCTION=0
# export FASTWAM_SDPA_BACKEND=cudnn
# export FASTWAM_CUDNN_BENCHMARK=1

accelerate launch \
  --config_file scripts/accelerate_configs/accelerate_zero2_ds_bucket200m.yaml \
  --num_processes 6 \
  scripts/train.py \
  task=libero_uncond_2cam224_1e-4 batch_size=20 num_workers=12 \
  log_every=1 eval_every=999999 save_every=999999 \
  seed=42 wandb.enabled=false
  # ++encode_overlap=true ++encode_prefetch_factor=4 \
  # ++dataloader_persistent_workers=true \
  # ++compile_mot=false ++compile_vae=true ++compile_vae_mode=default
