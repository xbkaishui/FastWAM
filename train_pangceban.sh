# bash scripts/train_zero1.sh 6 task=pangceban_uncond_2cam224_1e-4

export TORCHINDUCTOR_MIX_ORDER_REDUCTION=0
# export FASTWAM_SDPA_BACKEND=cudnn
# export FASTWAM_CUDNN_BENCHMARK=1
RUN_ID="$(date +%Y-%m-%d_%H-%M-%S)"
TASK_BASENAME="pangceban_uncond_2cam224_1e-4"

accelerate launch \
  --config_file scripts/accelerate_configs/accelerate_zero2_ds_bucket200m.yaml \
  --num_processes 6 \
  scripts/train.py \
  task=${TASK_BASENAME} batch_size=20 num_workers=12 \
  log_every=10 eval_every=999999 save_every=3000 \
  seed=42 wandb.enabled=true wandb.name=pangceban_new_0629 output_dir=/root/autodl-fs/ckpts/fast_wam/runs/${TASK_BASENAME}/${RUN_ID}
  # ++encode_overlap=true ++encode_prefetch_factor=4 \
  # ++dataloader_persistent_workers=true \
  # ++compile_mot=false ++compile_vae=true ++compile_vae_mode=default
