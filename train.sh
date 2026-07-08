#!/usr/bin/env bash
# train.sh <category> — fine-tune a per-category RealFill LoRA on MPDD normal images.
#
# Prereq:  python prep_data.py <category> --resolution 256   (builds runs/<cat>/data)
# Output:  runs/<category>/lora/   (unet + text_encoder LoRA adapters)
#
# Env overrides: RES (default 256), RUNS_DIR (default ./runs), STEPS (default 1000)
set -euo pipefail

CAT=${1:?usage: bash train.sh <category>}
RES=${RES:-256}
RUNS_DIR=${RUNS_DIR:-./runs}
STEPS=${STEPS:-1000}

DATA="$RUNS_DIR/$CAT/data"
OUT="$RUNS_DIR/$CAT/lora"
[ -d "$DATA/ref" ] || { echo "run first: python prep_data.py $CAT --resolution $RES"; exit 1; }

accelerate launch train_realfill.py \
  --pretrained_model_name_or_path=stabilityai/stable-diffusion-2-inpainting \
  --train_data_dir="$DATA" \
  --output_dir="$OUT" \
  --resolution="$RES" \
  --train_batch_size=4 \
  --gradient_accumulation_steps=2 \
  --gradient_checkpointing \
  --mixed_precision=fp16 \
  --unet_learning_rate=2e-4 \
  --text_encoder_learning_rate=4e-5 \
  --lr_scheduler=constant \
  --lr_warmup_steps=100 \
  --max_train_steps="$STEPS" \
  --checkpointing_steps="$STEPS" \
  --lora_rank=8 \
  --lora_alpha=16 \
  --lora_dropout=0.1 \
  --seed=42

echo "===== $CAT LoRA done -> $OUT ====="
