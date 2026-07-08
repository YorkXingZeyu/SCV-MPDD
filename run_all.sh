#!/usr/bin/env bash
# run_all.sh — end-to-end pipeline over all six MPDD categories:
#   prep_data -> train LoRA -> evaluate (image/pixel AUROC).
#
# Env overrides: RES (256), RUNS_DIR (./runs), DATA_ROOT (./data/MPDD),
#                STRENGTH (0.3), N_RECON (10), OUT (results.csv)
set -euo pipefail

RES=${RES:-256}
RUNS_DIR=${RUNS_DIR:-./runs}
DATA_ROOT=${DATA_ROOT:-./data/MPDD}
STRENGTH=${STRENGTH:-0.3}
N_RECON=${N_RECON:-10}
OUT=${OUT:-results.csv}

CATS="connector metal_plate bracket_black bracket_brown bracket_white tubes"

for CAT in $CATS; do
  echo "========== $CAT =========="
  python prep_data.py "$CAT" --data_root "$DATA_ROOT" --runs_dir "$RUNS_DIR" --resolution "$RES"
  RES="$RES" RUNS_DIR="$RUNS_DIR" bash train.sh "$CAT"
done

echo "========== evaluate =========="
python evaluate.py \
  --data_root "$DATA_ROOT" --runs_dir "$RUNS_DIR" \
  --categories "$(echo $CATS | tr ' ' ',')" \
  --resolution "$RES" --strength "$STRENGTH" --n_recon "$N_RECON" \
  --out "$OUT"

echo "===== all done -> $OUT ====="
