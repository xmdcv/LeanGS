#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

OUT="${OUT_BASE:-./output/leangs}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
SCENES=(bicycle flowers garden stump treehill room counter kitchen bonsai truck train playroom drjohnson)

for scene in "${SCENES[@]}"; do
  model="$OUT/$scene"
  [[ -f "$model/cfg_args" ]] || { echo "Model configuration missing: $model/cfg_args" >&2; exit 1; }
  mult=0.5
  case "$scene" in truck|train|playroom|drjohnson) mult=0.7 ;; esac
  {
    python render.py -m "$model" --iteration 30000 --skip_train \
      --use_lr_vdcf --lr_vdcf_k 5 --mult "$mult"
    python metrics.py -m "$model"
  } 2>&1 | tee "$OUT/${scene}_evaluation.log"
done
