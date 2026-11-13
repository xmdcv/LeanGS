#!/bin/bash
# Train the rank-5 configuration, one scene at a time.

set -euo pipefail
cd "$(dirname "$0")/.."
# Run after activating the environment documented in README.md.

OUT="${OUT_BASE:-./output/leangs}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
NERF360_ROOT="${NERF360_ROOT:-./datasets/mipnerf360}"
DB_ROOT="${DB_ROOT:-./datasets/deepblending}"
TANDT_ROOT="${TANDT_ROOT:-./datasets/tanksandtemples}"

for dataset_root in "$NERF360_ROOT" "$TANDT_ROOT" "$DB_ROOT"; do
  [[ -d "$dataset_root" ]] || { echo "Dataset directory missing: $dataset_root" >&2; exit 1; }
done
mkdir -p "$OUT"

run_train() {
  local name="$1"
  shift
  local log="$OUT/${name}.log"
  echo "[$(date +%H:%M:%S)] Starting $name -> $log"
  python train.py -m "$OUT/$name" "$@" 2>&1 | tee "$log"
  echo "[$(date +%H:%M:%S)] Finished $name"
}

# ---------- MipNeRF360 ----------
run_train bicycle  -s "$NERF360_ROOT/bicycle" -i images --eval --densification_interval 500 --optimizer_type default --test_iterations 30000 --grad_abs_thresh 0.0012 --use_lr_vdcf --lr_vdcf_k 5 --prune_ratio 0.25
run_train flowers  -s "$NERF360_ROOT/flowers" -i images --eval --densification_interval 500 --optimizer_type default --test_iterations 30000 --dense 0.005 --grad_abs_thresh 0.0015 --use_lr_vdcf --lr_vdcf_k 5 --prune_ratio 0.25
run_train garden   -s "$NERF360_ROOT/garden" -i images --eval --densification_interval 500 --optimizer_type default --test_iterations 30000 --highfeature_lr 0.02 --loss_thresh 0.06 --grad_abs_thresh 0.0008 --use_lr_vdcf --lr_vdcf_k 5 --prune_ratio 0.25
run_train stump    -s "$NERF360_ROOT/stump" -i images --eval --densification_interval 500 --optimizer_type default --test_iterations 30000 --dense 0.004 --grad_abs_thresh 0.0015 --use_lr_vdcf --lr_vdcf_k 5 --prune_ratio 0.25
run_train treehill -s "$NERF360_ROOT/treehill" -i images --eval --densification_interval 500 --optimizer_type default --test_iterations 30000 --dense 0.01 --grad_abs_thresh 0.002 --use_lr_vdcf --lr_vdcf_k 5 --prune_ratio 0.25
run_train room     -s "$NERF360_ROOT/room" -i images --eval --densification_interval 500 --optimizer_type default --test_iterations 30000 --highfeature_lr 0.02 --grad_abs_thresh 0.0008 --use_lr_vdcf --lr_vdcf_k 5 --prune_ratio 0.25
run_train counter  -s "$NERF360_ROOT/counter" -i images --eval --densification_interval 500 --optimizer_type default --test_iterations 30000 --highfeature_lr 0.02 --grad_abs_thresh 0.0008 --use_lr_vdcf --lr_vdcf_k 5 --prune_ratio 0.25
run_train kitchen  -s "$NERF360_ROOT/kitchen" -i images --eval --densification_interval 500 --optimizer_type default --test_iterations 30000 --highfeature_lr 0.02 --grad_abs_thresh 0.0006 --use_lr_vdcf --lr_vdcf_k 5 --prune_ratio 0.25
run_train bonsai   -s "$NERF360_ROOT/bonsai" -i images --eval --densification_interval 500 --optimizer_type default --test_iterations 30000 --highfeature_lr 0.02 --grad_abs_thresh 0.0006 --use_lr_vdcf --lr_vdcf_k 5 --prune_ratio 0.25

# ---------- Tanks and Temples ----------
run_train truck    -s "$TANDT_ROOT/truck" --eval --densification_interval 500 --optimizer_type default --test_iterations 30000 --highfeature_lr 0.04 --grad_abs_thresh 0.0009 --mult 0.7 --use_lr_vdcf --lr_vdcf_k 5 --prune_ratio 0.25
run_train train    -s "$TANDT_ROOT/train" --eval --densification_interval 500 --optimizer_type default --test_iterations 30000 --highfeature_lr 0.042 --grad_abs_thresh 0.0015 --dense 0.01 --mult 0.7 --use_lr_vdcf --lr_vdcf_k 5 --prune_ratio 0.25

# ---------- Deep Blending ----------
run_train playroom -s "$DB_ROOT/playroom" --eval --densification_interval 500 --optimizer_type default --test_iterations 30000 --highfeature_lr 0.0015 --dense 0.003 --mult 0.7 --use_lr_vdcf --lr_vdcf_k 5 --prune_ratio 0.25
run_train drjohnson -s "$DB_ROOT/drjohnson" --eval --densification_interval 500 --optimizer_type default --test_iterations 30000 --highfeature_lr 0.0025 --grad_abs_thresh 0.0012 --dense 0.013 --mult 0.7 --use_lr_vdcf --lr_vdcf_k 5 --prune_ratio 0.25

echo "All 13 scenes finished. Training logs: $OUT/*.log"
