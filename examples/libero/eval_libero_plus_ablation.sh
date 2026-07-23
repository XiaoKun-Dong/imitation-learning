#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 || $# -gt 4 ]]; then
  cat >&2 <<'EOF'
Usage:
  eval_libero_plus_ablation.sh POLICY_DIR SEED [GPU] [BASE_PORT]

Runs correct-mask, empty-mask, and wrong-mask Step1 ablations on the same
50 LIBERO-plus Objects Layout tasks using one GPU.
EOF
  exit 2
fi

POLICY_DIR=$1
SEED=$2
GPU=${3:-0}
BASE_PORT=${4:-8000}

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SINGLE="$ROOT/examples/libero/eval_libero_plus_single.sh"

for entry in correct:2d empty:2d_empty wrong:2d_wrong; do
  label=${entry%%:*}
  condition=${entry#*:}
  # The single-GPU runner evaluates all standard seeds. Restricting this script
  # to one seed is handled by its temporary SEEDS override below.
  LIBERO_PLUS_SEEDS="$SEED" bash "$SINGLE" \
    "step1_4999_ablation_${label}" \
    "$condition" \
    pi05_libero_object_2d_step1 \
    "$POLICY_DIR" \
    "$GPU" \
    "$BASE_PORT"
done
