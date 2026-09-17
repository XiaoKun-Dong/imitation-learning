#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 || $# -gt 2 ]]; then
  cat >&2 <<'EOF'
Usage:
  train_demovla_libero_stage_a_4gpu.sh EXP_NAME [--resume]

Runs DemoVLA recovery stage A on the complete 40-task LIBERO dataset. The
official pi0.5-LIBERO backbone and its normalization statistics are frozen;
only DemoVLA parameters other than the fixed scalar layer gates are trained.
EOF
  exit 2
fi

EXP_NAME=$1
MODE=${2:-}
if [[ -n "$MODE" && "$MODE" != "--resume" ]]; then
  echo "Second argument must be --resume when provided: $MODE" >&2
  exit 2
fi

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="$ROOT/.venv/bin/python"
CONFIG=demovla_libero_full_stage_a_fixed_gate
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export XLA_PYTHON_CLIENT_PREALLOCATE="${XLA_PYTHON_CLIENT_PREALLOCATE:-true}"
export XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.95}"
export OPENPI_PI05_LIBERO_CHECKPOINT="${OPENPI_PI05_LIBERO_CHECKPOINT:-/home/dongxiaokun/baseck/pi05_libero}"
export OPENPI_LIBERO_FULL_DATA_ROOT="${OPENPI_LIBERO_FULL_DATA_ROOT:-$ROOT/data/lerobot/physical-intelligence/libero}"

IFS=',' read -r -a visible_gpus <<<"$CUDA_VISIBLE_DEVICES"
if [[ ${#visible_gpus[@]} -ne 4 ]]; then
  echo "Exactly four GPUs are required; CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES" >&2
  exit 1
fi

STATS="$OPENPI_PI05_LIBERO_CHECKPOINT/assets/physical-intelligence/libero/norm_stats.json"
for required in \
  "$PYTHON" \
  "$OPENPI_PI05_LIBERO_CHECKPOINT/params" \
  "$OPENPI_LIBERO_FULL_DATA_ROOT/meta/info.json" \
  "$STATS"; do
  if [[ ! -e "$required" ]]; then
    echo "Required path missing: $required" >&2
    exit 1
  fi
done

AUDIT_DIR="$ROOT/outputs/demovla_stage_a/$EXP_NAME"
mkdir -p "$AUDIT_DIR"
"$PYTHON" "$ROOT/scripts/verify_libero_full_dataset.py" \
  "$OPENPI_LIBERO_FULL_DATA_ROOT" \
  "$STATS" \
  --allow-stats-mismatch \
  --output "$AUDIT_DIR/dataset_and_stats_audit.json"
{
  echo "config=$CONFIG"
  echo "exp_name=$EXP_NAME"
  echo "cuda_visible_devices=$CUDA_VISIBLE_DEVICES"
  echo "pi05_libero_checkpoint=$OPENPI_PI05_LIBERO_CHECKPOINT"
  echo "dataset_root=$OPENPI_LIBERO_FULL_DATA_ROOT"
  echo "fixed_gate_probability=0.05"
  echo "interaction_output_init_std=0.001"
  echo "trainable_scope=demovla_except_interaction_gates"
  sha256sum "$STATS"
} >"$AUDIT_DIR/training_contract.txt"

cd "$ROOT"
args=("$PYTHON" scripts/train.py "$CONFIG" --exp-name "$EXP_NAME")
if [[ "$MODE" == "--resume" ]]; then
  args+=(--resume)
fi

echo "Training $CONFIG as $EXP_NAME on GPUs $CUDA_VISIBLE_DEVICES"
echo "Global batch size: 128 (32 samples/GPU)"
echo "Checkpoint: checkpoints/$CONFIG/$EXP_NAME"
exec "${args[@]}"
