#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 || $# -gt 3 ]]; then
  cat >&2 <<'EOF'
Usage:
  train_libero_full_matched_4gpu.sh {control|demovla} EXP_NAME [--resume]

The two targets use the same full 40-task dataset, official normalization
statistics, pi0.5-base initialization, optimizer, batch size, seed, training
steps, and four-GPU FSDP layout. Only the model architecture differs.
EOF
  exit 2
fi

TARGET=$1
EXP_NAME=$2
MODE=${3:-}
if [[ "$TARGET" == "control" ]]; then
  CONFIG=pi05_libero_full_matched
elif [[ "$TARGET" == "demovla" ]]; then
  CONFIG=demovla_libero_full_matched
else
  echo "Target must be control or demovla: $TARGET" >&2
  exit 2
fi
if [[ -n "$MODE" && "$MODE" != "--resume" ]]; then
  echo "Third argument must be --resume when provided: $MODE" >&2
  exit 2
fi

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="$ROOT/.venv/bin/python"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export XLA_PYTHON_CLIENT_PREALLOCATE="${XLA_PYTHON_CLIENT_PREALLOCATE:-true}"
export XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.95}"
export OPENPI_PI05_BASE_CHECKPOINT="${OPENPI_PI05_BASE_CHECKPOINT:-/home/dongxiaokun/baseck/pi05_base}"
export OPENPI_PI05_LIBERO_CHECKPOINT="${OPENPI_PI05_LIBERO_CHECKPOINT:-/home/dongxiaokun/baseck/pi05_libero}"
export OPENPI_LIBERO_FULL_DATA_ROOT="${OPENPI_LIBERO_FULL_DATA_ROOT:-$ROOT/data/lerobot/physical-intelligence/libero}"
export OPENPI_LIBERO_FULL_STATS_ROOT="${OPENPI_LIBERO_FULL_STATS_ROOT:-$ROOT/assets/libero_full_matched}"

IFS=',' read -r -a visible_gpus <<<"$CUDA_VISIBLE_DEVICES"
if [[ ${#visible_gpus[@]} -ne 4 ]]; then
  echo "Exactly four GPUs are required; CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES" >&2
  exit 1
fi

STATS="$OPENPI_LIBERO_FULL_STATS_ROOT/physical-intelligence/libero/norm_stats.json"
for required in "$PYTHON" "$OPENPI_PI05_BASE_CHECKPOINT/params" "$OPENPI_LIBERO_FULL_DATA_ROOT/meta/info.json" "$STATS"; do
  if [[ ! -e "$required" ]]; then
    echo "Required path missing: $required" >&2
    exit 1
  fi
done

AUDIT_DIR="$ROOT/outputs/libero_full_matched/$EXP_NAME"
mkdir -p "$AUDIT_DIR"
"$PYTHON" "$ROOT/scripts/verify_libero_full_dataset.py" \
  "$OPENPI_LIBERO_FULL_DATA_ROOT" \
  "$STATS" \
  --output "$AUDIT_DIR/${TARGET}_dataset_and_stats_audit.json"
{
  echo "target=$TARGET"
  echo "config=$CONFIG"
  echo "exp_name=$EXP_NAME"
  echo "cuda_visible_devices=$CUDA_VISIBLE_DEVICES"
  echo "xla_python_client_preallocate=$XLA_PYTHON_CLIENT_PREALLOCATE"
  echo "xla_python_client_mem_fraction=$XLA_PYTHON_CLIENT_MEM_FRACTION"
  echo "pi05_base_checkpoint=$OPENPI_PI05_BASE_CHECKPOINT"
  echo "pi05_libero_checkpoint=$OPENPI_PI05_LIBERO_CHECKPOINT"
  echo "dataset_root=$OPENPI_LIBERO_FULL_DATA_ROOT"
  echo "stats_root=$OPENPI_LIBERO_FULL_STATS_ROOT"
  sha256sum "$STATS"
} >"$AUDIT_DIR/${TARGET}_training_contract.txt"

cd "$ROOT"
args=("$PYTHON" scripts/train.py "$CONFIG" --exp-name "$EXP_NAME")
if [[ "$MODE" == "--resume" ]]; then
  args+=(--resume)
fi

echo "Training $CONFIG as $EXP_NAME on GPUs $CUDA_VISIBLE_DEVICES"
echo "Global batch size: 32 (8 samples/GPU)"
echo "Checkpoint: checkpoints/$CONFIG/$EXP_NAME"
exec "${args[@]}"
