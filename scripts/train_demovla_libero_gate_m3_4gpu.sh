#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 || $# -gt 2 ]]; then
  cat >&2 <<'EOF'
Usage:
  train_demovla_libero_gate_m3_4gpu.sh EXP_NAME [--resume]

Examples:
  bash scripts/train_demovla_libero_gate_m3_4gpu.sh gate_m3_v1
  bash scripts/train_demovla_libero_gate_m3_4gpu.sh gate_m3_v1 --resume
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
CONFIG=demovla_libero_sparse_deep_dynamic_gate_m3

if [[ ! -x "$PYTHON" ]]; then
  echo "Project Python not found: $PYTHON" >&2
  exit 1
fi

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
IFS=',' read -r -a visible_gpus <<<"$CUDA_VISIBLE_DEVICES"
if [[ ${#visible_gpus[@]} -ne 4 ]]; then
  echo "Exactly four GPUs are required; CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES" >&2
  exit 1
fi

export XLA_PYTHON_CLIENT_PREALLOCATE="${XLA_PYTHON_CLIENT_PREALLOCATE:-false}"
export OPENPI_PI05_LIBERO_CHECKPOINT="${OPENPI_PI05_LIBERO_CHECKPOINT:-/home/dongxiaokun/baseck/pi05_libero}"
export OPENPI_LIBERO_DATA_ROOT="${OPENPI_LIBERO_DATA_ROOT:-$ROOT/data/lerobot/local/libero}"

if [[ ! -d "$OPENPI_PI05_LIBERO_CHECKPOINT/params" ]]; then
  echo "PI0.5-LIBERO params not found: $OPENPI_PI05_LIBERO_CHECKPOINT/params" >&2
  exit 1
fi
if [[ ! -d "$OPENPI_LIBERO_DATA_ROOT/meta" ]]; then
  echo "LIBERO LeRobot dataset not found: $OPENPI_LIBERO_DATA_ROOT" >&2
  exit 1
fi

cd "$ROOT"
args=(
  "$PYTHON" scripts/train.py "$CONFIG"
  --exp-name "$EXP_NAME"
)
if [[ "$MODE" == "--resume" ]]; then
  args+=(--resume)
fi

echo "Training $CONFIG as $EXP_NAME on GPUs $CUDA_VISIBLE_DEVICES"
echo "Checkpoint: checkpoints/$CONFIG/$EXP_NAME"
exec "${args[@]}"
