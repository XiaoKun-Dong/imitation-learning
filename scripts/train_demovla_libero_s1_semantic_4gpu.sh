#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "Usage: train_demovla_libero_s1_semantic_4gpu.sh {control|semantic} EXP_NAME" >&2
  exit 2
fi

ARM=$1
EXP_NAME=$2
case "$ARM" in
  control) CONFIG=demovla_libero_full_s1_semantic_control ;;
  semantic) CONFIG=demovla_libero_full_s1_semantic_ranking ;;
  *) echo "ARM must be control or semantic: $ARM" >&2; exit 2 ;;
esac

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="$ROOT/.venv/bin/python"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export XLA_PYTHON_CLIENT_PREALLOCATE="${XLA_PYTHON_CLIENT_PREALLOCATE:-true}"
export XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.95}"
export OPENPI_PI05_LIBERO_CHECKPOINT="${OPENPI_PI05_LIBERO_CHECKPOINT:-/home/dongxiaokun/baseck/pi05_libero}"
export OPENPI_LIBERO_FULL_DATA_ROOT="${OPENPI_LIBERO_FULL_DATA_ROOT:-$ROOT/data/lerobot/physical-intelligence/libero}"
export OPENPI_DEMOVLA_STAGE_A_PARAMS="${OPENPI_DEMOVLA_STAGE_A_PARAMS:-$ROOT/checkpoints/demovla_libero_full_stage_a_fixed_gate/stage_a_fixed005_seed42_v2/4999/params}"

IFS=',' read -r -a visible_gpus <<<"$CUDA_VISIBLE_DEVICES"
if [[ ${#visible_gpus[@]} -ne 4 ]]; then
  echo "Exactly four GPUs are required; CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES" >&2
  exit 1
fi

STATS="$OPENPI_PI05_LIBERO_CHECKPOINT/assets/physical-intelligence/libero/norm_stats.json"
for required in \
  "$PYTHON" \
  "$OPENPI_DEMOVLA_STAGE_A_PARAMS" \
  "$OPENPI_LIBERO_FULL_DATA_ROOT/meta/info.json" \
  "$STATS"; do
  if [[ ! -e "$required" ]]; then
    echo "Required path missing: $required" >&2
    exit 1
  fi
done

AUDIT_DIR="$ROOT/outputs/demovla_s1_semantic/$EXP_NAME"
mkdir -p "$AUDIT_DIR"
{
  echo "arm=$ARM"
  echo "config=$CONFIG"
  echo "exp_name=$EXP_NAME"
  echo "cuda_visible_devices=$CUDA_VISIBLE_DEVICES"
  echo "stage_a_params=$OPENPI_DEMOVLA_STAGE_A_PARAMS"
  echo "fixed_gate_probability=0.03"
  echo "global_batch_size=128"
  echo "seed=42"
  echo "num_train_steps=1200"
  echo "trainable_scope=original_readout_only"
  if [[ "$ARM" == semantic ]]; then
    echo "single_factor=same_image_prompt_hard_negative_ranking"
    echo "ranking_weight=0.25"
    echo "ranking_margin=0.00002"
  else
    echo "single_factor=none"
  fi
  sha256sum "$STATS"
} >"$AUDIT_DIR/training_contract.txt"

cd "$ROOT"
echo "Training $CONFIG as $EXP_NAME on GPUs $CUDA_VISIBLE_DEVICES"
exec "$PYTHON" scripts/train.py "$CONFIG" --exp-name "$EXP_NAME"
