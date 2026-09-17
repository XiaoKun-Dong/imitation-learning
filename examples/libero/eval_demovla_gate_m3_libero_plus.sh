#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 || $# -gt 2 ]]; then
  cat >&2 <<'EOF'
Usage:
  eval_demovla_gate_m3_libero_plus.sh EXP_NAME [CHECKPOINT_STEP]

Runs the 50-task LIBERO-plus Objects Layout evaluation for seeds 7, 42,
and 123 using only RGB, language, and robot state observations.
EOF
  exit 2
fi

EXP_NAME=$1
CHECKPOINT_STEP=${2:-29999}
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CONFIG=demovla_libero_sparse_deep_dynamic_gate_m3
POLICY_DIR="$ROOT/checkpoints/$CONFIG/$EXP_NAME/$CHECKPOINT_STEP"
LABEL="demovla_gate_m3_${EXP_NAME}_step${CHECKPOINT_STEP}"

if [[ ! -d "$POLICY_DIR/params" ]]; then
  echo "Checkpoint params not found: $POLICY_DIR/params" >&2
  exit 1
fi

exec bash "$ROOT/examples/libero/eval_libero_plus_parallel.sh" \
  "$LABEL" "$CONFIG" "$POLICY_DIR"
