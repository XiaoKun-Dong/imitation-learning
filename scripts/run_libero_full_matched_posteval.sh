#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 || $# -gt 3 ]]; then
  echo "Usage: run_libero_full_matched_posteval.sh EXP_NAME [TRIALS] [SEED]" >&2
  exit 2
fi

EXP_NAME=$1
TRIALS=${2:-10}
SEED=${3:-7}
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUN_DIR="$ROOT/outputs/libero_full_matched/$EXP_NAME"
SEQUENCE_STATUS="$RUN_DIR/sequence_status.txt"
POST_STATUS="$RUN_DIR/posteval_status.txt"
CONTROL_LABEL="control_${EXP_NAME}_pilot${TRIALS}"
DEMOVLA_LABEL="demovla_${EXP_NAME}_pilot${TRIALS}"

if [[ -e "$POST_STATUS" ]]; then
  echo "Post-eval status already exists; refusing duplicate launch: $POST_STATUS" >&2
  exit 1
fi

record() {
  printf '%s %s\n' "$(date --iso-8601=seconds)" "$*" | tee -a "$POST_STATUS"
}

record "posteval_status=waiting_for_training trials=$TRIALS seed=$SEED"
while ! grep -q "sequence_status=completed" "$SEQUENCE_STATUS" 2>/dev/null; do
  if grep -q "status=failed" "$SEQUENCE_STATUS" 2>/dev/null; then
    record "posteval_status=blocked reason=training_failed"
    exit 1
  fi
  sleep 60
done

CONTROL_CHECKPOINT="$ROOT/checkpoints/pi05_libero_full_matched/$EXP_NAME/29999"
DEMOVLA_CHECKPOINT="$ROOT/checkpoints/demovla_libero_full_matched/$EXP_NAME/29999"
record "posteval_status=running target=control"
bash "$ROOT/examples/libero/eval_libero_full_4gpu.sh" \
  "$CONTROL_LABEL" pi05_libero_full_matched "$CONTROL_CHECKPOINT" "$TRIALS" "$SEED" \
  >"$RUN_DIR/control_eval.log" 2>&1

record "posteval_status=running target=demovla"
bash "$ROOT/examples/libero/eval_libero_full_4gpu.sh" \
  "$DEMOVLA_LABEL" demovla_libero_full_matched "$DEMOVLA_CHECKPOINT" "$TRIALS" "$SEED" \
  >"$RUN_DIR/demovla_eval.log" 2>&1

CONTROL_ROOT="$ROOT/data/libero/videos/full_matched/$CONTROL_LABEL"
DEMOVLA_ROOT="$ROOT/data/libero/videos/full_matched/$DEMOVLA_LABEL"
"$ROOT/.venv/bin/python" "$ROOT/examples/libero/summarize_full_suite_eval.py" \
  "$DEMOVLA_ROOT" --reference-root "$CONTROL_ROOT" \
  >"$RUN_DIR/paired_pilot_summary.log" 2>&1
record "posteval_status=completed comparison=$DEMOVLA_ROOT/comparison.md"
