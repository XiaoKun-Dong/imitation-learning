#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 || $# -gt 4 ]]; then
  cat >&2 <<'EOF'
Usage:
  run_libero_official_sft_posteval.sh EXP_NAME [TRIALS] [SEED] [DEMOVLA_TRAIN_PID]

Waits for the full-data DemoVLA training process, then evaluates the trained
DemoVLA checkpoint and the official pi0.5 LIBERO SFT checkpoint on the same
four LIBERO suites with paired environment and flow-noise seeds.
EOF
  exit 2
fi

EXP_NAME=$1
TRIALS=${2:-10}
SEED=${3:-7}
TRAIN_PID=${4:-}
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUN_DIR="$ROOT/outputs/libero_full_matched/$EXP_NAME"
POST_STATUS="$RUN_DIR/official_sft_posteval_status.txt"
OFFICIAL_CHECKPOINT=${OPENPI_PI05_LIBERO_CHECKPOINT:-/home/dongxiaokun/baseck/pi05_libero}
DEMOVLA_CHECKPOINT="$ROOT/checkpoints/demovla_libero_full_matched/$EXP_NAME/29999"
OFFICIAL_LABEL="official_pi05_sft_${EXP_NAME}_pilot${TRIALS}"
DEMOVLA_LABEL="demovla_${EXP_NAME}_pilot${TRIALS}"

mkdir -p "$RUN_DIR"
if [[ -e "$POST_STATUS" ]]; then
  echo "Official-SFT post-eval status already exists; refusing duplicate launch: $POST_STATUS" >&2
  exit 1
fi

record() {
  printf '%s %s\n' "$(date --iso-8601=seconds)" "$*" | tee -a "$POST_STATUS"
}

record "posteval_status=waiting_for_demovla trials=$TRIALS seed=$SEED train_pid=${TRAIN_PID:-auto}"
if [[ -n "$TRAIN_PID" ]]; then
  while [[ -r "/proc/$TRAIN_PID/stat" ]]; do
    state=$(awk '{print $3}' "/proc/$TRAIN_PID/stat")
    [[ "$state" == "Z" ]] && break
    sleep 60
  done
else
  while pgrep -f "scripts/train.py demovla_libero_full_matched --exp-name $EXP_NAME" >/dev/null; do
    sleep 60
  done
fi

for checkpoint in "$OFFICIAL_CHECKPOINT" "$DEMOVLA_CHECKPOINT"; do
  if [[ ! -d "$checkpoint/params" ]]; then
    record "posteval_status=blocked reason=checkpoint_params_missing checkpoint=$checkpoint"
    exit 1
  fi
  stats_count=$(find "$checkpoint/assets" -type f -name norm_stats.json | wc -l)
  if [[ "$stats_count" -ne 1 ]]; then
    record "posteval_status=blocked reason=checkpoint_stats_count checkpoint=$checkpoint count=$stats_count"
    exit 1
  fi
done

record "posteval_status=running target=official_pi05_sft checkpoint=$OFFICIAL_CHECKPOINT"
bash "$ROOT/examples/libero/eval_libero_full_4gpu.sh" \
  "$OFFICIAL_LABEL" pi05_libero "$OFFICIAL_CHECKPOINT" "$TRIALS" "$SEED" \
  >"$RUN_DIR/official_pi05_sft_eval.log" 2>&1

record "posteval_status=running target=demovla checkpoint=$DEMOVLA_CHECKPOINT"
bash "$ROOT/examples/libero/eval_libero_full_4gpu.sh" \
  "$DEMOVLA_LABEL" demovla_libero_full_matched "$DEMOVLA_CHECKPOINT" "$TRIALS" "$SEED" \
  >"$RUN_DIR/demovla_eval.log" 2>&1

OFFICIAL_ROOT="$ROOT/data/libero/videos/full_matched/$OFFICIAL_LABEL"
DEMOVLA_ROOT="$ROOT/data/libero/videos/full_matched/$DEMOVLA_LABEL"
"$ROOT/.venv/bin/python" "$ROOT/examples/libero/summarize_full_suite_eval.py" \
  "$DEMOVLA_ROOT" --reference-root "$OFFICIAL_ROOT" \
  >"$RUN_DIR/official_sft_paired_pilot_summary.log" 2>&1
record "posteval_status=completed comparison=$DEMOVLA_ROOT/comparison.md"
