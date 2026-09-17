#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/dongxiaokun/imitation-learning/openpi
EXPERIMENT_ROOT="$ROOT/experiments/demovla_lvr_content_intervention_20260901"
RUN_ROOT="$EXPERIMENT_ROOT/runs"
CHECKPOINT="$ROOT/checkpoints/demovla_libero_full_stage_a_fixed_gate/stage_a_fixed005_seed42_v2/4999"
DONOR_ROOT="$ROOT/outputs/demovla_key_replan/stagea4999_libero10_task9_ep3_v1/correct/replans"

cd "$ROOT"
mkdir -p "$RUN_ROOT"
for target in p01_retry01 p02; do
  if [[ -e "$RUN_ROOT/$target" ]]; then
    echo "Refusing to overwrite existing output: $RUN_ROOT/$target" >&2
    exit 1
  fi
done
if ss -ltn | grep -Eq ':(8600|8601|8602|8700|8701|8702) '; then
  echo "A requested policy port is already in use" >&2
  exit 1
fi

nohup env GPU_OFFSET=0 \
  bash examples/libero/eval_demovla_content_lockstep_4gpu.sh \
  "$CHECKPOINT" libero_10 8 9 18 \
  "$DONOR_ROOT/replan_046_step_0230.npz" \
  "$RUN_ROOT/p01_retry01" 8600 \
  >"$RUN_ROOT/p01_retry01_launcher.log" 2>&1 &
p01_pid=$!
echo "$p01_pid" >"$RUN_ROOT/p01_retry01.pid"

nohup env GPU_OFFSET=4 \
  bash examples/libero/eval_demovla_content_lockstep_4gpu.sh \
  "$CHECKPOINT" libero_10 1 5 46 \
  "$DONOR_ROOT/replan_072_step_0360.npz" \
  "$RUN_ROOT/p02" 8700 \
  >"$RUN_ROOT/p02_launcher.log" 2>&1 &
p02_pid=$!
echo "$p02_pid" >"$RUN_ROOT/p02.pid"

printf 'p01_retry01_pid=%s\np02_pid=%s\n' "$p01_pid" "$p02_pid"
