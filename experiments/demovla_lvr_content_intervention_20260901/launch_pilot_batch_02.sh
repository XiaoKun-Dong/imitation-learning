#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/dongxiaokun/imitation-learning/openpi
EXPERIMENT_ROOT="$ROOT/experiments/demovla_lvr_content_intervention_20260901"
RUN_ROOT="$EXPERIMENT_ROOT/runs"
CHECKPOINT="$ROOT/checkpoints/demovla_libero_full_stage_a_fixed_gate/stage_a_fixed005_seed42_v2/4999"
DONOR_ROOT="$ROOT/outputs/demovla_key_replan/stagea4999_libero10_task9_ep3_v1/correct/replans"

cd "$ROOT"
mkdir -p "$RUN_ROOT"
for target in p03 p04; do
  if [[ -e "$RUN_ROOT/$target" ]]; then
    echo "Refusing to overwrite existing output: $RUN_ROOT/$target" >&2
    exit 1
  fi
done
if ss -ltn | grep -Eq ':(8800|8801|8802|8900|8901|8902) '; then
  echo "A requested policy port is already in use" >&2
  exit 1
fi

nohup env GPU_OFFSET=0 \
  bash examples/libero/eval_demovla_content_lockstep_4gpu.sh \
  "$CHECKPOINT" libero_10 5 2 72 \
  "$DONOR_ROOT/replan_013_step_0065.npz" \
  "$RUN_ROOT/p03" 8800 \
  >"$RUN_ROOT/p03_launcher.log" 2>&1 &
p03_pid=$!
echo "$p03_pid" >"$RUN_ROOT/p03.pid"

nohup env GPU_OFFSET=4 \
  bash examples/libero/eval_demovla_content_lockstep_4gpu.sh \
  "$CHECKPOINT" libero_10 3 6 13 \
  "$DONOR_ROOT/replan_018_step_0090.npz" \
  "$RUN_ROOT/p04" 8900 \
  >"$RUN_ROOT/p04_launcher.log" 2>&1 &
p04_pid=$!
echo "$p04_pid" >"$RUN_ROOT/p04.pid"

printf 'p03_pid=%s\np04_pid=%s\n' "$p03_pid" "$p04_pid"
