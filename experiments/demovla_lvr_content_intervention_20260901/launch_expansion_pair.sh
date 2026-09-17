#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 3 ]]; then
  echo "Usage: $0 POINT_A POINT_B BASE_PORT" >&2
  exit 2
fi

POINT_A=$1
POINT_B=$2
BASE_PORT=$3
ROOT=/home/dongxiaokun/imitation-learning/openpi
EXPERIMENT_ROOT="$ROOT/experiments/demovla_lvr_content_intervention_20260901"
MANIFEST="$EXPERIMENT_ROOT/expansion_manifest.json"
RUN_ROOT="$EXPERIMENT_ROOT/expansion_runs"
CHECKPOINT="$ROOT/checkpoints/demovla_libero_full_stage_a_fixed_gate/stage_a_fixed005_seed42_v2/4999"
DONOR_ROOT="$ROOT/outputs/demovla_key_replan/stagea4999_libero10_task9_ep3_v1/correct/replans"

load_point() {
  local point_id=$1
  jq -er --arg point_id "$point_id" '
    .points[] | select(.point_id == $point_id) |
    [.task_id, .episode_id, .branch_replan, .donor_id] | @tsv
  ' "$MANIFEST"
}

verify_donor() {
  local donor_id=$1
  local donor_file expected_sha actual_sha
  donor_file=$(jq -er --arg donor_id "$donor_id" '.donors[$donor_id].file' "$MANIFEST")
  expected_sha=$(jq -er --arg donor_id "$donor_id" '.donors[$donor_id].sha256' "$MANIFEST")
  actual_sha=$(sha256sum "$DONOR_ROOT/$donor_file" | cut -d ' ' -f 1)
  if [[ "$actual_sha" != "$expected_sha" ]]; then
    echo "Donor hash mismatch for $donor_id" >&2
    exit 1
  fi
  printf '%s\n' "$DONOR_ROOT/$donor_file"
}

cd "$ROOT"
mkdir -p "$RUN_ROOT"
for target in "$POINT_A" "$POINT_B"; do
  if [[ -e "$RUN_ROOT/$target" ]]; then
    echo "Refusing to overwrite existing output: $RUN_ROOT/$target" >&2
    exit 1
  fi
done

IFS=$'\t' read -r task_a episode_a replan_a donor_id_a < <(load_point "$POINT_A")
IFS=$'\t' read -r task_b episode_b replan_b donor_id_b < <(load_point "$POINT_B")
donor_a=$(verify_donor "$donor_id_a")
donor_b=$(verify_donor "$donor_id_b")
port_a=$BASE_PORT
port_b=$((BASE_PORT + 10))
if ss -ltn | grep -Eq ":(${port_a}|$((port_a + 1))|$((port_a + 2))|${port_b}|$((port_b + 1))|$((port_b + 2))) "; then
  echo "A requested policy port is already in use" >&2
  exit 1
fi

nohup env GPU_OFFSET=0 \
  bash examples/libero/eval_demovla_content_lockstep_4gpu.sh \
  "$CHECKPOINT" libero_10 "$task_a" "$episode_a" "$replan_a" \
  "$donor_a" "$RUN_ROOT/$POINT_A" "$port_a" \
  >"$RUN_ROOT/${POINT_A}_launcher.log" 2>&1 &
pid_a=$!
echo "$pid_a" >"$RUN_ROOT/${POINT_A}.pid"

nohup env GPU_OFFSET=4 \
  bash examples/libero/eval_demovla_content_lockstep_4gpu.sh \
  "$CHECKPOINT" libero_10 "$task_b" "$episode_b" "$replan_b" \
  "$donor_b" "$RUN_ROOT/$POINT_B" "$port_b" \
  >"$RUN_ROOT/${POINT_B}_launcher.log" 2>&1 &
pid_b=$!
echo "$pid_b" >"$RUN_ROOT/${POINT_B}.pid"

printf '%s_pid=%s\n%s_pid=%s\n' "$POINT_A" "$pid_a" "$POINT_B" "$pid_b"

