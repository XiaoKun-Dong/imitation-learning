#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/dongxiaokun/imitation-learning/openpi
EXPERIMENT_ROOT="$ROOT/experiments/demovla_lvr_content_intervention_20260901"
RUN_ROOT="$EXPERIMENT_ROOT/expansion_runs"
PAIR_LAUNCHER="$EXPERIMENT_ROOT/launch_expansion_pair.sh"
PROGRESS="$RUN_ROOT/progress.jsonl"

cd "$ROOT"
mkdir -p "$RUN_ROOT"
if [[ -e "$PROGRESS" ]]; then
  echo "Refusing to overwrite expansion progress: $PROGRESS" >&2
  exit 1
fi

for ((batch_index = 0; batch_index < 15; batch_index++)); do
  printf -v point_a 'e%02d' "$((batch_index * 2 + 1))"
  printf -v point_b 'e%02d' "$((batch_index * 2 + 2))"
  base_port=$((9000 + batch_index * 20))
  echo "Starting batch $((batch_index + 1))/15: $point_a $point_b"
  bash "$PAIR_LAUNCHER" "$point_a" "$point_b" "$base_port"
  read -r pid_a <"$RUN_ROOT/${point_a}.pid"
  read -r pid_b <"$RUN_ROOT/${point_b}.pid"

  while kill -0 "$pid_a" 2>/dev/null || kill -0 "$pid_b" 2>/dev/null; do
    sleep 5
  done

  for point_id in "$point_a" "$point_b"; do
    summary="$RUN_ROOT/$point_id/summary.json"
    if [[ ! -s "$summary" ]]; then
      echo "Missing summary for $point_id; stopping expansion" >&2
      exit 1
    fi
    jq -c --arg point_id "$point_id" \
      '{point_id:$point_id,status,intervention_reached,task_id,episode_id,branch_replan,conditions}' \
      "$summary" >>"$PROGRESS"
  done
done

echo "All 30 frozen expansion points completed"
