#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 4 || $# -gt 6 ]]; then
  cat >&2 <<'EOF'
Usage:
  eval_libero_plus_failures.sh LABEL OBJECT_CONDITION POLICY_CONFIG POLICY_DIR [GPU] [PORT]

The script builds a per-seed union of failures from official, frozen-none, and
object-2d final metrics, then reruns those paired cases with mask/bbox overlays.
EOF
  exit 2
fi

LABEL=$1
OBJECT_CONDITION=$2
POLICY_CONFIG=$3
POLICY_DIR=$4
GPU=${5:-0}
PORT=${6:-8000}

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON="$ROOT/.venv/bin/python"
METRICS_ROOT="$ROOT/data/libero_plus/videos/final"
OUTPUT_ROOT="$ROOT/data/libero_plus/videos/failure_overlay"
LOG_DIR="$ROOT/data/libero_plus/logs/failure_overlay/$LABEL"
SEEDS=(7 42 123)

mkdir -p "$OUTPUT_ROOT" "$LOG_DIR"
server_pid=""

cleanup() {
  if [[ -n "$server_pid" ]] && kill -0 "$server_pid" 2>/dev/null; then
    kill "$server_pid" 2>/dev/null || true
  fi
}
trap cleanup EXIT INT TERM

failure_ids_for_seed() {
  local seed=$1
  "$PYTHON" - "$METRICS_ROOT" "$seed" <<'PY'
import json
import pathlib
import sys

root = pathlib.Path(sys.argv[1])
seed = int(sys.argv[2])
failed_ids = set()
for label in ("official", "frozen_none", "object_2d"):
    path = root / f"{label}_seed{seed}" / "metrics.jsonl"
    if not path.exists():
        raise FileNotFoundError(path)
    for line in path.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        if not row["success"]:
            failed_ids.add(int(row["benchmark_task_id"]))
print(" ".join(map(str, sorted(failed_ids))))
PY
}

cd "$ROOT"
server_log="$LOG_DIR/server.log"
CUDA_VISIBLE_DEVICES=$GPU \
XLA_PYTHON_CLIENT_PREALLOCATE=false \
"$PYTHON" -u scripts/serve_policy.py \
  --port "$PORT" \
  policy:checkpoint \
  --policy.config "$POLICY_CONFIG" \
  --policy.dir "$POLICY_DIR" \
  >"$server_log" 2>&1 &
server_pid=$!

for ((elapsed = 0; elapsed < 300; elapsed++)); do
  if grep -q "Creating server" "$server_log" 2>/dev/null; then
    break
  fi
  if ! kill -0 "$server_pid" 2>/dev/null; then
    echo "Policy server failed. Log: $server_log" >&2
    tail -50 "$server_log" >&2 || true
    exit 1
  fi
  sleep 1
done
if ! grep -q "Creating server" "$server_log" 2>/dev/null; then
  echo "Timed out waiting for policy server. Log: $server_log" >&2
  exit 1
fi

for seed in "${SEEDS[@]}"; do
  read -r -a task_ids <<<"$(failure_ids_for_seed "$seed")"
  echo "Running $LABEL seed=$seed on ${#task_ids[@]} paired failure cases"

  CUDA_VISIBLE_DEVICES=$GPU \
  MUJOCO_GL=egl \
  MUJOCO_EGL_DEVICE_ID=$GPU \
  "$PYTHON" examples/libero/run_libero_plus.py \
    --args.host 127.0.0.1 \
    --args.port "$PORT" \
    --args.task-suite-name libero_object \
    --args.object-condition "$OBJECT_CONDITION" \
    --args.task-ids "${task_ids[@]}" \
    --args.max-tasks "${#task_ids[@]}" \
    --args.num-trials-per-task 1 \
    --args.seed "$seed" \
    --args.debug-object-overlay \
    --args.video-out-path "$OUTPUT_ROOT/${LABEL}_seed${seed}" \
    >"$LOG_DIR/client_seed${seed}.log" 2>&1
done

echo "Failure rerun complete. Metrics:"
wc -l "$OUTPUT_ROOT/${LABEL}_seed"*/metrics.jsonl
