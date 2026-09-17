#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 3 ]]; then
  cat >&2 <<'EOF'
Usage:
  eval_libero_plus_parallel.sh LABEL POLICY_CONFIG POLICY_DIR

Example:
  eval_libero_plus_parallel.sh demovla_eval \
    demovla_libero_sparse_deep_dynamic_gate \
    checkpoints/demovla_libero_sparse_deep_dynamic_gate/dynamic_gate_v1/29999
EOF
  exit 2
fi

LABEL=$1
POLICY_CONFIG=$2
POLICY_DIR=$3

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON="$ROOT/.venv/bin/python"
LOG_DIR="$ROOT/data/libero_plus/logs/$LABEL"
VIDEO_ROOT="$ROOT/data/libero_plus/videos/final"

GPUS=(0 1 2)
PORTS=(8000 8001 8002)
SEEDS=(7 42 123)
TASK_IDS=(
  1840 1844 1848 1852 1856
  1884 1888 1892 1896 1900
  1928 1932 1936 1940 1944
  1966 1970 1974 1978 1982
  1993 1997 2001 2005 2009
  2018 2022 2026 2030 2034
  2063 2067 2071 2075 2079
  2112 2116 2120 2124 2128
  2157 2161 2164 2167 2170
  2204 2208 2212 2214 2218
)

if [[ ! -x "$PYTHON" ]]; then
  echo "Project Python not found: $PYTHON" >&2
  exit 1
fi

mkdir -p "$LOG_DIR" "$VIDEO_ROOT"
server_pids=()
client_pids=()

cleanup() {
  local pid
  for pid in "${client_pids[@]:-}" "${server_pids[@]:-}"; do
    if [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null; then
      kill "$pid" 2>/dev/null || true
    fi
  done
}
trap cleanup EXIT INT TERM

wait_for_server() {
  local pid=$1
  local log_file=$2
  local timeout_seconds=300

  for ((elapsed = 0; elapsed < timeout_seconds; elapsed++)); do
    if grep -q "Creating server" "$log_file" 2>/dev/null; then
      return 0
    fi
    if ! kill -0 "$pid" 2>/dev/null; then
      echo "Policy server exited before becoming ready. Log: $log_file" >&2
      tail -50 "$log_file" >&2 || true
      return 1
    fi
    sleep 1
  done

  echo "Timed out waiting for policy server. Log: $log_file" >&2
  tail -50 "$log_file" >&2 || true
  return 1
}

cd "$ROOT"

echo "Starting three policy servers for $LABEL..."
for index in "${!GPUS[@]}"; do
  gpu=${GPUS[$index]}
  port=${PORTS[$index]}
  seed=${SEEDS[$index]}
  server_log="$LOG_DIR/server_seed${seed}.log"

  CUDA_VISIBLE_DEVICES=$gpu \
  XLA_PYTHON_CLIENT_PREALLOCATE=false \
  "$PYTHON" -u scripts/serve_policy.py \
    --port "$port" \
    policy:checkpoint \
    --policy.config "$POLICY_CONFIG" \
    --policy.dir "$POLICY_DIR" \
    >"$server_log" 2>&1 &
  server_pids+=("$!")
done

for index in "${!server_pids[@]}"; do
  wait_for_server "${server_pids[$index]}" "$LOG_DIR/server_seed${SEEDS[$index]}.log"
done

echo "All policy servers are ready. Starting evaluations..."
for index in "${!GPUS[@]}"; do
  gpu=${GPUS[$index]}
  port=${PORTS[$index]}
  seed=${SEEDS[$index]}
  client_log="$LOG_DIR/client_seed${seed}.log"

  CUDA_VISIBLE_DEVICES=$gpu \
  MUJOCO_GL=egl \
  MUJOCO_EGL_DEVICE_ID=$gpu \
  "$PYTHON" examples/libero/run_libero_plus.py \
    --args.host 127.0.0.1 \
    --args.port "$port" \
    --args.task-suite-name libero_object \
    --args.task-ids "${TASK_IDS[@]}" \
    --args.max-tasks 50 \
    --args.num-trials-per-task 1 \
    --args.seed "$seed" \
    --args.video-out-path "$VIDEO_ROOT/${LABEL}_seed${seed}" \
    >"$client_log" 2>&1 &
  client_pids+=("$!")
done

status=0
for index in "${!client_pids[@]}"; do
  if wait "${client_pids[$index]}"; then
    echo "seed ${SEEDS[$index]} completed"
  else
    echo "seed ${SEEDS[$index]} failed; see $LOG_DIR/client_seed${SEEDS[$index]}.log" >&2
    status=1
  fi
done

if [[ $status -eq 0 ]]; then
  echo "Evaluation complete. Metrics:"
  wc -l "$VIDEO_ROOT/${LABEL}_seed"*/metrics.jsonl
fi

exit "$status"
