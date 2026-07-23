#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 4 || $# -gt 6 ]]; then
  cat >&2 <<'EOF'
Usage:
  eval_libero_plus_single.sh LABEL OBJECT_CONDITION POLICY_CONFIG POLICY_DIR [GPU] [PORT]

Runs the 50-task LIBERO-plus Objects Layout evaluation sequentially for seeds
7, 42, and 123 on one GPU.
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
LOG_DIR="$ROOT/data/libero_plus/logs/$LABEL"
VIDEO_ROOT="$ROOT/data/libero_plus/videos/final"
read -r -a SEEDS <<<"${LIBERO_PLUS_SEEDS:-7 42 123}"
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
server_pid=""

cleanup() {
  if [[ -n "$server_pid" ]] && kill -0 "$server_pid" 2>/dev/null; then
    kill "$server_pid" 2>/dev/null || true
  fi
}
trap cleanup EXIT INT TERM

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
  echo "Running $LABEL seed=$seed on ${#TASK_IDS[@]} tasks"
  CUDA_VISIBLE_DEVICES=$GPU \
  MUJOCO_GL=egl \
  MUJOCO_EGL_DEVICE_ID=$GPU \
  "$PYTHON" examples/libero/run_libero_plus.py \
    --args.host 127.0.0.1 \
    --args.port "$PORT" \
    --args.task-suite-name libero_object \
    --args.object-condition "$OBJECT_CONDITION" \
    --args.task-ids "${TASK_IDS[@]}" \
    --args.max-tasks "${#TASK_IDS[@]}" \
    --args.num-trials-per-task 1 \
    --args.seed "$seed" \
    --args.video-out-path "$VIDEO_ROOT/${LABEL}_seed${seed}" \
    >"$LOG_DIR/client_seed${seed}.log" 2>&1
done

echo "Evaluation complete. Metrics:"
wc -l "$VIDEO_ROOT/${LABEL}_seed"*/metrics.jsonl
