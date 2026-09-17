#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 5 ]]; then
  cat >&2 <<'EOF'
Usage:
  eval_demovla_zero_fixed_gate_4gpu.sh POLICY_DIR GATE TRIALS SEED OUTPUT_ROOT

Evaluates zero memory at an explicitly fixed gate on all four LIBERO suites.
The suites run concurrently on GPUs 0-3. Set REFERENCE_ROOT to write a paired
comparison against an existing correct-memory evaluation.
EOF
  exit 2
fi

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON="$ROOT/.venv/bin/python"
POLICY_DIR=$(realpath "$1")
GATE=$2
TRIALS=$3
SEED=$4
OUTPUT_ROOT=$5
POLICY_CONFIG=${POLICY_CONFIG:-demovla_libero_full_stage_a_fixed_gate}
FLOW_NOISE_SEED=${FLOW_NOISE_SEED:-0}
BASE_PORT=${BASE_PORT:-8700}
REFERENCE_ROOT=${REFERENCE_ROOT:-}
SUITES=(libero_spatial libero_object libero_goal libero_10)

if [[ ! -x "$PYTHON" ]]; then
  echo "Project Python not found: $PYTHON" >&2
  exit 1
fi
if [[ ! -d "$POLICY_DIR/params" ]]; then
  echo "Checkpoint params missing: $POLICY_DIR/params" >&2
  exit 1
fi
if [[ -e "$OUTPUT_ROOT" ]]; then
  echo "Output root already exists; refusing to overwrite: $OUTPUT_ROOT" >&2
  exit 1
fi

if [[ -f "$ROOT/third_party/libero/libero/libero/__init__.py" ]]; then
  LIBERO_SOURCE_ROOT="$ROOT/third_party/libero"
elif [[ -f "$ROOT/third_party/libero/LIBERO/libero/libero/__init__.py" ]]; then
  LIBERO_SOURCE_ROOT="$ROOT/third_party/libero/LIBERO"
else
  echo "LIBERO checkout missing under $ROOT/third_party/libero" >&2
  exit 1
fi
export PYTHONPATH="$LIBERO_SOURCE_ROOT${PYTHONPATH:+:$PYTHONPATH}"

mkdir -p "$OUTPUT_ROOT/logs"
{
  echo "policy_config=$POLICY_CONFIG"
  echo "policy_dir=$POLICY_DIR"
  echo "condition=zero_memory_fixed_gate"
  echo "gate=$GATE"
  echo "trials_per_task=$TRIALS"
  echo "environment_seed=$SEED"
  echo "flow_noise_seed=$FLOW_NOISE_SEED"
  echo "reference_root=$REFERENCE_ROOT"
} >"$OUTPUT_ROOT/manifest.txt"

runner_pids=()
cleanup() {
  for pid in "${runner_pids[@]}"; do
    if kill -0 "$pid" 2>/dev/null; then
      kill "$pid" 2>/dev/null || true
    fi
  done
}
trap cleanup EXIT INT TERM

wait_for_server() {
  local pid=$1
  local log_file=$2
  for ((elapsed = 0; elapsed < 300; elapsed++)); do
    if grep -q "Creating server" "$log_file" 2>/dev/null; then
      return 0
    fi
    if ! kill -0 "$pid" 2>/dev/null; then
      tail -50 "$log_file" >&2 || true
      return 1
    fi
    sleep 1
  done
  echo "Timed out waiting for policy server: $log_file" >&2
  return 1
}

run_suite() (
  set -euo pipefail
  local suite=$1
  local gpu=$2
  local port=$3
  local server_log="$OUTPUT_ROOT/logs/${suite}_server.log"
  local server_pid=

  stop_server() {
    if [[ -n "$server_pid" ]]; then
      kill "$server_pid" 2>/dev/null || true
      wait "$server_pid" 2>/dev/null || true
    fi
  }
  trap stop_server EXIT INT TERM

  CUDA_VISIBLE_DEVICES=$gpu \
  XLA_PYTHON_CLIENT_PREALLOCATE=false \
  "$PYTHON" -u scripts/serve_policy.py \
    --port "$port" \
    --interaction-ablation zero_memory \
    --interaction-layer-mean-gates "$GATE" "$GATE" "$GATE" \
    policy:checkpoint \
    --policy.config "$POLICY_CONFIG" \
    --policy.dir "$POLICY_DIR" \
    >"$server_log" 2>&1 &
  server_pid=$!
  wait_for_server "$server_pid" "$server_log"

  CUDA_VISIBLE_DEVICES=$gpu \
  MUJOCO_GL=egl \
  MUJOCO_EGL_DEVICE_ID=$gpu \
  "$PYTHON" examples/libero/main.py \
    --args.host 127.0.0.1 \
    --args.port "$port" \
    --args.task-suite-name "$suite" \
    --args.resize-size 224 \
    --args.replan-steps 5 \
    --args.num-steps-wait 10 \
    --args.num-trials-per-task "$TRIALS" \
    --args.seed "$SEED" \
    --args.policy-noise-seed "$FLOW_NOISE_SEED" \
    --args.video-out-path "$OUTPUT_ROOT/$suite" \
    >"$OUTPUT_ROOT/logs/${suite}_client.log" 2>&1
)

cd "$ROOT"
for index in "${!SUITES[@]}"; do
  run_suite "${SUITES[$index]}" "$index" "$((BASE_PORT + index))" &
  runner_pids+=("$!")
done

status=0
for pid in "${runner_pids[@]}"; do
  if ! wait "$pid"; then
    status=1
  fi
done
runner_pids=()
if ((status != 0)); then
  echo "Zero-memory fixed-gate evaluation failed; inspect $OUTPUT_ROOT/logs" >&2
  exit "$status"
fi

"$PYTHON" examples/libero/summarize_full_suite_eval.py "$OUTPUT_ROOT" \
  >"$OUTPUT_ROOT/summary.stdout.txt"
if [[ -n "$REFERENCE_ROOT" ]]; then
  "$PYTHON" examples/libero/summarize_full_suite_eval.py \
    "$REFERENCE_ROOT" --reference-root "$OUTPUT_ROOT" \
    >"$OUTPUT_ROOT/correct_vs_zero_paired.txt"
fi

echo "Zero-memory fixed-gate evaluation complete: $OUTPUT_ROOT"
