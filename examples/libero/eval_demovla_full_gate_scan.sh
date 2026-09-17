#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 || $# -gt 4 ]]; then
  cat >&2 <<'EOF'
Usage:
  eval_demovla_full_gate_scan.sh POLICY_DIR [TRIALS] [SEED] [OUTPUT_ROOT]

Runs a paired full-LIBERO pilot on the same DemoVLA checkpoint with dynamic,
injection-off, and fixed gates 0.01, 0.03, and 0.05. Four conditions run in
parallel on GPUs 0-3; fixed-0.05 runs after the first wave. Defaults: one
trial per task, seed 7, stateless flow-noise seed 0.
EOF
  exit 2
fi

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON="$ROOT/.venv/bin/python"
POLICY_DIR=$(realpath "$1")
TRIALS=${2:-1}
SEED=${3:-7}
OUTPUT_ROOT=${4:-"$ROOT/data/libero/videos/full_matched_gate_scan_$(date +%Y%m%d_%H%M%S)"}
POLICY_CONFIG=${POLICY_CONFIG:-demovla_libero_full_matched}
FLOW_NOISE_SEED=${FLOW_NOISE_SEED:-0}
BASE_PORT=${BASE_PORT:-8500}
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
  echo "trials_per_task=$TRIALS"
  echo "environment_seed=$SEED"
  echo "flow_noise_seed=$FLOW_NOISE_SEED"
  echo "conditions=dynamic off fixed_001 fixed_003 fixed_005"
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

run_condition() {
  local label=$1
  local ablation=$2
  local gate=$3
  local gpu=$4
  local port=$5
  local condition_root="$OUTPUT_ROOT/$label"
  local server_log="$OUTPUT_ROOT/logs/${label}_server.log"
  local -a ablation_args=()

  if [[ "$ablation" == "off" ]]; then
    ablation_args=(--interaction-ablation off)
  elif [[ "$ablation" == "layer_mean" ]]; then
    ablation_args=(
      --interaction-ablation layer_mean
      --interaction-layer-mean-gates "$gate" "$gate" "$gate"
    )
  fi

  mkdir -p "$condition_root"
  CUDA_VISIBLE_DEVICES=$gpu \
  XLA_PYTHON_CLIENT_PREALLOCATE=false \
  "$PYTHON" -u scripts/serve_policy.py \
    --port "$port" \
    "${ablation_args[@]}" \
    policy:checkpoint \
    --policy.config "$POLICY_CONFIG" \
    --policy.dir "$POLICY_DIR" \
    >"$server_log" 2>&1 &
  local server_pid=$!
  trap 'kill "$server_pid" 2>/dev/null || true; wait "$server_pid" 2>/dev/null || true' EXIT INT TERM
  wait_for_server "$server_pid" "$server_log"

  for suite in "${SUITES[@]}"; do
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
      --args.video-out-path "$condition_root/$suite" \
      >"$OUTPUT_ROOT/logs/${label}_${suite}_client.log" 2>&1
  done

  kill "$server_pid" 2>/dev/null || true
  wait "$server_pid" 2>/dev/null || true
  trap - EXIT INT TERM
  "$PYTHON" examples/libero/summarize_full_suite_eval.py "$condition_root" \
    >"$condition_root/summary.stdout.txt"
}

cd "$ROOT"
run_condition dynamic normal "" 0 "$((BASE_PORT + 0))" & runner_pids+=("$!")
run_condition off off "" 1 "$((BASE_PORT + 1))" & runner_pids+=("$!")
run_condition fixed_001 layer_mean 0.01 2 "$((BASE_PORT + 2))" & runner_pids+=("$!")
run_condition fixed_003 layer_mean 0.03 3 "$((BASE_PORT + 3))" & runner_pids+=("$!")

status=0
for pid in "${runner_pids[@]}"; do
  if ! wait "$pid"; then
    status=1
  fi
done
runner_pids=()
if ((status != 0)); then
  echo "First gate-scan wave failed; inspect $OUTPUT_ROOT/logs" >&2
  exit "$status"
fi

run_condition fixed_005 layer_mean 0.05 0 "$((BASE_PORT + 4))"

"$PYTHON" examples/libero/summarize_full_suite_eval.py \
  "$OUTPUT_ROOT/dynamic" --reference-root "$OUTPUT_ROOT/off" \
  >"$OUTPUT_ROOT/dynamic_vs_off.txt"
for label in fixed_001 fixed_003 fixed_005; do
  "$PYTHON" examples/libero/summarize_full_suite_eval.py \
    "$OUTPUT_ROOT/$label" --reference-root "$OUTPUT_ROOT/off" \
    >"$OUTPUT_ROOT/${label}_vs_off.txt"
done

echo "Full-LIBERO gate scan complete: $OUTPUT_ROOT"
