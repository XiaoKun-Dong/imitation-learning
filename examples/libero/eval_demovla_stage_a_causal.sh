#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 || $# -gt 4 ]]; then
  cat >&2 <<'EOF'
Usage:
  eval_demovla_stage_a_causal.sh POLICY_DIR [TRIALS] [SEED] [OUTPUT_ROOT]

Runs the paired full-LIBERO Stage-A behavior scan. The first wave evaluates
correct memory with its trained fixed-0.05 gate, injection off, zero memory,
and fixed-0.01 on GPUs 0-3. Fixed-0.03 runs afterward. Shuffled memory is
evaluated by the paired batch-loss script because online rollout has batch 1.
EOF
  exit 2
fi

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON="$ROOT/.venv/bin/python"
POLICY_DIR=$(realpath "$1")
TRIALS=${2:-1}
SEED=${3:-7}
OUTPUT_ROOT=${4:-"$ROOT/data/libero/videos/demovla_stage_a_causal_$(date +%Y%m%d_%H%M%S)"}
POLICY_CONFIG=${POLICY_CONFIG:-demovla_libero_full_stage_a_fixed_gate}
FLOW_NOISE_SEED=${FLOW_NOISE_SEED:-0}
BASE_PORT=${BASE_PORT:-8600}
FOURTH_LABEL=${FOURTH_LABEL:-fixed_001}
FOURTH_GATE=${FOURTH_GATE:-0.01}
RUN_SECOND_GATE=${RUN_SECOND_GATE:-1}
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
  echo "first_wave_conditions=correct_fixed005 off zero_memory $FOURTH_LABEL"
  echo "fourth_gate=$FOURTH_GATE"
  echo "run_second_gate=$RUN_SECOND_GATE"
  echo "shuffled_memory=offline_paired_batch_loss_only"
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

  case "$ablation" in
    off|zero_memory)
      ablation_args=(--interaction-ablation "$ablation")
      ;;
    layer_mean)
      ablation_args=(
        --interaction-ablation layer_mean
        --interaction-layer-mean-gates "$gate" "$gate" "$gate"
      )
      ;;
  esac

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
run_condition correct_fixed005 normal "" 0 "$((BASE_PORT + 0))" & runner_pids+=("$!")
run_condition off off "" 1 "$((BASE_PORT + 1))" & runner_pids+=("$!")
run_condition zero_memory zero_memory "" 2 "$((BASE_PORT + 2))" & runner_pids+=("$!")
run_condition "$FOURTH_LABEL" layer_mean "$FOURTH_GATE" 3 "$((BASE_PORT + 3))" & runner_pids+=("$!")

status=0
for pid in "${runner_pids[@]}"; do
  if ! wait "$pid"; then
    status=1
  fi
done
runner_pids=()
if ((status != 0)); then
  echo "First Stage-A behavior wave failed; inspect $OUTPUT_ROOT/logs" >&2
  exit "$status"
fi

if [[ "$RUN_SECOND_GATE" == "1" ]]; then
  run_condition fixed_003 layer_mean 0.03 0 "$((BASE_PORT + 4))"
elif [[ "$RUN_SECOND_GATE" != "0" ]]; then
  echo "RUN_SECOND_GATE must be 0 or 1, got: $RUN_SECOND_GATE" >&2
  exit 2
fi

"$PYTHON" examples/libero/summarize_full_suite_eval.py \
  "$OUTPUT_ROOT/correct_fixed005" --reference-root "$OUTPUT_ROOT/off" \
  >"$OUTPUT_ROOT/correct_vs_off.txt"
"$PYTHON" examples/libero/summarize_full_suite_eval.py \
  "$OUTPUT_ROOT/correct_fixed005" --reference-root "$OUTPUT_ROOT/zero_memory" \
  >"$OUTPUT_ROOT/correct_vs_zero.txt"
gate_labels=("$FOURTH_LABEL")
if [[ "$RUN_SECOND_GATE" == "1" && "$FOURTH_LABEL" != "fixed_003" ]]; then
  gate_labels+=(fixed_003)
fi
for label in "${gate_labels[@]}"; do
  "$PYTHON" examples/libero/summarize_full_suite_eval.py \
    "$OUTPUT_ROOT/$label" --reference-root "$OUTPUT_ROOT/off" \
    >"$OUTPUT_ROOT/${label}_vs_off.txt"
done

echo "Stage-A full-LIBERO causal behavior scan complete: $OUTPUT_ROOT"
