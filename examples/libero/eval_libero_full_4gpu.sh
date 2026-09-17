#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 3 || $# -gt 6 ]]; then
  cat >&2 <<'EOF'
Usage:
  eval_libero_full_4gpu.sh LABEL POLICY_CONFIG POLICY_DIR [TRIALS] [SEED] [OUTPUT_ROOT]

Runs Spatial, Object, Goal, and LIBERO-10 concurrently on four GPUs with
paired stateless flow noise. Defaults: TRIALS=10, SEED=7.
EOF
  exit 2
fi

LABEL=$1
POLICY_CONFIG=$2
POLICY_DIR=$(realpath "$3")
TRIALS=${4:-10}
SEED=${5:-7}

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON="$ROOT/.venv/bin/python"
OUTPUT_ROOT=${6:-"$ROOT/data/libero/videos/full_matched/$LABEL"}
GPU_LIST=${GPU_LIST:-0,1,2,3}
BASE_PORT=${BASE_PORT:-8200}
FLOW_NOISE_SEED=${FLOW_NOISE_SEED:-0}
INTERACTION_ABLATION=${INTERACTION_ABLATION:-normal}
INTERACTION_GATE=${INTERACTION_GATE:-}
SUITES=(libero_spatial libero_object libero_goal libero_10)

IFS=',' read -r -a GPUS <<<"$GPU_LIST"
if [[ ${#GPUS[@]} -ne 4 ]]; then
  echo "Exactly four GPUs are required; GPU_LIST=$GPU_LIST" >&2
  exit 2
fi
if [[ ! -d "$POLICY_DIR/params" ]]; then
  echo "Checkpoint params missing: $POLICY_DIR/params" >&2
  exit 1
fi
mapfile -t stats_files < <(find "$POLICY_DIR/assets" -type f -name norm_stats.json -print)
if [[ ${#stats_files[@]} -ne 1 ]]; then
  echo "Expected exactly one checkpoint norm_stats.json, found ${#stats_files[@]}" >&2
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
  echo "label=$LABEL"
  echo "policy_config=$POLICY_CONFIG"
  echo "policy_dir=$POLICY_DIR"
  echo "trials_per_task=$TRIALS"
  echo "environment_seed=$SEED"
  echo "flow_noise_seed=$FLOW_NOISE_SEED"
  echo "gpu_list=$GPU_LIST"
  echo "interaction_ablation=$INTERACTION_ABLATION"
  echo "interaction_gate=$INTERACTION_GATE"
  sha256sum "${stats_files[0]}"
} >"$OUTPUT_ROOT/manifest.txt"

ablation_args=()
case "$INTERACTION_ABLATION" in
  normal)
    if [[ -n "$INTERACTION_GATE" ]]; then
      echo "INTERACTION_GATE must be empty for normal mode" >&2
      exit 2
    fi
    ;;
  off)
    ablation_args=(--interaction-ablation off)
    ;;
  layer_mean|zero_memory)
    if [[ -z "$INTERACTION_GATE" ]]; then
      echo "INTERACTION_GATE is required for $INTERACTION_ABLATION" >&2
      exit 2
    fi
    ablation_args=(
      --interaction-ablation "$INTERACTION_ABLATION"
      --interaction-layer-mean-gates "$INTERACTION_GATE" "$INTERACTION_GATE" "$INTERACTION_GATE"
    )
    ;;
  *)
    echo "Unsupported INTERACTION_ABLATION=$INTERACTION_ABLATION" >&2
    exit 2
    ;;
esac

job_pids=()
cleanup() {
  for pid in "${job_pids[@]}"; do
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

run_suite() {
  local suite=$1
  local gpu=$2
  local port=$3
  local server_log="$OUTPUT_ROOT/logs/${suite}_server.log"
  local client_log="$OUTPUT_ROOT/logs/${suite}_client.log"

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
    >"$client_log" 2>&1

  kill "$server_pid" 2>/dev/null || true
  wait "$server_pid" 2>/dev/null || true
  trap - EXIT INT TERM
}

cd "$ROOT"
for index in "${!SUITES[@]}"; do
  run_suite "${SUITES[$index]}" "${GPUS[$index]}" "$((BASE_PORT + index))" &
  job_pids+=("$!")
done

status=0
for pid in "${job_pids[@]}"; do
  if ! wait "$pid"; then
    status=1
  fi
done
job_pids=()
if ((status != 0)); then
  echo "At least one suite failed; inspect $OUTPUT_ROOT/logs" >&2
  exit "$status"
fi

"$PYTHON" examples/libero/summarize_full_suite_eval.py "$OUTPUT_ROOT" | tee "$OUTPUT_ROOT/summary.stdout.txt"
