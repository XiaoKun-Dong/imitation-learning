#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 3 || $# -gt 6 ]]; then
  cat >&2 <<'EOF'
Usage:
  eval_demovla_norm_matrix_4gpu.sh PHASE DYNAMIC_POLICY_DIR PI05_POLICY_DIR [TRIALS] [SEED] [OUTPUT_ROOT]

PHASE:
  equivalence  Compare pi0.5 with DemoVLA injection-off under official/local stats.
  causal       Compare one DemoVLA checkpoint with injection off/on under official/local stats.

Defaults:
  TRIALS=3, SEED=7
  OUTPUT_ROOT=data/libero/videos/demovla_norm_matrix_<phase>

Environment:
  GPU_LIST=0,1,2,3
  BASE_PORT=8100
  FLOW_NOISE_SEED=0
  DYNAMIC_CONFIG=demovla_libero_sparse_deep_dynamic_gate
  PI05_CONFIG=pi05_libero
EOF
  exit 2
fi

PHASE=$1
DYNAMIC_POLICY_DIR=$(realpath "$2")
PI05_POLICY_DIR=$(realpath "$3")
TRIALS=${4:-3}
SEED=${5:-7}

if [[ "$PHASE" != "equivalence" && "$PHASE" != "causal" ]]; then
  echo "PHASE must be equivalence or causal, got $PHASE" >&2
  exit 2
fi
if ((TRIALS < 1 || TRIALS > 50)); then
  echo "TRIALS must be between 1 and 50, got $TRIALS" >&2
  exit 2
fi

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON="$ROOT/.venv/bin/python"
OUTPUT_ROOT=${6:-"$ROOT/data/libero/videos/demovla_norm_matrix_${PHASE}"}
GPU_LIST=${GPU_LIST:-0,1,2,3}
BASE_PORT=${BASE_PORT:-8100}
FLOW_NOISE_SEED=${FLOW_NOISE_SEED:-0}
DYNAMIC_CONFIG=${DYNAMIC_CONFIG:-demovla_libero_sparse_deep_dynamic_gate}
PI05_CONFIG=${PI05_CONFIG:-pi05_libero}
OFFICIAL_STATS_DIR="$PI05_POLICY_DIR/assets/physical-intelligence/libero"
LOCAL_STATS_DIR="$DYNAMIC_POLICY_DIR/assets/local/libero"

IFS=',' read -r -a GPUS <<<"$GPU_LIST"
if [[ ${#GPUS[@]} -ne 4 ]]; then
  echo "Exactly four GPUs are required; GPU_LIST=$GPU_LIST" >&2
  exit 2
fi

for required in \
  "$PYTHON" \
  "$DYNAMIC_POLICY_DIR/params" \
  "$PI05_POLICY_DIR/params" \
  "$OFFICIAL_STATS_DIR/norm_stats.json" \
  "$LOCAL_STATS_DIR/norm_stats.json"; do
  if [[ ! -e "$required" ]]; then
    echo "Required path missing: $required" >&2
    exit 1
  fi
done

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
  echo "phase=$PHASE"
  echo "dynamic_checkpoint=$DYNAMIC_POLICY_DIR"
  echo "pi05_checkpoint=$PI05_POLICY_DIR"
  echo "trials_per_task=$TRIALS"
  echo "environment_seed=$SEED"
  echo "flow_noise_seed=$FLOW_NOISE_SEED"
  echo "gpu_list=$GPU_LIST"
  echo "official_stats_dir=$OFFICIAL_STATS_DIR"
  echo "local_stats_dir=$LOCAL_STATS_DIR"
  sha256sum "$OFFICIAL_STATS_DIR/norm_stats.json" "$LOCAL_STATS_DIR/norm_stats.json"
} >"$OUTPUT_ROOT/manifest.txt"

job_pids=()

cleanup() {
  for pid in "${job_pids[@]}"; do
    if [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null; then
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
      echo "Policy server failed. Log: $log_file" >&2
      tail -50 "$log_file" >&2 || true
      return 1
    fi
    sleep 1
  done
  echo "Timed out waiting for policy server. Log: $log_file" >&2
  return 1
}

run_condition() {
  local label=$1
  local gpu=$2
  local port=$3
  local config=$4
  local checkpoint=$5
  local stats_dir=$6
  local ablation=$7
  local server_log="$OUTPUT_ROOT/logs/${label}_server.log"
  local client_log="$OUTPUT_ROOT/logs/${label}_client.log"
  local video_dir="$OUTPUT_ROOT/$label"
  local -a ablation_args=()

  if [[ "$ablation" == "off" ]]; then
    ablation_args=(--interaction-ablation off)
  fi

  CUDA_VISIBLE_DEVICES=$gpu \
  XLA_PYTHON_CLIENT_PREALLOCATE=false \
  "$PYTHON" -u scripts/serve_policy.py \
    --port "$port" \
    --norm-stats-dir "$stats_dir" \
    "${ablation_args[@]}" \
    policy:checkpoint \
    --policy.config "$config" \
    --policy.dir "$checkpoint" \
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
    --args.task-suite-name libero_object \
    --args.resize-size 224 \
    --args.replan-steps 5 \
    --args.num-steps-wait 10 \
    --args.num-trials-per-task "$TRIALS" \
    --args.seed "$SEED" \
    --args.policy-noise-seed "$FLOW_NOISE_SEED" \
    --args.video-out-path "$video_dir" \
    >"$client_log" 2>&1

  kill "$server_pid" 2>/dev/null || true
  wait "$server_pid" 2>/dev/null || true
  trap - EXIT INT TERM
}

launch() {
  run_condition "$@" &
  job_pids+=("$!")
}

cd "$ROOT"
if [[ "$PHASE" == "equivalence" ]]; then
  launch pi05_official_stats "${GPUS[0]}" "$((BASE_PORT + 0))" "$PI05_CONFIG" "$PI05_POLICY_DIR" "$OFFICIAL_STATS_DIR" normal
  launch demovla_off_official_stats "${GPUS[1]}" "$((BASE_PORT + 1))" "$DYNAMIC_CONFIG" "$DYNAMIC_POLICY_DIR" "$OFFICIAL_STATS_DIR" off
  launch pi05_local_stats "${GPUS[2]}" "$((BASE_PORT + 2))" "$PI05_CONFIG" "$PI05_POLICY_DIR" "$LOCAL_STATS_DIR" normal
  launch demovla_off_local_stats "${GPUS[3]}" "$((BASE_PORT + 3))" "$DYNAMIC_CONFIG" "$DYNAMIC_POLICY_DIR" "$LOCAL_STATS_DIR" off
else
  launch demovla_off_official_stats "${GPUS[0]}" "$((BASE_PORT + 0))" "$DYNAMIC_CONFIG" "$DYNAMIC_POLICY_DIR" "$OFFICIAL_STATS_DIR" off
  launch dynamic_official_stats "${GPUS[1]}" "$((BASE_PORT + 1))" "$DYNAMIC_CONFIG" "$DYNAMIC_POLICY_DIR" "$OFFICIAL_STATS_DIR" normal
  launch demovla_off_local_stats "${GPUS[2]}" "$((BASE_PORT + 2))" "$DYNAMIC_CONFIG" "$DYNAMIC_POLICY_DIR" "$LOCAL_STATS_DIR" off
  launch dynamic_local_stats "${GPUS[3]}" "$((BASE_PORT + 3))" "$DYNAMIC_CONFIG" "$DYNAMIC_POLICY_DIR" "$LOCAL_STATS_DIR" normal
fi

status=0
for pid in "${job_pids[@]}"; do
  if ! wait "$pid"; then
    status=1
  fi
done
job_pids=()
if ((status != 0)); then
  echo "At least one evaluation condition failed; inspect $OUTPUT_ROOT/logs" >&2
  exit "$status"
fi

"$PYTHON" examples/libero/summarize_norm_matrix.py "$OUTPUT_ROOT" --phase "$PHASE" | tee "$OUTPUT_ROOT/summary.stdout.txt"
