#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 || $# -gt 6 ]]; then
  cat >&2 <<'EOF'
Usage:
  eval_demovla_gate_ablation.sh DYNAMIC_POLICY_DIR STATIC_POLICY_DIR [GPU] [PORT] [TRIALS] [SEED]

Runs four paired LIBERO-object evaluations with identical initial states and
stateless flow noise: dynamic, per-layer mean gate, injection off, and a
separately trained static-deep checkpoint.

Optional environment overrides:
  DYNAMIC_CONFIG       default: demovla_libero_sparse_deep_dynamic_gate
  STATIC_CONFIG        default: demovla_libero_sparse_deep_diverse
  LAYER_MEAN_GATES     default: "0.0211 0.0242 0.0277"
  FLOW_NOISE_SEED      default: 0
  ABLATION_OUTPUT_ROOT default: data/libero/videos/demovla_gate_ablation
EOF
  exit 2
fi

DYNAMIC_POLICY_DIR=$1
STATIC_POLICY_DIR=$2
GPU=${3:-0}
PORT=${4:-8000}
TRIALS=${5:-50}
SEED=${6:-7}

if ((TRIALS < 1 || TRIALS > 50)); then
  echo "TRIALS must be between 1 and 50, got $TRIALS" >&2
  exit 2
fi

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON="$ROOT/.venv/bin/python"
DYNAMIC_CONFIG=${DYNAMIC_CONFIG:-demovla_libero_sparse_deep_dynamic_gate}
STATIC_CONFIG=${STATIC_CONFIG:-demovla_libero_sparse_deep_diverse}
FLOW_NOISE_SEED=${FLOW_NOISE_SEED:-0}
OUTPUT_ROOT=${ABLATION_OUTPUT_ROOT:-$ROOT/data/libero/videos/demovla_gate_ablation}
read -r -a LAYER_GATES <<<"${LAYER_MEAN_GATES:-0.0211 0.0242 0.0277}"

if [[ ! -x "$PYTHON" ]]; then
  echo "Project Python not found: $PYTHON" >&2
  exit 1
fi

# Support both the normal LIBERO submodule and the older nested checkout
# without requiring an editable package installation.
if [[ -f "$ROOT/third_party/libero/libero/libero/__init__.py" ]]; then
  LIBERO_SOURCE_ROOT="$ROOT/third_party/libero"
elif [[ -f "$ROOT/third_party/libero/LIBERO/libero/libero/__init__.py" ]]; then
  LIBERO_SOURCE_ROOT="$ROOT/third_party/libero/LIBERO"
else
  echo "LIBERO checkout missing under $ROOT/third_party/libero" >&2
  exit 1
fi
export PYTHONPATH="$LIBERO_SOURCE_ROOT${PYTHONPATH:+:$PYTHONPATH}"

if [[ ${#LAYER_GATES[@]} -ne 3 ]]; then
  echo "LAYER_MEAN_GATES must contain three probabilities for layers 4, 9, and 14" >&2
  exit 2
fi

mkdir -p "$OUTPUT_ROOT/logs"
server_pid=""

cleanup() {
  if [[ -n "$server_pid" ]] && kill -0 "$server_pid" 2>/dev/null; then
    kill "$server_pid" 2>/dev/null || true
    wait "$server_pid" 2>/dev/null || true
  fi
  server_pid=""
}
trap cleanup EXIT INT TERM

wait_for_server() {
  local log_file=$1
  for ((elapsed = 0; elapsed < 300; elapsed++)); do
    if grep -q "Creating server" "$log_file" 2>/dev/null; then
      return 0
    fi
    if ! kill -0 "$server_pid" 2>/dev/null; then
      echo "Policy server failed. Log: $log_file" >&2
      tail -50 "$log_file" >&2 || true
      return 1
    fi
    sleep 1
  done
  echo "Timed out waiting for policy server. Log: $log_file" >&2
  return 1
}

run_one() {
  local label=$1
  local policy_config=$2
  local policy_dir=$3
  local ablation=$4
  local server_log="$OUTPUT_ROOT/logs/${label}_server.log"
  local client_log="$OUTPUT_ROOT/logs/${label}_client.log"
  local video_dir="$OUTPUT_ROOT/$label"
  local -a ablation_args=()

  if [[ "$ablation" == "layer_mean" ]]; then
    ablation_args=(
      --interaction-ablation layer_mean
      --interaction-layer-mean-gates "${LAYER_GATES[@]}"
    )
  elif [[ "$ablation" == "off" ]]; then
    ablation_args=(--interaction-ablation off)
  fi

  echo "Starting $label: config=$policy_config checkpoint=$policy_dir"
  CUDA_VISIBLE_DEVICES=$GPU \
  XLA_PYTHON_CLIENT_PREALLOCATE=false \
  "$PYTHON" -u scripts/serve_policy.py \
    --port "$PORT" \
    "${ablation_args[@]}" \
    policy:checkpoint \
    --policy.config "$policy_config" \
    --policy.dir "$policy_dir" \
    >"$server_log" 2>&1 &
  server_pid=$!
  wait_for_server "$server_log"

  CUDA_VISIBLE_DEVICES=$GPU \
  MUJOCO_GL=egl \
  MUJOCO_EGL_DEVICE_ID=0 \
  "$PYTHON" examples/libero/main.py \
    --args.host 127.0.0.1 \
    --args.port "$PORT" \
    --args.task-suite-name libero_object \
    --args.num-trials-per-task "$TRIALS" \
    --args.seed "$SEED" \
    --args.policy-noise-seed "$FLOW_NOISE_SEED" \
    --args.video-out-path "$video_dir" \
    >"$client_log" 2>&1

  cleanup
}

cd "$ROOT"
run_one dynamic "$DYNAMIC_CONFIG" "$DYNAMIC_POLICY_DIR" normal
run_one layer_mean "$DYNAMIC_CONFIG" "$DYNAMIC_POLICY_DIR" layer_mean
run_one injection_off "$DYNAMIC_CONFIG" "$DYNAMIC_POLICY_DIR" off
run_one static_deep "$STATIC_CONFIG" "$STATIC_POLICY_DIR" normal

echo "Paired gate ablation complete: $OUTPUT_ROOT"
"$PYTHON" examples/libero/summarize_gate_ablation.py "$OUTPUT_ROOT"
