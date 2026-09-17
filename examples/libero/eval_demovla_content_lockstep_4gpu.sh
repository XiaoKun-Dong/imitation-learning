#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 8 ]]; then
  cat >&2 <<'EOF'
Usage:
  eval_demovla_content_lockstep_4gpu.sh POLICY_DIR SUITE TASK_ID EPISODE_ID BRANCH_REPLAN SHUFFLED_MEMORY_FILE OUTPUT_ROOT BASE_PORT

Runs a one-replan correct/shuffled/zero/injection-off/wrong-prompt content
intervention from an exactly shared correct-memory prefix. Set GPU_OFFSET to
run a second independent launcher on GPUs 4-7.
EOF
  exit 2
fi

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON="$ROOT/.venv/bin/python"
POLICY_DIR=$(realpath "$1")
SUITE=$2
TASK_ID=$3
EPISODE_ID=$4
BRANCH_REPLAN=$5
SHUFFLED_MEMORY_FILE=$(realpath "$6")
OUTPUT_ROOT=$7
BASE_PORT=$8
POLICY_CONFIG=${POLICY_CONFIG:-demovla_libero_full_stage_a_fixed_gate}
ENV_SEED=${ENV_SEED:-7}
FLOW_NOISE_SEED=${FLOW_NOISE_SEED:-0}
GPU_OFFSET=${GPU_OFFSET:-0}

if ! [[ "$GPU_OFFSET" =~ ^[0-4]$ ]]; then
  echo "GPU_OFFSET must be an integer in [0, 4]" >&2
  exit 1
fi
if [[ -e "$OUTPUT_ROOT" ]]; then
  echo "Output root already exists; refusing to overwrite: $OUTPUT_ROOT" >&2
  exit 1
fi
if [[ ! -d "$POLICY_DIR/params" ]]; then
  echo "Checkpoint params missing: $POLICY_DIR/params" >&2
  exit 1
fi
if [[ ! -f "$SHUFFLED_MEMORY_FILE" ]]; then
  echo "Shuffled memory file missing: $SHUFFLED_MEMORY_FILE" >&2
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

server_pids=()
cleanup() {
  for pid in "${server_pids[@]}"; do
    kill "$pid" 2>/dev/null || true
    wait "$pid" 2>/dev/null || true
  done
}
trap cleanup EXIT INT TERM

start_server() {
  local label=$1 gpu=$2 port=$3 ablation=$4
  local -a gate_args=()
  if [[ "$ablation" != "off" ]]; then
    gate_args=(--interaction-layer-mean-gates 0.03 0.03 0.03)
  fi
  CUDA_VISIBLE_DEVICES=$gpu XLA_PYTHON_CLIENT_PREALLOCATE=false \
  "$PYTHON" -u scripts/serve_policy.py \
    --port "$port" \
    --interaction-ablation "$ablation" \
    "${gate_args[@]}" \
    policy:checkpoint --policy.config "$POLICY_CONFIG" --policy.dir "$POLICY_DIR" \
    >"$OUTPUT_ROOT/logs/${label}_server.log" 2>&1 &
  server_pids+=("$!")
}

wait_for_server() {
  local pid=$1 log_file=$2
  for ((elapsed = 0; elapsed < 300; elapsed++)); do
    if grep -q "Creating server" "$log_file" 2>/dev/null; then return 0; fi
    if ! kill -0 "$pid" 2>/dev/null; then tail -80 "$log_file" >&2 || true; return 1; fi
    sleep 1
  done
  echo "Timed out waiting for policy server: $log_file" >&2
  return 1
}

cd "$ROOT"
start_server correct "$((GPU_OFFSET + 0))" "$((BASE_PORT + 0))" layer_mean
start_server zero "$((GPU_OFFSET + 1))" "$((BASE_PORT + 1))" zero_memory
start_server injection_off "$((GPU_OFFSET + 2))" "$((BASE_PORT + 2))" off
labels=(correct zero injection_off)
for index in "${!server_pids[@]}"; do
  wait_for_server "${server_pids[$index]}" "$OUTPUT_ROOT/logs/${labels[$index]}_server.log"
done

CUDA_VISIBLE_DEVICES="$((GPU_OFFSET + 3))" MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID="$((GPU_OFFSET + 3))" \
"$PYTHON" examples/libero/run_demovla_content_lockstep.py \
  --task-suite-name "$SUITE" \
  --task-id "$TASK_ID" \
  --episode-id "$EPISODE_ID" \
  --branch-replan "$BRANCH_REPLAN" \
  --shuffled-memory-file "$SHUFFLED_MEMORY_FILE" \
  --output-dir "$OUTPUT_ROOT" \
  --correct-port "$((BASE_PORT + 0))" \
  --zero-port "$((BASE_PORT + 1))" \
  --injection-off-port "$((BASE_PORT + 2))" \
  --seed "$ENV_SEED" \
  --policy-noise-seed "$FLOW_NOISE_SEED" \
  >"$OUTPUT_ROOT/logs/content_lockstep_client.log" 2>&1

echo "DemoVLA content-lockstep evaluation complete: $OUTPUT_ROOT"
