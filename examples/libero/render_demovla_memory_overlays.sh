#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 7 || $# -gt 8 ]]; then
  cat >&2 <<'EOF'
Usage:
  render_demovla_memory_overlays.sh POLICY_DIR SUITE TASK_ID EPISODE_ID GPU PORT OUTPUT_DIR [GATE]

Replays one exact LIBERO task/initial-state pair and renders DemoVLA memory
extraction attention on every replan. TASK_ID is one-based; EPISODE_ID is
zero-based. GATE defaults to 0.03.
EOF
  exit 2
fi

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON="$ROOT/.venv/bin/python"
POLICY_DIR=$(realpath "$1")
SUITE=$2
TASK_ID=$3
EPISODE_ID=$4
GPU=$5
PORT=$6
OUTPUT_DIR=$7
GATE=${8:-0.03}
POLICY_CONFIG=${POLICY_CONFIG:-demovla_libero_full_stage_a_fixed_gate}
ENV_SEED=${ENV_SEED:-7}
FLOW_NOISE_SEED=${FLOW_NOISE_SEED:-0}

if [[ ! -x "$PYTHON" ]]; then
  echo "Project Python not found: $PYTHON" >&2
  exit 1
fi
if [[ ! -d "$POLICY_DIR/params" ]]; then
  echo "Checkpoint params missing: $POLICY_DIR/params" >&2
  exit 1
fi
if [[ -e "$OUTPUT_DIR" ]]; then
  echo "Output directory already exists; refusing to overwrite: $OUTPUT_DIR" >&2
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

mkdir -p "$OUTPUT_DIR/logs"
{
  echo "policy_config=$POLICY_CONFIG"
  echo "policy_dir=$POLICY_DIR"
  echo "suite=$SUITE"
  echo "task_id=$TASK_ID"
  echo "episode_id=$EPISODE_ID"
  echo "gate=$GATE"
  echo "environment_seed=$ENV_SEED"
  echo "flow_noise_seed=$FLOW_NOISE_SEED"
  echo "render=all_replans"
} >"$OUTPUT_DIR/manifest.txt"

server_pid=
cleanup() {
  if [[ -n "$server_pid" ]]; then
    kill "$server_pid" 2>/dev/null || true
    wait "$server_pid" 2>/dev/null || true
  fi
}
trap cleanup EXIT INT TERM

cd "$ROOT"
CUDA_VISIBLE_DEVICES=$GPU \
XLA_PYTHON_CLIENT_PREALLOCATE=false \
"$PYTHON" -u scripts/serve_policy.py \
  --port "$PORT" \
  --interaction-diagnostics \
  --interaction-ablation layer_mean \
  --interaction-layer-mean-gates "$GATE" "$GATE" "$GATE" \
  policy:checkpoint \
  --policy.config "$POLICY_CONFIG" \
  --policy.dir "$POLICY_DIR" \
  >"$OUTPUT_DIR/logs/server.log" 2>&1 &
server_pid=$!

for ((elapsed = 0; elapsed < 300; elapsed++)); do
  if grep -q "Creating server" "$OUTPUT_DIR/logs/server.log" 2>/dev/null; then
    break
  fi
  if ! kill -0 "$server_pid" 2>/dev/null; then
    tail -80 "$OUTPUT_DIR/logs/server.log" >&2 || true
    exit 1
  fi
  if ((elapsed == 299)); then
    echo "Timed out waiting for policy server" >&2
    exit 1
  fi
  sleep 1
done

CUDA_VISIBLE_DEVICES=$GPU \
MUJOCO_GL=egl \
MUJOCO_EGL_DEVICE_ID=$GPU \
"$PYTHON" examples/libero/main.py \
  --args.host 127.0.0.1 \
  --args.port "$PORT" \
  --args.task-suite-name "$SUITE" \
  --args.task-ids "$TASK_ID" \
  --args.episode-ids "$EPISODE_ID" \
  --args.resize-size 224 \
  --args.replan-steps 5 \
  --args.num-steps-wait 10 \
  --args.seed "$ENV_SEED" \
  --args.policy-noise-seed "$FLOW_NOISE_SEED" \
  --args.visualize-interaction-patches \
  --args.interaction-visualizations-per-episode -1 \
  --args.video-out-path "$OUTPUT_DIR/rollout" \
  >"$OUTPUT_DIR/logs/client.log" 2>&1

echo "DemoVLA memory overlays complete: $OUTPUT_DIR"
