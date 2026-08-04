#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 || $# -gt 6 ]]; then
  cat >&2 <<'EOF'
Usage:
  eval_demovla_regression_pilot.sh DYNAMIC_POLICY_DIR PI05_POLICY_DIR [GPU] [PORT] [TRIALS] [SEED]

Runs a small paired LIBERO-object regression check using the official OpenPI
rollout settings. The only deliberate deviation is stateless flow noise, which
makes every (task, episode, replan) use the same noise across policies.

Policies:
  dynamic        dynamic-gate checkpoint, normal inference
  injection_off  same checkpoint and norm stats, interaction injection disabled
  pi05_official  official pi05_libero checkpoint and its original norm stats

Defaults: GPU=0, PORT=8000, TRIALS=3 per task, SEED=7.

Optional environment overrides:
  DYNAMIC_CONFIG     default: demovla_libero_sparse_deep_dynamic_gate
  PI05_CONFIG        default: pi05_libero
  FLOW_NOISE_SEED    default: 0
  PILOT_OUTPUT_ROOT  default: data/libero/videos/demovla_regression_pilot
EOF
  exit 2
fi

DYNAMIC_POLICY_DIR=$1
PI05_POLICY_DIR=$2
GPU=${3:-0}
PORT=${4:-8000}
TRIALS=${5:-3}
SEED=${6:-7}

if ((TRIALS < 1 || TRIALS > 50)); then
  echo "TRIALS must be between 1 and 50, got $TRIALS" >&2
  exit 2
fi

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON="$ROOT/.venv/bin/python"
DYNAMIC_CONFIG=${DYNAMIC_CONFIG:-demovla_libero_sparse_deep_dynamic_gate}
PI05_CONFIG=${PI05_CONFIG:-pi05_libero}
FLOW_NOISE_SEED=${FLOW_NOISE_SEED:-0}
OUTPUT_ROOT=${PILOT_OUTPUT_ROOT:-$ROOT/data/libero/videos/demovla_regression_pilot}

if [[ ! -x "$PYTHON" ]]; then
  echo "Project Python not found: $PYTHON" >&2
  exit 1
fi
if [[ ! -d "$DYNAMIC_POLICY_DIR/params" ]]; then
  echo "Dynamic checkpoint params missing: $DYNAMIC_POLICY_DIR/params" >&2
  exit 1
fi
if [[ ! -f "$DYNAMIC_POLICY_DIR/assets/local/libero/norm_stats.json" ]]; then
  echo "Dynamic checkpoint norm stats missing: $DYNAMIC_POLICY_DIR/assets/local/libero/norm_stats.json" >&2
  exit 1
fi
if [[ ! -d "$PI05_POLICY_DIR/params" ]]; then
  echo "Official pi05 checkpoint params missing: $PI05_POLICY_DIR/params" >&2
  exit 1
fi
if [[ ! -f "$PI05_POLICY_DIR/assets/physical-intelligence/libero/norm_stats.json" ]]; then
  echo "Official pi05 norm stats missing: $PI05_POLICY_DIR/assets/physical-intelligence/libero/norm_stats.json" >&2
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

  if [[ "$ablation" == "off" ]]; then
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
    --args.object-condition none \
    --args.resize-size 224 \
    --args.replan-steps 5 \
    --args.num-steps-wait 10 \
    --args.num-trials-per-task "$TRIALS" \
    --args.seed "$SEED" \
    --args.policy-noise-seed "$FLOW_NOISE_SEED" \
    --args.video-out-path "$video_dir" \
    >"$client_log" 2>&1

  cleanup
}

cd "$ROOT"
run_one dynamic "$DYNAMIC_CONFIG" "$DYNAMIC_POLICY_DIR" normal
run_one injection_off "$DYNAMIC_CONFIG" "$DYNAMIC_POLICY_DIR" off
run_one pi05_official "$PI05_CONFIG" "$PI05_POLICY_DIR" normal

echo "Paired regression pilot complete: $OUTPUT_ROOT"
"$PYTHON" examples/libero/summarize_paired_eval.py \
  "$OUTPUT_ROOT" dynamic injection_off pi05_official --reference pi05_official
