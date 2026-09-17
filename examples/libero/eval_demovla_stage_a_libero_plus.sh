#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 || $# -gt 4 ]]; then
  cat >&2 <<'EOF'
Usage:
  eval_demovla_stage_a_libero_plus.sh POLICY_DIR OUTPUT_ROOT [BASE_PORT] [TASK_LIMIT]

Runs paired Stage-A fixed-0.03 correct / zero-memory / injection-off
evaluations on LIBERO-plus Objects Layout. The default task set contains
10 objects x 5 target-displacement levels. Set LIBERO_PLUS_SEEDS to select
environment seeds (default: "7 42 123"). TASK_LIMIT is intended for smoke
tests and takes a prefix of the frozen 50-task list.
EOF
  exit 2
fi

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON="$ROOT/.venv/bin/python"
POLICY_DIR=$(realpath "$1")
OUTPUT_ROOT=$2
BASE_PORT=${3:-8400}
TASK_LIMIT=${4:-50}
POLICY_CONFIG=${POLICY_CONFIG:-demovla_libero_full_stage_a_fixed_gate}
GPU_LIST=${GPU_LIST:-0,1,2}
FLOW_NOISE_SEED=${FLOW_NOISE_SEED:-0}
INTERACTION_GATE=${INTERACTION_GATE:-0.03}
read -r -a SEEDS <<<"${LIBERO_PLUS_SEEDS:-7 42 123}"
IFS=',' read -r -a GPUS <<<"$GPU_LIST"

# Frozen target-displacement panel: 10 LIBERO Object targets x the five
# LIBERO-plus level1..level5 variants, always sample1. IDs are ordered by
# level so TASK_LIMIT=10 covers all ten targets instead of only two targets.
ALL_TASK_IDS=(
  1840 1884 1928 1966 1993 2018 2063 2112 2157 2204
  1844 1888 1932 1970 1997 2022 2067 2116 2161 2208
  1848 1892 1936 1974 2001 2026 2071 2120 2164 2212
  1852 1896 1940 1978 2005 2030 2075 2124 2167 2214
  1856 1900 1944 1982 2009 2034 2079 2128 2170 2218
)
CONDITIONS=(correct zero_memory injection_off)

if [[ ${#GPUS[@]} -ne 3 ]]; then
  echo "Exactly three GPUs are required; GPU_LIST=$GPU_LIST" >&2
  exit 2
fi
if ! [[ "$TASK_LIMIT" =~ ^[1-9][0-9]*$ ]] || ((TASK_LIMIT > ${#ALL_TASK_IDS[@]})); then
  echo "TASK_LIMIT must be an integer in [1, ${#ALL_TASK_IDS[@]}]" >&2
  exit 2
fi
if [[ ${#SEEDS[@]} -eq 0 ]]; then
  echo "LIBERO_PLUS_SEEDS must contain at least one integer seed" >&2
  exit 2
fi
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

LIBERO_PLUS_ROOT="$ROOT/third_party/libero-plus/LIBERO-plus"
LIBERO_PLUS_ASSETS="$LIBERO_PLUS_ROOT/libero/libero/assets"
if [[ ! -f "$LIBERO_PLUS_ROOT/libero/libero/__init__.py" ]]; then
  echo "LIBERO-plus checkout missing: $LIBERO_PLUS_ROOT" >&2
  exit 1
fi
if [[ ! -d "$LIBERO_PLUS_ASSETS" ]]; then
  echo "LIBERO-plus assets missing: $LIBERO_PLUS_ASSETS" >&2
  exit 1
fi

TASK_IDS=("${ALL_TASK_IDS[@]:0:TASK_LIMIT}")
mkdir -p "$OUTPUT_ROOT/logs"

stats_file=$(find "$POLICY_DIR/assets" -type f -name norm_stats.json -print -quit)
if [[ -z "$stats_file" ]]; then
  echo "Checkpoint norm_stats.json not found under $POLICY_DIR/assets" >&2
  exit 1
fi

{
  echo "benchmark=LIBERO-plus"
  echo "category=Objects Layout"
  echo "panel=target displacement level1-level5 sample1"
  echo "policy_config=$POLICY_CONFIG"
  echo "policy_dir=$POLICY_DIR"
  echo "conditions=${CONDITIONS[*]}"
  echo "interaction_gate=$INTERACTION_GATE"
  echo "environment_seeds=${SEEDS[*]}"
  echo "flow_noise_seed=$FLOW_NOISE_SEED"
  echo "task_ids=${TASK_IDS[*]}"
  echo "gpu_list=$GPU_LIST"
  echo "openpi_git_commit=$(git -C "$ROOT" rev-parse HEAD)"
  echo "libero_plus_git_commit=$(git -C "$LIBERO_PLUS_ROOT" rev-parse HEAD)"
  sha256sum "$stats_file"
} >"$OUTPUT_ROOT/manifest.txt"

server_pids=()
client_pids=()

cleanup() {
  local pid
  for pid in "${client_pids[@]:-}" "${server_pids[@]:-}"; do
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
      echo "Policy server failed: $log_file" >&2
      tail -80 "$log_file" >&2 || true
      return 1
    fi
    sleep 1
  done
  echo "Timed out waiting for policy server: $log_file" >&2
  return 1
}

start_server() {
  local condition=$1
  local gpu=$2
  local port=$3
  local -a ablation_args
  case "$condition" in
    correct)
      ablation_args=(
        --interaction-ablation layer_mean
        --interaction-layer-mean-gates "$INTERACTION_GATE" "$INTERACTION_GATE" "$INTERACTION_GATE"
      )
      ;;
    zero_memory)
      ablation_args=(
        --interaction-ablation zero_memory
        --interaction-layer-mean-gates "$INTERACTION_GATE" "$INTERACTION_GATE" "$INTERACTION_GATE"
      )
      ;;
    injection_off)
      ablation_args=(--interaction-ablation off)
      ;;
    *)
      echo "Unsupported condition: $condition" >&2
      return 2
      ;;
  esac

  CUDA_VISIBLE_DEVICES=$gpu XLA_PYTHON_CLIENT_PREALLOCATE=false \
  "$PYTHON" -u scripts/serve_policy.py \
    --port "$port" \
    "${ablation_args[@]}" \
    policy:checkpoint \
    --policy.config "$POLICY_CONFIG" \
    --policy.dir "$POLICY_DIR" \
    >"$OUTPUT_ROOT/logs/${condition}_server.log" 2>&1 &
  server_pids+=("$!")
}

cd "$ROOT"
for index in "${!CONDITIONS[@]}"; do
  start_server "${CONDITIONS[$index]}" "${GPUS[$index]}" "$((BASE_PORT + index))"
done
for index in "${!CONDITIONS[@]}"; do
  wait_for_server "${server_pids[$index]}" "$OUTPUT_ROOT/logs/${CONDITIONS[$index]}_server.log"
done

for seed in "${SEEDS[@]}"; do
  client_pids=()
  for index in "${!CONDITIONS[@]}"; do
    condition=${CONDITIONS[$index]}
    gpu=${GPUS[$index]}
    port=$((BASE_PORT + index))
    condition_dir="$OUTPUT_ROOT/$condition/seed_$seed"
    mkdir -p "$condition_dir"
    CUDA_VISIBLE_DEVICES=$gpu \
    MUJOCO_GL=egl \
    MUJOCO_EGL_DEVICE_ID=$gpu \
    "$PYTHON" examples/libero/run_libero_plus.py \
      --args.host 127.0.0.1 \
      --args.port "$port" \
      --args.task-suite-name libero_object \
      --args.task-ids "${TASK_IDS[@]}" \
      --args.max-tasks "$TASK_LIMIT" \
      --args.num-trials-per-task 1 \
      --args.seed "$seed" \
      --args.policy-noise-seed "$FLOW_NOISE_SEED" \
      --args.track-grasp-diagnostics \
      --args.video-out-path "$condition_dir" \
      >"$OUTPUT_ROOT/logs/${condition}_seed${seed}_client.log" 2>&1 &
    client_pids+=("$!")
  done

  status=0
  for index in "${!client_pids[@]}"; do
    if ! wait "${client_pids[$index]}"; then
      echo "${CONDITIONS[$index]} seed=$seed failed" >&2
      status=1
    fi
  done
  client_pids=()
  if ((status != 0)); then
    exit "$status"
  fi
done

"$PYTHON" examples/libero/summarize_libero_plus_paired.py "$OUTPUT_ROOT" \
  --conditions "${CONDITIONS[@]}" \
  --reference correct \
  --seeds "${SEEDS[@]}"

echo "LIBERO-plus paired evaluation complete: $OUTPUT_ROOT"
