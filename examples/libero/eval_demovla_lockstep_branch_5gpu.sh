#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 8 ]]; then
  cat >&2 <<'EOF'
Usage:
  eval_demovla_lockstep_branch_5gpu.sh POLICY_DIR SUITE TASK_ID EPISODE_ID BRANCH_REPLAN DONOR_DIR OUTPUT_ROOT BASE_PORT

Runs all memory and layer interventions in independent LIBERO environments with
an exactly shared, state-checked zero-memory prefix before BRANCH_REPLAN.
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
DONOR_DIR=$(realpath "$6")
OUTPUT_ROOT=$7
BASE_PORT=$8
POLICY_CONFIG=${POLICY_CONFIG:-demovla_libero_full_stage_a_fixed_gate}
ENV_SEED=${ENV_SEED:-7}
FLOW_NOISE_SEED=${FLOW_NOISE_SEED:-0}
NUM_REPEATS=${NUM_REPEATS:-1}
WRONG_PHASE_REPLAN=${WRONG_PHASE_REPLAN:-74}

if ! [[ "$NUM_REPEATS" =~ ^[1-9][0-9]*$ ]]; then
  echo "NUM_REPEATS must be a positive integer" >&2
  exit 1
fi
if ! [[ "$WRONG_PHASE_REPLAN" =~ ^[0-9]+$ ]]; then
  echo "WRONG_PHASE_REPLAN must be a non-negative integer" >&2
  exit 1
fi

if [[ -e "$OUTPUT_ROOT" ]]; then
  echo "Output root already exists; refusing to overwrite: $OUTPUT_ROOT" >&2
  exit 1
fi
if [[ -f "$ROOT/third_party/libero/libero/libero/__init__.py" ]]; then
  LIBERO_SOURCE_ROOT="$ROOT/third_party/libero"
else
  LIBERO_SOURCE_ROOT="$ROOT/third_party/libero/LIBERO"
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
  local label=$1 gpu=$2 port=$3 ablation=$4 gate4=$5 gate9=$6 gate14=$7
  CUDA_VISIBLE_DEVICES=$gpu XLA_PYTHON_CLIENT_PREALLOCATE=false \
  "$PYTHON" -u scripts/serve_policy.py \
    --port "$port" \
    --interaction-ablation "$ablation" \
    --interaction-layer-mean-gates "$gate4" "$gate9" "$gate14" \
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
  return 1
}

cd "$ROOT"
start_server correct 0 "$((BASE_PORT + 0))" layer_mean 0.03 0.03 0.03
start_server zero 1 "$((BASE_PORT + 1))" zero_memory 0.03 0.03 0.03
start_server knockout_layer4 2 "$((BASE_PORT + 2))" layer_mean 0.0 0.03 0.03
start_server knockout_layer9 3 "$((BASE_PORT + 3))" layer_mean 0.03 0.0 0.03
start_server knockout_layer14 4 "$((BASE_PORT + 4))" layer_mean 0.03 0.03 0.0
labels=(correct zero knockout_layer4 knockout_layer9 knockout_layer14)
for index in "${!server_pids[@]}"; do
  wait_for_server "${server_pids[$index]}" "$OUTPUT_ROOT/logs/${labels[$index]}_server.log"
done

for ((repeat = 1; repeat <= NUM_REPEATS; repeat++)); do
  run_output="$OUTPUT_ROOT"
  client_log="$OUTPUT_ROOT/logs/lockstep_client.log"
  if ((NUM_REPEATS > 1)); then
    printf -v repeat_name 'repeat_%02d' "$repeat"
    run_output="$OUTPUT_ROOT/$repeat_name"
    client_log="$OUTPUT_ROOT/logs/${repeat_name}_client.log"
  fi
  CUDA_VISIBLE_DEVICES=5 MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=5 \
  "$PYTHON" examples/libero/run_demovla_lockstep_branch.py \
    --args.task-suite-name "$SUITE" \
    --args.task-id "$TASK_ID" \
    --args.episode-id "$EPISODE_ID" \
    --args.branch-replan "$BRANCH_REPLAN" \
    --args.donor-memory-dir "$DONOR_DIR" \
    --args.output-dir "$run_output" \
    --args.correct-port "$((BASE_PORT + 0))" \
    --args.zero-port "$((BASE_PORT + 1))" \
    --args.knockout-layer4-port "$((BASE_PORT + 2))" \
    --args.knockout-layer9-port "$((BASE_PORT + 3))" \
    --args.knockout-layer14-port "$((BASE_PORT + 4))" \
    --args.seed "$ENV_SEED" \
    --args.policy-noise-seed "$FLOW_NOISE_SEED" \
    --args.wrong-phase-replan "$WRONG_PHASE_REPLAN" \
    >"$client_log" 2>&1
done

if ((NUM_REPEATS > 1)); then
  "$PYTHON" -c 'import json,pathlib,sys; root=pathlib.Path(sys.argv[1]); rows=[json.loads(p.read_text()) for p in sorted(root.glob("repeat_*/summary.json"))]; (root/"repeat_summaries.json").write_text(json.dumps(rows,indent=2)+"\n")' "$OUTPUT_ROOT"
fi

echo "DemoVLA lockstep branch evaluation complete: $OUTPUT_ROOT"
