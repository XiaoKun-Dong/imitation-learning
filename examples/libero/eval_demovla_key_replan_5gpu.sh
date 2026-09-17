#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 7 ]]; then
  cat >&2 <<'EOF'
Usage:
  eval_demovla_key_replan_5gpu.sh POLICY_DIR SUITE TASK_ID EPISODE_ID DONOR_MEMORY_DIR OUTPUT_ROOT BASE_PORT

Starts matched fixed-0.03 correct, zero-memory, and three single-layer-knockout
servers on GPUs 0-4. It then traces the exact episode twice, once following the
correct policy and once following zero memory. Every replan queries all causal
conditions with identical observation and stateless flow noise.
EOF
  exit 2
fi

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON="$ROOT/.venv/bin/python"
POLICY_DIR=$(realpath "$1")
SUITE=$2
TASK_ID=$3
EPISODE_ID=$4
DONOR_MEMORY_DIR=$(realpath "$5")
OUTPUT_ROOT=$6
BASE_PORT=$7
POLICY_CONFIG=${POLICY_CONFIG:-demovla_libero_full_stage_a_fixed_gate}
ENV_SEED=${ENV_SEED:-7}
FLOW_NOISE_SEED=${FLOW_NOISE_SEED:-0}
POST_BRANCH_REPLAN=${POST_BRANCH_REPLAN:-}
POST_BRANCH_NUM_REPEATS=${POST_BRANCH_NUM_REPEATS:-1}
POST_WRONG_PHASE_REPLAN=${POST_WRONG_PHASE_REPLAN:-10}

if [[ -n "$POST_BRANCH_REPLAN" && "$POST_BRANCH_REPLAN" != "auto" ]] && ! [[ "$POST_BRANCH_REPLAN" =~ ^[0-9]+$ ]]; then
  echo "POST_BRANCH_REPLAN must be empty, 'auto', or a non-negative integer" >&2
  exit 1
fi
if ! [[ "$POST_BRANCH_NUM_REPEATS" =~ ^[1-9][0-9]*$ ]]; then
  echo "POST_BRANCH_NUM_REPEATS must be a positive integer" >&2
  exit 1
fi
if ! [[ "$POST_WRONG_PHASE_REPLAN" =~ ^[0-9]+$ ]]; then
  echo "POST_WRONG_PHASE_REPLAN must be a non-negative integer" >&2
  exit 1
fi

if [[ ! -d "$POLICY_DIR/params" ]]; then
  echo "Checkpoint params missing: $POLICY_DIR/params" >&2
  exit 1
fi
if [[ ! -d "$DONOR_MEMORY_DIR" ]]; then
  echo "Donor memory directory missing: $DONOR_MEMORY_DIR" >&2
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
server_pids=()
cleanup() {
  for pid in "${server_pids[@]}"; do
    kill "$pid" 2>/dev/null || true
    wait "$pid" 2>/dev/null || true
  done
}
trap cleanup EXIT INT TERM

start_server() {
  local label=$1
  local gpu=$2
  local port=$3
  local ablation=$4
  local gate4=$5
  local gate9=$6
  local gate14=$7
  local diagnostics=$8
  local -a diagnostics_args=()
  if [[ "$diagnostics" == "1" ]]; then
    diagnostics_args=(--interaction-diagnostics)
  fi
  CUDA_VISIBLE_DEVICES=$gpu \
  XLA_PYTHON_CLIENT_PREALLOCATE=false \
  "$PYTHON" -u scripts/serve_policy.py \
    --port "$port" \
    "${diagnostics_args[@]}" \
    --interaction-ablation "$ablation" \
    --interaction-layer-mean-gates "$gate4" "$gate9" "$gate14" \
    policy:checkpoint \
    --policy.config "$POLICY_CONFIG" \
    --policy.dir "$POLICY_DIR" \
    >"$OUTPUT_ROOT/logs/${label}_server.log" 2>&1 &
  server_pids+=("$!")
}

wait_for_server() {
  local pid=$1
  local log_file=$2
  for ((elapsed = 0; elapsed < 300; elapsed++)); do
    if grep -q "Creating server" "$log_file" 2>/dev/null; then
      return 0
    fi
    if ! kill -0 "$pid" 2>/dev/null; then
      tail -80 "$log_file" >&2 || true
      return 1
    fi
    sleep 1
  done
  echo "Timed out waiting for policy server: $log_file" >&2
  return 1
}

cd "$ROOT"
start_server correct 0 "$((BASE_PORT + 0))" layer_mean 0.03 0.03 0.03 1
start_server zero 1 "$((BASE_PORT + 1))" zero_memory 0.03 0.03 0.03 0
start_server knockout_layer4 2 "$((BASE_PORT + 2))" layer_mean 0.0 0.03 0.03 0
start_server knockout_layer9 3 "$((BASE_PORT + 3))" layer_mean 0.03 0.0 0.03 0
start_server knockout_layer14 4 "$((BASE_PORT + 4))" layer_mean 0.03 0.03 0.0 0

labels=(correct zero knockout_layer4 knockout_layer9 knockout_layer14)
for index in "${!server_pids[@]}"; do
  wait_for_server "${server_pids[$index]}" "$OUTPUT_ROOT/logs/${labels[$index]}_server.log"
done

for execution in correct zero; do
  execution_donor_dir="$DONOR_MEMORY_DIR"
  if [[ "$execution" == "zero" ]]; then
    execution_donor_dir=$(realpath "$OUTPUT_ROOT/correct/replans")
  fi
  CUDA_VISIBLE_DEVICES=5 \
  MUJOCO_GL=egl \
  MUJOCO_EGL_DEVICE_ID=5 \
  "$PYTHON" examples/libero/run_demovla_key_replan.py \
    --args.task-suite-name "$SUITE" \
    --args.task-id "$TASK_ID" \
    --args.episode-id "$EPISODE_ID" \
    --args.output-dir "$OUTPUT_ROOT/$execution" \
    --args.correct-port "$((BASE_PORT + 0))" \
    --args.zero-port "$((BASE_PORT + 1))" \
    --args.knockout-layer4-port "$((BASE_PORT + 2))" \
    --args.knockout-layer9-port "$((BASE_PORT + 3))" \
    --args.knockout-layer14-port "$((BASE_PORT + 4))" \
    --args.execute-condition "$execution" \
    --args.donor-memory-dir "$execution_donor_dir" \
    --args.seed "$ENV_SEED" \
    --args.policy-noise-seed "$FLOW_NOISE_SEED" \
    >"$OUTPUT_ROOT/logs/${execution}_client.log" 2>&1
done

if [[ "$POST_BRANCH_REPLAN" == "auto" ]]; then
  "$PYTHON" scripts/analyze_demovla_key_replans.py "$OUTPUT_ROOT" \
    >"$OUTPUT_ROOT/logs/key_replan_analysis.log" 2>&1
  POST_BRANCH_REPLAN=$("$PYTHON" -c 'import json,pathlib,sys; data=json.loads((pathlib.Path(sys.argv[1])/"analysis/summary.json").read_text()); value=data["zero_trajectory_selection"]["earliest_candidate_replan"]; print("" if value is None else value)' "$OUTPUT_ROOT")
fi

if [[ -n "$POST_BRANCH_REPLAN" ]] && "$PYTHON" -c 'import json,pathlib,sys; root=pathlib.Path(sys.argv[1]); correct=json.loads((root/"correct/summary.json").read_text()); zero=json.loads((root/"zero/summary.json").read_text()); raise SystemExit(0 if correct["success"] and not zero["success"] else 1)' "$OUTPUT_ROOT"; then
  branch_root="$OUTPUT_ROOT/lockstep_r$(printf '%03d' "$POST_BRANCH_REPLAN")"
  mkdir -p "$branch_root"
  for ((repeat = 1; repeat <= POST_BRANCH_NUM_REPEATS; repeat++)); do
    printf -v repeat_name 'repeat_%02d' "$repeat"
    CUDA_VISIBLE_DEVICES=5 \
    MUJOCO_GL=egl \
    MUJOCO_EGL_DEVICE_ID=5 \
    "$PYTHON" examples/libero/run_demovla_lockstep_branch.py \
      --args.task-suite-name "$SUITE" \
      --args.task-id "$TASK_ID" \
      --args.episode-id "$EPISODE_ID" \
      --args.branch-replan "$POST_BRANCH_REPLAN" \
      --args.donor-memory-dir "$OUTPUT_ROOT/correct/replans" \
      --args.output-dir "$branch_root/$repeat_name" \
      --args.correct-port "$((BASE_PORT + 0))" \
      --args.zero-port "$((BASE_PORT + 1))" \
      --args.knockout-layer4-port "$((BASE_PORT + 2))" \
      --args.knockout-layer9-port "$((BASE_PORT + 3))" \
      --args.knockout-layer14-port "$((BASE_PORT + 4))" \
      --args.seed "$ENV_SEED" \
      --args.policy-noise-seed "$FLOW_NOISE_SEED" \
      --args.wrong-phase-replan "$POST_WRONG_PHASE_REPLAN" \
      >"$OUTPUT_ROOT/logs/lockstep_r$(printf '%03d' "$POST_BRANCH_REPLAN")_${repeat_name}.log" 2>&1
  done
  "$PYTHON" -c 'import json,pathlib,sys; root=pathlib.Path(sys.argv[1]); rows=[json.loads(p.read_text()) for p in sorted(root.glob("repeat_*/summary.json"))]; (root/"repeat_summaries.json").write_text(json.dumps(rows,indent=2)+"\n")' "$branch_root"
elif [[ -n "$POST_BRANCH_REPLAN" ]]; then
  "$PYTHON" -c 'import json,pathlib,sys; root=pathlib.Path(sys.argv[1]); correct=json.loads((root/"correct/summary.json").read_text()); zero=json.loads((root/"zero/summary.json").read_text()); payload={"branch_replan":int(sys.argv[2]),"status":"skipped","reason":"requires correct success and zero failure","correct_success":correct["success"],"zero_success":zero["success"]}; (root/"post_branch_status.json").write_text(json.dumps(payload,indent=2)+"\n")' "$OUTPUT_ROOT" "$POST_BRANCH_REPLAN"
fi

echo "DemoVLA key-replan trace complete: $OUTPUT_ROOT"
