#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "Usage: run_libero_full_matched_sequence_4gpu.sh EXP_NAME" >&2
  exit 2
fi

EXP_NAME=$1
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUN_DIR="$ROOT/outputs/libero_full_matched/$EXP_NAME"
STATUS_FILE="$RUN_DIR/sequence_status.txt"

mkdir -p "$RUN_DIR"
if [[ -e "$STATUS_FILE" ]]; then
  echo "Sequence status already exists; refusing duplicate launch: $STATUS_FILE" >&2
  exit 1
fi

record() {
  printf '%s %s\n' "$(date --iso-8601=seconds)" "$*" | tee -a "$STATUS_FILE"
}

run_target() {
  local target=$1
  local log_file="$RUN_DIR/${target}_train.log"
  record "target=$target status=starting log=$log_file"
  if bash "$ROOT/scripts/train_libero_full_matched_4gpu.sh" "$target" "$EXP_NAME" >"$log_file" 2>&1; then
    record "target=$target status=completed"
  else
    local status=$?
    record "target=$target status=failed exit_code=$status"
    return "$status"
  fi
}

record "sequence_status=starting order=demovla,control"
run_target demovla
run_target control
record "sequence_status=completed"
