#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WITH_RLDS=0
SKIP_SUBMODULES=0

usage() {
  cat <<'EOF'
Usage: bash scripts/setup_demovla.sh [--with-rlds] [--skip-submodules]

  --with-rlds       Install TensorFlow/TFDS dependencies used only for data conversion.
  --skip-submodules Do not initialize the LIBERO git submodule.
EOF
}

while (($#)); do
  case "$1" in
    --with-rlds)
      WITH_RLDS=1
      ;;
    --skip-submodules)
      SKIP_SUBMODULES=1
      ;;
    -h | --help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
  shift
done

cd "$ROOT"

if ! command -v uv >/dev/null 2>&1; then
  echo "uv is required. Install it from https://docs.astral.sh/uv/ and rerun this script." >&2
  exit 1
fi

for command_name in git ffmpeg; do
  if ! command -v "$command_name" >/dev/null 2>&1; then
    echo "Warning: $command_name is missing; install the Ubuntu system dependencies listed in README.md." >&2
  fi
done
if ! git lfs version >/dev/null 2>&1; then
  echo "Warning: Git LFS is missing; install git-lfs before downloading LFS-managed assets." >&2
fi
if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "Warning: nvidia-smi is unavailable; GPU training and EGL evaluation will not work yet." >&2
fi
if command -v ldconfig >/dev/null 2>&1 && ! ldconfig -p 2>/dev/null | grep "libEGL.so" >/dev/null; then
  echo "Warning: libEGL is missing; install libegl1 for headless LIBERO evaluation." >&2
fi

if ((SKIP_SUBMODULES == 0)); then
  if [[ ! -f third_party/libero/setup.py && ! -f third_party/libero/LIBERO/setup.py ]]; then
    git submodule update --init --recursive third_party/libero
  fi
fi

if [[ ! -f third_party/libero/setup.py && ! -f third_party/libero/LIBERO/setup.py ]]; then
  echo "LIBERO checkout is missing. Initialize the submodule or set it up under third_party/libero." >&2
  exit 1
fi

sync_args=(sync --frozen --inexact --group dev)
if ((WITH_RLDS == 1)); then
  sync_args+=(--group rlds)
fi

GIT_LFS_SKIP_SMUDGE=1 uv "${sync_args[@]}"

echo
echo "Running environment checks..."
.venv/bin/python scripts/check_demovla_env.py

echo
echo "DemoVLA environment is ready."
echo "Next: prepare data, compute norm stats, then run the training command in README.md."
