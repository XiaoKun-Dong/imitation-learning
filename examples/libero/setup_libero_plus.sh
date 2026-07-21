#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CHECKOUT="$ROOT/third_party/libero-plus/LIBERO-plus"
PACKAGE_ROOT="$CHECKOUT/libero/libero"
ARCHIVE="$ROOT/third_party/libero-plus/assets.zip"
ARCHIVE_ASSETS="$PACKAGE_ROOT/inspire/hdd/project/embodied-multimodality/public/syfei/libero_new/release/dataset/LIBERO-plus-0/assets"
REPO_URL="${LIBERO_PLUS_REPO_URL:-https://github.com/sylvestf/LIBERO-plus.git}"
ASSETS_URL="${LIBERO_PLUS_ASSETS_URL:-https://huggingface.co/datasets/Sylvest/LIBERO-plus/resolve/main/assets.zip}"

mkdir -p "$(dirname "$CHECKOUT")"
if [[ ! -d "$CHECKOUT/.git" ]]; then
  git clone --depth 1 "$REPO_URL" "$CHECKOUT"
fi

uv pip install --python "$ROOT/.venv/bin/python" "scikit-image>=0.25.0" "wand>=0.6.13"
if ! ldconfig -p 2>/dev/null | grep -q "libMagickWand"; then
  echo "MagickWand shared library is missing." >&2
  echo "Install it on Ubuntu with: sudo apt-get install libmagickwand-dev" >&2
  exit 1
fi

if [[ ! -d "$PACKAGE_ROOT/assets" ]]; then
  if [[ ! -d "$ARCHIVE_ASSETS" ]]; then
    echo "Downloading LIBERO-plus assets (about 6.4 GB; interrupted downloads can be resumed)..."
    curl_args=(
      --fail
      --location
      --continue-at -
      --retry 12
      --retry-delay 5
      --output "$ARCHIVE"
    )
    if curl --help all 2>/dev/null | grep -q -- "--retry-all-errors"; then
      curl_args+=(--retry-all-errors)
    fi
    curl "${curl_args[@]}" "$ASSETS_URL"
    unzip -tq "$ARCHIVE"
    unzip -q "$ARCHIVE" -d "$PACKAGE_ROOT"
  fi
  mv "$ARCHIVE_ASSETS" "$PACKAGE_ROOT/assets"
fi

if [[ ! -d "$PACKAGE_ROOT/assets" ]]; then
  echo "assets.zip did not create $PACKAGE_ROOT/assets" >&2
  exit 1
fi

echo "LIBERO-plus is ready at $CHECKOUT"
