#!/bin/bash
# Apply the flash-attention sm_75 patch and install the resulting
# Python package into the comfyui container's venv.
#
# Run from the workstation host (NOT inside the container).
#
# Usage:
#   bash apply-patch-and-install.sh
#
# After this script completes, --use-flash-attention in ComfyUI
# should work on sm_75 (Turing) hardware.
#
# This script is designed for Turing-only hosts (i.e. hosts whose
# only NVIDIA GPU is sm_75). It is also correct on heterogeneous
# hosts, but the build is then ~90 min instead of ~8 min.

set -euo pipefail

# --- Configuration ---------------------------------------------------------

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PATCH_FILE="$REPO_ROOT/patches/0001-add-sm75-arch.patch"

# Inside-container working directory.
CONTAINER_WORK="/tmp/flash-attention"

# Upstream we patch.
UPSTREAM_REPO="https://github.com/Dao-AILab/flash-attention.git"
UPSTREAM_COMMIT="965c921"

# CUTLASS submodule. The comfyui container has no network to github.com,
# so we clone CUTLASS on the host (where the network works) and copy
# it into the container.
CUTLASS_REPO="https://github.com/NVIDIA/cutlass.git"
CUTLASS_COMMIT="df18f5e4"
CUTLASS_HOST_DIR="/tmp/cutlass-host"

# nvidia cu13 wheel paths inside the container.
NVIDIA_CU13_INCLUDE="/opt/venv/lib/python3.12/site-packages/nvidia/cu13/include"
NVIDIA_CU13_LIB="/opt/venv/lib/python3.12/site-packages/nvidia/cu13/lib"

# Build matrix. Override by setting this env var before running the
# script. Default is sm_75 only (Turing-only hosts).
# For heterogeneous hosts, set FLASH_ATTN_CUDA_ARCHS=75;80;90
# (warning: ~90 min build instead of ~8 min).
: "${FLASH_ATTN_CUDA_ARCHS:=75}"
export FLASH_ATTN_CUDA_ARCHS

# --- Sanity checks ---------------------------------------------------------

if ! command -v docker >/dev/null 2>&1; then
    echo "ERROR: docker not on PATH" >&2; exit 1
fi
if ! docker ps --format '{{.Names}}' | grep -qx comfyui; then
    echo "ERROR: comfyui container is not running." >&2
    echo "Start it first:  cd /opt/comfyui && docker compose up -d" >&2
    exit 1
fi
if [ ! -f "$PATCH_FILE" ]; then
    echo "ERROR: patch file not found: $PATCH_FILE" >&2; exit 1
fi

# --- Step 1: Clone (or update) upstream inside the container ---------------

echo "[1/7] Cloning upstream into the container at $CONTAINER_WORK..."
docker exec comfyui bash -c '
    if [ -d "$1/.git" ]; then
        echo "  existing clone found, fetching $2"
        cd "$1"
        git fetch --depth=1 origin "$2"
        git checkout "$2"
    else
        mkdir -p "$1"
        cd "$1"
        git init
        git remote add origin "$3"
        git fetch --depth=1 origin "$2"
        git checkout "$2"
    fi
' bash "$CONTAINER_WORK" "$UPSTREAM_COMMIT" "$UPSTREAM_REPO"
echo "  upstream at $UPSTREAM_COMMIT"

# --- Step 2: Verify the patch applies cleanly (host-side) ------------------

echo "[2/7] Verifying patch applies..."
TMP_APPLY_DIR="$(mktemp -d)"
trap "rm -rf '$TMP_APPLY_DIR'" EXIT
git clone --quiet --depth=1 "$UPSTREAM_REPO" "$TMP_APPLY_DIR/upstream"
cd "$TMP_APPLY_DIR/upstream"
git fetch --quiet --depth=1 origin "$UPSTREAM_COMMIT"
git checkout --quiet "$UPSTREAM_COMMIT"
if ! git apply --check "$PATCH_FILE"; then
    echo "ERROR: patch does not apply cleanly to upstream $UPSTREAM_COMMIT." >&2
    echo "Either the patch is stale (regenerate against current upstream)" >&2
    echo "or the upstream commit hash in this script is wrong." >&2
    exit 1
fi
git apply "$PATCH_FILE"
echo "  patch applies cleanly to $UPSTREAM_COMMIT"

# --- Step 3: Apply the patch inside the container --------------------------

echo "[3/7] Applying patch inside the container..."
PATCHED_FILES="$(cd "$TMP_APPLY_DIR/upstream" && git diff --name-only)"
for f in $PATCHED_FILES; do
    docker cp "$TMP_APPLY_DIR/upstream/$f" "comfyui:$CONTAINER_WORK/$f"
done
echo "  applied $(echo $PATCHED_FILES | wc -w) patched file(s)"

# --- Step 4: Vendor CUTLASS -------------------------------------------------

echo "[4/7] Vendoring CUTLASS $CUTLASS_COMMIT..."
if [ ! -d "$CUTLASS_HOST_DIR/.git" ]; then
    git clone --quiet --no-checkout "$CUTLASS_REPO" "$CUTLASS_HOST_DIR"
fi
cd "$CUTLASS_HOST_DIR"
git fetch --quiet --depth=1 origin "$CUTLASS_COMMIT"
git checkout --quiet "$CUTLASS_COMMIT"
docker exec comfyui mkdir -p "$CONTAINER_WORK/csrc/cutlass"
tar -C "$CUTLASS_HOST_DIR" -cf - --exclude=.git . \
    | docker exec -i comfyui tar -C "$CONTAINER_WORK/csrc/cutlass" -xf -
echo "  CUTLASS vendored"

# --- Step 5: Build & install inside the container --------------------------
#
# Note on bash -c quoting: docker exec is called with bash -c followed
# by a SINGLE-quoted argument. This protects against inner double
# quotes (e.g. in user comments or example env-var values) being
# parsed by the host shell. We pass host-side variable values
# ($CONTAINER_WORK, $FLASH_ATTN_CUDA_ARCHS, $NVIDIA_CU13_INCLUDE,
# $NVIDIA_CU13_LIB) as positional arguments to bash so they are
# interpolated by the host shell, then re-exported inside.

echo "[5/7] Building and installing (this takes ~8 min for sm_75 only)..."
docker exec comfyui bash -c '
    cd "$1"
    export FLASH_ATTN_CUDA_ARCHS="$2"
    export FLASH_ATTENTION_FORCE_BUILD=TRUE
    export CPATH="$3"
    export LIBRARY_PATH="$4"
    /opt/venv/bin/pip install --no-build-isolation --no-deps -e . 2>&1
' bash "$CONTAINER_WORK" "$FLASH_ATTN_CUDA_ARCHS" "$NVIDIA_CU13_INCLUDE" "$NVIDIA_CU13_LIB"

# --- Step 6: Verify import -------------------------------------------------

echo "[6/7] Verifying import..."
docker exec comfyui /opt/venv/bin/python3 -c '
from flash_attn import flash_attn_func
print("flash_attn_func imported OK")
print("module file:", flash_attn_func.__module__)
'

# --- Step 7: Sanity-check the sm_75 kernel compiled ------------------------

echo "[7/7] Sanity-checking sm_75 kernel is in the build..."
if docker exec comfyui /opt/venv/bin/python3 -c '
import torch
from flash_attn import flash_attn_func
q = torch.randn(1, 1, 64, 64, dtype=torch.float16, device="cuda")
k = torch.randn(1, 1, 64, 64, dtype=torch.float16, device="cuda")
v = torch.randn(1, 1, 64, 64, dtype=torch.float16, device="cuda")
out = flash_attn_func(q, k, v)
print("sm_75 forward OK, output shape:", tuple(out.shape))
' 2>&1 | tail -5; then
    echo "  sm_75 kernel is callable"
else
    echo "WARNING: sm_75 sanity check failed. The build is installed but" >&2
    echo "the kernel may not be reachable. Run:" >&2
    echo "  docker exec comfyui /opt/venv/bin/python3 -c \"import torch; print(torch.cuda.get_device_capability())\"" >&2
    exit 1
fi

echo
echo "Done. --use-flash-attention should now work on sm_75."
echo "Add --use-flash-attention to COMFYUI_ARGS in compose.yaml and restart:"
echo "  cd /opt/comfyui && docker compose restart"
