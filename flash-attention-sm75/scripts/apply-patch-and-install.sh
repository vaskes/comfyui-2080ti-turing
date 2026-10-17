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

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PATCH_FILE="$REPO_ROOT/patches/0001-add-sm75-arch.patch"
UPSTREAM_REPO="https://github.com/Dao-AILab/flash-attention.git"
UPSTREAM_COMMIT="965c921"  # the commit this patch was generated against
WORK_DIR="/opt/research/flash-attention"

NVIDIA_CU13_INCLUDE="/opt/venv/lib/python3.12/site-packages/nvidia/cu13/include"
NVIDIA_CU13_LIB="/opt/venv/lib/python3.12/site-packages/nvidia/cu13/lib"

# 1. Clone or update upstream
if [ -d "$WORK_DIR/.git" ]; then
    echo "[1/5] Updating existing clone at $WORK_DIR..."
    cd "$WORK_DIR"
    git fetch --depth=1 origin "$UPSTREAM_COMMIT" 2>&1 | tail -3
    git checkout "$UPSTREAM_COMMIT" 2>&1 | tail -3
else
    echo "[1/5] Cloning upstream to $WORK_DIR..."
    mkdir -p "$(dirname "$WORK_DIR")"
    git clone --depth=1 "$UPSTREAM_REPO" "$WORK_DIR" 2>&1 | tail -3
    cd "$WORK_DIR"
    git fetch --depth=1 origin "$UPSTREAM_COMMIT" 2>&1 | tail -3
    git checkout "$UPSTREAM_COMMIT" 2>&1 | tail -3
fi

# 2. Verify the patch applies cleanly
echo "[2/5] Verifying patch applies..."
if ! git apply --check "$PATCH_FILE"; then
    echo "ERROR: patch does not apply cleanly. The upstream HEAD may have moved." >&2
    echo "Expected commit: $UPSTREAM_COMMIT" >&2
    echo "Actual HEAD:    $(git rev-parse HEAD)" >&2
    exit 1
fi
git apply "$PATCH_FILE"
echo "  patch applied."

# 3. Run pip install inside the comfyui container
echo "[3/5] Running pip install inside the comfyui container..."
docker exec -w "$WORK_DIR" comfyui \
    bash -c "
        export FLASH_ATTN_CUDA_ARCHS="${FLASH_ATTN_CUDA_ARCHS:-75}"  # default: sm_75 only (Turing-only hosts). For heterogeneous GPUs, set to e.g. "75;80;90" (will take ~90 min to build)
        export FLASH_ATTENTION_FORCE_BUILD=TRUE
        export CPATH='$NVIDIA_CU13_INCLUDE'
        export LIBRARY_PATH='$NVIDIA_CU13_LIB'
        /opt/venv/bin/pip install --no-build-isolation --no-deps -e . 2>&1
    " | tail -20

# 4. Verify import works
echo "[4/5] Verifying import..."
docker exec comfyui /opt/venv/bin/python3 -c "
from flash_attn import flash_attn_func
print('flash_attn_func imported OK')
print('module file:', flash_attn_func.__module__)
" 2>&1

# 5. Done
echo "[5/5] Done. --use-flash-attention should now work on sm_75."
echo "      Add --use-flash-attention to COMFYUI_ARGS in compose.yaml and restart:"
echo "        cd /opt/comfyui && docker compose restart"
