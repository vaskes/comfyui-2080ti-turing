#!/bin/bash
# ComfyUI entrypoint — baked into the image at /usr/local/bin/comfyui-entrypoint.sh
# Auto-installs SageAttention 1.0.6 + pytest if missing (idempotent).
# Then exec ComfyUI with $COMFYUI_ARGS.

set -e

# SageAttention 1.0.6 — Kijai-patch fallback backend
if ! python3 -c "import sageattention" 2>/dev/null; then
  echo "[entrypoint] Installing SageAttention 1.0.6..."
  pip install sageattention==1.0.6 2>&1 | tail -3
fi

# pytest — for in-container test runs (survives container recreates)
if ! python3 -c "import pytest" 2>/dev/null; then
  echo "[entrypoint] Installing pytest..."
  pip install pytest 2>&1 | tail -3
fi

cd /opt/ComfyUI
exec python3 main.py $COMFYUI_ARGS
