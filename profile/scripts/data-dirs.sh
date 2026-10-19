#!/bin/bash
# Create per-instance and shared data directories for the 2x 2080 Ti
# production deployment. Run once on host (not in container).
#
# Layout (mirrored from /opt/comfyui_bad/):
#   /opt/comfyui/                  shared (models, input, workflows, custom_nodes)
#   /opt/comfyui-2080_1/{output,user}  per-instance
#   /opt/comfyui-2080_2/{output,user}  per-instance
set -e

SHARED=/opt/comfyui
INSTANCES=(/opt/comfyui-2080_1 /opt/comfyui-2080_2)

# Shared dirs (must exist with correct owner)
mkdir -p "$SHARED/models" "$SHARED/input" "$SHARED/workflows" "$SHARED/custom_nodes"
echo "[data-dirs] shared: $SHARED/{models,input,workflows,custom_nodes} ready"

# Per-instance dirs
for inst in "${INSTANCES[@]}"; do
  mkdir -p "$inst/output" "$inst/user"
  echo "[data-dirs] instance: $inst/{output,user} ready"
done

# Symlink for ergonomics: /opt/comfyui-2080_X/user/default -> if user wants
# pre-existing user90/__manager etc, they can rsync from /opt/comfyui_bad/.

echo "[data-dirs] DONE"
echo
echo "Optional: clone production custom_nodes from /opt/comfyui_bad/custom_nodes/ to"
echo "  /opt/comfyui/custom_nodes/"
echo "Skip if you only want the 2 production backends (ComfyUI-Manager + SageAttention-T4)."
