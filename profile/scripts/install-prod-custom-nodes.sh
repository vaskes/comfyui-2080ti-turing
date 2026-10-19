#!/bin/bash
# Install the production custom_nodes from /opt/comfyui_bad/custom_nodes/
# to /opt/comfyui/custom_nodes/ (which is bind-mounted into both
# production containers at /opt/ComfyUI/custom_nodes/).
#
# The image already has the pip deps baked (see profile/Dockerfile.production),
# so this just rsyncs the code. Both instances pick up the nodes
# immediately on the next `docker compose up -d` or `docker restart`.
#
# Run once per host. Re-runnable; rsync preserves anything already
# present that isn't in the source.
set -e

SRC=/opt/comfyui_bad/custom_nodes
DST=/opt/comfyui/custom_nodes

if [ ! -d "$SRC" ]; then
  echo "[install-prod-nodes] ERROR: $SRC not found." >&2
  echo "[install-prod-nodes] This is the vaskes-specific prod custom_nodes" >&2
  echo "[install-prod-nodes] stash on this host. Check the path or copy the" >&2
  echo "[install-prod-nodes] nodes manually." >&2
  exit 1
fi

mkdir -p "$DST"
echo "[install-prod-nodes] rsyncing $SRC -> $DST"
rsync -av "$SRC/" "$DST/"

echo
echo "[install-prod-nodes] DONE. $(ls "$DST" | wc -l) custom_nodes in $DST"
echo "[install-prod-nodes] Restart containers to pick up new nodes:"
echo "  cd /opt/comfyui-git && bash profile/scripts/restart-both.sh"
