#!/bin/bash
# Stop both ComfyUI production instances.
set -e
cd "$(dirname "$0")/../.."

echo "[stop] instance 1"
docker compose -p comfyui2080-1 -f profile/compose-2080_1.yaml down
echo
echo "[stop] instance 2"
docker compose -p comfyui2080-2 -f profile/compose-2080_2.yaml down
echo
echo "[stop] DONE"
