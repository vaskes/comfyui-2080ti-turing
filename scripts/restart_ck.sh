#!/bin/bash
# Restart ComfyUI with production contract (v1.0.5+).
# The compose.yaml has the working defaults baked in, so a plain
# `docker compose up -d` is enough. This script is a convenience wrapper.
set -e
cd /opt/comfyui-git
docker compose down
docker compose up -d
sleep 8
for i in 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15; do
  if docker exec comfyui curl -m 3 -sf http://127.0.0.1:8188/system_stats >/dev/null 2>&1; then
    echo "ready after ${i}*2s"
    break
  fi
  sleep 2
done
echo "=== Cmdline ==="
docker exec comfyui sh -c 'tr "\0" " " < /proc/1/cmdline' 2>&1
