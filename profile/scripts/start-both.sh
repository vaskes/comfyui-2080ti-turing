#!/bin/bash
# Start both ComfyUI production instances.
set -e
cd "$(dirname "$0")/../.."

echo "[start] instance 1 (RTX 2080 Ti #1, port 8188)"
docker compose -p comfyui2080-1 -f profile/compose-2080_1.yaml up -d

echo
echo "[start] instance 2 (RTX 2080 Ti #2, port 8189)"
docker compose -p comfyui2080-2 -f profile/compose-2080_2.yaml up -d

echo
echo "[start] waiting for both to be healthy..."
for i in 1 2 3 4 5 6 7 8 9 10; do
  sleep 3
  H1=$(docker inspect --format='{{.State.Health.Status}}' comfyui2080_1 2>/dev/null || echo "missing")
  H2=$(docker inspect --format='{{.State.Health.Status}}' comfyui2080_2 2>/dev/null || echo "missing")
  echo "  [$((i*3))s] instance 1: $H1, instance 2: $H2"
  if [ "$H1" = "healthy" ] && [ "$H2" = "healthy" ]; then
    echo "[start] BOTH HEALTHY"
    break
  fi
done

echo
echo "[start] ports:"
ss -tlnp 2>/dev/null | grep -E ':(8188|8189)' || netstat -tlnp 2>/dev/null | grep -E ':(8188|8189)' || true
echo
echo "[start] access:"
echo "  instance 1: http://127.0.0.1:8188"
echo "  instance 2: http://127.0.0.1:8189"
