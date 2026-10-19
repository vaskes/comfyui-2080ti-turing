#!/bin/bash
# Restart both production ComfyUI containers.
set -e
cd "$(dirname "$0")/../.."
echo "[restart] instance 1"
docker compose -p comfyui2080-1 -f profile/compose-2080_1.yaml restart
echo "[restart] instance 2"
docker compose -p comfyui2080-2 -f profile/compose-2080_2.yaml restart
echo
echo "[restart] waiting for both to be healthy..."
for i in 1 2 3 4 5 6 7 8 9 10; do
  sleep 3
  H1=$(docker inspect --format='{{.State.Health.Status}}' comfyui2080_1 2>/dev/null || echo "missing")
  H2=$(docker inspect --format='{{.State.Health.Status}}' comfyui2080_2 2>/dev/null || echo "missing")
  echo "  [$((i*3))s] instance 1: $H1, instance 2: $H2"
  if [ "$H1" = "healthy" ] && [ "$H2" = "healthy" ]; then
    echo "[restart] BOTH HEALTHY"
    break
  fi
done
