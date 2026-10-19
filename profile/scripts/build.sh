#!/bin/bash
# Build the production image. Build context is the repo root so we
# can COPY docker/entrypoint.sh from the universal repo.
set -e
cd "$(dirname "$0")/../.."
docker build -f profile/Dockerfile.production \
  -t comfyui-workstation:cu130-torch-stable-gcc-prod .
echo "[build] image: comfyui-workstation:cu130-torch-stable-gcc-prod"
docker images | grep comfyui-workstation
