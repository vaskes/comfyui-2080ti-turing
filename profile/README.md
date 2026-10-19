# Profile: Vaskes's 2× RTX 2080 Ti production setup

This directory is **Vaskes-specific production config** that uses the
universal `comfyui-2080ti-turing` repo as its base. The main repo
(Dockerfile at the root, single-instance `compose.yaml`, attention
backend matrix, fragmentation fix) is generic and works for any
RTX 2080 Ti / sm_75 workstation. **This profile is not generic.**

The split exists because:

1. **Main repo** (`/`) is the public vaskes/comfyui-2080ti-turing
   release: matrix tests, fragmentation fix, three attention backends,
   documentation. Anyone can clone and use it.

2. **profile/** is Vaskes's specific production deployment: two
   containers (`comfyui2080_1`, `comfyui2080_2`) on one workstation
   (one codebase, two GPU instances). Per-instance output dirs, separate
   ComfyUI-Manager SQLite DBs (`__manager/` lives in each `user*/`),
   shared models / input / custom_nodes.

What lives in `profile/`:

| File | Purpose |
|---|---|
| `Dockerfile.production` | Production image with prod custom_node dependencies baked in. Builds from the same base as the main Dockerfile + adds the pip deps for the 22 custom_nodes in `/opt/comfyui_bad/custom_nodes/`. |
| `compose-2080_1.yaml` | Instance 1: container `comfyui2080_1`, port 8188, GPU 0, `/opt/comfyui-2080_1/{output,user}` volumes. |
| `compose-2080_2.yaml` | Instance 2: container `comfyui2080_2`, port 8189, GPU 1, `/opt/comfyui-2080_2/{output,user}` volumes. |
| `scripts/data-dirs.sh` | Creates the per-instance and shared data directories with correct ownership. |
| `scripts/build.sh` | Builds `comfyui-workstation:cu130-torch-stable-gcc-prod` from this profile. |
| `scripts/start-both.sh` | Starts both instances via docker compose. |
| `scripts/stop-both.sh` | Stops both. |

What lives outside the repo (host paths, not in git):

| Path | Purpose | Shared? |
|---|---|---|
| `/opt/comfyui/models/` | Diffusion, text encoders, VAE, LoRA, GGUF | shared by both instances |
| `/opt/comfyui/input/` | Input images / videos | shared |
| `/opt/comfyui/workflows/` | Workflow JSON files | shared |
| `/opt/comfyui/custom_nodes/` | Custom nodes (cloned here, deps in image) | shared |
| `/opt/comfyui-2080_1/output/` | Instance 1 outputs | per-instance |
| `/opt/comfyui-2080_1/user/` | Instance 1 user data (incl. `__manager/` ComfyUI-Manager DB) | per-instance |
| `/opt/comfyui-2080_2/output/` | Instance 2 outputs | per-instance |
| `/opt/comfyui-2080_2/user/` | Instance 2 user data (incl. `__manager/` ComfyUI-Manager DB) | per-instance |

## Why two instances of ComfyUI-Manager DB?

ComfyUI-Manager stores installed-node state, snapshot history, channels
list, and batch history in a SQLite DB at `__manager/config.ini` +
`__manager/cache/`. If two instances share one DB, races corrupt it
and the "Installed" tab becomes inconsistent.

The prod pattern (mirrored from `/opt/comfyui_bad/user80` + `user90`)
keeps each instance's `__manager/` in its own `user*/` bind-mount.

## Why are container names 2080_1 and 2080_2 (not 2080_0 and 2080_1)?

Future hardware: this server will get a third GPU. `nvidia-smi` reorders
when GPUs are added/removed. Using `1` and `2` as suffixes reserves
slot 0 for the future-added card; existing containers won't slide.

**Current GPU mapping** (only 2 RTX 2080 Ti present in this host):
- `comfyui2080_1` -> GPU 0 in `nvidia-smi` (NVIDIA_VISIBLE_DEVICES=0)
- `comfyui2080_2` -> GPU 1 in `nvidia-smi` (NVIDIA_VISIBLE_DEVICES=1)

When the third card is added (and nvidia-smi renumbers), the existing
two stay mapped to their physical GPUs because the suffix is just a
name, not a GPU index. The `device_ids` in the compose files can be
updated at that point.

## Quick start

```bash
# 1. Create data dirs (one-time)
cd /opt/comfyui-git
bash profile/scripts/data-dirs.sh

# 2. Build production image (one-time, ~30 min)
bash profile/scripts/build.sh

# 3. Start both
bash profile/scripts/start-both.sh

# 4. Verify
curl -m 3 http://127.0.0.1:8188/system_stats   # instance 1
curl -m 3 http://127.0.0.1:8189/system_stats   # instance 2
```

## Production contract (v1.0.5+)

Both instances run with the same attention backend + memory flags:

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
python3 main.py \
  --extra-model-paths-config /opt/ComfyUI/user/extra_model_paths.yaml \
  --listen 0.0.0.0 --port 8188 \
  --lowvram --reserve-vram 0 \
  --enable-triton-backend --enable-manager \
  --use-ck-attention \
  --disable-dynamic-vram \
  --disable-cuda-malloc \
  --disable-smart-memory
```

See root [README.md](../README.md) for the rationale and
[CHANGELOG.md](../CHANGELOG.md) v1.0.5 entry.

## Architecture reference

This profile's layout (separate user/output dirs, shared models +
custom_nodes, two compose files) is mirrored from
`/opt/comfyui_bad/docker-compose.comfyui{80,90}.yml` on this host.
The `comfyui_bad` setup is a single-image, two-instance deployment
of `comfyui-turing:v0.37` for an earlier ComfyUI; this profile
updates the pattern to use the v1.0.7 universal base.
