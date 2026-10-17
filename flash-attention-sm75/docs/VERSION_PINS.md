# Exact dependency versions used to build and validate flash-attention-sm75

Pinned to specific commits / versions so the next Mavis session can
reproduce the build byte-for-byte. The host on which this was
built is `workstation` (RTX 2080 Ti, sm_75); the container image
is `comfyui-workstation:cu130-torch-stable-gcc` (this repo).

## System

| Component | Version | Notes |
|---|---|---|
| Host OS | Ubuntu 26.04 | 5.15+ kernel |
| Container | `comfyui-workstation:cu130-torch-stable-gcc` | this repo, image tag built 2025-12 |
| Python | 3.12.3 | in-container `/opt/venv/bin/python3` |
| GCC / G++ | 13.3.0 | required for Triton JIT (lower versions fail) |
| NVCC | 13.0.88 (CUDA toolkit 13.0) | `/usr/local/cuda/bin/nvcc` |
| Driver | 615.71.09 (apt `nvidia-driver-615-open`) | supports up to CUDA 13.0 |

## Pip packages (relevant)

| Package | Version | Notes |
|---|---|---|
| `torch` | **2.14.1+cu130** | the build target; `at::symint::sizes<T>` works in C++20 only |
| `triton` | 3.8.0 | for pytorch bmm kernels; sm_75 dispatch disabled by comfy_kitchen by design |
| `torchaudio` | 2.11.0+cu130 | unused in our path, included for completeness |
| `torchvision` | 0.29.1+cu130 | ditto |
| `torchsde` | 0.2.6 | ditto |
| `comfy-kitchen` | **0.2.38** | the ck-attention backend we compare against; has `sol_attn` capability |
| `comfyui-frontend-package` | 1.55.17 | web UI |
| `comfyui-workflow-templates` | 0.11.87 | matches ComfyUI commit 5cc4b87 |
| `comfyui_frontend_package` | 1.55.17 | ditto |
| `comfyui-aimdo` | 0.5.6 | unused |
| `comfyui-angle` | 0.1.1 | unused |

## nvidia cu13 wheel packages (system)

| Package | Version | Why we need it |
|---|---|---|
| `nvidia-cuda-runtime-cu13` | 13.0.96 | base CUDA runtime |
| `nvidia-cudnn-cu13` | **9.24.0.43** | cudnn library |
| `nvidia-cusparse-cu13` | 12.6.3.3 | cuSparse headers; the `libcusparse-dev-13-0` apt package does not exist, only `libcusparse-dev-12-5`/`12-6` (verified 2026-10-17). The nvidia cu13 wheel is the only source of `cusparse.h` on Ubuntu 24.04 |
| `nvidia-cublas-cu13` | 13.1.1.3 | cuBLAS |
| `nvidia-cufft-cu13` | 12.0.0.61 | cuFFT |
| `nvidia-curand-cu13` | 10.4.0.35 | cuRAND |
| `nvidia-cusolver-cu13` | 12.0.4.66 | cuSOLVER |
| `nvidia-cuda-nvrtc-cu13` | 13.0.88 | NVRTC |
| `nvidia-cuda-cupti-cu13` | 13.0.85 | profiling |
| `nvidia-nccl-cu13` | 2.30.7 | multi-GPU (unused, single-GPU in this build) |
| `nvidia-nvjitlink-cu13` | 13.3.33 | JIT linking for runtime compiled kernels |
| `nvidia-nvshmem-cu13` | 3.4.5 | multi-GPU shmem (unused) |
| `nvidia-nvtx-cu13` | 13.0.85 | profiling |

These wheels live at
`/opt/venv/lib/python3.12/site-packages/nvidia/cu13/{include,lib}` inside
the comfyui container. `setup.py` for flash-attention must be
extended with `include_dirs=[NVIDIA_CU13_INCLUDE]` and
`library_dirs=[NVIDIA_CU13_LIB]` (the patch in
`scripts/apply-patch-and-install.sh` does this).

## Upstream dependencies

| Repo | Commit | Pinned by |
|---|---|---|
| `Dao-AILab/flash-attention` | **965c921** | `patches/0001-add-sm75-arch.patch` is generated against this commit; do not move without regenerating |
| `NVIDIA/cutlass` | **df18f5e4** | vendored at `/opt/flash-attention-turing/csrc/cutlass/` on the workstation host. Pinned by content, not by git (we copied the directory, not the repo state). If you re-clone from GitHub you may get a different commit; verify `CHANGELOG.md` and pin in `scripts/apply-patch-and-install.sh` if you do. |
| ComfyUI | `5cc4b87` | `chore: update workflow templates to v0.11.87 (#17110)` |
| `THU-ML/SageAttention` | (2.1.1, sm75 fork) | already built into `custom_nodes/SageAttention-T4/` of this repo; not used for the `--use-flash-attention` path but kept installed for the per-warp legacy path |

## Built artifacts

| Artifact | Where | How |
|---|---|---|
| `flash_attn-2.8.4-*.whl` | `/tmp/pip-ephem-wheel-cache-*/wheels/...` during build | `pip install --no-build-isolation --no-deps -e .` with `FLASH_ATTN_CUDA_ARCHS=75` and C++20 |
| `flash_attn_2_cuda.cpython-312-x86_64-linux-gnu.so` | `/tmp/flash-attention/` (editable install, 11.7 MB) | the actual `.so` that `from flash_attn import flash_attn_func` loads |
| `flash_attn/__init__.py` | `/opt/venv/lib/python3.12/site-packages/flash_attn/` | editable install — the file comes from `/tmp/flash-attention/sageattention/__init__.py` in the build tree, NOT from the source `/opt/venv/lib/python3.12/site-packages/flash_attn/__init__.py` template |

The `pip install --no-build-isolation --no-deps -e .` step **must**
happen with `FLASH_ATTN_CUDA_ARCHS=75` (or `75;80;...` if you
want all archs; the 6-arch build took 90+ minutes, the
sm_75-only build took 8 minutes). The cu13 include path and
library path must be set as env vars or via the include_dirs/
library_dirs arguments in `setup.py` (see
`patches/0001-add-sm75-arch.patch` for the working pattern).

## Why torch 2.14 specifically

The build fails on torch 2.13 and earlier with:

```
error: #error C++20 or later compatible compiler is required to use PyTorch.
```

`torch 2.14` requires C++20 in the host compiler. The image's
GCC 13.3 supports C++20, so this is fine for us. The
`at::symint::sizes<T>` and `c10::strong_ordering` symbols
referenced in `torch/extension.h` are also C++20-only and
would otherwise be missing.

## Why CUDA 13.0 specifically

`flash-attention 2.8.x` (Dao-AILab upstream) builds against
`cuda-toolkit-13-0` (we verified `nvcc -V` reports V13.0.88
during the build). The image has both the apt
`cuda-nvcc-13-0` package and the `nvidia-cu13` pip wheels, so
both compilation (`nvcc`) and runtime (`cudart`, `cudnn`,
`cusparse`, etc.) are available.

We do **not** need `cuda-12-*` for the flash-attention build,
but the older `libcusparse-dev-12-5`/`12-6` apt packages are
listed here for diagnostic purposes only — they were checked
and found to be the wrong major version (we need 13).

## Reproducibility checklist

To rebuild from scratch on a fresh container:

```bash
# 1. apt packages (already in image)
apt list --installed 2>/dev/null | grep -E "cuda-nvcc-13-0|gcc-13|nvidia-driver-615"

# 2. In the comfyui container
docker exec -w /tmp comfyui bash -c '
# CUTLASS workaround (no network to github)
rm -rf /opt/flash-attention-turing   # only if you need to re-vendor
# (CUTLASS is pre-staged on the workstation host at this path;
#  the install script also vendors it from a fresh clone if available)

# 3. Apply patch and build
cd /tmp/flash-attention
git apply /path/to/0001-add-sm75-arch.patch
FLASH_ATTN_CUDA_ARCHS=75 \
FLASH_ATTENTION_FORCE_BUILD=TRUE \
CPATH=/opt/venv/lib/python3.12/site-packages/nvidia/cu13/include \
LIBRARY_PATH=/opt/venv/lib/python3.12/site-packages/nvidia/cu13/lib \
/opt/venv/bin/pip install --no-build-isolation --no-deps -e .

# 4. Verify
/opt/venv/bin/python3 -c "from flash_attn import flash_attn_func; print(flash_attn_func.__doc__[:80])"
```

If the build fails with `cusparse.h: No such file or directory`,
the cu13 wheel path is missing. If it fails with
`at::symint::sizes<T>` or `c10::strong_ordering`, the
torch/C++20 combo is wrong.
