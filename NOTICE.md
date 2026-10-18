# Notice — Third-Party Components

This document lists the third-party open-source components that the Work
incorporates, depends on, or otherwise interacts with, along with each
component's version, source, and license. It exists to satisfy attribution
and notice obligations imposed by the open-source licenses of those
components. The Work itself is **MIT licensed** (see [`LICENSE`](LICENSE))
and the upstream components retain their own licenses — there are no
proprietary restrictions in the Work.

## 1. Core Runtime Components

| Component | Version | Source | License |
|---|---|---|---|
| **ComfyUI** | master (tracking 0.41.0) | https://github.com/comfyanonymous/ComfyUI | **GPL-3.0** |
| **ComfyUI-Manager** | main (tracking `comfy_kitchen` 0.2.38) | https://github.com/ltdrdata/ComfyUI-Manager | GPL-3.0 (inherited from ComfyUI ecosystem) |
| **comfy_kitchen** | 0.2.38 (bundled with ComfyUI-Manager) | https://github.com/comfyanonymous/comfy3d-research | see upstream repo (no explicit top-level license) |
| **PyTorch** | 2.14.1+cu130 | https://pytorch.org / https://github.com/pytorch/pytorch | **BSD-3-Clause** |
| **torchvision** | matching pytorch | https://github.com/pytorch/vision | **BSD-3-Clause** |
| **torchaudio** | matching pytorch | https://github.com/pytorch/audio | **BSD-3-Clause** |

## 2. Attention / Kernel Components

| Component | Version | Source | License |
|---|---|---|---|
| **flash_attn** (Dao-AILab, patched for sm_75) | 2.8.4 (sm_75 build) | https://github.com/Dao-AILab/flash-attention | **BSD-3-Clause** |
| **flash-attention-turing** (ssiu) | reference only | https://github.com/ssiu/flash-attention-turing | BSD-3-Clause (inherited from upstream) |
| **SageAttention** | 1.0.6 | https://github.com/thu-ml/SageAttention | **Apache-2.0** |
| **NVIDIA CUTLASS** | submodule tracking (built into flash_attn build) | https://github.com/NVIDIA/cutlass | **BSD-3-Clause** |
| **composable_kernel** (ROCm) | submodule (unused on NVIDIA) | https://github.com/ROCm/composable_kernel | MIT |
| **Triton** | 3.8.0 | https://github.com/triton-lang/triton | **MIT** |

## 3. System / Driver / CUDA

| Component | Version | Source | License |
|---|---|---|---|
| **NVIDIA driver** | 615.71.09 (`nvidia-driver-615-open`) | https://www.nvidia.com/Download/index.aspx | NVIDIA proprietary driver EULA |
| **CUDA Toolkit** | 13.0.88 (runtime libs) + 13.0.96 (cudart-dev) | https://developer.nvidia.com/cuda-toolkit | NVIDIA CUDA Toolkit EULA |
| **NVIDIA cuBLAS / cuDNN / cuFFT / cuRand** | 13.0.x (bundled with runtime image) | https://developer.nvidia.com | NVIDIA CUDA Toolkit EULA |
| **nvidia-container-toolkit** | 1.20.1 | https://github.com/NVIDIA/nvidia-container-toolkit | **Apache-2.0** |
| **NVIDIA Container Runtime** | matching toolkit | https://github.com/NVIDIA/nvidia-container-runtime | **Apache-2.0** |

## 4. Build Toolchain (inside container)

| Component | Version | Source | License |
|---|---|---|---|
| **gcc / g++** | 13.3.0 (Ubuntu 24.04) | https://gcc.gnu.org | **GPL-3.0** (with runtime library exception) |
| **build-essential** | 12.10 (Ubuntu 24.04) | apt package | GPL-3.0 (meta-package) |
| **Python** | 3.12.3 (Ubuntu 24.04) | https://www.python.org | **PSF-2.0** |
| **pip** | 26.2.1 | https://github.com/pypa/pip | **MIT** |
| **ninja** | 1.13.2 | https://ninja-build.org | **Apache-2.0** |
| **setuptools** | 84.0.0 | https://github.com/pypa/setuptools | **MIT** |
| **wheel** | (latest) | https://github.com/pypa/wheel | **MIT** |

## 5. Container / Orchestration

| Component | Version | Source | License |
|---|---|---|---|
| **Docker Engine** | 29.1.3 (`docker.io` on Ubuntu 24.04) | https://www.docker.com / https://github.com/moby/moby | **Apache-2.0** |
| **Docker Compose v2** | 2.40.3 (`docker-compose-v2`) | https://github.com/docker/compose | **Apache-2.0** |
| **nvidia/cuda base image** | `13.0.3-cudnn-runtime-ubuntu24.04` | https://hub.docker.com/r/nvidia/cuda | NVIDIA CUDA Image EULA |

## 6. License Compatibility

The Work is **MIT** ([`LICENSE`](LICENSE)). It depends on open-source
components under various licenses. MIT is permissive and does not impose
copyleft, so this Work can be combined with components under any license,
including GPL-3.0, BSD-3-Clause, MIT, and Apache-2.0. The upstreams' license
terms apply to those components only.

- **GPL-3.0** (ComfyUI, gcc, build-essential): combined here as a runtime
  dependency pulled by the user from upstream. Source not redistributed
  in this Work.
- **BSD-3-Clause** (PyTorch, torchvision, torchaudio, CUTLASS, flash_attn):
  copyright notice + license text preserved per upstream requirements.
- **MIT** (Triton, composable_kernel, pip, setuptools, wheel, SageAttention
  build deps): same as BSD-3-Clause but shorter notice.
- **Apache-2.0** (nvidia-container-toolkit, nvidia-container-runtime, Docker,
  Docker Compose, ninja): attribution + notice preserved.
- **NVIDIA EULAs** (driver, CUDA toolkit, cuBLAS, cuDNN, cuFFT, cuRand,
  base image): apply to the NVIDIA components only. No NVIDIA source is
  redistributed; NVIDIA binaries are pulled from official NVIDIA package
  repositories.

## 7. Derivative Works

You are free to copy, modify, merge, publish, distribute, sublicense, and/or
sell copies of the Work under the terms of the MIT license (see
[`LICENSE`](LICENSE)). When redistributing, the only requirements are:

1. Preserve the copyright notice and license text in the Work and its
   derivatives.
2. Comply with the upstream license terms of any third-party components
   you distribute or statically link, per the licenses listed in Sections
   1-5.
3. The upstream project names (ComfyUI, SageAttention, comfy_kitchen,
   Dao-AILab flash-attention, etc.) remain the trademarks of their
   respective owners.

## 8. Updates to This Notice

This NOTICE.md reflects the dependency state at the time of each release.
To audit the actual current state, run inside the container:

```bash
docker exec comfyui pip list
docker exec comfyui dpkg -l | grep -E "(nvidia|cuda|gcc|build-ess)"
docker exec comfyui nvidia-smi --query-gpu=driver_version --format=csv
```

---

*Maintained by Vaskes for the community — see [`CREDITS.md`](CREDITS.md) for the people behind this work.*
