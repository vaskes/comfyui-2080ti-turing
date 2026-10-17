# Notice — Third-Party Components

This document lists the third-party open-source components that the Work
incorporates, depends on, or otherwise interacts with, along with each
component's version, source, and license. It exists to satisfy attribution
and notice obligations imposed by the open-source licenses of those
components and to make clear which parts of the stack are proprietary
(Kuduza Ai Lab) versus which parts are governed by external licenses.

The Work itself (Dockerfile, compose.yaml, README, CREDITS, LICENSE, and
the flash_attn shim) is **proprietary** and is governed by the LICENSE file
in this repository. Nothing in this NOTICE file grants any rights to the
Work itself; rights to third-party components are governed by the licenses
listed below.

## 1. Core Runtime Components

| Component | Version | Source | License |
|---|---|---|---|
| **ComfyUI** | master (tracking 0.41.0) | https://github.com/comfyanonymous/ComfyUI | **GPL-3.0** |
| **ComfyUI-Manager** | main (tracking `comfy_kitchen` 0.2.38) | https://github.com/ltdrdata/ComfyUI-Manager | GPL-3.0 (inherited from ComfyUI ecosystem) |
| **comfy_kitchen** | 0.2.38 (bundled with ComfyUI-Manager) | https://github.com/comfyanonymous/comfy3d-research | see upstream repo (no explicit top-level license; treat as proprietary to comfy3d unless stated otherwise) |
| **PyTorch** | 2.14.1+cu130 | https://pytorch.org / https://github.com/pytorch/pytorch | **BSD-3-Clause** |
| **torchvision** | matching pytorch | https://github.com/pytorch/vision | **BSD-3-Clause** |
| **torchaudio** | matching pytorch | https://github.com/pytorch/audio | **BSD-3-Clause** |

## 2. Attention / Kernel Components

| Component | Version | Source | License |
|---|---|---|---|
| **flash_attn** (KLM-corporation build) | 2.7.3 (sm_75 port) | https://github.com/KLM-corporation/flash-attention-2080ti (branch `2080ti-port`) | **BSD-3-Clause** (inherited from upstream Dao-AILab/flash-attention) |
| **flash-attention** (Dao-AILab, reference) | n/a (build only) | https://github.com/Dao-AILab/flash-attention | **BSD-3-Clause** |
| **flash-attention-turing** (ssiu) | n/a (build only) | https://github.com/ssiu/flash-attention-turing | License not declared in repository; **treated as "no license, all rights reserved" by default**. Used only as a reference for the sm_75 build pipeline. |
| **SageAttention** (planned, not currently built) | n/a | https://github.com/thu-ml/SageAttention | **Apache-2.0** |
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
| **Python** | 3.12.3 (Ubuntu 24.04) | https://www.python.org | **PSF-2.0** (Python Software Foundation) |
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

## 6. Standard Python Libraries (used in flash_attn_turing shim and the comfyui_run.py script)

| Library | Purpose | License |
|---|---|---|
| `urllib` (stdlib) | API client | PSF-2.0 |
| `json` (stdlib) | JSON parsing | PSF-2.0 |
| `pathlib` (stdlib) | Path handling | PSF-2.0 |
| `time` (stdlib) | Timing | PSF-2.0 |
| `argparse` (stdlib) | CLI parsing | PSF-2.0 |

All standard library modules are part of CPython and are governed by the
PSF-2.0 license.

## 7. License Compatibility Summary

The Work is **proprietary** (Kuduza Ai Lab, see `LICENSE`). It uses
open-source components under the following licenses:

- **GPL-3.0** (ComfyUI, gcc, build-essential, possibly ComfyUI-Manager):
  permits use, modification, and distribution in source form **provided**
  the resulting work is also distributed under GPL-3.0 (or compatible).
  Combining GPL-3.0 components with the proprietary Work is permitted
  because the Work does not statically link or distribute the GPL-3.0
  source — users obtain the GPL-3.0 components independently from their
  upstream sources and combine them in their own deployment.

- **BSD-3-Clause** (PyTorch, torchvision, torchaudio, CUTLASS, flash_attn):
  permits redistribution in source or binary form, with or without
  modification, **provided** the copyright notice and license text are
  preserved. The Work's NOTICE.md and accompanying attribution satisfy
  this requirement.

- **MIT** (Triton, composable_kernel, pip, setuptools, wheel): same as
  BSD-3-Clause but with shorter notice.

- **Apache-2.0** (nvidia-container-toolkit, nvidia-container-runtime, Docker,
  Docker Compose, ninja, SageAttention): permits redistribution under a
  similar attribution + notice regime. SageAttention is **not currently
  built** but is on the roadmap (see Section 2).

- **NVIDIA EULAs** (driver, CUDA toolkit, cuBLAS, cuDNN, cuFFT, cuRand,
  base image): proprietary EULAs. The Work's use of these components is
  covered by the user's prior acceptance of the corresponding NVIDIA EULAs
  on the host system. No NVIDIA source is redistributed; NVIDIA binaries
  are pulled from official NVIDIA package repositories.

## 8. Derivative Works

If you create a derivative work of the Work and distribute it, you must:

1. Comply with the Kuduza Ai Lab proprietary LICENSE (Section 5 of LICENSE).
2. Comply with the open-source license terms of every third-party component
   you distribute or statically link, particularly:
   - **GPL-3.0** (ComfyUI, gcc, build-essential): provide complete
     corresponding source code for any GPL-3.0 component, or a written
     offer to provide such source, valid for at least three years.
   - **BSD-3-Clause / MIT / Apache-2.0**: preserve copyright notices and
     license texts.
3. Not use the names "Kuduza Ai Lab", "Kuduza", or associated trademarks
   in any way that suggests endorsement of your derivative work without
   separate written permission.

## 9. Updates to This Notice

This NOTICE.md reflects the dependency state at the time of each release.
To audit the actual current state, run inside the container:

```bash
docker exec comfyui pip list
docker exec comfyui dpkg -l | grep -E "(nvidia|cuda|gcc|build-ess)"
docker exec comfyui nvidia-smi --query-gpu=driver_version --format=csv
```

---

*Maintained by Vaskes for Kuduza Ai Lab — https://kuduza.com*
