# Credits — `flash-attention-sm75/` patch

## This work

- **Vaskes** — Vibecoder, hardware & infrastructure, project lead, integration QA. Maintainer of `vaskes/comfyui-2080ti-turing` (proprietary).
- **Mavis** — AI implementation agent, powered by MiniMax M3. Wrote the sm_75 build-matrix patch and the integration tooling.

## Upstream: Dao-AILab/flash-attention

This work is a **derivative work** of the official
[`Dao-AILab/flash-attention`](https://github.com/Dao-AILab/flash-attention)
project. The patch is 5 lines in `setup.py`; the rest of the source
code is upstream and governed by the BSD 3-Clause License (see
`LICENSE` in this directory).

### People (per upstream `AUTHORS` file, commit `965c921`)

- **Tri Dao** — primary author and maintainer of flash-attention. Lead at Together AI.
- **Daniel Y. Fu** — early contributor; co-author of the FlashAttention paper.
- **Stefano Ermon** — advisor; co-author of the FlashAttention paper; Stanford CS faculty.
- **Atri Rudra** — advisor; co-author of the FlashAttention paper; University of Buffalo CS faculty.
- **Christopher Ré** — advisor; co-author of the FlashAttention paper; Stanford CS faculty.
- Plus ~40 community contributors acknowledged in the upstream `AUTHORS` file at
  https://github.com/Dao-AILab/flash-attention/blob/main/AUTHORS

### Papers (cited in the upstream README, not authored by this work)

- FlashAttention (NeurIPS 2022): Tri Dao, Daniel Y. Fu, Stefano Ermon, Atri Rudra, Christopher Ré.
- FlashAttention-2 (NeurIPS 2023): Tri Dao.
- FlashAttention-3 (NeurIPS 2024): Jay Shah, Ganesh Bikshandi, Ying Zhang, Vijay Thakkar, Pradeep Ramani, Tri Dao.
- FlashAttention-4 (2025): Team at Dao-AILab.

## Tools and infrastructure (in this repo, not new for this work)

- **ComfyUI** — © 2023–present, comfyanonymous. GPL-3.0. https://github.com/comfyanonymous/ComfyUI
- **comfy_kitchen** — © 2024–present, comfyui-kitchen. Apache-2.0 (or similar). Bundled with ComfyUI-Manager.
- **PyTorch** — © 2016–present, Meta. BSD-3-Clause. https://github.com/pytorch/pytorch
- **Triton** — © 2018–present, Philippe Tillet et al. MIT. https://github.com/triton-lang/triton
- **CUDA Toolkit + cuDNN** — © NVIDIA. https://developer.nvidia.com/cuda-toolkit
- **CUTLASS** — © NVIDIA. BSD-3-Clause. https://github.com/NVIDIA/cutlass

## Test models and workflows (proprietary to vaskes)

The end-to-end tests in `tests/test_sm75_e2e.py` exercise this patch
on the following real models and workflows on the workstation:

- **Qwen-Image 2.1** by Alibaba Qwen team. Apache-2.0.
  - Diffusers / ComfyUI workflow: `/opt/comfyui/image_qwen_image_2_1_t2i_api.json`
  - Models: `/opt/comfyui/models/diffusion_models/qwen_image_2.1/`
- **Minimax H3** by MiniMax. Apache-2.0.
  - ComfyUI workflow: `/opt/comfyui/video_minimax_h3_t2v_api.json`
  - Models: `/opt/comfyui/models/diffusion_models/minimax_h3/`

We do not redistribute any model weights. The models are downloaded
directly from their official sources by the user.
