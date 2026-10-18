# Credits — `sage-attention-sm75/`

## This work

- **Vaskes** — Vibecoder, hardware & infrastructure, project lead, integration QA. Maintainer of `vaskes/comfyui-2080ti-turing` (MIT).
- **Mavis** — AI implementation agent, powered by MiniMax M3. Wrote the integration docs and validation scripts.

## Upstream: thu-ml/SageAttention

SageAttention 1.0.6 is the upstream component; this subdirectory is a
**thin wrapper** that documents the sm_75 + Triton 3.x deployment path
on RTX 2080 Ti. The upstream project is consumed at runtime via
`pip install sageattention==1.0.6` (idempotent install baked into the
container entrypoint at `/usr/local/bin/comfyui-entrypoint.sh`).

- **Repository:** https://github.com/thu-ml/SageAttention
- **License:** Apache-2.0
- **Paper:** Jintao Zhang et al., "SageAttention: Accurate 8-Bit
  Attention for Plug-and-play Inference Acceleration", arXiv:2410.02367

### Authors (per upstream)

- **Jintao Zhang** — primary author
- **Jia Wei** — co-author
- **Pengle Zhang** — co-author
- **Xiaoming Xu** — co-author
- **Haofeng Huang** — co-author
- **Yujie Qian** — co-author
- **Yang (Ted) Gao** — co-author
- **Christoph Meinel** — advisor
- **Jianfei Chen** — corresponding author

Plus community contributors acknowledged in the upstream
[`AUTHORS`](https://github.com/thu-ml/SageAttention/blob/main/AUTHORS)
file.

## Tools and infrastructure (not new for this work)

- **ComfyUI** — © 2023–present, comfyanonymous. GPL-3.0.
- **comfy_kitchen** — © 2024–present, comfyui-kitchen. Apache-2.0 (bundled with ComfyUI-Manager).
- **PyTorch** — © 2016–present, Meta. BSD-3-Clause.
- **Triton** — © 2018–present, Philippe Tillet et al. MIT.
- **Kijai Sage Attn patch** — © Kijai (custom node workflow).

## Test models and workflows (vaskes hardware testbed)

The end-to-end tests in `tests/test_backends.py` and the matrix runner
at `tests/matrix/run_matrix.py` exercise SageAttention 1.0.6 on the
following real models and workflows on the workstation:

- **Qwen-Image 2.1** by Alibaba Qwen team. Apache-2.0.
  - Diffusers / ComfyUI workflow: `/opt/comfyui/workflows/image_qwen_image_2_1_t2i_api.json`
  - Models: `/opt/comfyui/models/diffusion_models/qwen_image_2.1/`
- **Minimax H3** by MiniMax. Apache-2.0.
  - ComfyUI workflow: `/opt/comfyui/workflows/video_minimax_h3_t2v_api.json`
  - Models: `/opt/comfyui/models/diffusion_models/minimax_h3/`

We do not redistribute any model weights. The models are downloaded
directly from their official sources by the user.
