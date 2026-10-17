# Credits

This work would not exist without the contributions of many people,
organizations, and open-source projects. The following credits reflect the
contributions of each role and external dependency as of the most recent
release of this repository.

## Project Roles

| Role | Person / Entity | Contribution |
|---|---|---|
| **Project Lead / Hardware & Infrastructure** | **Vaskes** (Vibecoder) | Hardware provisioning, infrastructure, system administration, deployment, integration testing, project direction, licensing decisions |
| **Vibecoder** | **Vaskes** | Workflow design, model curation, prompt engineering, end-to-end validation, integration of attention backends, QA |
| **AI Developer / Implementation Agent** | **Mavis** (agent), powered by **MiniMax M3** | Dockerfile authoring, build pipeline, attention shim, debugging, validation scripts, repository maintenance, documentation |
| **Copyright Holder** | **Kuduza Ai Lab** (https://kuduza.com) | Proprietary license steward |

Vaskes is the human vibecoder who designs the workflows, supplies the
hardware, validates end-to-end behavior, and makes the final go/no-go calls.
Mavis is the agentic developer executing the implementation under Vaskes'
direction. The model powering Mavis is MiniMax M3.

## Why "Vibecoder"?

Vaskes prefers the term *vibecoder* over *developer*: a vibecoder is a
hands-on technical decision-maker who shapes the architecture, picks the
components, runs the experiments, and decides when a result is good enough
to ship. Vaskes does not write every line of code in this repository but
makes the calls that determine what gets built, how it gets tested, and
whether it ships.

## Acknowledgments

Special thanks to:

- **ssiu** for the original `flash-attention-turing` kernel port that made
  Turing-specific attention research possible (even though that specific
  port was unsuitable for bf16 models on this hardware).
- **KLM-corporation / farnghwai** for the more comprehensive Tri-Dao-style
  `flash-attention-2080ti` port, which validated the build pipeline and
  provided a reference for the sm_75 + bf16 hardware limitation.
- **Tri Dao** and the **Dao-AILab** team for the original FlashAttention /
  FlashAttention-2 papers and reference implementation, without which none of
  the sm_75 ports would have existed.
- The **comfyanonymous** team for the ComfyUI reference implementation that
  this entire stack is built around.
- The **ltdrdata** project for the ComfyUI-Manager extension that provides
  the comfy_kitchen attention backend used here.
- The **NVIDIA CUTLASS** team for the foundational template library used
  by every flash-attention port.
- The **PyTorch** core team for the framework that makes any of this
  practical at all.
- The **NVIDIA** driver and CUDA toolkit teams for the actual silicon and
  software that the kernels run on.

## Third-Party Components

The full list of third-party open-source components, their versions, and
their respective licenses is documented in [`NOTICE.md`](NOTICE.md).

---

*Kuduza Ai Lab — https://kuduza.com*
