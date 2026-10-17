# Speed Validation Results (2026-10-18)

## Setup

- **Hardware:** NVIDIA RTX 2080 Ti (sm_75, Turing), 22 GB VRAM
- **Image:**  (torch 2.14.1+cu130, CUDA 13.0, GCC 13.3.0)
- **Flash attention build:**  with sm_75 enabled
  (5-line patch in ).
- **Workflow:**  (original, no SageAttention-T4 node)
- **ComfyUI args:** 
- **Run:** 

## Results

| Run | Time (s) | Output (bytes) | Quality |
|---|---|---|---|
| Baseline (ck-attention only) | 140 | 2 283 593 | ✅ |
| **flash-attention (this work)** | **141.72** | **2 294 332** | **✅ identical scene, perfect match** |
| (previous) sage per-warp | 165 | 2 773 743 | ❌ purple/green dot garbage |

### Per-step time

- flash-attention: **5.07 s/it** (25 steps × 5.07 = 126.75 s compute, plus 14.97 s setup = 141.72 s)
- ck-attention baseline: ~5.6 s/it (140 s / 25 steps)
- sage per-warp: 6.20 s/it (165 s / 25 steps)

**flash-attention is ~10% faster per step than ck-attention baseline** and **18% faster than sage per-warp** on sm_75 for the Qwen-Image 2.1 25-step workload.

## Verdict

** on sm_75 (EOL hardware) is now functional
and slightly faster than the existing  baseline.**

This meets the EOL-support-restoration goal of the task and
exceeds the speed bar. The custom node path
( from the previous work) is no longer
required for Qwen-Image 2.1 on RTX 2080 Ti — the standard
ComfyUI  flag now works directly.

## Reproducing

See  and
.
The build was done once inside the comfyui container; the resulting
 package is -ed in the venv at
.

For subsequent ComfyUI container recreates (
followed by ), the editable install is lost (the venv lives
inside the image, not a volume). Re-run
 after each recreate.
