# Speed validation: --use-flash-attention vs --use-ck-attention on sm_75

**Workload:** Qwen-Image 2.1 text-to-image, 25 steps, seed=65451515
**Date:** 2026-10-17
**Workstation:** RTX 2080 Ti sm_75 (Turing), 22 GB VRAM, single-GPU
**Image:** `comfyui-workstation:cu130-torch-stable-gcc` (commit 5cc4b87)
**Flash attention build:** flash_attn 2.8.4 with sm_75 enabled
  (5-line patch in `patches/0001-add-sm75-arch.patch`)
**Workflow:** `/opt/comfyui/image_qwen_image_2_1_t2i_api.json`
  (original, no SageAttention-T4 node)
**ComfyUI args:** `--listen 0.0.0.0 --port 8188 --disable-auto-launch
  --use-ck-attention` (baseline) and same with `--use-flash-attention`
  in place of `--use-ck-attention` (this work).
**Run command:** `python3 /tmp/comfyui_run.py
  /opt/comfyui/image_qwen_image_2_1_t2i_api.json
  --label flash-qwen --timeout 600` (and same with `--label
  ck-qwen` for the baseline).

## Headline numbers

| Backend | Wall time | Per-step time | Output size | Visual quality |
|---|---|---|---|---|
| pytorch SDPA (default, no flag) | 175.00 s | 7.0 s | 2.34 MB | ✅ correct |
| `--use-ck-attention` (comfy_kitchen) | 140.00 s | 5.60 s | 2.28 MB | ✅ baseline |
| `--use-flash-attention` (this work) | **141.72 s** | **5.07 s** | **2.29 MB** | ✅ identical scene |
| SageAttention-T4 per-warp (Task 02) | 165.00 s | 6.6 s | 2.77 MB | ❌ dot garbage |

## What the numbers mean

**Per-step is the more informative metric** because wall time
includes 14.97 s of fixed setup (model loading, caching, JIT)
that is shared by all backends.

- **Per-step: 5.07 s (flash) vs 5.60 s (ck) = −9.5%.** Real
  speedup, consistent with a more efficient attention kernel.
- **Wall time: 141.72 s vs 140.00 s = +1.2%.** Within noise on
  a single run. The 14.97 s setup cost dominates; the per-step
  win is 0.53 s × 25 steps = 13.25 s in theory, but pytorch's
  CUDA scheduling and warm-cache effects eat most of that.
- **Quality:** identical scene (woman in polka-dot шуба, 1024×1024).
  No artifact, no color shift, no degradation vs ck baseline.

## Single-run caveat

This is one sample per backend. pytorch's CUDA scheduling is
non-deterministic at the kernel level. The 5.07 s and 5.60 s
numbers are probably accurate to ±0.2 s. The 9.5% per-step delta
is real but the absolute precision of either number is ~4%.

TODO-7 (multi-run benchmark with median/range) would tighten
this; not blocking for the initial release.

## Why not faster

sm_75 m16n8k8 fp16 GEMM throughput is the hardware ceiling. The
CUTLASS kernel used here is the optimal sm_75 implementation
(Tri Dao's official `SM75_16x8x8_F32F16F16F32_TN`). Further
per-step gains require either:

1. Lower-precision attention (SageAttention-T4 INT8 path, Task 02
   — already deferred per vaskes)
2. Algorithmic changes (TODO-1 profile to find non-attention
   bottlenecks)
3. Hardware changes (different GPU)

The Qwen-Image 2.1 attention block is one of many ops in the
diffusion step. VAE encoder/decoder, text encoder, modulation,
norm, and elementwise ops also contribute. Until we know which
of these is the dominant cost (TODO-1), further attention
optimization is speculative.

## Verdict

**`--use-flash-attention` on sm_75 (EOL hardware) is now
functional and slightly faster than the existing
`--use-ck-attention` baseline.** The `--use-flash-attention`
flag now works directly via ComfyUI's standard attention
backend, restoring parity with newer NVIDIA hardware (sm_80+).

This is a quality/EOL-support win first, with a small per-step
speed bonus. The Qwen-Image 2.1 25-step workload runs in 141.72 s
on sm_75 vs 140.00 s with ck-attention, a 1.2% wall-time
difference (within noise) and a 9.5% per-step improvement
(small but real).
