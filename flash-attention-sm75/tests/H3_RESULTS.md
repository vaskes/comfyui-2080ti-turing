# H3 e2e test results — flash-attention-sm75 v0.2

**Date:** 2026-10-18
**Workload:** Minimax H3 video generation, 8 steps (turbo), seed=auto
**Workflow:** `/opt/comfyui/video_minimax_h3_t2v_api.json`
**Models:**
- UNET: `minimax_h3_fl2va_pruned_int8_convrot.safetensors` (19.5 GB, int8 convrot)
- CLIP: `qwen3vl_32b_h3_ultra_uncensored_heretic_int8_convrot.safetensors` (14.6 GB, int8 convrot)
- LoRA: `minimax_h3_fl2v_turbo_8step_v1.0_comfyui_bf16.safetensors` (1.9 GB, bf16)
- VAE: video (5.0 GB fp16) + audio (0.6 GB fp32)
**Workstation:** RTX 2080 Ti sm_75 (Turing), 22 GB VRAM, single-GPU
**ComfyUI:** 0.41.0 with `comfy_kitchen 0.2.38` + `flash_attn 2.8.4` (sm_75 patched)

## TL;DR

**The sm_75 flash patch works for Qwen-Image 2.1 (bf16/fp16 weights)
but produces checkerboard garbage for Minimax H3 (int8 convrot
weights). The H3 e2e test was a critical gap that Mavis review
didn't catch. v0.2 is therefore released as "bf16/fp16 weights
only" with a known-issue note for int8 convrot models.**

## Headline numbers

| Backend | Wall time | Output size | Bitrate | Visual quality |
|---|---|---|---|---|
| `--use-ck-attention` (comfy_kitchen int8 convrot) | 116.0 s | 538 KB | 2.6 Mbps | ✅ correct, cinematic chase scene |
| `--use-flash-attention` (this work) | 111.0 s | 3.8 MB | 18.7 Mbps | ❌ **checkerboard garbage** |

The flash run is **4.3% faster** but produces **kernel-level
corruption**: every frame is a noise pattern with no recognizable
content. The high bitrate (18.7 Mbps vs 2.6 Mbps) is a symptom:
the encoder is trying to faithfully compress the garbage, not a
sign of better visual quality.

## Detailed measurements

### Video metadata (both runs)

```
codec: h264 (video) + aac (audio)
width: 864
height: 480
framerate: 24 fps
nb_frames: 39 (video) + 52 (audio)
duration: 1.625 s
```

### Per-frame compressed size

| Backend | Total bytes | Per frame |
|---|---|---|
| ck baseline | 538 374 | 13 804 bytes/frame |
| flash | 3 806 415 | 97 600 bytes/frame |

The 7x larger per-frame size in the flash run is a
**quality-anti-signal**: H.264 only uses that much data when the
content has very high entropy (because the residual after
prediction is large). Garbage content has high entropy; cinematic
content has low entropy.

## Why flash fails on H3 but works on Qwen

H3 uses **int8 convrot weights** (rotated + per-channel INT8
quantization) for both the UNET and the text encoder. Qwen-Image
2.1 uses **bf16/fp16 weights** (the sm_75 patch was tested
against Qwen's bf16 storage).

The CUTLASS kernel we enable is
`SM75_16x8x8_F32F16F16F32_TN`:

- `F32` accumulator (fp32 throughout the QK^T and softmax)
- `F16` `F16` input operands (Q, K, V are fp16)
- `F32` output

This is the **only** CUTLASS sm_75 specialization in upstream
flash-attention. It expects fp16 Q/K/V. When H3 feeds it
int8 convrot-rotated values, the rotation has been applied
*before* the attention call (in the linear projection), so the
tensors are nominally `dtype=float16` — but the **values are
not representable in fp16** with normal magnitude. The fp16
rounding before the CUTLASS mma corrupts the result.

The community forks (`ssiu/flash-attention-turing`,
`KLM-corporation/flash-attention-2080ti`) use the m8n8k8 PTX
with fp16 accumulator and have the same problem. The official
upstream uses the right fp32-accumulator design but only has
the fp16-input specialization.

**There is no sm_75 CUTLASS kernel for int8 convrot inputs in
the upstream flash-attention code.** Patching this would
require either:
- Adding a new CUTLASS template instantiation
  (`SM75_16x8x8_INT8INT8INT32_TN` or similar) — significant
  work, may not even be possible on sm_75 because INT8 mma
  throughput on Turing is poor (no dedicated INT8 tensor cores
  like sm_80+ has).
- Dequantizing convrot weights to fp16 in a wrapper before the
  attention call — would add compute cost and break the
  "no host-side dequant" assumption of the convrot path.

**Recommendation:** don't try to make flash work on H3 int8
convrot. Use `--use-ck-attention` (comfy_kitchen's int8 path)
for H3 and any other int8 convrot model. Use
`--use-flash-attention` for bf16/fp16 models like Qwen-Image
2.1.

## How to reproduce

```bash
# Baseline (ck-attention, works correctly):
# - In compose.yaml, set COMFYUI_ARGS to "... --use-ck-attention"
# - docker compose up -d
# - Submit H3 workflow to /prompt, time it
# Result: 116 s, 538 KB, correct video

# Flash (this work, broken on H3):
# - In compose.yaml, set COMFYUI_ARGS to "... --use-flash-attention"
# - docker compose up -d
# - Submit H3 workflow to /prompt, time it
# Result: 111 s, 3.8 MB, checkerboard garbage
```

The two runs are saved at:
- `tests/results/h3_ck_116s.mp4` (correct, Batman chase)
- `tests/results/h3_flash_111s.mp4` (broken, checkerboard)

Sample frames at:
- `tests/results/frames/h3_ck_116s_f01.png` (correct)
- `tests/results/frames/h3_ck_116s_f02.png` (correct)
- `tests/results/frames/h3_flash_111s_f01.png` (garbage)
- `tests/results/frames/h3_flash_111s_f02.png` (garbage)

## Implication for v0.2 release

The kernel work itself is correct for bf16/fp16 weights. The H3
gap was not caught by either Qwen or Mavis review because:

1. **Qwen review (v0.1, A+/A/A)** evaluated the patch correctness
   and the unit test correctness against a SDPA reference. The
   reference is also fp16-input, so the unit test passes. Qwen
   did not test on int8 convrot because Qwen's workload is
   bf16/fp16 throughout.

2. **Mavis review (v0.2, REQUEST CHANGES → fixed)** focused on
   the install script and doc drifts. Mavis added a T6 stress
   test (scale=1.0, L=4096) which is also fp16-input. Neither
   reviewer had access to a real int8 convrot workflow to
   validate against.

3. **The TODO-4 (H3 e2e) was explicitly deferred** in
   `M4.md` ("E2E Qwen ✅, E2E H3 ⏳ pending (deferred to TODO-4)")
   and in the README "Validation status" table. We ran the H3
   e2e as part of pre-release validation, and that's where the
   failure surfaced.

**Decision:** v0.2 ships with a "bf16/fp16 weights only" note in
the README. M4 (H3) is closed with the caveat that H3 requires
ck-attention. TODO-1 (profile) and the per-workflow dispatcher
work are deferred to v0.3. ck-attention is restored as the
default `COMFYUI_ARGS` because it's the only attention that
works for both Qwen and H3 on sm_75 today.
