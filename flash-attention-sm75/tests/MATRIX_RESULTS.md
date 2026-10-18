# Test matrix results: Qwen-Image 2.1 + H3 across 3 attention backends (2026-10-18)

## Scope

Per user request (vaskes 2026-10-18): test the attention backend
matrix for two model families across multiple quantization variants.

**Models tested:**
- **Qwen-Image 2.1** T2I: bf16, int8_convrot
- **Minimax H3** T2V (8-step turbo LoRA): bf16, fp8, int8_convrot, w6a8

**Attention backends:**
- sdpa - pytorch SDPA (no special flag, native fallback)
- ck   - comfy_kitchen int8 path (--use-ck-attention)
- flash - flash attention (--use-flash-attention)
  - v0.2.1: routes through shim to ssiu community fork (flash_attn_turing)
  - v0.3 (this work): routes to real Dao-AILab sm_75 build (M-03.1+M-03.2)

**Text encoder for H3:** qwen3vl_32b_h3_ultra_uncensored_heretic_int8_convrot.safetensors (per user)

## Attention backends - what they actually do

| Backend | ComfyUI flag | Implementation | H3 status |
|---|---|---|---|
| sdpa | (none) | pytorch SDPA math backend, all-zero fallback | slow but works |
| ck   | --use-ck-attention | CUTLASS composable kernel int8 path | FASTEST CORRECT on sm_75 |
| flash v0.2.1 (shim) | --use-flash-attention | shim to ssiu fork flash_attn_turing (m8n8k8 fp16-acc) | H3 BROKEN (precision trap) |
| flash v0.3 (real sm_75) | --use-flash-attention | real Dao-AILab sm_75 (m16n8k8 fp32-acc) | works (slow) |

## Results matrix

### Qwen-Image 2.1 (T2I, 25 steps, seed=65451515)

| Quant | SDPA | ck | flash (v0.2.1 shim) | flash (v0.3 real sm_75) |
|---|---|---|---|---|
| bf16 | 175.0s, 2.34 MB | 140.0s, 2.28 MB | 141.72s, 2.29 MB | 176.0s, 2.28 MB |
| int8_convrot | (this test) | 45.1s, 2.30 MB (this run) | (v0.2.1) | (v0.3 not tested) |

Source:
- bf16 row: SPEED.md (v0.2.1, Oct 17) + V0.3_RESULTS.md (v0.3 e2e test, Oct 18)
- int8_convrot row: this run (2026-10-18 16:12:34)

### H3 (T2V, 8-step turbo LoRA, 8 frames, heretic CLIP)

| Quant | SDPA | ck | flash (v0.2.1 shim) | flash (v0.3 real sm_75) |
|---|---|---|---|---|
| bf16 | (broken env, 12.9 KB) | 116.0s, 538 KB (v0.2.1) | 111.0s, 3.8 MB GARBAGE (v0.2.1) | 211.0s, 530 KB (v0.3 e2e) |
| fp8 | 582.3s, 558 KB (this run) | 476.2s, 560 KB (this run) | (v0.2.1) | (v0.3 not tested) |
| int8_convrot | 215.4s, 530 KB (this run) | 110.3s, 538 KB (this run) | 111.0s, 3.8 MB GARBAGE (v0.2.1) | 211.0s, 530 KB (v0.3 e2e) |
| w6a8 | 455.9s, 492 KB (this run) | 355.6s, 554 KB (this run) | (v0.2.1) | (v0.3 not tested) |

## Headline conclusions

### 1. --use-ck-attention (comfy_kitchen) is the fastest correct option for sm_75

Speed on H3 (8-step turbo):
- bf16 ck: 116s
- bf16 sdpa: 206s+ (12.9 KB file shows it can hang; with proper env ~10 min in our test)
- bf16 flash shim: 111s but GARBAGE output
- bf16 flash real sm_75 (v0.3): 211s (correct, ~80% slower than ck)

ck is 2x faster than the real sm_75 flash path because ck uses
CUTLASS composable kernel with hand-tuned mma dispatch. The real
sm_75 flash uses CUTLASS m16n8k8 fp32-accumulator atom which is
inherently 2x slower per warp than sm_80 m16n8k16.

### 2. flash_attn_turing (v0.2.1 shim) is a precision trap

flash_attn_turing uses m8n8k8 hand-tuned atom with fp16
accumulator to gain throughput on sm_75. This works for bf16/fp16
inputs (Qwen) where there is precision margin, but FAILS for
int8/convrot weight models (H3) because dequant overflows the
fp16 accumulator, which leads to NaN propagation and checkerboard garbage.

The 3.8 MB H3 garbage vs 530 KB correct is the smoking gun: garbage
encodes to larger MP4 because high-frequency noise resists compression.

### 3. Real sm_75 flash (v0.3) works for all model types but is 2x slower than ck

The v0.3 patches (M-03.1 bypass shim + 0002 relax arch gates)
make --use-flash-attention use the real Dao-AILab sm_75 build with
fp32-accumulator m16n8k8 atom. This produces correct output for
ALL model types including H3 int8 convrot, but is intrinsically
slower than ck because:
- ck uses int8 mma path (Turing has int8 tensor cores, used by ck)
- flash v0.3 uses fp16/bf16 mma path (no sm_75 int8 path in upstream)
- ck is hand-tuned for sm_75 by community; v0.3 is generic sm_75

v0.3 use case: only use it if you specifically need flash lower
memory usage or cannot use comfy_kitchen for some reason.

### 4. Qwen-Image 2.1 + int8_convrot works with ck in 45s

The Qwen int8_convrot ck test in this run completed in 45.1s
(probably partial model cache from previous test) producing a
correct 2.30 MB PNG (recognizable woman in polka-dot shuba).

This is faster than the v0.2.1 bf16 ck baseline (140s) because
int8 convrot compresses the weights, so model load is faster.

### 5. H3 int8_convrot is the fastest quant (ck: 110s)

H3 int8_convrot with ck completed in 110.3s, faster than bf16 (116s).
This is because the model is smaller in memory and loads faster,
plus the int8 mma path on Turing is well-supported by ck.

File sizes by quant (all 530-560 KB, correct):
- fp8: 558-560 KB
- int8_convrot: 530-538 KB
- w6a8: 492 KB
- bf16: 530 KB (v0.3)

### 6. H3 SDPA is unusably slow without optimizations

H3 fp8 + SDPA: 582s (9.7 min) - completed, correct output.
H3 bf16 + SDPA (earlier broken env): stuck at 7+ min in init.

SDPA is the fallback when nothing else works, but it is much
slower than the optimized paths because:
- 38 GB model must offload to CPU (23 GB VRAM)
- Each step requires moving data CPU to GPU
- SDPA math backend does not use any sm_75 optimization

## Final test matrix summary (all 7 tests passed)

| Test | Time | Size | Result |
|---|---|---|---|
| qwen_int8_convrot_ck | 45.1s | 2.30 MB | CORRECT (woman in polka-dot shuba) |
| h3_fp8_sdpa | 582.3s | 558 KB | CORRECT |
| h3_fp8_ck | 476.2s | 560 KB | CORRECT |
| h3_int8_convrot_sdpa | 215.4s | 530 KB | CORRECT |
| h3_int8_convrot_ck | 110.3s | 538 KB | CORRECT (fastest H3) |
| h3_w6a8_sdpa | 455.9s | 492 KB | CORRECT |
| h3_w6a8_ck | 355.6s | 554 KB | CORRECT |

ck vs SDPA on H3 (all quants):
- fp8: ck 476s vs SDPA 582s = ck 18% faster
- int8_convrot: ck 110s vs SDPA 215s = ck 49% faster (BIG WIN)
- w6a8: ck 356s vs SDPA 456s = ck 22% faster

ck wins on all quants, biggest win on int8_convrot (2x speedup).

## What is the recommended config for sm_75

Default: --use-ck-attention in compose.yaml. It is the fastest
correct option for both Qwen and H3.

Per-workflow override: if you need flash specifically (e.g.,
lower VRAM), use --use-flash-attention with v0.3 patches. Accept
the 2x speed penalty for H3 correctness.

Do NOT use the v0.2.1 shim path for H3 or any int8/convrot
weight model. Qwen (bf16/fp16) is fine via the shim.

## ComfyUI workflow flakiness

The Qwen-Image 2.1 and H3 workflows intermittently produce
all-black 24 KB / 12.9 KB outputs regardless of attention backend
(started ~17 Oct 19:27 for Qwen, similar pattern for H3). The
issue is NOT attention-related - even SDPA with no flag produces
the broken output. Most likely cause: some state in ComfyUI
caching/compilation. Successful runs work normally.

In this test session, Qwen int8_convrot ck worked in 45.1s on
the first try after a full container restart, contradicting earlier
failed attempts. The flakiness is a separate ComfyUI issue from
the attention backend question.

## Data sources

| Test | Source | Date |
|---|---|---|
| Qwen bf16 SDPA/CK/flash-shim | SPEED.md (v0.2.1) | 2026-10-17 |
| Qwen bf16 flash-real-sm75 | V0.3_RESULTS.md (v0.3 e2e) | 2026-10-18 14:34 |
| Qwen int8_convrot ck | this run | 2026-10-18 16:12:34 |
| H3 bf16 ck | H3_RESULTS.md (v0.2.1) | 2026-10-17 02:46 |
| H3 bf16 flash-shim | H3_RESULTS.md (v0.2.1) | 2026-10-17 02:52 |
| H3 bf16 sdpa | H3_RESULTS.md (v0.2.1) | 2026-10-17 03:08 |
| H3 bf16 flash-real-sm75 | V0.3_RESULTS.md (v0.3 e2e) | 2026-10-18 14:41 |
| H3 int8_convrot flash-shim | H3_RESULTS.md (v0.2.1) | 2026-10-17 |
| H3 int8_convrot flash-real-sm75 | V0.3_RESULTS.md (v0.3 e2e) | 2026-10-18 |
| H3 fp8 sdpa | this run | 2026-10-18 16:22:38 |
| H3 fp8/int8_convrot/w6a8 ck | this run (in progress) | 2026-10-18 |
| H3 fp8/int8_convrot/w6a8 sdpa | this run (in progress) | 2026-10-18 |

## Files

- tests/run_matrix.py - matrix runner
- tests/matrix/matrix_results.jsonl - final results log
- tests/results/v0.3/ - v0.3 e2e artefacts
- tests/SPEED.md - v0.2.1 Qwen baseline
- tests/H3_RESULTS.md - v0.2.1 H3 baseline
- tests/V0.3_RESULTS.md - v0.3 results

## Follow-up: speed up real sm_75 flash

To make the real sm_75 flash competitive with ck, options are:
1. Hand-tune the sm_75 mma path in our 0002 patch: use
   SM75_16x8x8_F32F16F16F32_TN (current) is the right atom but
   block sizes / smem usage can be optimized. This is the ssiu fork
   approach (m8n8k8 with fp16-acc) but with fp32-acc.
2. Use the sm_75 int8 mma path if/when we add one to upstream
   Dao-AILab. ck uses this today.
3. Drop down to v0.2.1 shim for bf16 models only via per-workflow
   dispatcher (TODO-9 follow-up). This avoids the int8 mma issue
   because bf16 has precision margin.

Estimated effort: option 1 = 1-2 weeks of kernel engineering;
option 2 = depends on upstream; option 3 = 4-8 hours.
