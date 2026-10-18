# Changelog

All notable changes to this fork. Versions are tagged as
`comfyui-2080ti-turing-vX.Y.Z` on the corresponding commit.

## v1.0.0 — 2026-10-18 — production release

**Attention backends**: 3 backends, sage as default, ck for speed, flash v0.3 experimental.

### Added
- **SageAttention 1.0.6** as production default (`--use-sage-attention` in compose.yaml).
  - INT8 Q/K + FP16 P/V quantization-aware attention via Triton.
  - Works correctly for all tested (model, quant) combinations on sm_75.
  - Auto-installs on container start via entrypoint script.
- **comfy_kitchen (ck)** as fastest correct option (`--use-ck-attention`).
- **FlashAttention v0.3** (real sm_75 Dao-AILab build with 2-line patch) as
  experimental fallback (`--use-flash-attention`).
  - Patches preserved in `flash-attention-sm75/patches/0001-add-sm75-arch.patch`
    and `patches/0002-relax-arch-gates.patch` for reference.
- Entrypoint script (`comfyui-entrypoint.sh`) bind-mounted to
  `/tmp/flash-attention-build` inside the container, idempotent sage install.
- `tests/MATRIX.md` — full 18-cell matrix report (3 backends × 2 models × 4 quants).
- `tests/TESTING.md` — test runner guide.
- `tests/test_backends.py` — production smoke + import tests.
- `tests/matrix/run_matrix.py` — production matrix runner.

### Verified (all 18 cells, 2026-10-18)
- Qwen bf16: ck 140s, flash_v0.2.1 141.72s, flash_v0.3 176s, sage 175s — all correct.
- Qwen int8_convrot: ck 45.1s, sdpa 70.2s, sage 70.1s, flash variants GARBAGE.
- H3 bf16: ck 116s, flash_v0.2.1 GARBAGE 3.8MB, flash_v0.3 211s, sage 591s.
- H3 fp8: sdpa 582s, ck 476s, flash_v0.2.1 BROKEN 12.9KB, flash_v0.3 577s, sage 586s.
- H3 int8_convrot: sdpa 215s, ck 110s, flash_v0.2.1 GARBAGE 3.8MB, flash_v0.3 210s, sage 231s.
- H3 w6a8: sdpa 456s, ck 356s, flash_v0.3 456s, sage 466s.

**Recommendation**: sage as default (works for all), ck for speed
when only ck-supported models are used.

## v0.3.0 — 2026-10-18 (research, not release)

Real sm_75 build of upstream Dao-AILab flash-attention 2.8.4.
Discovered that sm_75 mma atom exists in `kernel_traits.h:30-36` but
runtime gates `cc_major >= 8` and `__CUDA_ARCH__ >= 800` block it.
2-line fix in `patches/0002-relax-arch-gates.patch`.

**Findings**:
- Works correctly for H3 all quants.
- **Broken for Qwen int8_convrot** (precision trap: dequantization
  overflow in softmax exp, even with fp32-accumulator).
- 2x slower than ck on sm_75 (intrinsic m16n8k8 vs sm_80 m16n8k16).

**Conclusion**: flash v0.3 is correct for H3 but not universal.
Sage is the universal correct backend for sm_75.

## v0.2.1 — 2026-10-18 (research, not release)

`flash-attention-sm75` shim layer that routes to ssiu community fork
`flash_attn_turing` (m8n8k8 fp16-accumulator). Marked as experimental
because the fp16-accumulator is a precision trap for any model with
int8/convrot weights (H3 int8_convrot = 3.8MB checkerboard garbage,
H3 fp8 = 12.9KB broken workflow output).

**Findings**: shim path is fastest for bf16/fp16 (Qwen bf16 141.72s
vs ck 140s) but broken for any int8/convrot weight model.

## v0.2.0 — 2026-10-17 (initial)

5-line patch to setup.py to add sm_75 to CUDA arch list. Compilation
works but the sm_80 .cu files (named `flash_fwd_hdim*_sm80.cu`) use
`__CUDA_ARCH__ >= 800` gates and the sm_80 mma atom that doesn't
exist on sm_75. For sm_75, kernels print "FATAL: requires sm80-sm90"
and return uninitialized memory.

## v0.1 — 2026-10-17 (initial)

Pure-config fork of comfyui-2080ti-turing, no attention optimizations.
Qwen and H3 ran with pytorch SDPA, 175-582s per workflow depending
on model/quant.
