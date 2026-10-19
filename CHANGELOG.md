# Changelog

All notable changes to this fork. Versions are tagged as
`comfyui-2080ti-turing-vX.Y.Z` on the corresponding commit.


## v1.0.5 — 2026-10-19 — H3 32B 15 sec production fix on sm_75

**Fragmentation fix** for H3 32B heretic int8_convrot on RTX 2080 Ti (sm_75).
Without this fix, the workflow OOMs in 5-20 sec on sm_75 because
`cudaMallocAsync` pool fragmentation compounds with each sampling step,
making runtime scale **exponentially** with video duration (10s → 47m,
15s → 8-12h projected OOM).

### The fix
- `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` — PyTorch native allocator
  uses one big virtual segment per pool. Pages are mapped on demand.
- `--disable-cuda-malloc` — required. `expandable_segments:True` is a no-op
  under `cudaMallocAsync` (the ComfyUI default on torch 2.0+).
- `--disable-dynamic-vram` — forces legacy `ModelPatcher`, disables
  comfy-aimdo's 308 MB fp32 cast buffer.
- `--disable-smart-memory` — disables the pressure-cache layer.
- `--reserve-vram 0` — gives the model ALL 22 GB; 0.5 GB headroom is not
  enough for H3 32B int8_convrot.

### Recipe origin
[AMXELA-Official/ltx25-minimax-h3-12gb-vram-comfyui](https://github.com/AMXELA-Official/ltx25-minimax-h3-12gb-vram-comfyui) — same H3 model family on 12 GB VRAM. Adapted for sm_75 / 22 GB.

### Results (RTX 2080 Ti + H3 32B heretic int8_convrot)
| Duration | Before fix | After fix |
|---|---|---|
| 5 sec | ~5 min (129s/it, fragmentation) | **6:13** (12.4s/it) |
| 10 sec | 47:43 (129.6s/it, exp. frag) | (not re-tested) |
| 15 sec | 8-12h projected OOM | **19:56** (52.7s/it, linear) |

15 sec 19:56 vs prod (RTX 5080 sage) 15:40 — workstation is 1.27x slower
but **production-viable**. Linear scaling, no exponential degradation.
Output: 864×480, 362 frames, 15.08 sec, 3.87 MB MP4 with audio. Correct.

### compose.yaml changes
- `COMFYUI_ARGS` default: now includes the full `--use-ck-attention
  --lowvram --reserve-vram 0 --disable-dynamic-vram --disable-cuda-malloc
  --disable-smart-memory` launch args.
- `PYTORCH_CUDA_ALLOC_CONF` default: `expandable_segments:True`.
- Both can be overridden via env: `COMFYUI_ARGS=...` and
  `PYTORCH_CUDA_ALLOC_CONF=...`.

### Files changed
- `compose.yaml` (defaults baked in)
- `README.md` (new "Launch args (v1.0.5+)" section + "Why these flags"
  + "Anti-patterns" tables; updated Provenance)
- `scripts/restart_ck.sh` (now a thin wrapper around `docker compose`)
- `CHANGELOG.md` (this entry)

## v1.0.0 — 2026-10-18 — production release

**Attention backends**: 3 backends, **ck as default**, sage as Kijai-patch fallback, flash v0.3 experimental.

### Production contract
- **Default**: `--use-ck-attention` (fastest correct for every model on sm_75)
- **Fallback**: `--use-sage-attention` (for workflows with Kijai Sage Attn patch node)
- **Experimental**: `--use-flash-attention` (v0.3 real sm_75 build, H3 only)

### Added
- **comfy_kitchen (ck)** as production default (`--use-ck-attention` in compose.yaml).
  - Fastest correct for all (model, quant) combinations on sm_75.
  - Uses Turing int8 tensor cores via CUTLASS composable kernel.
- **SageAttention 1.0.6** as Kijai-patch fallback (`--use-sage-attention`).
  - INT8 Q/K + FP16 P/V quantization-aware attention via Triton.
  - 1.5-2x slower than ck, but correct for any model.
  - Required for workflows that use the Kijai Sage Attn patch custom node.
  - Auto-installs on container start via entrypoint script.
- **FlashAttention v0.3** (real sm_75 Dao-AILab build with 5+1 line patch) as
  experimental (`--use-flash-attention`).
  - Patches preserved in `flash-attention-sm75/patches/0001-add-sm75-arch.patch`
    and `patches/0002-relax-arch-gates.patch` for reference.
  - Works for H3, broken for Qwen int8_convrot (precision trap).
- Entrypoint script (`comfyui-entrypoint.sh`) baked into the image at
  `/usr/local/bin/comfyui-entrypoint.sh` (Dockerfile ENTRYPOINT).
  SageAttention 1.0.6 + pytest are pre-installed in the image layer; the
  entrypoint does an idempotent `import` check for safety.
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

**Recommendation**: ck as default (fastest), sage for Kijai-patch
workflows, flash only for H3 experimental work.

## v0.3.0 — 2026-10-18 (research, not release)

Real sm_75 build of upstream Dao-AILab flash-attention 2.8.4.
Discovered that sm_75 mma atom exists in `kernel_traits.h:30-36` but
runtime gates `cc_major >= 8` and `__CUDA_ARCH__ >= 800` block it.
5-site fix in flash_api.cpp + 1-site fix in flash_fwd_launch_template.h (6 lines total) in `patches/0002-relax-arch-gates.patch`.

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
