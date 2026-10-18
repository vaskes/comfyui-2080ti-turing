# flash-attention-sm75 (experimental v0.3)

Experimental **Dao-AILab flash-attention 2.8.4** build patched for sm_75
(Turing) hardware. **Status: experimental, H3 only.**

⚠️  **Not for production.** Use `--use-ck-attention` (default) or
`--use-sage-attention` (Kijai patch fallback). Flash v0.3 is broken
for Qwen int8_convrot (precision trap even with fp32-accumulator mma).
See [`../tests/MATRIX.md`](../tests/MATRIX.md) for the full comparison.

## When to use

```bash
# Only if you specifically need flash attention for H3 (experimental)
COMFYUI_ARGS="--use-flash-attention" docker compose -f /opt/comfyui-git/compose.yaml up -d
```

For H3 fp8/int8_convrot/w6a8, the v0.3 build is correct and
~2x faster than sage (210s vs 231s for int8_convrot). For Qwen bf16,
the v0.3 build is correct but slow. **For Qwen int8_convrot, v0.3
produces garbage — use `--use-ck-attention` instead.**

## Patches

`patches/0001-add-sm75-arch.patch` — adds sm_75 to the CUDA arch
list in setup.py. 5 lines.

`patches/0002-relax-arch-gates.patch` — relaxes the runtime gates
that check `cc_major >= 8` and `__CUDA_ARCH__ >= 800`. Without
this patch, the sm_75 build loads but the kernels print "FATAL:
requires sm80-sm90" and return uninitialized memory. 5 sites in flash_api.cpp + 1 site in flash_fwd_launch_template.h (6 lines total)
fix in `flash_api.cpp`.

## Why v0.3 is slow on sm_75

Turing's mma atom is `SM75_16x8x8_F32F16F16F32_TN` (k=8 per warp).
Ampere/Ada/Hopper use `SM80_16x8x16_F32F16F16F32_TN` (k=16). The
k-dim is half, so the sm_75 atom does 2x more mma instructions to
process the same data. This is intrinsic to the hardware, not a
patch limitation.

Hand-tuning the sm_75 path (smaller block sizes, different smem
layout, fused dequant) is possible but is multi-day kernel work.
The ssiu community fork does this with m8n8k8 + fp16-accumulator
but loses precision for any int8/convrot weight model — the same
problem this fork is trying to solve.

## Build (for reference only)

The `.so` artifact is preserved in `/opt/flash-sm75-build/` on the
workstation, bind-mounted to `/tmp/flash-attention-build/` inside the
container. To rebuild from scratch:

```bash
# This is the build recipe used on the workstation. For v1.0 production
# users do NOT need to rebuild — the bind-mounted .so is sufficient.
# The .so is pre-built at /opt/flash-sm75-build/flash_attn_2_cuda.cpython-312-x86_64-linux-gnu.so
# and bind-mounted into the container at /tmp/flash-attention-build/.

cd /opt/flash-sm75-build
git apply patches/0001-add-sm75-arch.patch
git apply patches/0002-relax-arch-gates.patch
FLASH_ATTN_CUDA_ARCHS=75 \
  FLASH_ATTENTION_FORCE_BUILD=TRUE \
  CPATH=/opt/venv/lib/python3.12/site-packages/nvidia/cu13/include \
  LIBRARY_PATH=/opt/venv/lib/python3.12/site-packages/nvidia/cu13/lib \
  pip install --no-build-isolation --no-deps -e .
```

Build time: ~2h 14min for 96 .cu files. Output: 400 MB `.so` in
`flash_attn/` subdir.

## Diagnostic history

Earlier v0.2.1 attempts:
- The ComfyUI image ships a shim package at
  `site-packages/flash_attn/__init__.py` that routes to
  `flash_attn_turing` (ssiu community fork, m8n8k8 fp16-acc).
- The shim shadows any editable install of the real sm_75 build.
- The shim works for Qwen (bf16/fp16 inputs have enough precision
  margin) but is a precision trap for int8/convrot weight models (H3).

v0.3 bypasses the shim by:
1. Renaming `site-packages/flash_attn/` to `flash_attn_shim_backup/`
2. Adding a `.pth` file pointing to `/tmp/flash-attention-build`
3. The `.so` is bind-mounted from the host

This is done automatically by the matrix runner's
`setup_v0_3_in_container()` helper on every container restart.

## Test results

See [`../tests/MATRIX.md`](../tests/MATRIX.md) for the full 18-cell
comparison (sage vs ck vs flash for every model/quant combination).
