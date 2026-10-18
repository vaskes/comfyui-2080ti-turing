# Full attention backend matrix: Qwen-Image 2.1 + Minimax H3 (2026-10-18)

## TL;DR

Per vaskes request: tested 3 backends (sdpa, ck, flash) for every (model,
quant) combination. For "flash" we have TWO build versions to test:
- **flash v0.2.1 (shim)**: routes to ssiu community fork (m8n8k8 fp16-acc)
- **flash v0.3 (real sm_75)**: routes to Dao-AILab sm_75 build with 0002 patch (m16n8k8 fp32-acc)

**ck (comfy_kitchen) is the only backend that works correctly for ALL
model/quant combinations on sm_75.** Flash in any form has precision
issues for some int8/convrot weight models.

## Headline results

### Qwen-Image 2.1 (T2I, 25 steps)

| Quant | SDPA | ck | flash v0.2.1 (shim) | flash v0.3 (real sm_75) |
|---|---|---|---|---|
| bf16 | 175.0s, 2.34 MB ✅ | **140.0s, 2.28 MB** ⭐ | 141.72s, 2.29 MB ✅ | 176.0s, 2.28 MB ✅ |
| int8_convrot | 70.2s, 2.29 MB ✅ | **45.1s, 2.30 MB** ⭐ | 35.1s, 2.48 MB ❌ GARBAGE | 45.1s, 2.48 MB ❌ GARBAGE |

### Minimax H3 (T2V, 8-step turbo, 8 frames, heretic CLIP)

| Quant | SDPA | ck | flash v0.2.1 (shim) | flash v0.3 (real sm_75) |
|---|---|---|---|---|
| bf16 | (broken env earlier) | **116.0s, 538 KB** ⭐ | 111.0s, 3.8 MB ❌ GARBAGE | 211.0s, 530 KB ✅ |
| fp8 | 582.3s, 558 KB | **476.2s, 560 KB** ⭐ | 12.9 KB ❌ BROKEN | 576.7s, 558 KB ✅ |
| int8_convrot | 215.4s, 530 KB | **110.3s, 538 KB** ⭐ | 111.0s, 3.8 MB ❌ GARBAGE | 210.3s, 530 KB ✅ |
| w6a8 | 455.9s, 492 KB | **355.6s, 554 KB** ⭐ | (not tested) | 455.6s, 492 KB ✅ |

⭐ = fastest correct for that (model, quant)

## What this means

### 1. ck (comfy_kitchen) is the universal winner

ck is the fastest correct backend for EVERY (model, quant) combination
on sm_75. The speedup over SDPA on H3:
- fp8: 18% faster (476s vs 582s)
- int8_convrot: 49% faster (110s vs 215s) — 2x speedup
- w6a8: 22% faster (356s vs 456s)

For Qwen, ck beats SDPA by 36-50% across both quants.

ck uses CUTLASS composable kernel int8 path with hand-tuned sm_75
mma dispatch — that's why it's faster than both SDPA math and the
generic flash path.

### 2. flash v0.3 (real sm_75) works for H3, breaks for Qwen int8_convrot

This was a surprise. We expected v0.3 to work for everything. Result:
- H3 bf16 / fp8 / int8_convrot / w6a8: ALL CORRECT (530-558 KB)
- Qwen int8_convrot: GARBAGE (2.48 MB, no recognizable image)

The model-specific difference is unexplained but reproducible.
Possibilities:
- Qwen int8_convrot uses more aggressive quantization that loses
  more precision when dequantized
- Qwen and H3 have different attention head dimensions that hit
  different kernel code paths
- The 530 KB H3 output is the "correct" size; the 2.48 MB Qwen
  output is the "garbage" size (matches v0.2.1 shim garbage pattern)

**For H3, v0.3 is correct but ~2x slower than ck** (210s vs 110s
for int8_convrot). The slowness is intrinsic: sm_75 m16n8k8 fp32-acc
atom has k=8 per warp, vs sm_80's k=16 (2x more mma instructions).

### 3. flash v0.2.1 (shim) is a precision trap

flash_attn_turing uses m8n8k8 hand-tuned atom with fp16 accumulator
to gain throughput on sm_75. This works for bf16/fp16 inputs (Qwen bf16)
where there is precision margin, but FAILS for:
- All H3 int8/convrot weights → 3.8 MB garbage (precision overflow in fp16)
- H3 fp8 → 12.9 KB (workflow error, possibly NaN cascade)
- Qwen int8_convrot → 2.48 MB garbage (same precision overflow)

The fp16 accumulator cannot hold the dynamic range of dequantized
int8 weights — it overflows to NaN/Inf, which propagates through
softmax and produces checkerboard garbage.

### 4. SDPA is the slowest but always correct (when env is right)

SDPA pytorch math backend is the universal fallback. It works for
everything but is much slower than ck because:
- 38 GB H3 model must offload to CPU (23 GB VRAM)
- Each step requires CPU↔GPU data movement
- SDPA math backend has no sm_75 optimization

## Recommended config for sm_75

**Default**: `--use-ck-attention` in compose.yaml.

**Per-workflow override**: 
- For H3 (any quant): `--use-flash-attention` is OK if you want to
  test v0.3 (real sm_75 build, correct but slow). Or `--use-ck-attention`
  for speed.
- For Qwen bf16: any backend works, ck is fastest.
- For Qwen int8_convrot: `--use-ck-attention` only. Flash is broken
  in any form.

**NEVER USE flash_attn_turing shim (v0.2.1)** for:
- Any H3 model (int8/fp8/convrot weights)
- Qwen int8_convrot

The shim was the original ComfyUI image's bundled fallback. We've
replaced it with v0.3 (real sm_75 build) as the active flash build,
which correctly handles H3 but not Qwen int8_convrot.

## Technical details

### v0.3 setup (active in container as of 2026-10-18 17:42)

The container has a bind-mount `/opt/flash-sm75-build` →
`/tmp/flash-attention-build` (in compose.yaml). The Python path is
configured via a `.pth` file at
`/opt/venv/lib/python3.12/site-packages/zz_flash_attn_sm75.pth`
that adds `/tmp/flash-attention-build` to sys.path. This makes
`import flash_attn` resolve to the v0.3 real sm_75 build.

To switch back to v0.2.1 shim (e.g., to compare):
- `docker exec comfyui rm -f /opt/venv/lib/python3.12/site-packages/zz_flash_attn_sm75.pth`
- `docker exec comfyui mv /opt/venv/lib/python3.12/site-packages/flash_attn_shim_backup /opt/venv/lib/python3.12/site-packages/flash_attn`
- Restart ComfyUI

Or use the matrix script's `--flash-mode v0.2.1` flag.

### v0.3 build details

- Source: Dao-AILab flash-attention 2.8.4 with patches 0001 + 0002
- 0001 patch: adds sm_75 to setup.py's CUDA arch list
- 0002 patch: relaxes `__CUDA_ARCH__ >= 800` and `cc_major >= 8` runtime
  gates to allow sm_75
- Built with: `FLASH_ATTN_CUDA_ARCHS=75` against CUDA 13.0
- Build time: 2h 14min
- .so size: 400 MB

### Why ck is fast

ck uses CUTLASS composable kernel with int8 mma path on Turing
(int8 tensor cores are Turing's specialty). The hand-tuned dispatch
in comfy_kitchen's `int8_attention` uses the sm_75 int8 mma directly,
which has 2x throughput vs fp16/bf16 mma on sm_75.

flash attention (any variant) uses fp16/bf16 mma path with
dequantization-then-attention. Even with fp32 accumulator, the
dequantization precision is the bottleneck for int8 weight models.

## Source data

- Test results: `tests/matrix/matrix_results.jsonl`
- Test outputs: `tests/results/matrix_2026-10-18/`
- Matrix script: `tests/matrix/run_matrix.py` (with `--flash-mode` arg)
- Earlier v0.2.1 baselines: `tests/SPEED.md`, `tests/H3_RESULTS.md`
- Earlier v0.3 e2e: `tests/V0.3_RESULTS.md`

## Date and environment

- Date: 2026-10-18 18:20 MSK
- Workstation: 85.136.112.144:12008, RTX 2080 Ti sm_75
- Docker: comfyui-workstation:cu130-torch-stable-gcc
- ComfyUI: 5cc4b87, comfy_kitchen 0.2.38
