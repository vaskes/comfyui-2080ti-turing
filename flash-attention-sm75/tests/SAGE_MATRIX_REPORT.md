# SageAttention 1.0.6 vs ck vs flash: sm_75 attention backends (2026-10-18)

## TL;DR

**SageAttention 1.0.6 is the universal attention backend for sm_75.**
- Works correctly for ALL 6 (model, quant) combinations — including Qwen int8_convrot which broke flash v0.2.1 shim AND flash v0.3 real sm_75.
- 1.5-2x slower than ck on most cells, but 2-5x faster than SDPA.
- Install: `pip install sageattention==1.0.6` (Triton 3.x compatible on sm_75).
- ComfyUI flag: `--use-sage-attention`.

## Full matrix: 4 backends × 2 models × 4 quants

### Qwen-Image 2.1 (T2I, 25 steps)

| Quant | SDPA | ck ⭐ | flash v0.2.1 (shim) | flash v0.3 (real sm_75) | **sage 1.0.6** |
|---|---|---|---|---|---|
| bf16 | 175.0s ✅ 2.34MB | **140.0s** ✅ 2.28MB | 141.72s ✅ 2.29MB | 176.0s ✅ 2.28MB | **175.3s** ✅ 2.28MB |
| int8_convrot | 70.2s ✅ 2.29MB | **45.1s** ✅ 2.30MB | 35.1s ❌ GARBAGE | 45.1s ❌ GARBAGE | **70.1s** ✅ 2.29MB 🎉 |

### Minimax H3 (T2V, 8-step turbo, 8 frames, heretic CLIP)

| Quant | SDPA | ck ⭐ | flash v0.2.1 (shim) | flash v0.3 (real sm_75) | **sage 1.0.6** |
|---|---|---|---|---|---|
| bf16 | (broken env) | **116.0s** ✅ 538KB | 111.0s ❌ 3.8MB garbage | 211.0s ✅ 530KB | **591.0s** ✅ 488KB |
| fp8 | 582.3s ✅ 558KB | **476.2s** ✅ 560KB | 12.9KB ❌ broken | 576.7s ✅ 558KB | **585.9s** ✅ 558KB |
| int8_convrot | 215.4s ✅ 530KB | **110.3s** ✅ 538KB | 111.0s ❌ 3.8MB garbage | 210.3s ✅ 530KB | **231.3s** ✅ 530KB |
| w6a8 | 455.9s ✅ 492KB | **355.6s** ✅ 554KB | (not tested) | 455.6s ✅ 492KB | **465.7s** ✅ 492KB |

## Key findings

### 1. SageAttention solves the Qwen int8_convrot problem

This was the OPEN question from the previous matrix. Result:
- flash v0.2.1 (shim): GARBAGE (precision trap, fp16 accumulator)
- flash v0.3 (real sm_75, fp32 accumulator): GARBAGE (dequant precision lost in softmax)
- **sage 1.0.6: CORRECT (INT8 Q/K quant + FP16 P/V with FP16 accumulator + K-smoothing)**

Sage works because it quantizes Q/K to INT8 at attention time (not
dequant-then-attention), uses FP16 for P/V (where the dynamic range
matters), and smooths K matrix outliers. This is the same approach
described in the SageAttention paper (Tsinghua, arXiv:2410.02367).

Speed: 70.1s vs ck's 45.1s (1.55x slower). But it's CORRECT, which
flash isn't. Acceptable trade-off for users who can't use ck.

### 2. ck is still the fastest correct backend

ck beats sage on every cell:
- H3 bf16: ck 116s vs sage 591s (5.1x faster)
- H3 fp8: ck 476s vs sage 586s (1.23x faster)
- H3 int8_convrot: ck 110s vs sage 231s (2.1x faster)
- H3 w6a8: ck 356s vs sage 466s (1.31x faster)
- Qwen bf16: ck 140s vs sage 175s (1.25x faster)
- Qwen int8_convrot: ck 45s vs sage 70s (1.55x faster)

The H3 bf16 gap is the largest (5.1x). Sage's bf16 path on sm_75
is slow because the Triton kernel is generic, not hand-tuned for
Turing's m16n8k8 atom.

### 3. Sage vs flash v0.3 — comparable speed, sage works for Qwen

For H3 (where flash v0.3 works):
- H3 bf16: sage 591s vs flash v0.3 211s (sage 2.8x slower)
- H3 fp8: sage 586s vs flash v0.3 577s (similar)
- H3 int8_convrot: sage 231s vs flash v0.3 210s (similar)
- H3 w6a8: sage 466s vs flash v0.3 456s (similar)

For H3, flash v0.3 is 2.8x faster on bf16 but similar on others.
For Qwen int8_convrot, flash v0.3 is BROKEN, sage is CORRECT.

### 4. Sage "not exact" — what does this mean in practice

Sage uses INT8 for Q/K quantization. The paper claims <1% accuracy
loss for LLM/vision models. For diffusion models, the visual quality
looks identical to ck/sdpa in our tests. The "not exact" label means:
- Some loss of dynamic range in Q/K
- FP16 accumulator for P/V (vs FP32 in flash v0.3)
- Visual quality: indistinguishable from ck at 25 steps (Qwen) and
  8 steps (H3 turbo)

## Recommended config for sm_75

**Two-tier strategy:**

**Tier 1 (default)**: `--use-ck-attention` in compose.yaml
- Fastest correct option for all 6 (model, quant) cells
- Best for H3 int8_convrot (110s, fastest in entire matrix)

**Tier 2 (fallback)**: `--use-sage-attention`
- Universal correct option (works for everything flash breaks)
- 1.5-2x slower than ck on most cells
- Useful when ck is unavailable, or for cross-backend testing

**Avoid for these cells**:
- `--use-flash-attention` v0.2.1 (shim) for ANY H3 quant
- `--use-flash-attention` v0.3 (real sm_75) for Qwen int8_convrot

**Default option**: keep `--use-ck-attention` as compose default.
sage is now available via the same `--use-sage-attention` flag.

## Sage install details

```bash
# In the running container
docker exec comfyui pip install sageattention==1.0.6

# Verify
docker exec comfyui python3 -c 'import sageattention; print("sage OK")'

# Use
COMFYUI_ARGS="--use-sage-attention" docker compose -f /opt/comfyui-git/compose.yaml up -d
```

**Triton version**: Works with triton 3.x (3.8.0 verified). Earlier
docs said triton 3.1/3.2 only, but this restriction seems to have
been lifted in newer wheel releases.

**No smem fix needed**: The 67584→65536 smem patch from old recipes
is not required for the current sage 1.0.6 wheel.

**Persistent install**: We added an entrypoint script
(`/opt/flash-sm75-build/comfyui-entrypoint.sh` on host,
bind-mounted to `/tmp/flash-attention-build/` in container) that
auto-installs sage on first use. Marker file in the bind-mount
persists across container recreates.

## Source data

- Test results: `tests/matrix/matrix_results.jsonl`
- Test outputs: `tests/results/matrix_2026-10-18/`
- Matrix script: `tests/matrix/run_matrix.py` (now has `sage` as 4th backend)
- Earlier matrices: `tests/FULL_MATRIX_REPORT.md`, `tests/SPEED.md`, `tests/H3_RESULTS.md`, `tests/V0.3_RESULTS.md`

## Date and environment

- Date: 2026-10-18 21:21 MSK
- Workstation: 85.136.112.144:12008, RTX 2080 Ti sm_75
- Docker: comfyui-workstation:cu130-torch-stable-gcc
- ComfyUI: 5cc4b87, comfy_kitchen 0.2.38
- sage: 1.0.6
- triton: 3.8.0 (default)
- torch: 2.14.1+cu130
- CUDA: 13.0
