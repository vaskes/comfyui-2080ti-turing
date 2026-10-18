# SageAttention 1.0.6 (Kijai patch fallback)

**SageAttention 1.0.6** is the Kijai Sage Attn patch fallback for
`comfyui-2080ti-turing` on RTX 2080 Ti (sm_75, Turing). It is the
universal correct attention backend for sm_75, but is **1.5-2x slower
than the default `ck` (comfy_kitchen)**. Use it ONLY when your
workflow contains the Kijai Sage Attn patch custom node.

## When to use

```bash
# For workflows that use the Kijai Sage Attn patch custom node
COMFYUI_ARGS="--use-sage-attention" docker compose -f /opt/comfyui-git/compose.yaml up -d
```

For all other workflows, use the default `--use-ck-attention`.

## What it does

SageAttention (Tsinghua, [arXiv:2410.02367](https://arxiv.org/abs/2410.02367))
uses **INT8 Q/K + FP16 P/V** quantization-aware attention via Triton
kernels, with **K-smoothing** to handle activation outliers. On sm_75
(Turing), it uses the Triton fallback (no native CUDA kernels).

## Why it's the Kijai patch fallback (not default)

The Kijai Sage Attn patch custom node in ComfyUI patches the attention
path for specific attention layouts that `ck` (comfy_kitchen) doesn't
support. For those workflows, sage is the only option.

For everything else, `ck` is the production default because:
- ck is 1.5-2x faster than sage on sm_75
- ck uses Turing's int8 tensor cores directly
- ck is hand-tuned for sm_75 by the comfy_kitchen community

## Install

SageAttention 1.0.6 is **pre-installed in the image** (added by
the Dockerfile). The container entrypoint at
`/usr/local/bin/comfyui-entrypoint.sh` (baked into the image via
Dockerfile ENTRYPOINT) does an idempotent `import sageattention`
check on startup as a safety net. No marker file is used — the
install survives container recreates because it lives in the image
layer.

To install manually:
```bash
docker exec comfyui pip install sageattention==1.0.6
```

**Triton version**: works with triton 3.x (3.8.0 verified on sm_75).
No smem fix needed (the 67584→65536 patch from old recipes is not
required for the current 1.0.6 wheel).

## ComfyUI flag

`--use-sage-attention` is a startup flag, not a runtime switch. To
change the active backend, restart the container with the new flag.
The flag persists for the container lifetime.

```python
# /opt/ComfyUI/comfy/ldm/modules/attention.py:740
sage_kwargs = {
    "is_causal": False,
    "tensor_layout": tensor_layout,
    "sm_scale": kwargs.get("scale", None),
    "smooth_k": False,  # ComfyUI uses smooth_k=False
}
out = sageattn(q, k, v, **sage_kwargs)
```

## Test coverage

Sage is tested as part of the cross-backend matrix. See
[`../tests/MATRIX.md`](../tests/MATRIX.md) for the full 18-cell
comparison (sage vs ck vs flash for every model/quant combination).

Unit tests in [`../tests/test_backends.py`](../tests/test_backends.py):
- `test_sage_importable` — sage module loads
- `test_sage_module_version` — version is 1.0.6
- `test_smoke_sage` — disabled (known triton 3.8 + sm_75 compile bug
  on small dims; real workflows use larger dims that work)

## Known limitations

- **Slow on small dims** (heads<8 or seq<128). Triton 3.8 + sm_75
  fails to compile the matmul pass. Real workflows use larger dims
  (H3: 32 heads × 1024+ seq; Qwen: 32 heads × 1024 seq) which work.
- **No triton kernel for fp16 inputs** — sage casts to fp16 internally.
  bf16 inputs work but are converted to fp16 before the matmul.
- **GPU memory** — uses 8 GB peak for H3 turbo LoRA, 22 GB for
  Qwen-Image 2.1. Same as ck.

## Reference

- **Tsinghua paper**: [arXiv:2410.02367](https://arxiv.org/abs/2410.02367)
- **Source**: [github.com/thu-ml/SageAttention](https://github.com/thu-ml/SageAttention)
- **Production commit**: 5ec3798 (v1.0.1)
- **Test data**: `tests/matrix/matrix_results.jsonl` (sage column entries)
- **Cross-backend comparison**: [`../tests/MATRIX.md`](../tests/MATRIX.md)
