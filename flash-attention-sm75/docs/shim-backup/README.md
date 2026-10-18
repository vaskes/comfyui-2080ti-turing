# Shim backup (2026-10-18)

This directory preserves the `flash_attn_turing` shim that shipped
with the original ComfyUI workstation image
(`comfyui-workstation:cu130-torch-stable`).

## What was the shim

A 124-line `__init__.py` placed at
`/opt/venv/lib/python3.12/site-packages/flash_attn/__init__.py` that
shadowed any real `flash_attn` install by routing all imports to
`flash_attn_turing` (ssiu community fork, m8n8k8 fp16-accumulator).

**The shim was added by a previous Mavis session** (not in any
Dockerfile in this repo — the existing Dockerfile at
`/opt/comfyui-git/Dockerfile` never installs flash_attn).

## Why we removed it

The shim worked for bf16/fp16 models (Qwen-Image 2.1) but produced
checkerboard garbage for H3 (int8 convrot), because the m8n8k8
fp16-accumulator has insufficient precision for out-of-fp16-range
values that int8 convrot dequant can produce.

The real Dao-AILab sm_75 build at `/opt/flash-sm75-build/` was
present (57 MB `flash_attn_2_cuda.cpython-312-x86_64-linux-gnu.so`)
but never registered (`pip install -e .` was never run successfully),
so the shim won by site-packages priority.

## Files in this directory

- `__init__.py.2026-10-18.bak` — the original shim, sha256
  `27f9b63a498b552050b66fe2e9594dce7061486fe5e84a48ee5630ade362e8a0`

## When to restore the shim

If you need to roll back to v0.2.1 behavior (bf16/fp16 only, H3
broken) — e.g. the new sm_75 kernel port (TODO-9.2) doesn't work
out — restore the shim with:

```bash
docker cp /opt/comfyui-git/flash-attention-sm75/docs/shim-backup/__init__.py.2026-10-18.bak \
    comfyui:/opt/venv/lib/python3.12/site-packages/flash_attn/__init__.py
docker restart comfyui
```

(You'll also need to remove the real `flash_attn.egg-info` from the
venv's site-packages if it's there.)

## Diagnostic reference

Full root-cause analysis: `../H3_DIAGNOSTIC.md` (194 lines).
Plan to make the sm_75 build work for all model types:
`../tasks/03-sm75-mma-kernel-port.md` (257 lines).
