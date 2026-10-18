# H3 flash attention — diagnostic and root cause

**Date:** 2026-10-18
**Status:** Open issue. Not a v0.2 blocker. Requires sm_75 mma kernel port.

## TL;DR

H3 (Minimax video) with `--use-flash-attention` produces **checkerboard garbage**.
The root cause is **not the sm_75 patch** — it's that the shim package in
the ComfyUI image routes `import flash_attn` to `flash_attn_turing` (ssiu
community fork, m8n8k8 fp16-accumulator), which has insufficient precision
for H3's int8 convrot weights. Qwen-Image 2.1 works because it uses
bf16/fp16 weights, where the m8n8k8 fp16 accumulator has enough margin.

The "sm_75 patch" in v0.2 (`patches/0001-add-sm75-arch.patch`) is **shadowed
by the shim** at import time. To make H3 work, we need to either:
1. Replace the shim's underlying implementation with a proper sm_75 kernel
   (using `SM75_16x8x8_F32F16F16F32_TN` m16n8k8 fp32-accumulator, fp16-input)
2. Add per-workflow dispatcher — but vaskes explicitly rejected this as
   "костыль"

Option 1 is a multi-day kernel port (similar to ssiu/KLM-corporation
forks, but with fp32 accumulator and proper int8 input handling).
See TODO.md §TODO-9.

## Diagnostic log (pre-fix)

### Symptom

| Setup | Time | Output | Visual |
|---|---|---|---|
| `--use-ck-attention` (baseline) | 116 s | 538 KB | ✅ Batman chase scene |
| `--use-flash-attention` (v0.2) | 111 s | 3.8 MB | ❌ checkerboard garbage |
| `--use-pytorch-cross-attention` (SDPA) | 206 s | 529 KB | ✅ Batman chase scene |

The 3.8 MB is high entropy noise, not better detail — H.264 only
uses that much bitrate when content has very high entropy (i.e. noise).

### Root cause analysis

#### Step 1: where does flash_attn come from?

The ComfyUI image has a **shim package** at
`/opt/venv/lib/python3.12/site-packages/flash_attn/__init__.py`
that routes the standard `flash_attn` namespace to `flash_attn_turing`
(community fork, ssiu/flash-attention-turing):

```python
# /opt/venv/lib/python3.12/site-packages/flash_attn/__init__.py
import flash_attn_turing as _gpu
from flash_attention_interface import (flash_attn_func as _flash_attn_func_native, ...)

def flash_attn_func(q, k, v, dropout_p=0.0, softmax_scale=None, causal=False, ...):
    ...
    return _flash_attn_func_native(q, k, v, softmax_scale=softmax_scale, causal=causal)
```

This shim is in the image (`comfyui-workstation:cu130-torch-stable-gcc`)
and **shadows any real flash_attn package** because site-packages is
searched first.

The shim's docstring says: "If the real flash_attn package is installed,
this shim is shadowed because both occupy the flash_attn namespace."
**This is wrong** — site-packages takes priority over editable installs.

#### Step 2: what does flash_attn_turing do?

`flash_attn_turing` is the ssiu community fork at
`https://github.com/ssiu/flash-attention-turing`. It uses the
`mma.sync.aligned.m8n8k8` PTX with **fp16 accumulator** (vs the
Dao-AILab official which uses `m16n8k8` with **fp32 accumulator**).

This is exactly the precision-loss issue that the v0.2 patch
documentation warned about for the broken community forks.

#### Step 3: what does v0.2 actually do?

The v0.2 patch (`patches/0001-add-sm75-arch.patch`) modifies
`Dao-AILab/flash-attention:setup.py` to add sm_75 to the build matrix.
This **builds a real sm_75 flash_attn.so** at
`/tmp/flash-attention/flash_attn_2_cuda.cpython-312-x86_64-linux-gnu.so`.

But because of the shim, this build is **never imported** when
ComfyUI runs. The shim's `import flash_attn` resolves to
`flash_attn_turing`, not the real Dao-AILab package.

**Net effect of v0.2 on H3**: zero. The same `flash_attn_turing`
runs before and after the patch.

#### Step 4: why does H3 fail?

H3's UNET is `minimax_h3_fl2va_pruned_int8_convrot.safetensors` —
int8 convrot weights, dequantized to fp16/bf16 at runtime. The
dequantization can produce NaN/Inf in some attention Q projections
(call 2 of 200+ in H3's diffusion step) when scaling factor is 0 or
extreme.

`flash_attn_turing`'s m8n8k8 fp16 accumulator: NaN in Q → NaN in QK^T
→ NaN in softmax → NaN in output. The NaN propagates through all
downstream layers, producing the checkerboard pattern in the final
output (because NaN cast to int = garbage values that the encoder
faithfully represents at high bitrate).

`pytorch SDPA` (math backend): NaN handling is more robust — the math
backend computes softmax in fp64 for the score, fp32 for the V
multiplication, and gracefully handles NaN in some inputs. The output
is well-defined (NaN rows get masked or output as 0).

**Why Qwen works**: Qwen uses bf16 storage throughout. No int8
dequantization, no NaN injection. m8n8k8 fp16 accumulator has enough
margin for bf16 inputs and produces correct output.

## Why a "fix" wasn't applied

Three attempted fixes in the diagnostic session were reverted because
they don't address the real problem:

1. **Removing the shim** — exposes the real Dao-AILab sm_75 build.
   But the upstream `.cu` source files are named `flash_fwd_hdim*_sm80.cu`
   and use sm_80-specific mma atom. They have `__CUDA_ARCH__ >= 800` gates.
   For sm_75, the kernel just prints "FATAL: requires sm80-sm90" and
   returns uninitialized memory. Still garbage, just different garbage.

2. **Patching `cc_major >= 8` to `cc_major >= 8 || (cc_major == 7 && cc_minor == 5)`**
   in `flash_api.cpp` — lets the build *load* on sm_75, but the kernel
   still doesn't execute (mma atom missing).

3. **Patching `ARCH_SUPPORTS_FLASH` to define for `__CUDA_ARCH__ >= 750`**
   in `flash_fwd_launch_template.h` — causes the build to take >1 hour
   and never finish (probably tries to compile sm_80 mma atom for sm_75
   and chokes).

The only path that works for sm_75 + any model is to **add new `.cu`
source files** with sm_75-specific instantiations, using
`SM75_16x8x8_F32F16F16F32_TN` (m16n8k8 fp32-accumulator) atom for
fp16-input and (ideally) a new `SM75_..._INT8..._TN` atom for int8-input.

## Path forward

### TODO-9: real sm_75 mma kernel port

Three options (in order of preference):

1. **Add sm_75 .cu files** using `SM75_16x8x8_F32F16F16F32_TN` atom for
   fp16 input. This is what ssiu and KLM-corporation forks do, but
   with fp32 accumulator (precision fix vs the broken community forks).
   Estimated work: 1-2 days of CUDA/Triton work, plus testing.

2. **Defer H3 to a separate ticket** with explicit "sm_75 is bf16/fp16
   only" limitation. Document in README. ck-attention (int8 convrot
   path) is the recommended backend for H3. This is the current state
   of v0.2.

3. **Per-workflow dispatcher** — explicitly rejected by vaskes as
   "костыль".

**Recommendation**: option 2 (current state) for v0.2 release. Option
1 is a real engineering project for v0.3.

## Files and state

### Current state (v0.2.1, after this diagnostic)

- `/opt/comfyui/compose.yaml` uses `--use-ck-attention` (safe default,
  works for both Qwen and H3)
- Shim package at site-packages is intact (the v0.2 default)
- `flash_attn_turing` is the runtime flash backend
- H3 with `--use-flash-attention` = garbage (known issue)
- H3 with `--use-ck-attention` = correct
- Qwen with `--use-flash-attention` = correct (9.5% per-step speedup
  vs ck)

### Diagnostic session artefacts (not committed to repo)

- Workstation: `/opt/flash-sm75-build/` (Dao-AILab sm_75 build,
  with attempted ARCH_SUPPORTS_FLASH patch — not used at runtime)
- Workstation: `/opt/cutlass-vendored/` (CUTLASS df18f5e4 source)
- Container: `/opt/flash-sm75-build` and `/opt/cutlass-vendored` are
  bind-mounted to `/tmp/flash-attention` and `/tmp/cutlass-host` for
  the build, but **not used at runtime** due to the shim

### Cleanup needed

- Revert `/opt/flash-sm75-build/csrc/flash_attn/flash_api.cpp` to
  upstream `is_sm8x_min` check
- Revert `/opt/flash-sm75-build/csrc/flash_attn/src/flash_fwd_launch_template.h`
  to upstream `__CUDA_ARCH__ >= 800` check
- Revert `/opt/flash-sm75-build/csrc/flash_attn/src/flash_bwd_launch_template.h`
  to upstream `__CUDA_ARCH__ >= 800` check
- Optional: keep `/opt/flash-sm75-build/` for future kernel port
  work, or remove to free disk space

The reverted source should be checked in to `/opt/flash-sm75-build/`
for reproducibility, but it is **not** used at runtime.
