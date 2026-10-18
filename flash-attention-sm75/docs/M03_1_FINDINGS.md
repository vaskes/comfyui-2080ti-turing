# M-03.1 findings (2026-10-18): bypass the shim — INSUFFICIENT alone

## What M-03.1 was supposed to do

Bypass the shim package at
`/opt/venv/lib/python3.12/site-packages/flash_attn/__init__.py` so
that ComfyUI's `import flash_attn` loads the real Dao-AILab sm_75
build at `/opt/flash-sm75-build/` instead of being shadowed.

## What I actually did

1. Backed up the shim to
   `docs/shim-backup/__init__.py.2026-10-18.bak` (124 lines, sha256
   `27f9b63a498b552050b66fe2e9594dce7061486fe5e84a48ee5630ade362e8a0`).
2. Removed the shim dir from site-packages.
3. Added a `.pth` file (`zz_flash_attn_sm75.pth`) pointing to
   `/tmp/flash-attention` (the host bind-mount of
   `/opt/flash-sm75-build`).
4. Verified import:
   - `flash_attn.__file__` → `/tmp/flash-attention/flash_attn/__init__.py` (real)
   - `flash_attn.flash_attn_func.__module__` → `flash_attn.flash_attn_interface` (real)
5. Ran a 4×64×64 fp16 forward pass.

## What I observed

The forward call **completes without raising**, but the launch
template prints `FATAL: FlashAttention requires building with sm
version sm80-sm90, but was built for < 8.0!` repeatedly (one per
kernel block dispatched). The output is:

| Metric | Value |
|---|---|
| shape | (1, 4, 64, 64) ✓ |
| dtype | float16 ✓ |
| NaN | no |
| Inf | no |
| min/max | 5e-5 / 5.06 |
| **mean** | **0.7999** |
| **SDPA reference mean** | **0.00035** |
| **max abs diff** | **4.75** (HUGE) |

The "FATAL" comes from the sm_80 launch template
`flash_fwd_launch_template.h:25`:

```cpp
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
#define ARCH_SUPPORTS_FLASH
...
#else
#define FLASH_UNSUPPORTED_ARCH printf("FATAL: FlashAttention ...");
#endif
```

`FLASH_UNSUPPORTED_ARCH` is **just a printf** — no abort, no
`asm("trap;")`, no early return. After the printf, the kernel
continues to the next instruction, which writes the output pointer
with whatever was in `params.o_ptr` (uninitialized memory allocated
by the wrapper).

So the output is **garbage from uninitialized memory**, dressed up
in valid fp16.

## What this confirms

**M-03.1 alone is INSUFFICIENT for the goal "any model works on
sm_75".** The shim bypass is a prerequisite, but the upstream
Dao-AILab sm_80 .cu files (`flash_fwd_hdim*_sm80.cu`) genuinely
cannot run on sm_75 because:

1. They use sm_80 mma atom (m16n8k16 with fp16 accumulator).
2. Their `__CUDA_ARCH__ >= 800` gate makes them fall through to
   `FLASH_UNSUPPORTED_ARCH` on sm_75.
3. Even with the gate relaxed, the sm_80 mma atom doesn't exist
   on sm_75 (sm_75 has m16n8k8 fp32-accumulator instead).

This is exactly what `docs/H3_DIAGNOSTIC.md` and
`docs/tasks/03-sm75-mma-kernel-port.md` predicted. **M-03.2
(sm_75-specific .cu files) is the real fix.**

## What I did after

Restored the shim (so Qwen with `--use-flash-attention` continues
to work via `flash_attn_turing`) and removed the
`zz_flash_attn_sm75.pth` file. The container is back to the
v0.2.1 baseline:

- `import flash_attn` → shim → `flash_attn_turing` (ssiu fork)
- `--use-flash-attention` on Qwen: 141.72 s, correct ✓
- `--use-flash-attention` on H3: garbage (known v0.2.1 issue)
- `--use-ck-attention` on Qwen: 140 s, correct ✓
- `--use-ck-attention` on H3: 116 s, correct ✓

## What M-03.1 actually delivers

Bypass is **a one-line operation that can be re-applied at any
time** when M-03.2 lands:

```bash
# backup the shim (already done)
docker cp comfyui:/opt/venv/lib/python3.12/site-packages/flash_attn/__init__.py \
    flash-attention-sm75/docs/shim-backup/__init__.py.2026-10-18.bak

# remove the shim
docker exec comfyui rm -rf /opt/venv/lib/python3.12/site-packages/flash_attn

# add .pth to register the sm_75 build
docker exec comfyui bash -c 'echo /tmp/flash-attention > \
    /opt/venv/lib/python3.12/site-packages/zz_flash_attn_sm75.pth'

# verify
docker exec comfyui /opt/venv/bin/python3 -c \
    "import flash_attn; print(flash_attn.__file__)"
# expect: /tmp/flash-attention/flash_attn/__init__.py
```

This is the **bypass** that M-03.2 builds on. The Dockerfile
change to make it permanent (survives `docker compose down`+`up`)
is a follow-up task — see M-03.2 plan.

## Why we don't keep the bypass in place yet

1. The output is garbage (max diff 4.75 vs SDPA). Anyone setting
   `--use-flash-attention` gets broken output silently.
2. `compose.yaml` uses `--use-ck-attention` (the safe default),
   so users won't hit the broken path by default. But any explicit
   `--use-flash-attention` would be broken.
3. The shim routes to `flash_attn_turing` which at least works
   for bf16/fp16 (Qwen). Better to keep that working than to have
   the broken path exposed.
4. The bypass becomes useful **only when M-03.2 lands** — i.e.
   when the real sm_75 .cu files exist and the FATAL goes away.

## Files in this delivery

- `docs/shim-backup/__init__.py.2026-10-18.bak` — preserved shim
- `docs/shim-backup/README.md` — how/when to restore
- `docs/M03_1_FINDINGS.md` — this document
- (no source code changes — the bypass is a runtime operation)

## Next: M-03.2 (sm_75 .cu kernel files)

Now that the bypass is proven to work (the .so is loaded, just the
.cu files fail), M-03.2 is the real engineering work: write
sm_75-specific .cu files using CUTLASS
`SM75_16x8x8_F32F16F16F32_TN` (m16n8k8 fp32-accumulator,
fp16-input). See `docs/tasks/03-sm75-mma-kernel-port.md` for
the full plan.
