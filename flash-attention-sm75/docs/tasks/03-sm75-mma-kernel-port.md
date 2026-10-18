# Task 03 — sm_75 mma kernel port (v0.3)

**Status:** ⏳ Awaiting vaskes approval. Plan complete; coding not started.

## Abstract

Write sm_75-specific `.cu` source files for the Dao-AILab
flash-attention build, using CUTLASS
`SM75_16x8x8_F32F16F16F32_TN` (m16n8k8 fp32-accumulator, fp16-input)
atom. This makes `--use-flash-attention` actually execute our sm_75
build instead of falling through to `flash_attn_turing` (community
fork) via the shadowing shim package.

## Problem we solve

**Current state (v0.2.1):**
- `--use-flash-attention` works for Qwen-Image 2.1 (bf16/fp16 weights)
  via the image's pre-installed shim + `flash_attn_turing` community
  fork. The shim routes `import flash_attn` away from our sm_75 build.
- `--use-flash-attention` produces checkerboard garbage for H3
  (int8 convrot weights). `flash_attn_turing` uses m8n8k8
  fp16-accumulator, which has insufficient precision for the
  out-of-fp16-range values that int8 convrot dequant can produce.
- `--use-ck-attention` (comfy_kitchen int8 path) is the correct
  path for H3. Recommended default in `compose.yaml`.

**What the user wants:**
- One attention backend (`--use-flash-attention`) that works for
  **any** model on sm_75, including H3 int8 convrot.
- Per-workflow dispatcher was explicitly rejected as "костыль".

**Why the v0.2.1 patch didn't help:**
The patch enables sm_75 in the build matrix of Dao-AILab, but the
shim shadows it. The Dao-AILab sm_75 build never executes.

See `docs/H3_DIAGNOSTIC.md` (194 lines) for the full root-cause
analysis.

## How to solve

Three sub-tasks in dependency order. Total estimated effort:
**1.5–2.5 working days**.

### Sub-task 03.1 — Bypass the shim (4–8h)

The shim at
`/opt/venv/lib/python3.12/site-packages/flash_attn/__init__.py`
routes to `flash_attn_turing`. Make our sm_75 build actually run.

**Recommended approach: modify the ComfyUI Dockerfile.**

In `vaskes/comfyui-2080ti-turing:Dockerfile`, find the section that
installs the shim (look for `flash_attn_turing` in `pip install`
lines) and either:
- Remove the shim install (let real flash_attn be used)
- Add a check: install the shim only if no real flash_attn is
  present (skip shim if we're going to install a real one)
- Replace the shim install with a thin no-op wrapper that defers
  to whatever is actually importable

This is a single-line change in the Dockerfile + an image rebuild.
Once merged, every container built from the new image will use our
sm_75 build when `--use-flash-attention` is set.

**Alternative (workaround):** keep the shim, override via
`docker-compose.yaml` bind mount
(`/opt/flash-sm75-build/flash_attn:/opt/venv/lib/python3.12/site-packages/flash_attn`).
Already tested; works for our build but is fragile (depends on
host paths existing).

### Sub-task 03.2 — sm_75 fp16-input kernel files (1–2 days)

The current `Dao-AILab/flash-attention:csrc/flash_attn/src/` has
`flash_fwd_hdim*_sm80.cu` files. These hard-code sm_80 mma atom
(m16n8k16 with fp16 accumulator) and have `__CUDA_ARCH__ >= 800`
guards. For sm_75, the kernel just prints "FATAL" and returns
uninitialized memory.

**Add new sm_75 `.cu` files** using the existing CUTLASS
`SM75_16x8x8_F32F16F16F32_TN` (m16n8k8 fp32-accumulator, fp16-input)
atom. The atom is already in `kernel_traits.h` from upstream — we
just need to instantiate it.

**Files to add** (alongside existing `flash_fwd_hdim*_sm80.cu`):
- `flash_fwd_hdim32_fp16_sm75.cu`
- `flash_fwd_hdim32_bf16_sm75.cu`
- `flash_fwd_hdim64_fp16_sm75.cu`
- `flash_fwd_hdim64_bf16_sm75.cu`
- `flash_fwd_hdim128_fp16_sm75.cu`
- `flash_fwd_hdim128_bf16_sm75.cu`
- (causal variants as needed)
- Same for bwd if the workspace needs backward pass

**Pattern (per file):**
```cpp
// Copyright (c) 2024, Tri Dao + sm_75 port.
// Auto-generated. See "generate_kernels.py"
#include "namespace_config.h"
#include "flash_fwd_launch_template.h"

namespace FLASH_NAMESPACE {

template<>
void run_mha_fwd_<cutlass::half_t, 128, false>(Flash_fwd_params &params, cudaStream_t stream) {
    run_mha_fwd_hdim128<cutlass::half_t, false>(params, stream);
}

} // namespace FLASH_NAMESPACE
```

**Build invocation:**
- Add `if cc_major == 7 && cc_minor == 5: add_cuda_gencode(cc_flag, "compute_75", "sm_75")` in `setup.py`
- Build with `FLASH_ATTN_CUDA_ARCHS=75` for sm_75-only build
- Or `FLASH_ATTN_CUDA_ARCHS=75;80;90;100;120` for heterogeneous

**Pass criteria:**
- 6/6 unit tests pass with the new build
- E2E Qwen-Image 2.1 with `--use-flash-attention` produces output
  within 1% of v0.2.1 baseline (we expect output close to identical
  since the underlying sm_80 vs sm_75 math is comparable for bf16
  inputs)
- Speed within 5% of v0.2.1 (sm_75 mma throughput is the same class)
- No "FATAL" messages in kernel output

### Sub-task 03.3 — sm_75 int8 convrot input (DEFERRED)

H3 uses int8 convrot weights. The flash kernel would need to
accept int8 or pre-dequantized fp16 values that may be out of the
normal fp16 range.

**Option A:** add a CUTLASS `SM75_16x8x8_INT8INT8INT32_TN`
instantiation. Sm_75 has no dedicated INT8 tensor cores, so
throughput will be ~1/4 of fp16 path. Probably not worth the
engineering cost.

**Option B:** dequantize convrot to fp16 in a wrapper before the
attention call. Preserves flash kernel design but adds compute
overhead. May not be faster than `--use-ck-attention` for H3.

**Recommendation:** defer TODO-9.3 indefinitely. Document H3 as
"requires ck-attention" in README. If H3 becomes critical, revisit
with a dedicated port (this would be a separate Task 04, ~1 week
of work).

## Milestones

- [ ] **M-03.1** (4-8h) — shim bypass; install script tested on fresh
  host; image rebuilt
- [ ] **M-03.2** (1-2 days) — sm_75 fp16-input `.cu` files; build
  succeeds; 6/6 unit tests pass; E2E Qwen correct
- [ ] **M-03.3** (deferred) — int8 convrot input; if needed

## Test cases

For M-03.2, the existing `tests/test_sm75_correctness.py` covers
the kernel correctness. New tests needed:

- `tests/test_sm75_real_build.py` — verifies the installed
  `flash_attn` is the Dao-AILab sm_75 build (not the shim). Check:
  `flash_attn.flash_attn_func.__module__` should be
  `flash_attn.flash_attn_interface`, NOT `flash_attn_turing`.
- E2E Qwen-Image 2.1 e2e test must produce 2.29 MB PNG (matches
  v0.2.1 baseline)
- E2E H3 e2e test still produces garbage (since TODO-9.3 deferred)

## Progress

| Date | Status | Notes |
|---|---|---|
| 2026-10-18 | Plan written | Awaiting vaskes approval |
| (future) | Branch created | `feature/sm75-mma-kernel-port` |
| (future) | M-03.1 done | shim bypassed, image rebuilt |
| (future) | M-03.2 done | sm_75 fp16-input files, tests pass |
| (future) | v0.3 tagged | supersedes v0.2.1 |
| (deferred) | M-03.3 done? | only if H3 is critical |

## Resolution

(To be filled when complete.)

## Files to be created/modified

In `vaskes/comfyui-2080ti-turing`:
- `Dockerfile` — modify to not install shim (or install conditionally)
- (no change to compose.yaml needed if Dockerfile is fixed)

In `vaskes/flash-attention-sm75` (this repo):
- `patches/0002-sm75-fp16-kernels.patch` (new) — adds the sm_75 `.cu`
  files
- `patches/0003-sm75-fwd-dispatch.patch` (new) — modifies
  `flash_fwd_launch_template.h` to use the sm_75 kernels for
  `cc_major == 7`
- `tests/test_sm75_real_build.py` (new) — verifies the right
  flash_attn is loaded
- `docs/H3_DIAGNOSTIC.md` (existing) — updated to "resolved" once
  M-03.2 is done

In workstation `/opt/flash-sm75-build/` (not committed to git,
but used as the build source):
- New `.cu` files in `csrc/flash_attn/src/`
- Modified `flash_fwd_launch_template.h` and `flash_bwd_launch_template.h`
  to dispatch to sm_75 paths

## Risks and mitigations

- **Risk:** sm_75 mma atom may have lower throughput than sm_80 mma
  for fp16 (the m16n8k8 vs m16n8k16 difference is 2x peak FLOPs).
  **Mitigation:** benchmark E2E before/after; if significantly
  slower, document and consider whether 9.5% per-step speedup is
  worth the sm_75-specific code path. (Likely yes — the actual
  speedup vs ck was small in v0.2.1 anyway.)

- **Risk:** the sm_80 files may have hidden sm_80-specific PTX
  inline assembly that doesn't translate to sm_75. **Mitigation:**
  start with hdim=64 (smallest), get it working, then scale up
  to hdim=128 incrementally.

- **Risk:** 1-2 day estimate may slip if there are subtle issues
  with the CUTLASS sm_75 atom and our existing templates.
  **Mitigation:** use the upstream `kernel_traits.h` test as a
  template; sm_75 mma is documented in CUTLASS.

- **Risk:** build time. The new `.cu` files double compilation time
  (one set for sm_80, one for sm_75). With CCache, this is
  amortized. **Mitigation:** use `ccache` if not already configured.

## Not in scope

- Backward pass (bwd) on sm_75 — only forward is needed for
  inference. Training on sm_75 is not a use case.
- sm_70 (Volta) or older — not on the roadmap.
- New int8 mma atom — sm_75 has no dedicated INT8 tensor cores.
  Not worth the engineering cost.
- Generic flash-attention improvements unrelated to sm_75.

## Cross-references

- `docs/H3_DIAGNOSTIC.md` — full root-cause analysis
- `docs/TODO.md` §TODO-9 — high-level plan
- `docs/CHANGELOG.md` v0.2.1 — release notes for the bf16/fp16-only
  freeze
- `docs/VERSION_PINS.md` — version pinning (torch 2.14.1+cu130,
  CUDA 13.0, nvidia cu13 wheels, CUTLASS df18f5e4)
- `tests/RESULTS.md` — current 6/6 unit test pass results
- `tests/SPEED.md` — current Qwen speed measurements
- `docs/upstream-analysis.md` — file-by-file walkthrough of sm_75
  in upstream Dao-AILab

## Decision needed from vaskes

1. **Approve the plan above** to start work on M-03.1 (shim bypass).
2. **Or** reject and ask for changes (e.g. different scope,
   different priority, or "release v0.2.1 as-is and never
   revisit H3").

If approved, I'll create a branch `feature/sm75-mma-kernel-port`
and start with the Dockerfile change (M-03.1).
