# M-03.2 discovery: the kernel port is MUCH simpler than the plan estimated (2026-10-18)

## What I expected (per the original plan)

Per `docs/tasks/03-sm75-mma-kernel-port.md` M-03.2:

> "Add new sm_75 .cu files alongside existing flash_fwd_hdim*_sm80.cu
> using the existing CUTLASS SM75_16x8x8_F32F16F16F32_TN
> (m16n8k8 fp32-accumulator, fp16-input) atom."
>
> "Files to add (alongside existing flash_fwd_hdim*_sm80.cu):
> - flash_fwd_hdim32_fp16_sm75.cu
> - flash_fwd_hdim32_bf16_sm75.cu
> - flash_fwd_hdim64_fp16_sm75.cu
> - flash_fwd_hdim64_bf16_sm75.cu
> - flash_fwd_hdim128_fp16_sm75.cu
> - flash_fwd_hdim128_bf16_sm75.cu
> - (causal variants as needed)"
>
> Total: 1-2 days.

## What I actually found

**The sm_75 infrastructure is ALREADY in upstream Dao-AILab
flash-attention.** I had assumed the sm_75 mma atom was missing and
needed to be added. It isn't.

### Finding 1: `kernel_traits.h` has sm_75 mma atom for `__CUDA_ARCH__ < 800`

In `csrc/flash_attn/src/kernel_traits.h:30-36`:

```cpp
#if defined(__CUDA_ARCH__) &&  __CUDA_ARCH__ >= 800
    using MMA_Atom_Arch = std::conditional_t<
        std::is_same_v<elem_type, cutlass::half_t>,
        MMA_Atom<SM80_16x8x16_F32F16F16F32_TN>,
        MMA_Atom<SM80_16x8x16_F32BF16BF16F32_TN>
    >;
#else
    using MMA_Atom_Arch = MMA_Atom<SM75_16x8x8_F32F16F16F32_TN>;
#endif
```

When compiled for sm_75 (`__CUDA_ARCH__ == 750`), the kernel
automatically uses `SM75_16x8x8_F32F16F16F32_TN` (m16n8k8,
fp32-accumulator, fp16-input). Same for `SmemCopyAtom` which uses
`SM75_U32x4_LDSM_N`.

### Finding 2: The .cu files are NOT sm_80-specific

`flash_fwd_hdim64_fp16_sm80.cu` is 8 lines:

```cpp
#include "namespace_config.h"
#include "flash_fwd_launch_template.h"

namespace FLASH_NAMESPACE {
template<>
void run_mha_fwd_<cutlass::half_t, 64, false>(Flash_fwd_params &params, cudaStream_t stream) {
    run_mha_fwd_hdim64<cutlass::half_t, false>(params, stream);
}
}
```

It just instantiates a template. The template is in
`flash_fwd_launch_template.h` which uses
`Flash_fwd_kernel_traits<Headdim, kBlockM, kBlockN, 4, false, false, T>`.
That trait, in `kernel_traits.h`, adapts to `__CUDA_ARCH__`.

So the .cu files are 100% sm_75-compatible. The `_sm80` suffix in
the filename is just a label, not a constraint.

### Finding 3: Two runtime gates block sm_75

Only two things prevent sm_75 from working at runtime:

1. `csrc/flash_attn/src/flash_fwd_launch_template.h:17`:
   ```cpp
   #if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
   #define ARCH_SUPPORTS_FLASH
   ```
   Gates the kernel from being compiled-in for sm_75 (else the
   kernel is replaced with a printf-only stub).

2. `csrc/flash_attn/flash_api.cpp:393, 572, 836, 1050, 1271`:
   ```cpp
   bool is_sm8x_min = cc_major >= 8;
   TORCH_CHECK(is_sm8x_min, "FlashAttention only supports Ampere GPUs or newer.");
   ```
   Raises an error if called on a non-Ampere GPU.

### The fix

- `patches/0002-relax-arch-gates.patch` (62 lines):
  - Change `__CUDA_ARCH__ >= 800` to `__CUDA_ARCH__ >= 750` in
    `flash_fwd_launch_template.h` (1 line)
  - Change `cc_major >= 8` to `cc_major >= 8 || (cc_major == 7 && cc_minor == 5)`
    in 5 places in `flash_api.cpp` (5 lines)
- Rebuild (50+ min for full hdim/dtype set on sm_75).

That's it. NO new .cu files. NO new CUTLASS dispatch. NO sm_75
kernel port. The 1-2 day plan was based on a wrong assumption.

## What this means for the project

1. **v0.3 is feasible in 1-2 hours of work, not 1-2 days.** The
   plan was over-engineered.

2. **The M-03.3 sub-task (int8 convrot) is still independent of
   M-03.1/M-03.2.** Whether sm_75 mma atom handles H3 int8 convrot
   is a separate question — to be tested by running H3 e2e after
   M-03.2 build.

3. **bwd (backward pass) is NOT fixed by patch 0002.** I only
   changed the fwd launch template, not the bwd one. If/when bwd
   is needed for sm_75, the same fix applies to
   `flash_bwd_launch_template.h:19`.

4. **Build is slow on sm_75 (50+ min).** Each .cu takes 1-2 min
   due to heavy template instantiations. 4-thread parallelism is
   bottlenecked by the heaviest file. Could be optimized by
   building only hdim 32/64/96/128 (skip 192, 256) but that's a
   future optimization.

## Test plan after build completes

1. **Unit tests**: 6/6 from `tests/test_sm75_correctness.py`
2. **E2E Qwen-Image 2.1**: 25 steps, --use-flash-attention
3. **E2E H3**: 8 steps, --use-flash-attention (the test that was
   failing in v0.2.1)
4. If H3 still broken, document as TODO-9.3 (sm_75 int8 convrot
   path — separate work, deferred per plan)

## Cross-project lesson

**When porting GPU kernels to a new architecture, FIRST check if
the upstream codebase already has the architecture-adaptive
template infrastructure (e.g. `#if __CUDA_ARCH__` switches in
kernel_traits.h).** I spent 4+ hours planning the kernel port
without verifying whether the underlying mma atom was already in
the template. Always inspect the existing trait/templating code
before designing new files.

**Applies to any future EOL hardware port**: Volta (sm_70),
Maxwell (sm_50), etc. The pattern is the same — there's likely
already a compile-time branch for the older arch that "just works"
once the runtime gates are relaxed.
