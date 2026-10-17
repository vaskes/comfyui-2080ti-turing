# Upstream analysis: how sm_75 support is wired in Dao-AILab/flash-attention

**Repo:** https://github.com/Dao-AILab/flash-attention
**Commit analyzed:** `965c921` (master HEAD as of 2026-10-17)
**License:** BSD 3-Clause (© 2023 Tri Dao)

This document describes exactly **how** the sm_75 support is wired
in the official flash-attention repository, so a future Mavis
implementer can verify that the 5-line patch in `patches/0001-add-sm75-arch.patch`
is sufficient to re-enable it.

## File-by-file analysis

### `setup.py:75` — the arch list source

```python
@functools.lru_cache(maxsize=None)
def cuda_archs() -> str:
    return os.getenv("FLASH_ATTN_CUDA_ARCHS", "80;90;100;110;120").split(";")
```

The default excludes `75`. Setting `FLASH_ATTN_CUDA_ARCHS=75;80;90;100;120`
is the **first** of two changes needed.

### `setup.py:103-150` — the `add_cuda_gencodes` function

This function converts a list of arch strings (e.g. `["75", "80", "90"]`)
into a list of `-gencode arch=compute_X,code=sm_X` flags for nvcc.
The current code has explicit branches for `"80"`, `"90"`, `"100"`,
`"120"`, `"121"`, `"110"`, but **no branch for `"75"`**. This is
the **second** of two changes needed (and the only one in `setup.py`
proper).

### `csrc/flash_attn/src/kernel_traits.h:30-40` — the sm_75 mma atom

```cpp
using MMA_Atom_Arch = std::conditional_t<
    std::is_same_v<elem_type, cutlass::half_t>,
    MMA_Atom<SM80_16x8x16_F32F16F16F32_TN>,
    MMA_Atom<SM80_16x8x16_F32BF16BF16F32_TN>
>;
#else
using MMA_Atom_Arch = MMA_Atom<SM75_16x8x8_F32F16F16F32_TN>;
#endif
```

This is the **CUTLASS `SM75_16x8x8_F32F16F16F32_TN`** mma atom:
- Shape: **m=16, n=8, k=8** (Turing's native fp16 mma shape with
  fp32 accumulator)
- Input: `fp16` (half)
- Accumulator: **`fp32`** (the key design point — softmax in fp32
  is what makes this kernel precision-correct for sm_75)
- Layout: TN (Q in row-major, K in column-major)

This is the **opposite** of the community forks' approach. The
forks use the raw `mma.sync.aligned.m8n8k8.f16.f16.f16.f16` PTX
instruction (the one without an fp32 accumulator) and pay the
precision price. The official flash-attention uses the CUTLASS
wrapper which **does** have an fp32 accumulator on sm_75.

### `csrc/flash_attn/src/kernel_traits.h:39` — the sm_75 ldmatrix

```cpp
#if defined(__CUDA_ARCH__) &&  __CUDA_ARCH__ >= 750
    using SmemCopyAtom = Copy_Atom<SM75_U32x4_LDSM_N, elem_type>;
    using SmemCopyAtomTransposed = Copy_Atom<SM75_U16x8_LDSM_T, elem_type>;
#else
    using SmemCopyAtom = Copy_Atom<DefaultCopy, elem_type>;
    using SmemCopyAtomTransposed = Copy_Atom<DefaultCopy, elem_type>;
#endif
```

This is the sm_75-specific `ldmatrix` instruction, which is the
correct way to load fp16 fragments from shared memory into the
mma atom's register layout on Turing. The `__CUDA_ARCH__ >= 750`
guard means it is selected **at compile time** when the sm_75
arch flag is passed to nvcc.

### Other files

- `csrc/flash_attn/src/philox.cuh` — checked at upstream
  `965c921`, no sm_75-specific `#if __CUDA_ARCH__ >= 750` guard.
  The unconditional code is correct on sm_75 because the sm_75
  mma atom is selected entirely by the kernel_traits.h template
  specialization; philox is a separate concern.
- `csrc/flash_attn/src/flash_fwd_kernel.h` and friends — these are
  the actual flash attention forward kernel implementations, which
  use the sm_75 code paths automatically when compiled with
  `-gencode arch=compute_75,code=sm_75`.

## What the patch needs to do

The complete patch is:

1. **`setup.py`:** add a branch in `add_cuda_gencodes` for the
   string `"75"` (5 lines, see `patches/0001-add-sm75-arch.patch`).
2. **Build invocation:** pass `FLASH_ATTN_CUDA_ARCHS=75;80;...`
   when running `pip install` or `python setup.py build_ext`.

That is the entire patch. No CUDA source changes, no CUTLASS
patching, no kernel rewriting.

## What the patch does **not** do

The patch does not:

- Touch any file under `csrc/`. All sm_75 source code is already
  upstream.
- Add new tests. The official flash-attention test suite will be
  run on sm_75 as part of `python3 -m pytest` (assuming the user
  runs it; we do not enable the test suite by default in the
  `pip install` flow).
- Document sm_75 in the upstream README. We document it in this
  directory's `README.md` and in the parent repo's `README.md`.

## What could go wrong (and the diagnostic)

If after applying the patch, the build fails, the most likely
causes are:

1. **CUTLASS headers not found.** The `csrc/flash_attn/CMakeLists.txt`
   or `setup.py` may reference CUTLASS headers that need to be
   available. Check `/usr/local/cuda/include/cutlass/` and the
   cu13 wheel at `/opt/venv/lib/python3.12/site-packages/nvidia/cu13/`.
2. **C++ standard mismatch.** Flash-attention uses C++20 (per
   upstream commit `edb5c76` "compile with c++20 for pytorch 2.13+").
   This is consistent with our existing
   `comfyui-workstation:cu130-torch-stable-gcc` image which has
   GCC 13.3.0 supporting C++20.
3. **cuSparse/cuBlas headers.** The same workaround as for
   `SageAttention-T4` applies: add
   `/opt/venv/lib/python3.12/site-packages/nvidia/cu13/{include,lib}`
   to the include/library paths.

These are all build-side issues, not algorithmic. The kernel
itself, if it builds, is expected to produce correct output because
it uses fp32 accumulators throughout the softmax path.

## References

- `Dao-AILab/flash-attention:965c921:csrc/flash_attn/src/kernel_traits.h:30-40`
- `Dao-AILab/flash-attention:965c921:setup.py:75,103-150`
- CUTLASS documentation on `SM75_16x8x8_F32F16F16F32_TN`:
  https://github.com/NVIDIA/cutlass/blob/main/include/cutlass/arch/mma_sm75.h
- `docs/strategy-decision.md` in this directory — why we chose
  Strategy B over Strategy A.
