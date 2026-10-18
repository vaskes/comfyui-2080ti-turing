# Third-Party Notices — `flash-attention-sm75/`

This subdirectory is part of the [vaskes/comfyui-2080ti-turing](https://github.com/vaskes/comfyui-2080ti-turing)
repository and is **MIT-licensed** under the same terms as the root
[`LICENSE`](../../LICENSE). The `LICENSE` file in this directory is a
copy of the root MIT LICENSE for provenance; it has the same legal
effect.

This subdirectory contains a **5-line patch** to
[`Dao-AILab/flash-attention`](https://github.com/Dao-AILab/flash-attention)
that re-enables sm_75 (Turing) support in the build matrix. The patch
is too small to have a meaningful copyright status on its own. The
upstream Dao-AILab/flash-attention project (whose source is consumed
at build time via `pip install` but is **not redistributed** by this
subdirectory) is **BSD-3-Clause** — see
[Dao-AILab/flash-attention](https://github.com/Dao-AILab/flash-attention)
for upstream license terms.

## What this patch does

- Adds 5 lines to `setup.py` to include `sm_75` in the list of
  supported architectures when the user requests it via
  `FLASH_ATTN_CUDA_ARCHS`.
- Does not modify any CUDA source code in `csrc/`. The sm_75 code
  paths in `csrc/flash_attn/src/kernel_traits.h` (CUTLASS
  `SM75_16x8x8_F32F16F16F32_TN` mma atom, `SM75_U32x4_LDSM_N` and
  `SM75_U16x8_LDSM_T` ldmatrix wrappers) are upstream code, unchanged.

See [`patches/0001-add-sm75-arch.patch`](patches/0001-add-sm75-arch.patch)
and [`patches/0002-relax-arch-gates.patch`](patches/0002-relax-arch-gates.patch).

## Upstream: Dao-AILab/flash-attention

- **Repository:** https://github.com/Dao-AILab/flash-attention
- **License:** BSD-3-Clause (Copyright © 2023, Tri Dao)
- **Upstream commit this patch was generated against:** `965c921`

The full upstream BSD-3-Clause license text is at
https://github.com/Dao-AILab/flash-attention/blob/main/LICENSE. It
applies to the upstream kernel code only (which this subdirectory
does not redistribute).

## How to obtain the upstream source

```bash
git clone https://github.com/Dao-AILab/flash-attention.git
cd flash-attention
git checkout 965c921   # commit this patch was generated against
```

The patch is applied by the matrix runner via `docker exec` against
the bind-mounted `/tmp/flash-attention-build` source directory —
no separate install script is needed.
