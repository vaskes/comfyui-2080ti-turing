# Strategy Decision: B (fork upstream + patch) over A (new kernel from scratch)

**Date:** 2026-10-17
**Decision:** Strategy B
**Made by:** Mavis coder, with vaskes approval

## Background

The original task document
[`docs/tasks/01-flash-sm75-native.md`](../tasks/01-flash-sm75-native.md)
proposed two strategies for restoring `--use-flash-attention` support
on sm_75 in current ComfyUI:

- **Strategy A:** write a new CUDA kernel from scratch in this repo,
  ship as a ComfyUI custom node.
- **Strategy B:** fork [`Dao-AILab/flash-attention`](https://github.com/Dao-AILab/flash-attention)
  and re-enable sm_75 in the build matrix (the kernel code is
  already sm_75-aware).

The M0 milestone (research, 0.5 day) was dedicated to choosing
between these.

## What the research found

`Dao-AILab/flash-attention` was cloned to `/opt/research/flash-attention`
on the workstation host. The relevant findings:

1. **License: BSD 3-Clause** — © 2023 Tri Dao. Compatible with our
   proprietary Kuduza license (we keep the BSD-3 copyright on the
   upstream code and our proprietary work is governed by Kuduza).

2. **Current build matrix:** `setup.py:75` reads
   `FLASH_ATTN_CUDA_ARCHS` env var, defaulting to `"80;90;100;110;120"`.
   The default **excludes sm_75**. Setting
   `FLASH_ATTN_CUDA_ARCHS=75;80;...` is necessary but not sufficient;
   the arch list parsing in `add_cuda_gencodes` does not have a
   branch for `"75"`.

3. **The sm_75 code path is already in the kernel.** The file
   `csrc/flash_attn/src/kernel_traits.h:39` contains:

   ```cpp
   #if defined(__CUDA_ARCH__) &&  __CUDA_ARCH__ >= 750
       using SmemCopyAtom = Copy_Atom<SM75_U32x4_LDSM_N, elem_type>;
       using SmemCopyAtomTransposed = Copy_Atom<SM75_U16x8_LDSM_T, elem_type>;
   #else
       using SmemCopyAtom = Copy_Atom<DefaultCopy, elem_type>;
       using SmemCopyAtomTransposed = Copy_Atom<DefaultCopy, elem_type>;
   #endif
   ```

   And earlier (line ~30):

   ```cpp
   #else
       using MMA_Atom_Arch = MMA_Atom<SM75_16x8x8_F32F16F16F32_TN>;
   #endif
   ```

   This is the **CUTLASS m16n8k8 fp16 → fp32 mma atom**, which is
   exactly the right design for EOL hardware: fp16 storage, fp32
   accumulator in the softmax, no precision-loss trap.

4. **The patch is 5 lines.** Re-enabling sm_75 in the build matrix
   requires adding a branch to `add_cuda_gencodes` in `setup.py`:

   ```python
   if "75" in archs:
       cc_flag += ["-gencode", "arch=compute_75,code=sm_75"]
       supported_archs.append("75")
   ```

   This is the **entire** patch. Nothing in `csrc/` needs to change.

## Why Strategy B is the right choice

1. **5 minutes of work vs ~7-10 days of CUDA dev.** The kernel code
   already exists; only the build matrix was missing the entry. The
   precision-loss trap that the existing community forks
   (`ssiu/flash-attention-turing`, `KLM-corporation/flash-attention-2080ti`)
   fall into is the fp16-accumulator mma instruction
   (`mma.sync.aligned.m8n8k8.f16.f16.f16.f16`), which the official
   repo **does not** use for sm_75 — it uses the fp32-accumulator
   m16n8k8 via CUTLASS.

2. **ComfyUI integration is free.** `--use-flash-attention` already
   exists in ComfyUI and works through the standard
   `from flash_attn import flash_attn_func` interface. No custom
   node, no `optimized_attention_override` patch, no ComfyUI code
   change. The deliverable is a `pip install` of the patched
   upstream.

3. **Future flash-attention algorithmic improvements (FA3, FA4,
   sparsity, sliding window) flow in automatically.** We do not
   have to maintain the algorithmic frontier for sm_75 — we just
   have to keep the build green for sm_75 when new flash-attention
   versions are released.

4. **The official flash-attention is well-tested.** The community
   forks exist because their authors tried to re-implement what the
   official repo already has. We are avoiding that re-implementation
   entirely.

5. **License is compatible.** BSD-3-Clause allows derivative works
   including proprietary ones, as long as the BSD-3 copyright is
   preserved (which we do, by keeping the upstream `LICENSE` file
   verbatim in our directory and the `NOTICE.md` listing the
   upstream copyright).

## When Strategy B would be the wrong choice

If the patched flash-attention, when built and run on sm_75 for a
diffusion model (Qwen-Image 2.1, H3), still produces garbage output,
the failure mode is **algorithmic, not build-related**. In that case
Strategy A becomes the fallback:

- The flash-attention m16n8k8 fp16 path, while fp32-accumulator, may
  still have an algorithmic precision loss specific to diffusion
  models (vs LLMs which flash-attention was originally designed for).
- A custom kernel targeted at diffusion attention patterns
  (e.g. with per-row quantization of attention scores) might be
  needed.

We do not yet have evidence that Strategy A would succeed where B
fails; the M2 milestone (kernel correctness) will produce that
evidence. If M2 fails, we revisit Strategy A as a fallback, with the
option of forking the sm_75-aware kernel code from official
flash-attention (which is BSD-3-Clause, also compatible with our
proprietary license) into a custom kernel in this repo.

## References

- [`Dao-AILab/flash-attention`](https://github.com/Dao-AILab/flash-attention) — BSD-3-Clause.
- [`docs/tasks/01-flash-sm75-native.md`](../tasks/01-flash-sm75-native.md) — original task with M0 milestone.
- The two community forks that this approach **does not** try to
  replicate: `ssiu/flash-attention-turing`,
  `KLM-corporation/flash-attention-2080ti`.
