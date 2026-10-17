# FlashAttention sm_75 (Turing) — EOL support restoration

Restores `--use-flash-attention` support for NVIDIA Turing (sm_75) GPUs
(RTX 2080 Ti, RTX 2080, RTX 2070, RTX 2060, Tesla T4, Quadro RTX) in
**current ComfyUI** (`comfyui-2080ti-turing` repo).

## What this is

The official [`Dao-AILab/flash-attention`](https://github.com/Dao-AILab/flash-attention)
repository dropped sm_75 support in current releases — not because
sm_75 is broken, but because NVIDIA marked Turing as EOL and the
maintainer (Tri Dao) decided not to keep the sm_75 path tested.

This directory contains a **minimal patch** that re-enables sm_75 in
the build matrix. The patch is 5 lines in `setup.py`:

```python
# sm_75 (Turing) - EOL hardware, but kernel code in csrc/flash_attn/src/
# already supports it via CUTLASS SM75_16x8x8_F32F16F16F32_TN (m16n8k8 fp16->fp32)
if "75" in archs:
    cc_flag += ["-gencode", "arch=compute_75,code=sm_75"]
    supported_archs.append("75")
```

**The kernel code itself is already sm_75-aware.** The
`csrc/flash_attn/src/kernel_traits.h` file at the official repo
includes:

```cpp
#else
    using MMA_Atom_Arch = MMA_Atom<SM75_16x8x8_F32F16F16F32_TN>;
#endif

#if defined(__CUDA_ARCH__) &&  __CUDA_ARCH__ >= 750
    using SmemCopyAtom = Copy_Atom<SM75_U32x4_LDSM_N, elem_type>;
    using SmemCopyAtomTransposed = Copy_Atom<SM75_U16x8_LDSM_T, elem_type>;
```

This means sm_75 is supported via the **m16n8k8 fp16 → fp32** mma
instruction, which has **fp32 accumulator** — the right design for
EOL hardware.

All that was missing was the arch list entry in `setup.py`. The
patch here adds it back.

## Quick start

```bash
# Inside the comfyui container:
cd /tmp
git clone https://github.com/Dao-AILab/flash-attention.git
cd flash-attention
git checkout 965c921   # the commit this patch was generated against
patch -p1 < /opt/comfyui-git/flash-attention-sm75/patches/0001-add-sm75-arch.patch
FLASH_ATTN_CUDA_ARCHS="75;80;90;100;120" FLASH_ATTENTION_FORCE_BUILD=TRUE \
    /opt/venv/bin/pip install --no-build-isolation --no-deps -e .
# Now test:
python3 -c "from flash_attn import flash_attn_func; print('OK')"
```

In ComfyUI, `--use-flash-attention` should now work on sm_75 with
correct output.

## File layout

```
flash-attention-sm75/
├── README.md                              # this file
├── patches/
│   └── 0001-add-sm75-arch.patch           # the 5-line setup.py patch
├── scripts/
│   └── apply-patch-and-install.sh         # convenience installer
├── tests/
│   ├── test_sm75_correctness.py           # unit tests vs SDPA reference
│   └── test_sm75_e2e.py                   # end-to-end with real workflows
├── docs/
│   ├── strategy-decision.md               # why Strategy B over A
│   ├── upstream-analysis.md               # how the sm_75 support is wired
│   └── tickets/
│       ├── M0-strategy-selection.md       # closed, Strategy B chosen
│       ├── M1-build-infra.md              # this is what M1 is
│       ├── M2-kernel-correctness.md
│       ├── M3-comfyui-integration.md
│       ├── M4-e2e-qwen-h3.md
│       ├── M5-speed-validation.md
│       └── M6-docs-cleanup.md
├── LICENSE                                # BSD-3-Clause (inherited)
└── CREDITS.md                             # Dao-AILab, Tri Dao, etc.
```

## Validation status

| Milestone | Status |
|---|---|
| M0: Strategy selection | ✅ done (Strategy B chosen) |
| M1: Build infra | 🚧 in progress |
| M2: Kernel correctness | ⏳ pending |
| M3: ComfyUI integration | ⏳ pending |
| M4: E2E Qwen + H3 | ⏳ pending |
| M5: Speed validation | ⏳ pending |
| M6: Docs + cleanup | ⏳ pending |

## References

- **Upstream:** [`Dao-AILab/flash-attention`](https://github.com/Dao-AILab/flash-attention) — BSD-3-Clause, © 2023 Tri Dao.
- **Task document:** [`docs/tasks/01-flash-sm75-native.md`](../tasks/01-flash-sm75-native.md) — the original task with full context.
