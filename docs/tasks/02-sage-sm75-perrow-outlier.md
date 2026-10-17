# Task 02: SageAttention sm_75 with per-row INT8 + outlier smoothing

| Field | Value |
|---|---|
| **Repo** | `vaskes/comfyui-2080ti-turing` (in `docs/tasks/`) |
| **Target HW** | NVIDIA RTX 2080 Ti, sm_75 (Turing), 22 GB VRAM |
| **Date created** | 2026-10-17 |
| **Status** | NOT STARTED |
| **Owner (proposer)** | vaskes |
| **Implementation** | Mavis coder (this Mavis session or future one) |
| **Code review** | Qwen (separate agent, called per checkpoint) |
| **Branch convention** | `sage-sm75-perrow-v0.1`, `...-v0.2`, ... |
| **Estimated effort** | 5–7 working days (one engineer) |
| **Outcome type** | Replacement kernel for the existing `SageAttention-T4` custom node, in this repo |

---

## Abstract

We need **`SageAttention` to work on sm_75 (Turing) hardware in current
ComfyUI**, for **any model**, with correct quality. This is **not** a
"make Qwen-Image 2.1 garbage-free in 90 s" task — it is an **EOL
hardware support restoration** task, the same framing as Task 01
(`docs/tasks/01-flash-sm75-native.md`).

The official `thu-ml/SageAttention` repository has **no sm_75 support**
in any of its current versions (the maintainer excluded sm_75 from the
build matrix because NVIDIA marked Turing as EOL — same policy
decision that caused `Dao-AILab/flash-attention` to drop sm_75). The
sm_75 community has tried three times to keep the path alive, and all
three attempts produce garbage output for diffusion models:

- `Ph0rk0z/SageAttention2` sm_75 fork — compiles, "the CUDA kernel
  can run, but the results are pretty wrong"
- `1506086927/SageAttention2_Optimized_Test` — same pattern
- `THE-ANGEL-AI/SageAttention-SM75-path` (the one currently shipped
  in `custom_nodes/SageAttention-T4/` of this repo) — applies the
  critical `.satfinite` PTX fix, gets the kernel *running*, but still
  produces purple/green dot garbage for Qwen-Image 2.1 (validated
  2026-10-17, 165 s for 25 steps, output 2.77 MB but visually wrong)

The reason all three fail is that they use **per-warp INT8
quantization** — one scale factor per group of 32 threads. This is
the same precision the sm_80+ versions of SageAttention use, but
sm_75 is less forgiving: the bf16 storage of Qwen's attention scores
has a wider dynamic range than the fp16 storage used by sm_80+
models, and per-warp quantization cannot capture the outliers that
show up row-by-row.

The fix is **per-row INT8 quantization** (one scale factor per Q row
and one per K row, instead of per warp), plus **outlier smoothing**
(moving the largest values toward the median to bring them into the
INT8 range without truncation). This is the algorithm from the
SageAttention 2 paper (Zhang et al. 2024, arXiv 2411.10958), adapted
to sm_75's `mma.m8n8k16.satfinite.s32.s8.s8.s32` INT8 path.

The scope of "any model" is intentionally broad. Concretely, the
kernel must work for (at minimum):

- **Qwen-Image 2.1** (bf16 + INT8-quant text encoders) — the test bench
  used throughout this repo
- **Minimax H3** (INT8 convrot) — also a known test bench in this repo
- Any future model that ComfyUI master ships — the kernel must be
  general, not Qwen-specific

Reference for "correct" is whatever the current `--use-ck-attention`
(comfy_kitchen `sol_attn`) produces today for the same model + prompt.
We are not trying to match some abstract "fp16 reference" — we are
trying to match the **existing production output** so that switching
from `--use-ck-attention` to a sage-based backend does not visibly
change the image.

Speed is a secondary, aspirational concern:

- If the new sage kernel produces correct output and is **the same
  speed as** the ck-attention baseline (~140 s for Qwen), that is a
  success. The user can then choose either backend based on other
  considerations (e.g. memory footprint, future model compatibility).
- If the new sage kernel is **faster** than ck-attention, that is a
  bonus — and unlike Task 01, the INT8 path has a realistic
  *chance* of being faster (per-row INT8 on a tensor-core path uses
  ~1.6× the throughput of fp16 GEMM, which is exactly the kind of
  speedup that beats the ck-attention fp16 GEMM baseline).
- If the new sage kernel is **slower** than ck-attention, it is still
  a success as long as it is correct, because the goal is EOL support,
  not a speed record.

The goal of this task is to **build the per-row + outlier version
inside the existing `custom_nodes/SageAttention-T4/` directory**,
validate it beats the per-warp quality, and merge back so the same
custom node is more useful for sm_75.

---

## Context (as of 2026-10-17, validated and committed)

**Image:** `comfyui-workstation:cu130-torch-stable-gcc` (built in this repo)
**Stack:** torch 2.14.1 + cu130, GCC 13.3.0, CUDA 13.0, nvidia cu13 wheel.
**Container:** `comfyui`, port 8188, GPU 0, 22 GB VRAM.
**Existing `SageAttention-T4` custom node:** built, runs, but produces
garbage quality on Qwen. Lives in
`custom_nodes/SageAttention-T4/`. **This is the file tree we extend,
not replace.**

**Validated measurements (Qwen-Image 2.1, 25 steps, seed=65451515):**

| Setup | Time | Output | Verdict |
|---|---|---|---|
| `ck-attention` (comfy_kitchen) | 140 s | 2.28 MB ✅ | baseline |
| `SageAttention-T4` per-warp INT8 (current THE-ANGEL-AI) | 165 s | dot garbage ❌ | the thing we improve |
| `SageAttention-T4 smooth_k=False` | 160 s | dot garbage ❌ | confirms smooth_k not the fix |

**Why the per-warp kernel fails — the math:**

`SageAttention-T4` does INT8 quantization **per warp** — all 32 threads
in a warp share one scale factor and one zero point. This means **all
values in a warp must fit in the same INT8 range** `[-127, 127]`. The
scale is `max(|warp|) / 127`. If one thread has a value of 100.0 and
another has 0.5, the smaller one becomes a quantized integer of `0.5 *
127/100 = 0.6`, which rounds to 1, losing half the precision.

For sm_80+ this is fine because the Q/K activations there are typically
in fp16 storage with a more uniform distribution. For Qwen on sm_75
(bf16 storage, larger dynamic range, the qwen3_8b text encoder
specifically has `linear_attn.in_proj_a/b.weight` with values up to 65k),
the warp-level scale factor is dominated by outliers and the bulk of the
values get crushed into a few quantization bins, destroying the softmax
precision.

**Per-row** quantization fixes this because the outliers in one Q row
do not affect the scale of a different Q row. The cost is a small
metadata overhead: instead of one scale per warp, we now have one scale
per Q row (and one per K row), which is `O(L)` extra storage per
attention call. For typical diffusion attention with `L ≤ 4096` and
32 heads, this is at most a few hundred KB — negligible.

**Outlier smoothing** (from the SageAttention 2 paper) further improves
this by replacing each value `x` with `x - mean(x) * alpha + mean(x) * alpha`
(a small constant shift toward the row mean), which pulls outliers
closer to the bulk distribution. The paper claims this makes INT4
quantization "lossless" for many LLMs; for our INT8 path, the smoothing
should make a 4–10× reduction in quantization error on the tails.

---

## Problem we solve

**The single concrete problem:** *make the SageAttention-T4 kernel
produce correct output for any diffusion model on RTX 2080 Ti (sm_75,
EOL hardware) in the current ComfyUI.* Speed is secondary; correctness
is primary. Qwen-Image 2.1 is the **test case**, not the **goal** —
the same EOL-support framing as Task 01.

Today, the SageAttention-T4 INT8 attention backend is **effectively
broken for any diffusion model on sm_75**: it produces purple/green
dot garbage (Qwen-Image 2.1, validated 2026-10-17), and H3 has the same
problem on the sm_75 INT8 path. The three community attempts
(`Ph0rk0z`, `1506086927`, `THE-ANGEL-AI`) all converge on the same
failure mode — per-warp INT8 quantization loses too much precision
for bf16 storage models. This is a **kernel design** problem, not a
hardware impossibility.

The scope of "any model" is intentionally broad. Concretely, the
kernel must work for (at minimum):

- **Qwen-Image 2.1** (bf16 + INT8-quant text encoders) — the test
  bench used throughout this repo
- **Minimax H3** (INT8 convrot) — also a known test bench in this repo
- Any future model that ComfyUI master ships — the kernel must be
  general, not Qwen-specific

Reference for "correct" is whatever the current `--use-ck-attention`
(comfy_kitchen `sol_attn`) produces today for the same model + prompt.
We are not trying to match some abstract "fp16 reference" — we are
trying to match the **existing production output** so that switching
from `--use-ck-attention` to a sage-based backend does not visibly
change the image.

Speed is a secondary, aspirational concern:

- If the new sage kernel produces correct output and is **the same
  speed as** the ck-attention baseline (~140 s for Qwen), that is a
  success. The user can then choose either backend based on other
  considerations.
- If the new sage kernel is **faster** than ck-attention, that is a
  bonus. Unlike Task 01, the INT8 path has a realistic chance of
  being faster — per-row INT8 on a tensor-core path uses ~1.6× the
  throughput of fp16 GEMM, which is exactly the kind of speedup
  that beats the ck-attention fp16 GEMM baseline.
- If the new sage kernel is **slower** than ck-attention, it is still
  a success as long as it is correct, because the goal is EOL support,
  not a speed record.

What we **do not** try to solve in this task:

- Writing a fp16-only flash kernel. That is **Task 01** (FlashAttention
  sm_75 native). Do not duplicate the architecture diagram from
  `docs/tasks/01-flash-sm75-native.md` here; this task assumes the
  INT8 mma path is the only viable one for sm_75 attention
  acceleration.
- bf16 → fp16 model weight conversion. Already proven to break
  Qwen-Image 2.1 in a previous commit on this repo.

---

## How to solve

### Architecture

We extend the existing `custom_nodes/SageAttention-T4/` directory with
new files. The existing per-warp kernel stays as a fallback option
(selected by a new parameter `quant_gran = "per_warp"`); the new
per-row + outlier kernel is the default (`quant_gran = "per_row"`).

**No new directory.** No new custom node class name. The user-facing
API of the ComfyUI node gains one new parameter; everything else is
unchanged. This is critical — vaskes's workflows reference
`SageAttentionT4_Apply` by name, and we must not break them.

### File layout (added to existing tree, paths relative to repo root)

```
custom_nodes/SageAttention-T4/
├── (existing files unchanged)
│
├── csrc/qattn/
│   ├── (existing)
│   │   ├── attn_cuda_sm75.h              # per-warp kernel (THE-ANGEL-AI original)
│   │   ├── qk_int_sv_f16_cuda_sm75.cu    # per-warp kernel source
│   │   └── pybind_sm75.cpp
│   │
│   ├── perrow/                           # NEW: this task's kernel
│   │   ├── __init__.py                   # not a Python dir, just a marker
│   │   ├── pybind_perrow.cpp             # pybind11 module
│   │   ├── qk_int_sv_f16_cuda_sm75_perrow.cu    # per-row kernel source
│   │   ├── attn_cuda_sm75_perrow.h       # per-row kernel template
│   │   ├── quant_perrow.cuh              # per-row INT8 quant + outlier smoothing
│   │   ├── smooth_k.cuh                  # outlier smoothing (SageAttention 2 paper)
│   │   └── mma_perrow.cuh                # sm_75 INT8 mma wrappers, per-row layout
│   │
│   └── quant_test.py                     # NEW: Python-side check that
│                                          # quantize/dequantize round-trip
│                                          # is within tolerance
│
├── sageattention/
│   ├── core.py                           # dispatch: per_row (default) or per_warp
│   ├── quant_perrow.py                   # NEW: Python reference impl of the
│   │                                      # per-row quantization, used in
│   │                                      # quant_test.py
│   └── (existing __init__.py exports sageattn, sageattn_qk_int8_pv_fp16_cuda_sm75)
│
├── comfyui_nodes/
│   ├── sageattention_node.py             # add `quant_gran` parameter
│   └── quant_gran_options.py             # NEW: enum, validation
│
├── test_quant_perrow.py                  # NEW: correctness vs per-warp + pytorch SDPA
├── bench/
│   └── bench_perrow_vs_perwarp.py        # NEW: speed comparison
│
└── KNOWN_ISSUES.md                       # append: per-row known issues
```

### Kernel design (per-row INT8 + outlier smoothing)

The algorithm follows the SageAttention 2 paper (Section 3.1) but
**adapted to sm_75's `mma.m8n8k16.satfinite.s32.s8.s8.s32` shape**:

```
Input: q, k, v in fp16, shape [B, H, L, D]  (D ∈ {64, 128})
Output: o in fp16, shape [B, H, L, D]

For each (B, H) head:
  # Step 1: outlier smoothing (SageAttention 2, Eq. 2)
  # alpha = sigmoid(mean(|q|) - some threshold)
  # alpha is per-row, computed from |q[row]|'s mean.
  # We use alpha=1.0 (full smoothing) when |q| mean > 5.0, else 0.0.
  q_smoothed = q - alpha * (q.mean(dim=-1, keepdim=True) - some_target)
  
  # Step 2: per-row INT8 quantization of q
  # One scale + zero per Q row.
  q_max = q_smoothed.abs().max(dim=-1, keepdim=True)  # [B, H, L, 1]
  q_scale = q_max / 127.0                             # [B, H, L, 1]
  q_zero = 0.0                                         # symmetric quant
  q_int8 = round(q_smoothed / q_scale).clamp(-127, 127).to(int8)
  
  # Step 3: per-row INT8 quantization of k
  k_max = k.abs().max(dim=-2, keepdim=True)            # [B, H, 1, L]  (per-K-row scale)
  # NOTE: K scale is per K row, not per K col, because the mma.m8n8k16
  # shape processes K in 16-element chunks per warp. Per-K-row scale aligns
  # with this chunking.
  k_scale = k_max / 127.0                              # [B, H, 1, L]
  k_int8 = round(k / k_scale).clamp(-127, 127).to(int8)
  
  # Step 4: QK^T via sm_75 INT8 mma
  # mma.m8n8k16.satfinite.s32.s8.s8.s32 (Turing native)
  # Accumulator fp32 (the .satfinite is on the input operands, not the accum).
  qk_int32 = q_int8 @ k_int8.T                         # [B, H, L, L] in int32
  qk_fp32 = qk_int32.to(fp32) * q_scale * k_scale.T    # dequantize
  
  # Step 5: scale by 1/sqrt(D)
  qk_fp32 *= (1.0 / sqrt(D))
  
  # Step 6: online softmax in fp32 (same as Task 01)
  m = qk_fp32.max(dim=-1, keepdim=True)
  p = exp(qk_fp32 - m)
  l = p.sum(dim=-1, keepdim=True)
  p_fp16 = (p / l).to(fp16)                            # normalize and cast
  
  # Step 7: PV via sm_75 FP16 mma (m8n8k4)
  # v is fp16, no quantization (V is read-only once, no recompute benefit from quant).
  o = p_fp16 @ v                                        # [B, H, L, D] in fp16
```

**Key differences from the per-warp THE-ANGEL-AI kernel:**

| | per-warp (current) | per-row (this task) |
|---|---|---|
| INT8 scale scope | one per warp (32 threads) | one per Q row, one per K row |
| Outlier handling | `smooth_k` global flag | per-row alpha = sigmoid-style |
| `q_scale` storage | implicit (recomputed) | explicit tensor `[B, H, L, 1]` |
| `k_scale` storage | implicit (recomputed) | explicit tensor `[B, H, 1, L]` |
| Memory cost (Qwen) | 0 extra bytes | ~256 KB extra per attention call (negligible) |
| Expected quality | garbage for Qwen | per SageAttention 2 paper: ~1.8× lower MRE than per-tensor |

### ComfyUI integration

The custom node `SageAttentionT4_Apply` gains one new parameter:

```python
"quant_gran": (["per_row", "per_warp"], {
    "default": "per_row",
    "label": "Quantization granularity (per_row=less precision loss, per_warp=legacy)"
})
```

The `sageattn` Python function in `sageattention/core.py` gains a new
keyword argument `quant_gran="per_row"` and dispatches:

- `quant_gran="per_row"` → calls `sageattn_qk_int8_pv_fp16_cuda_sm75_perrow`
  (new, in `csrc/qattn/perrow/`).
- `quant_gran="per_warp"` → calls `sageattn_qk_int8_pv_fp16_cuda_sm75`
  (existing, in `csrc/qattn/`).

The existing `smooth_k` parameter is kept (it remains useful in the
per-warp path; in the per-row path it can be redundant with the new
per-row smoothing, but having both is harmless).

The self-disable-after-3-fails pattern is unchanged. Add a separate
failure counter for the per-row path so that a per-row failure does
not disable the per-warp fallback (and vice versa).

### Build system

The `setup.py` is extended with a new extension:

```python
perrow_extension = CUDAExtension(
    name="sageattention._qattn_sm75_perrow",
    sources=[
        "csrc/qattn/perrow/pybind_perrow.cpp",
        "csrc/qattn/perrow/qk_int_sv_f16_cuda_sm75_perrow.cu",
    ],
    include_dirs=[NVIDIA_CU13_INCLUDE],
    library_dirs=[NVIDIA_CU13_LIB],
    extra_compile_args={
        "cxx": CXX_FLAGS,
        "nvcc": NVCC_FLAGS,
    },
)
ext_modules.append(perrow_extension)
```

Build:
```bash
cd custom_nodes/SageAttention-T4
TORCH_CUDA_ARCH_LIST="7.5" /opt/venv/bin/python3 setup.py build_ext --inplace
TORCH_CUDA_ARCH_LIST="7.5" /opt/venv/bin/pip install --no-build-isolation --no-deps -e .
```

The C++20 + cu13 include workaround (validated 2026-10-17) is unchanged.
See `docs/tasks/01-flash-sm75-native.md` for the rationale.

### What NOT to do

- **Do not** modify the existing per-warp kernel. It is the reference
  and the fallback. The two kernels must coexist.
- **Do not** change the `SageAttentionT4_Apply` node's existing
  parameters. Only add the new `quant_gran` parameter.
- **Do not** add new Python dependencies beyond what the existing
  `SageAttention-T4` already needs.
- **Do not** attempt INT4 quantization in this task. SageAttention 2
  paper does INT4, but that requires per-thread granularity that is
  much harder to get right than per-row. INT8 with per-row is the
  smallest viable improvement.
- **Do not** smooth V. Only smooth Q (and K, if we add K smoothing).
  Smoothing V would change the output distribution visibly.
- **Do not** use symmetric quant for K. Asymmetric (with zero point)
  would be marginally better, but symmetric is simpler and matches the
  per-warp kernel. Asymmetric is a future task.
- **Do not** change the `optimized_attention_override` payload
  structure. ComfyUI workflows that use the node must continue to work
  without modification (modulo the new parameter being optional).

---

## Milestones

### M1 — Build the per-row extension alongside the existing per-warp one (1 day)

- [ ] `custom_nodes/SageAttention-T4/csrc/qattn/perrow/` directory
  created with the file layout above.
- [ ] `pybind_perrow.cpp` declares the new function
  `qk_int8_sv_f16_accum_f32_attn_sm75_perrow` and registers it in
  the same pybind module as the existing per-warp function
  (or in a separate module — TBD by Mavis coder based on
  `sageattention/core.py` dispatching).
- [ ] `setup.py` extended to build the new extension.
- [ ] `pip install -e .` succeeds, both `_qattn_sm75.so` and
  `_qattn_sm75_perrow.so` exist in
  `custom_nodes/SageAttention-T4/sageattention/`.
- [ ] A smoke-test in `quant_test.py` imports the new module and
  calls the function with a 4×4×16×16 input — output shape correct.
- [ ] **Qwen code review:** review the build setup diff, confirm the
  C++20 + cu13 include pattern is reused, not changed.

**Pass:** `python3 -c "from sageattention._qattn_sm75_perrow import qk_int8_sv_f16_accum_f32_attn_sm75_perrow; print('OK')"`
prints `OK`.

### M2 — Per-row quantization round-trip correctness (1.5 days)

- [ ] `sageattention/quant_perrow.py` is a **reference implementation**
  in pure PyTorch of the per-row quantization + outlier smoothing,
  with a `@torch.no_grad()` decorator.
- [ ] `test_quant_perrow.py` runs the following cases, each must
  produce `max_abs_diff < 0.05` between the reference fp16 input
  and the dequantized output:
  - `B=1, H=4, L=128, D=64`, q/k = `randn * 0.1`
  - `B=1, H=4, L=128, D=128`, q/k = `randn * 0.1`
  - Same with `randn * 1.0` (wider distribution, no smoothing)
  - Same with `randn * 0.1 + outliers` (synthetic outliers, full
    smoothing should kick in)
  - qwen-realistic: load a slice of Qwen's actual q/k tensors from
    `/opt/comfyui/models/diffusion_models/qwen_image_2.1/qwen_image_2.1_bf16.safetensors`,
    convert to fp16, run quant/dequant
- [ ] **Qwen code review:** review the outlier smoothing formula
  against the SageAttention 2 paper Eq. 2.

**Pass:** all 5 cases produce `max_abs_diff < 0.05` for the
quantization round-trip alone (this is the **unit test for the
quantization, not the full attention** — that comes in M3).

### M3 — Kernel correctness on small inputs (2 days)

- [ ] `qk_int_sv_f16_cuda_sm75_perrow.cu` implements the full
  attention algorithm using the per-row quantization from M2.
- [ ] `test_quant_perrow.py` extended to run the kernel against
  pytorch SDPA reference. Test cases:
  - `B=1, H=4, L=128, D=64`, fp16 storage
  - `B=1, H=4, L=128, D=128`, fp16 storage
  - `B=2, H=8, L=256, D=128`, fp16 storage
  - `B=1, H=4, L=512, D=128`, fp16 storage
  - With causal mask, head_dim=128
- [ ] Acceptance: `max_abs_diff < 0.1` (note: looser than Task 01's
  `1e-2`, because INT8 attention *will* lose some precision; the
  goal is "visually correct output" not "bit-identical to fp16").
- [ ] **Qwen code review:** review the kernel, specifically check:
  - per-row scale is recomputed correctly (not cached from a wrong
    shape)
  - outlier smoothing is **off** when not needed (alpha=0) and
    **on** when needed (alpha=1) — both code paths tested
  - the sm_75 INT8 mma `.satfinite` PTX modifier is present on both
    the `kInplaceUpdate` and `kInit` branches (lesson learned from
    THE-ANGEL-AI's 2026-07-03 fix)

**Pass:** all 5 test cases produce `max_abs_diff < 0.1` vs SDPA
reference.

### M4 — ComfyUI node integration with `quant_gran` parameter (0.5 day)

- [ ] `comfyui_nodes/sageattention_node.py` adds the `quant_gran`
  parameter with default `"per_row"`.
- [ ] `sageattention/core.py` dispatches to the per-row or per-warp
  kernel based on the parameter.
- [ ] The `sageattn` Python function exposes `quant_gran` as a
  keyword argument with default `"per_row"`.
- [ ] ComfyUI restart shows the new parameter in the node UI.
- [ ] **Qwen code review:** review the parameter plumbing, confirm
  existing workflows without the new parameter still work (default
  to `"per_row"`).

**Pass:** the node UI shows `quant_gran` as a dropdown with
`"per_row"` (default) and `"per_warp"` (legacy).

### M5 — End-to-end correctness on Qwen and H3 (1 day)

- [ ] A workflow `test_wf_qwen_perrow.json` is created, identical
  to `image_qwen_image_2_1_t2i_api_SAGE.json` but with the new
  `quant_gran="per_row"` (default, so could be omitted) in the
  `SageAttentionT4_Apply` node.
- [ ] Running the workflow with the new node on **Qwen-Image 2.1**
  produces a PNG that is **visually correct** — recognizable subject,
  no purple/green dots, no gray square. Visual inspection by vaskes.
- [ ] Running the workflow with the new node on **Minimax H3**
  (`test_wf_h3_perrow.json` is a copy of
  `video_minimax_h3_t2v_api.json` with the new node added) produces
  a video that is **visually correct** — H3 INT8 convrot with the
  per-row kernel must not regress the existing H3 output quality.
- [ ] **Qwen code review:** review the workflows, the node parameter
  change, both test outputs.

**Pass:** the workflow output is a recognizable image (Qwen) and a
recognizable video (H3). The bar is "not garbage"; pixel-level match
to the ck-attention baseline is not required.

### M6 — Speed validation (1 day)

- [ ] `bench/bench_perrow_vs_perwarp.py` runs both kernels on the
  Qwen and H3 workflows and reports timing.
- [ ] Per-row kernel time on Qwen must be **at most 5% slower** than
  the ck-attention baseline (140 s × 1.05 = 147 s). This is the
  *correctness-and-compatibility* bar — the user can choose the new
  backend as a quality-correct alternative, even at the same speed.
  Unlike Task 01, the INT8 path has a realistic chance of being
  faster than fp16 GEMM (per-row INT8 m8n8k16.satfinite has ~1.6×
  the throughput of m8n8k4 fp16 on sm_75), so the bonus bar is
  "strictly faster than ck-attention."
- [ ] If the per-row kernel is **more than 5% slower**, investigate
  the cause (memory traffic for the scale tensors? dequantization
  not fused with the mma?) and document in `KNOWN_ISSUES.md`.
- [ ] Per-row kernel time on H3 must be **at most 5% slower** than
  the ck-attention baseline (1:45 × 1.05). Same logic.
- [ ] (Optional, bonus) If per-row is **strictly faster** than
  ck-attention on both Qwen and H3, document the speedup in the
  commit message and in the root `README.md`.
- [ ] **Qwen code review:** review the bench script, confirm
  methodology (warm cache, median of 3).

**Pass:** Qwen per-row time ≤ 147 s, H3 per-row time ≤ 1:45 × 1.05.
The 140 s ceiling is **not** a pass criterion (it is a stretch goal).
Per-row being ≤ per-warp × 1.05 is **not** a pass criterion either
(per-warp is itself broken on Qwen, so per-row could be much faster
or much slower depending on the bottleneck — both are valid outcomes
on the correctness axis).

### M7 — Documentation, README, commit (0.5 day)

- [ ] `custom_nodes/SageAttention-T4/README.md` updated with a
  section "Per-row quantization (since v0.2)" describing the new
  mode, the algorithm, the expected quality improvement.
- [ ] `custom_nodes/SageAttention-T4/KNOWN_ISSUES.md` updated with
  per-row specific issues if any.
- [ ] Commit with message referencing this task file:
  `Task: docs/tasks/02-sage-sm75-perrow-outlier.md`.
- [ ] Tag: `sage-sm75-v0.2` (the existing `sage-sm75-v0.1` is
  reserved for the initial THE-ANGEL-AI port; this is the
  per-row improvement).
- [ ] Push to `origin main`.
- [ ] Update root `README.md` to reflect the new quality verdict
  for `SageAttention-T4`.

**Pass:** `git tag -l` includes `sage-sm75-v0.2`; root README's
"SageAttention-T4 native sm_75 kernel" section is rewritten to
reflect the per-row + outlier version's results.

---

## Test cases

### Test case T1 — Qwen baseline (sanity)

Same as Task 01 / T1. Verifies the comfy_kitchen baseline still works
after the new code is merged.

### Test case T2 — Per-row kernel, Qwen

**Setup:** workflow
`/opt/comfyui/test_wf_qwen_perrow.json` (the new file from M5).
The workflow has one `SageAttentionT4_Apply` node with default
parameters (which means `quant_gran="per_row"`, `smooth_k=True`,
`enable=True`).

**Run:**
```bash
ssh workstation "python3 /tmp/comfyui_run.py /opt/comfyui/test_wf_qwen_perrow.json --label perrow-qwen --timeout 600"
```

**Pass:**
- PNG file size > 1.5 MB.
- PNG visually shows a recognizable subject (visual inspection by
  vaskes). **The bar is "not garbage"; pixel-level match to baseline
  is not required.**
- Time ≤ 165 s (the per-warp kernel time) × 1.05.
- ComfyUI log contains
  `sageattn_qk_int8_pv_fp16_cuda_sm75_perrow` invocations (i.e. the
  new kernel was actually called, not the per-warp one).
- ComfyUI log does **not** contain `SELF-DISABLED` for the per-row
  kernel.

### Test case T3 — Per-warp kernel still works (regression)

**Setup:** same workflow as T2, but with `quant_gran="per_warp"` set
explicitly in the node.

**Run:**
```bash
ssh workstation "python3 /tmp/comfyui_run.py /opt/comfyui/test_wf_qwen_perrow.json --label perwarp-regression --timeout 600"
```

**Pass:** the output is the same garbage as the previous
`SageAttention-T4` setup (purple/green dots). This is a **regression
check** that the per-warp kernel was not broken by the per-row work.
Visual inspection: garbage as before, time ~165 s.

### Test case T4 — M2 unit tests

**Run:** inside the container,
`python3 custom_nodes/SageAttention-T4/test_quant_perrow.py`.

**Pass:** all 5 quantization round-trip cases produce
`max_abs_diff < 0.05`. All 5 attention cases produce
`max_abs_diff < 0.1` vs SDPA reference.

### Test case T5 — H3 regression

**Setup:** workflow
`/opt/comfyui/video_minimax_h3_t2v_api.json` (the existing H3
workflow) with no SageAttention node.

**Run:**
```bash
ssh workstation "python3 /tmp/comfyui_run.py /opt/comfyui/video_minimax_h3_t2v_api.json --label h3-no-sage --timeout 1500"
```

**Pass:** H3 video output is recognizable, time ≤ 1:45 × 1.05.
This is the regression check: H3 must not get worse after the
SageAttention-T4 directory is modified.

Then add a `SageAttentionT4_Apply` node (with `quant_gran="per_row"`)
to a copy of the H3 workflow and re-run. Time should be ≤ 1:45 × 1.05
(i.e. the per-row kernel should be no worse than baseline for H3,
where the per-warp kernel already happened to produce OK output).

### Test case T6 — No-crash stress

**Setup:** any workflow, run 5 times in a row.
**Pass:** no CUDA OOM, no kernel panic, no
`cudaErrorIllegalAddress`. The kernel's failure handler should
fall back to pytorch if anything goes wrong (in this case the
fallback is **the per-warp kernel** as a tertiary fallback, not
just pytorch).

---

## Progress

Mavis coder updates this section.

### Build

- [ ] M1: Build per-row extension alongside per-warp
- [ ] M2: Per-row quantization round-trip correctness
- [ ] M3: Kernel correctness on small inputs
- [ ] M4: ComfyUI node integration with `quant_gran` parameter
- [ ] M5: End-to-end correctness on Qwen and H3
- [ ] M6: Speed validation
- [ ] M7: Documentation, README, commit

### Test results

| Test | Time | Output size | Quality OK? | Notes |
|---|---|---|---|---|
| T1 baseline | | | ✅/❌ | |
| T2 per-row Qwen | | | ✅/❌ | the main result |
| T3 per-warp regression | | | ❌ (expected) | confirms per-warp is unchanged |
| T5a H3 no sage | | | ✅/❌ | |
| T5b H3 with per-row | | | ✅/❌ | |
| T6 stress x5 | | | ✅/❌ | |

### Qwen code review checkpoints

- [ ] After M1: build setup approved
- [ ] After M2: per-row quantization formula approved
- [ ] After M3: kernel logic approved
- [ ] After M4: node parameter plumbing approved
- [ ] After M5: workflow integration approved (Qwen + H3)
- [ ] After M6: bench result approved

### Issues encountered

*(Mavis appends here as work progresses.)*

---

## Resolution

A task is **resolved** when:

1. All milestones M1–M7 are checked off.
2. All test cases T1–T6 pass.
3. Qwen code review has approved all six checkpoints.
4. The commit is pushed and tagged.
5. The root `README.md` is updated.

The bar for "resolved" is **EOL support restored**, not "fastest
sage kernel." A working, correct (quality OK) sage kernel for sm_75
that is the same speed as ck-attention is a successful resolution of
this task.

If **M5 fails** (the kernel still produces garbage even with per-row
quantization + outlier smoothing), the most likely causes are:

- The outlier smoothing formula is wrong. Cross-check against the
  SageAttention 2 paper Eq. 2.
- The per-row scale is being applied to the wrong axis. The K scale
  is `per K row`, which means the scale tensor is shaped
  `[B, H, 1, L]` and gets broadcast over the Q axis. If the shape is
  wrong, the dequantization will be off by orders of magnitude on
  the rows that get the wrong scale.
- The INT8 mma has the missing `.satfinite` issue (see
  `custom_nodes/SageAttention-T4/KNOWN_ISSUES.md` 2026-07-03 fix).
  Copy the `.satfinite` PTX from `csrc/mma.cuh` in the existing
  per-warp kernel; do not skip this.

Investigate systematically. The per-row quantization round-trip
unit test (M2) is the first check — if it fails, the kernel cannot
succeed. The M3 microbench `max_abs_diff` is the second check. M5
end-to-end is the third. The diagnostic chain is: M2 → M3 → M5.

If **M3 fails** (the microbench shows `max_abs_diff > 0.1`), this is
the same as M5 failure but at a smaller scale. The diagnostic is the
same: localize via the test cases.

If **M6 fails** (per-row is *more than 5% slower* than the
ck-attention baseline), it is still worth committing because the
correctness improvement is the main goal of this task. Document the
speed result honestly in the commit message and in `KNOWN_ISSUES.md`.
A future task could investigate why per-row adds overhead (likely:
extra memory traffic for the scale tensors, and the dequantization
is not fused with the mma). The repo still gains a working
EOL-support sage kernel, and future work can target speed
separately.

---

## References (for Mavis coder, in this repo)

- `custom_nodes/SageAttention-T4/KNOWN_ISSUES.md` — the existing
  per-warp fix history. Specifically the
  `Missing .satfinite in INT8 MMA PTX instruction` section explains
  why the existing kernel was producing NaN before the fix.
- `custom_nodes/SageAttention-T4/csrc/qattn/attn_cuda_sm75.h` —
  the existing per-warp kernel. **Read this entire file before
  writing the per-row kernel.** The mma wrappers, the ldmatrix
  helpers, the softmax structure are all reusable.
- `custom_nodes/SageAttention-T4/csrc/mma.cuh` — the sm_75 MMA
  wrappers, including the fixed `.satfinite` PTX.
- `custom_nodes/SageAttention-T4/sageattention/core.py` — the
  Python dispatch layer. Add a new entry point for the per-row
  kernel.
- `docs/tasks/01-flash-sm75-native.md` — the companion task.
  **Match the file layout, the build setup, the node API, the
  test methodology.** The two tasks should feel like they were
  written by the same hand.

## References (external, for context)

- SageAttention 2 paper: Zhang et al. 2024, "SageAttention2: Efficient
  Attention with Thorough Outlier Smoothing and Per-thread INT4
  Quantization." arXiv 2411.10958. Section 3.1 (Outlier Smoothing)
  and Section 3.2 (Per-thread Quantization) are the direct
  references for this task. The INT4 path is *not* what we are
  implementing (we use INT8 for sm_75's `mma.m8n8k16.s8s8s32`
  path), but the outlier smoothing formula is directly applicable.
- INT-FlashAttention paper: arXiv 2409.16997. The per-token
  quantization strategy is what we are approximating with per-row.
  Section 3.2.1 ("Per-token Quantization") is the relevant
  reference; the "82% smaller quantization error" claim from the
  abstract is the benchmark we are aiming for (relative to
  per-tensor, not relative to per-warp — the per-warp comparison
  is what we measure here).
- Ph0rk0z/SageAttention2 sm_75 fork — read the README, understand
  what was tried, what failed, and what was committed but not
  benchmarked. The fork's `mma nan is fixed but speeds still not
  beating xformers when compiled` note from 2025-12-13 is the
  honest reality check on what's possible with per-warp
  quantization on sm_75.
- sm_75 mma reference: NVIDIA PTX ISA documentation,
  "mma.sync.aligned.m8n8k16.row.col.satfinite.s32.s8.s8.s32" —
  the INT8 mma with fp32 accumulator. **Must** use the
  `.satfinite` modifier on the input operands (per THE-ANGEL-AI's
  2026-07-03 fix in `KNOWN_ISSUES.md`).
