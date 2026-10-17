# Task 01: FlashAttention native for sm_75 (Turing)

| Field | Value |
|---|---|
| **Repo** | `vaskes/comfyui-2080ti-turing` (in `docs/tasks/`) |
| **Target HW** | NVIDIA RTX 2080 Ti, sm_75 (Turing), 22 GB VRAM |
| **Date created** | 2026-10-17 |
| **Status** | NOT STARTED |
| **Owner (proposer)** | vaskes |
| **Implementation** | Mavis coder (this Mavis session or future one) |
| **Code review** | Qwen (separate agent, called per checkpoint) |
| **Branch convention** | `flash-sm75-native-v0.1`, `...-v0.2`, ... |
| **Estimated effort** | 7–10 working days (one engineer) |
| **Outcome type** | New ComfyUI custom node + CUDA extension in this repo |

---

## Abstract

We need a **native sm_75 (Turing) FlashAttention kernel** that produces **correct
output** on RTX 2080 Ti in **less time than the current 140 s baseline**
(`--use-ck-attention` + original bf16 Qwen-Image 2.1 models, comfy_kitchen
`sol_attn`).

The current state of attention on sm_75 is:

- `pytorch SDPA` (default) — 175 s for 25 steps, correct output
- `comfy_kitchen sol_attn` (`--use-ck-attention`) — 140 s, correct output
- `flash_attn` (any fork — ssiu, KLM-corporation) — 107–130 s, **gray square garbage**
- `SageAttention-T4` (THE-ANGEL-AI) — 165 s, **purple/green dot garbage**

We need a **new kernel** that beats 140 s while keeping quality equivalent to
the comfy_kitchen baseline.

**Why this is hard and interesting:** sm_75 (Turing) has *no* fp16 attention
tensor-core path. All existing `flash_attn` forks fail on this architecture
because they convert bf16 storage → fp16 for compute, and fp16's ~7-bit
mantissa cannot hold the bf16-scale attention scores' softmax precision. The
output of those kernels has std 0.81 vs reference 0.12 (a 6.6× distortion in
output distribution).

We have two real architectural choices for a sm_75 flash-style kernel:

1. **fp16 tensor core + fp32 accumulation in softmax** — use the *general*
   mma.fp16.fp16.fp16 instruction that Turing has for normal GEMM, with K
   broken into m8n8k4 tiles, and run softmax in fp32 with online correction
   (FlashAttention algorithm). This avoids the precision-loss trap.
2. **cp.async + manual fp16 GEMM with online softmax** — same as above but
   using `cp.async.bulk` for K/V loads to keep the SM busy. This is the
   Ampere-style FlashAttention pattern; Turing has `cp.async` (added in
   CUDA 11.4+ for sm_75, with reduced throughput vs sm_80).

The goal of this task is to **build option 1 first** (simpler, no
cp.async surface), validate it, and then optionally add option 2 as a
separate kernel in the same extension for comparison.

---

## Context (as of 2026-10-17, validated and committed)

**Image:** `comfyui-workstation:cu130-torch-stable-gcc` (built in this repo)
**Stack:** torch 2.14.1 + cu130, GCC 13.3.0, Triton 3.8.0, CUDA toolkit 13.0
(headers in `/usr/local/cuda/include`), nvidia cu13 wheel headers in
`/opt/venv/lib/python3.12/site-packages/nvidia/cu13/include`.
**Container:** `comfyui`, port 8188, GPU 0, 22 GB VRAM, 64 GB host RAM.
**ComfyUI:** master + comfy_kitchen 0.2.38 in `custom_nodes/comfy_kitchen/`.
**Existing `SageAttention-T4` custom node:** built, runs, but produces garbage
quality. Lives in `custom_nodes/SageAttention-T4/`. Reused as **build reference
for the new kernel** (same setup.py, same include paths, same torch patches).

**Validated measurements (Qwen-Image 2.1, 25 steps, seed=65451515):**

| Setup | Time | Output (bytes) | Verdict |
|---|---|---|---|
| pytorch SDPA (default) | 175 s | 2 339 947 | ✅ correct |
| `ck-attention` (comfy_kitchen) | 140 s | 2 283 593 | ✅ correct, **baseline** |
| `flash_attn` ssiu fork | 107 s | gray 24 KB | ❌ |
| `flash_attn` KLM fork | 130 s | gray 24 KB | ❌ |
| `SageAttention-T4` (THE-ANGEL-AI) | 165 s | 2 773 743 (dot garbage) | ❌ |
| `SageAttention-T4 smooth_k=False` | 160 s | 2 769 084 (dot garbage) | ❌ |

Reference output of the **baseline** (ck-attention, 140 s, 2.28 MB) is
captured at `/workspace/output/qwen-history/Qwen_image_2.1_best_140s.png`
in the agent's local workspace, and at `/opt/comfyui/output/Qwen_image_2.1_00016.png`
on the workstation host. These are the pixel reference for "quality OK" in
the test cases below.

**flash_attn sm_75 precision diagnosis (do not re-derive):**

| | flash_attn sm_75 kernel | pytorch SDPA reference |
|---|---|---|
| input dtype | bf16 | bf16 |
| output dtype | fp16 (kernel converts internally) | bf16 |
| output std | 0.8101 | 0.1226 |
| max abs diff | 2.87 | — |

This was measured by Mavis 2026-10-17 and committed as a tripwire
in `tests/test_drift_release_claims.py`-style approach. The conclusion in
plain language: sm_75 fp16-only kernels cannot hold bf16 attention scores.
The new kernel must avoid fp16 for the softmax path; it must use fp32 for
softmax accumulator and online correction.

---

## Problem we solve

**The single concrete problem:** *make Qwen-Image 2.1 on RTX 2080 Ti
faster than 140 s without breaking quality.*

Secondary problem (do not lose sight of): the kernel should also work for
**Minimax H3 (INT8 convrot)**. H3 currently runs at 1:45 with
`--use-ck-attention`. A native flash kernel for H3 should at minimum not
make it slower; ideally faster (target: ≤ 1:20, ~25% reduction).

What we **do not** try to solve in this task:

- bf16 → fp16 model weight conversion (proven to break qwen-image 2.1, see
  previous commit on this repo).
- Model-side quantization (w4a8, w6a8, etc.) — out of scope, not an
  attention problem.
- New comfy_kitchen backend integration — out of scope, the new kernel
  ships as a standalone custom node that patches `optimized_attention_override`
  exactly like `SageAttention-T4` does.

---

## How to solve

### Architecture

A single new Python package + CUDA extension under
`custom_nodes/FlashAttention-sm75/`. Mirrors the structure of
`SageAttention-T4/` so a future Mavis can diff them. Two CUDA kernels:

1. `qk_fp16_accum_fp32_sm75` — flash attention with fp16 mma for QK^T and
   PV, fp32 accumulator for both. Online softmax in fp32, with running
   max and denominator. Compatible with head_dim ∈ {64, 128} (the two
   values used by Qwen-Image 2.1 and H3 attention layers).
2. `qk_int8_pv_fp16_accum_fp32_perrow_sm75` — the *good* version of
   SageAttention-T4 with per-row INT8 scaling and outlier smoothing.
   This is the **Sage task** (Task 02) — listed here for completeness so
   the build system has one entry point; the per-row work is owned by
   Task 02 and reviewed separately.

For this task (Task 01) we build **only kernel 1** plus the ComfyUI
custom-node glue.

### File layout (all paths relative to repo root `comfyui-2080ti-turing/`)

```
custom_nodes/FlashAttention-sm75/
├── __init__.py                 # re-export NODE_CLASS_MAPPINGS
├── flash_sm75_nodes.py         # bridge for ComfyUI auto-discovery
├── flashattention/
│   ├── __init__.py             # from .core import flash_attn
│   ├── core.py                 # Python-side dispatch, auto-dtype conversion
│   └── quant.py                # placeholder (not used in this task)
├── csrc/
│   ├── flash/
│   │   ├── pybind.cpp          # pybind11 module
│   │   ├── flash_attn_fp16_sm75.cu  # the actual kernel
│   │   ├── flash_attn_fp16_sm75.h   # kernel template
│   │   ├── mma.cuh             # sm_75 mma wrappers (m8n8k4 fp16, m8n8k16 int8)
│   │   ├── smem.cuh            # shared memory layout
│   │   ├── softmax.cuh         # online softmax with running max
│   │   └── numeric.cuh         # half ↔ float helpers
│   └── fused/                  # empty placeholder for Task 02
├── setup.py                    # mirrors SageAttention-T4 setup.py
├── pyproject.toml              # torch>=2.3,<3; triton; pybind11
├── README.md                   # what this is, how to build, expected perf
├── KNOWN_ISSUES.md             # maintenance log
├── test_flash_sm75_kernel.py   # microbench + correctness vs pytorch SDPA
├── bench/
│   └── bench_flash_sm75.py     # speed comparison vs comfy_kitchen sol_attn
└── scripts/
    └── build_sm75.sh           # convenience: build_ext --inplace + install
```

The Python `__init__.py` and `flash_sm75_nodes.py` must mirror the **exact
pattern** in `custom_nodes/SageAttention-T4/__init__.py` and
`sageattn_t4_nodes.py`. Reuse these files. Do not invent a new way to
register the custom node — ComfyUI has exactly one auto-discovery
mechanism and we already know it works.

### ComfyUI integration

The custom node exposes exactly two node classes, identical contract to
`SageAttentionT4_Apply` / `SageAttentionT4_Remove`:

- `FlashAttentionT4_Apply` — wraps MODEL with `optimized_attention_override`
  pointing at our `attention_flash_sm75(q, k, v, heads, mask, ...)` function.
  Parameters: `head_dim` (default 128), `block_m`, `block_n` (tuning knobs,
  default 64/64), `enable` (default True), `mask_fallback` (default True
  — fall back to pytorch SDPA when mask is present, same as
  SageAttention-T4 does).
- `FlashAttentionT4_Remove` — strips the override.

Both nodes use the **same self-disable-after-3-fail pattern** as
`SageAttention-T4/comfyui_nodes/sageattention_node.py`. Copy it. Do not
re-invent the failure-handling logic.

The attention function must:
- Accept q, k, v in **NHD layout** (`[B, L, H*D]`) and **HND layout**
  (`[B, H, L, D]`) — ComfyUI passes both, depending on the model.
  Detect via `skip_reshape` argument (matches ComfyUI convention).
- Accept `torch.float32`, `torch.float16`, `torch.bfloat16` q/k/v.
  Internally cast to fp16 for the mma path, cast result back to input
  dtype. **Never use bf16 mma** — that is the sm_80+ path and does not
  exist on sm_75.
- If `mask is not None` and `mask_fallback=True`, return
  `comfy.ldm.modules.attention.attention_pytorch(...)`. Same fallback
  rule as SageAttention-T4.
- On any kernel exception: log warning, increment failure counter, return
  the pytorch fallback. Self-disable after `_max_fails = 3` consecutive
  failures.

### CUDA kernel design (kernel 1: `qk_fp16_accum_fp32_sm75`)

**Block layout** (per CTA, "cooperative thread array" = CUDA block):
- 1 CTA computes 1 output tile of shape `[BM, BN]` of the attention
  output, where `BM = 64` (Q rows) and `BN = 64` (K/V cols in the
  running softmax).
- 1 CTA has 4 warps = 128 threads. 2 warps cooperate on QK^T, 2 warps
  cooperate on PV.
- Each warp owns `BM/2 = 32` Q rows.

**Inner loop over K/V tiles** (FlashAttention algorithm):

```
for each k_tile in range(0, N, BN):
    # Load K[BM, BN] and V[BN, D] into shared memory using ldmatrix
    # (sm_75 has ldmatrix since CUDA 11.0, used correctly by THE-ANGEL-AI
    #  in csrc/qattn/attn_cuda_sm75.h — copy the load pattern).
    cp_async(K_smem, K_global + offset)
    cp_async(V_smem, V_global + offset)
    cp_async_commit()
    cp_async_wait_all()
    ldmatrix(Q_smem_frag, Q_smem)   # load Q tile into registers
    ldmatrix(K_smem_frag, K_smem)   # load K tile into registers
    
    # QK^T: mma.m8n8k4.f16.f16.f16.f16 (Turing native fp16 mma)
    # Compute in m8n8 tiles, accumulate in fp32
    for kk in range(BN, 0, -4):    # k-step 4
        qk_frag += mma_m8n8k4(q_frag, k_frag)  # fp16->fp16 mma, fp32 accum
    
    # Scale by 1/sqrt(D)
    qk_frag *= scale
    
    # Apply mask if present (else skip — add to the running max for safety)
    if mask is not None:
        qk_frag += mask  # fp32 add
    
    # Online softmax in fp32 (FlashAttention trick)
    m_new = max(m_old, max(qk_frag, axis=-1))  # per-row max in fp32
    alpha = exp(m_old - m_new)  # per-row scale in fp32
    p_frag = exp(qk_frag - m_new)  # per-row softmax in fp32
    l_new = alpha * l_old + sum(p_frag, axis=-1)
    
    # Cast p_frag to fp16, do PV via mma.m8n8k4
    p_fp16 = p_frag.to(fp16)
    ldmatrix(V_smem_frag, V_smem)
    o_frag = alpha * o_frag  # fp32
    o_frag += mma_m8n8k4(p_fp16_frag, v_frag)  # fp16 mma, fp32 accum
    
    m_old = m_new
    l_old = l_new

# Final: o_frag /= l_old (per-row), write to global memory
```

**Critical correctness points** (these are the things that the existing
flash_attn forks get wrong on sm_75 — do not repeat):

1. `l_frag` and `m_frag` (running max and denominator) are **fp32**, never
   fp16. Even though K was stored as fp16.
2. The exp() is computed in **fp32**, and the result is cast to fp16
   *after* softmax, not before.
3. The `alpha` rescaling of `o_frag` is **fp32 multiply** (no precision
   loss between iterations).
4. `mma.m8n8k4.f16.f16.f16.f16` accumulates in fp32 (PTX `mma.sync.aligned.m8n8k4.row.col.f32.f16.f16.f16`).
   This is the *only* fp16 mma shape Turing supports that has fp32
   accumulator. m8n8k8 fp16 accumulates in fp16 — **do not use** it.

**Memory loads** (Q, K, V): use `cp.async` for K, V (not for Q — Q is
loaded once and stays in registers/smem). `cp.async` is available on sm_75
since CUDA 11.4 with reduced bandwidth vs sm_80; the expected behavior
is that it still helps overlap memory with compute on the inner K loop.

**Block sizes** (tuning knobs exposed in the ComfyUI node UI):
- `BM` (Q tile rows): 64 (fixed — 4 warps × 16 Q rows per warp)
- `BN` (K/V tile cols): 64 (fixed — 4 warps × 16 K rows per warp, 8 K cols per warp × 2)
- `BD` (head_dim): 64 or 128 (parameterize; 128 is the most common for
  diffusion models)

For H3: head_dim may be 64 or 128 depending on the layer. Test both.

For Qwen-Image 2.1: head_dim = 128 throughout. The kernel will have a
compile-time switch `HEAD_DIM_64` / `HEAD_DIM_128`; runtime dispatches
based on the input tensor.

### Build system

The `setup.py` is essentially a **copy** of
`custom_nodes/SageAttention-T4/setup.py` with the following differences:

- Module name: `flashattention._flash_sm75` (not `sageattention._qattn_sm75`)
- Sources: `csrc/flash/pybind.cpp`, `csrc/flash/flash_attn_fp16_sm75.cu`
- Same arch flags: `arch=compute_75,code=sm_75`
- Same C++ standard: `std=c++20` (this is **required** for torch 2.14;
  C++17 fails with `at::symint::sizes<T>` and `c10::strong_ordering` errors
  in `ExpandUtils.h`. Mavis 2026-10-17 spent 4 hours discovering this;
  do not undo it.)
- Same include_dirs/library_dirs: `/opt/venv/lib/python3.12/site-packages/nvidia/cu13/{include,lib}`
  (because `libcusparse-dev-13-0` does not exist as an apt package on
  Ubuntu 24.04; the cu13 wheel is the only source of cusparse headers.)
- Same flags: `-U__CUDA_NO_HALF_OPERATORS__`, `-U__CUDA_NO_HALF_CONVERSIONS__`,
  `--use_fast_math`, `--threads=8`, `-Xptxas=-v`, `-diag-suppress=174`

The build is run as:
```bash
cd custom_nodes/FlashAttention-sm75
TORCH_CUDA_ARCH_LIST="7.5" /opt/venv/bin/python3 setup.py build_ext --inplace
TORCH_CUDA_ARCH_LIST="7.5" /opt/venv/bin/pip install --no-build-isolation --no-deps -e .
```

Both commands are needed. The first compiles; the second registers the
package so `from flashattention import flash_attn` works.

**Important runtime note** (validated 2026-10-17): `/opt/venv` lives
inside the image filesystem, not a Docker volume. **`docker compose down`
followed by `up` resets `/opt/venv` and the editable install is lost.**
Use `docker compose restart` to keep the install. If a full recreate
happens, re-run the `pip install -e .` above.

### What NOT to do

- **Do not** use `mma.m8n8k8.f16.f16.f16.f16` — accumulates in fp16,
  same precision-loss trap that kills the existing flash_attn forks.
- **Do not** use `mma.m16n8k16` fp16 — does not exist on sm_75 (only
  m8n8k4 fp16 with fp32 accum is available in the fp16 path).
- **Do not** try `mma.m8n8k16.s8.s8.s32` for QK^T in **this** task —
  that is INT8 quantization, which is **Task 02** (Sage per-row).
- **Do not** modify the existing `SageAttention-T4` code in this task.
  It is a separate kernel with separate semantics. Cross-pollinate only
  by reading the build setup, the node registration pattern, and the
  self-disable pattern.
- **Do not** add new Python dependencies beyond what `SageAttention-T4`
  already needs (`torch`, `pybind11`, `triton`). No `numpy` math kernels
  in the hot path.
- **Do not** add `bf16` to the dtype list of supported inputs. The kernel
  will accept it (cast to fp16 internally), but do not advertise it as
  a first-class path — bf16 storage models are the *cause* of the
  existing flash_attn precision issue, and we want to be explicit that
  we cast to fp16 for math.

---

## Milestones

Each milestone is a **discrete deliverable** with explicit pass criteria.
Do not advance to the next milestone until the current one's pass
criteria are met and Qwen code review has approved.

### M1 — Build infrastructure (1 day)

- [ ] `custom_nodes/FlashAttention-sm75/` directory created with the file
  layout above.
- [ ] `setup.py` and `pyproject.toml` copy-validated against
  `SageAttention-T4/setup.py` (diff is intentional: module name,
  source list).
- [ ] `__init__.py`, `flash_sm75_nodes.py`, `flashattention/__init__.py`,
  `flashattention/core.py` present as stubs.
- [ ] `csrc/flash/pybind.cpp` compiles against torch headers (with the
  same `std=c++20` + cu13 include workaround).
- [ ] `csrc/flash/flash_attn_fp16_sm75.cu` is a stub that returns an
  empty tensor and logs "FlashAttention sm_75 stub".
- [ ] `pip install -e .` succeeds, `from flashattention import flash_attn`
  returns a callable that prints the stub message.
- [ ] ComfyUI restart shows `FlashAttention-sm75` import in the
  `custom_nodes` import log without errors.
- [ ] **Qwen code review:** review the build setup diff against
  SageAttention-T4, approve the std=c++20 + cu13 include pattern as
  a reusable template.

**Pass:** `python3 -c "from flashattention import flash_attn; print(flash_attn())"`
prints the stub message. ComfyUI starts cleanly.

### M2 — Kernel correctness on small inputs (2 days)

- [ ] `csrc/flash/flash_attn_fp16_sm75.cu` implements the FlashAttention
  algorithm with fp16 mma + fp32 accumulator, online softmax in fp32,
  per-row rescaling.
- [ ] `test_flash_sm75_kernel.py` runs the following cases (each must
  produce `max_abs_diff < 1e-2` against pytorch SDPA):
  - `B=1, H=4, L=128, D=64`, q/k/v = `randn * 0.1`, fp16 storage
  - `B=1, H=4, L=128, D=128`, fp16 storage
  - `B=2, H=8, L=256, D=128`, fp16 storage
  - `B=1, H=4, L=512, D=128`, fp16 storage (longer context)
  - `B=1, H=4, L=128, D=128`, fp32 storage (cast to fp16 inside)
  - `B=1, H=4, L=128, D=128`, bf16 storage (cast to fp16 inside)
  - With causal mask, head_dim=128
  - With causal mask, head_dim=64
  - With arbitrary additive mask (Mavis should generate the mask; expected:
    same result as SDPA)
- [ ] The `mma.m8n8k4.f16.f16.f16.f16` PTX inline assembly is **only**
  in `csrc/flash/mma.cuh`. The `.cu` file calls a wrapper, never raw PTX.
- [ ] **Qwen code review:** review the kernel logic against the
  FlashAttention paper (Dao et al. 2022), specifically check:
  - online softmax numerical stability (running max, running denominator)
  - fp32 accumulator discipline (no fp16 path in the softmax)
  - `l_frag /= l_old` final step is in fp32

**Pass:** all 9 test cases produce `max_abs_diff < 1e-2` against
`F.scaled_dot_product_attention(q, k, v, is_causal=True/False)` reference.

### M3 — ComfyUI node glue (1 day)

- [ ] `flash_sm75_nodes.py` exposes `FlashAttentionT4_Apply` and
  `FlashAttentionT4_Remove`, mirroring `SageAttentionT4_Apply` exactly
  in API and behavior.
- [ ] Self-disable-after-3-fails logic copied verbatim from
  `SageAttention-T4/comfyui_nodes/sageattention_node.py`.
- [ ] ComfyUI restart shows the node class registered. `curl -s
  http://localhost:8188/object_info` includes `FlashAttentionT4_Apply`
  in the node list.
- [ ] **Qwen code review:** confirm the node's optimized_attention_override
  payload is identical in shape to SageAttention-T4's. No new
  ComfyUI APIs used.

**Pass:** the node appears in the ComfyUI node menu under category
`"FlashAttention-T4"`, parameters match the spec.

### M4 — Qwen-Image 2.1 end-to-end (2 days)

- [ ] A workflow `custom_nodes/FlashAttention-sm75/test_wf_qwen.json`
  is checked in. It is a copy of the validated baseline workflow
  `image_qwen_image_2_1_t2i_api.json` with a `FlashAttentionT4_Apply`
  node inserted between the `QwenImage21Cache` node and the `KSampler`
  (exact same insertion point as `image_qwen_image_2_1_t2i_api_SAGE.json`
  uses for `SageAttentionT4_Apply`).
- [ ] Running the workflow with the new node **enabled** and original
  bf16 Qwen models produces a PNG that is **visually correct** (a
  recognizable image, not a gray square and not a dot-pattern garbage).
  Pixel-by-pixel comparison with the baseline at 140 s is not required;
  visual inspection by vaskes is.
- [ ] Running the workflow with the new node **disabled** (enable=False
  in the node UI) produces the **exact same** image as the
  `--use-ck-attention` baseline (sanity check: the passthrough path
  is unchanged from the comfy_kitchen backend).
- [ ] **Qwen code review:** review the workflow JSON diff, the node
  position in the graph, the `optimized_attention_override` payload.

**Pass:** the enabled workflow produces a recognizable image of the
woman-in-glasses subject (the canonical Qwen-Image 2.1 demo output),
and the disabled workflow matches the baseline at 140 s.

### M5 — Speed validation (1 day)

- [ ] `bench/bench_flash_sm75.py` runs the workflow 3 times with
  warm cache, reports median time.
- [ ] The new kernel **strictly beats 140 s** for Qwen-Image 2.1
  (25 steps, seed=65451515, prompt as in the baseline).
- [ ] If the new kernel is **slower** than 140 s, this milestone fails
  and Mavis must investigate: is the kernel being called at all? Is
  the attention shape compatible? Is the per-call overhead of
  fp32→fp16→fp32 casts eating the speedup?
- [ ] H3 (Minimax H3) end-to-end speed at most 5% slower than the
  baseline 1:45. This is the lower bar — H3 has more attention ops
  total, so the kernel's per-op overhead matters more.
- [ ] **Qwen code review:** review the bench script, validate the
  methodology (warm cache, median of 3, same prompt and seed).

**Pass:** Qwen median time < 140 s, H3 median time ≤ 1:45 × 1.05.

### M6 — Documentation, README, commit (0.5 day)

- [ ] `custom_nodes/FlashAttention-sm75/README.md` documents the build,
  the node API, the test commands, the known issues.
- [ ] `custom_nodes/FlashAttention-sm75/KNOWN_ISSUES.md` lists open
  issues (e.g. if mask fallback is slow, if certain head_dims are
  not yet supported).
- [ ] Commit with a message in the format of the existing commits in
  this repo (see `git log --oneline`). Reference this task file:
  `Task: docs/tasks/01-flash-sm75-native.md`.
- [ ] Tag the commit: `flash-sm75-v0.1`.
- [ ] Push to `origin main`.
- [ ] Update root `README.md` with one paragraph in the "What works"
  table listing this kernel with its measured speed and quality.

**Pass:** `git tag -l` includes `flash-sm75-v0.1`; `git log --oneline -1`
shows the descriptive commit.

---

## Test cases

All test cases are end-to-end runs of real workflows. The workflow JSON
files referenced are the actual files on the workstation host
(`/opt/comfyui/`). They are not fabricated — they are the validated
files used in the existing bench.

### Test case T1 — Qwen-Image 2.1 baseline check (sanity)

**Setup:**
- Container running, `docker compose restart` only (no recreate, to
  preserve editable install).
- Workflow: `/opt/comfyui/image_qwen_image_2_1_t2i_api.json` (the
  existing baseline workflow, no FlashAttention node).
- COMFYUI_ARGS: `--listen 0.0.0.0 --port 8188 --disable-auto-launch --use-ck-attention`.
- Models: original bf16 files in
  `/opt/comfyui/models/{diffusion_models,text_encoders,vae}/qwen_image_2.1/`.
  No `_fp16_converted` or `.deprecated` files used.

**Run:**
```bash
ssh workstation "python3 /tmp/comfyui_run.py /opt/comfyui/image_qwen_image_2_1_t2i_api.json --label baseline-check --timeout 600"
```

**Pass:** PNG file `/opt/comfyui/output/Qwen_image_2.1_NNNNN.png` of
size 2.28 MB, recognizable subject, 140 s ±5 s. Copy to
`/workspace/output/qwen-history/T1_baseline.png` in the agent workspace.

### Test case T2 — FlashAttention sm_75 enabled, Qwen

**Setup:** same as T1, but with workflow
`/opt/comfyui/test_wf_qwen_flash.json` (the new file from M4).
The workflow has one extra `FlashAttentionT4_Apply` node between
`QwenImage21Cache` and `KSampler`, with `enable=True`.

**Run:**
```bash
ssh workstation "python3 /tmp/comfyui_run.py /opt/comfyui/test_wf_qwen_flash.json --label flash-enabled --timeout 600"
```

**Pass:**
- PNG file size > 1.5 MB (anything less means the kernel produced
  near-constant output, which is a failure mode).
- PNG visually shows a recognizable subject (visual inspection by
  vaskes — no automated check needed beyond size).
- Time < 140 s.
- ComfyUI log contains `FlashAttention sm_75: call #1: q.shape=(...)`
  and at least 100 OK calls without errors, similar to the
  `SageAttention-T4` log pattern that Mavis 2026-10-17 captured.

### Test case T3 — FlashAttention sm_75 disabled, Qwen

**Setup:** same as T2, but with `enable=False` in the
`FlashAttentionT4_Apply` node UI. This is the passthrough path.

**Run:**
```bash
ssh workstation "python3 /tmp/comfyui_run.py /opt/comfyui/test_wf_qwen_flash.json --label flash-disabled --timeout 600"
```

**Pass:**
- PNG file byte-identical to T1's output (the passthrough must not
  change anything).
- Time approximately 140 s (no kernel overhead).
- Log contains `FlashAttention: passthrough` (not the kernel call log).

### Test case T4 — head_dim coverage

**Setup:** run `test_flash_sm75_kernel.py` (the unit test from M2).
**Run:** inside the container, run the test.
**Pass:** all 9 test cases in M2 produce `max_abs_diff < 1e-2`.

### Test case T5 — H3 regression check

**Setup:** workflow `/opt/comfyui/video_minimax_h3_t2v_api.json`
(the existing H3 workflow). Models in
`/opt/comfyui/models/diffusion_models/minimax_h3/` (original bf16
and int8_convrot files, **not** the `_fp16.deprecated` or
`_fp16_converted.deprecated` files).

**Run:**
```bash
ssh workstation "python3 /tmp/comfyui_run.py /opt/comfyui/video_minimax_h3_t2v_api.json --label h3-no-flash --timeout 1500"
```

**Pass:** H3 video output is recognizable (visually correct, not
garbage) and finishes in ≤ 1:45 × 1.05 (i.e. ≤ 110 s for 20 steps).
This is the regression check: H3 should not get worse with the new
kernel installed but disabled.

Then enable the FlashAttention node in a copy of the H3 workflow
and re-run. Time should be ≤ 1:45 × 1.05 (i.e. the new kernel
should be no worse than baseline for H3).

### Test case T6 — Stress / no-crash

**Setup:** any workflow, run 5 times in a row.
**Pass:** no CUDA OOM, no kernel panic, no `cudaErrorIllegalAddress`.
The kernel's failure handler should kick in and fall back to pytorch
if anything goes wrong.

---

## Progress

Mavis coder updates this section. Each item links to a commit hash.

### Build

- [ ] M1: Build infrastructure
- [ ] M2: Kernel correctness on small inputs
- [ ] M3: ComfyUI node glue
- [ ] M4: Qwen-Image 2.1 end-to-end
- [ ] M5: Speed validation
- [ ] M6: Documentation, README, commit

### Test results (M5 onward, fill in after each run)

| Test | Time | Output size | Quality OK? | Notes |
|---|---|---|---|---|
| T1 baseline | | | ✅/❌ | |
| T2 flash enabled | | | ✅/❌ | |
| T3 flash disabled | | | ✅/❌ | |
| T5a H3 no flash | | | ✅/❌ | |
| T5b H3 with flash | | | ✅/❌ | |
| T6 stress x5 | | | ✅/❌ | |

### Qwen code review checkpoints

- [ ] After M1: build setup approved
- [ ] After M2: kernel logic approved
- [ ] After M3: node glue approved
- [ ] After M4: workflow integration approved
- [ ] After M5: bench methodology + speed result approved

### Issues encountered

*(Mavis appends here as work progresses. Format: short title, one paragraph
of context, link to commit that addresses it, link to follow-up if open.)*

---

## Resolution

A task is **resolved** when:

1. All milestones M1–M6 are checked off.
2. All test cases T1–T6 pass.
3. Qwen code review has approved all five checkpoints.
4. The commit is pushed and tagged.
5. The root `README.md` is updated.

If **M5 fails** (the new kernel is slower than 140 s despite M1–M4
passing), this is **not** a failure of the task in the sense of "wasted
work" — it is a **negative result** that should still be committed and
documented. The repo gains:

- A working, correct (quality OK) sm_75 flash kernel that happens to
  not be faster than the comfy_kitchen baseline. This is **notable**
  because it rules out the simple "write a flash kernel" path for
  further speedup, and points future work at **model-side optimization**
  (w4a8 quantization, turbo variants) or **a different attention
  algorithm** entirely (sparse attention, linear attention).

If **M4 fails** (the kernel produces garbage even with fp32 softmax
accumulator), the most likely cause is:

- An mma shape mismatch (e.g. using `m8n8k8` instead of `m8n8k4`).
- An accumulator discipline bug (e.g. exp() in fp16 instead of fp32).
- A wrong `ldmatrix` layout assumption (Turing has subtle differences
  vs Ampere in how fragments are laid out — the THE-ANGEL-AI code
  in `csrc/qattn/attn_cuda_sm75.h` has the sm_75-specific
  `ldmatrix_m8n8x1` / `m8n8x2` / `m8n8x4` wrappers; reuse them and
  do not invent new ones).

Investigate systematically with `test_flash_sm75_kernel.py`. The
`max_abs_diff` value is the diagnostic. A diff > 0.1 in any of the 9
M2 test cases means a bug; bisect the test cases to localize.

**Important:** do not skip M2 because the microbench "looks right."
The microbench measures performance; **the unit tests in M2 measure
correctness.** They are different. If the unit tests fail, the
microbench result is meaningless.

---

## References (for Mavis coder, in this repo)

- `custom_nodes/SageAttention-T4/csrc/qattn/attn_cuda_sm75.h` — the
  sm_75 MMA wrappers, `ldmatrix` wrappers, the `mma.sync` PTX inline
  assembly. **Read this carefully before writing any PTX.** The `.satfinite`
  modifier on the INT8 mma is *not* relevant to this task (we use fp16),
  but the `cp.async` and `ldmatrix` patterns are.
- `custom_nodes/SageAttention-T4/comfyui_nodes/sageattention_node.py` —
  the ComfyUI custom node template, including the self-disable-after-3
  pattern and the fp32→fp16 dtype cast. **Reuse this verbatim.**
- `custom_nodes/SageAttention-T4/setup.py` — the build setup. **Copy,
  only change module name and source list.**
- `README.md` (root) — the "What works" table format. Add a row for
  this kernel with the measured time and quality verdict.
- `docs/tasks/02-sage-sm75-perrow-outlier.md` (if/when created) —
  the companion task. **Read it once to make sure the file layout
  and node API match between Task 01 and Task 02.**

## References (external, for context)

- FlashAttention paper: Dao et al. 2022, "FlashAttention: Fast and
  Memory-Efficient Exact Attention with IO-Awareness."
- sm_75 mma instruction reference: NVIDIA PTX ISA documentation,
  "8.7.13 Matrix Multiply-Accumulate Instructions." Specifically:
  - `mma.sync.aligned.m8n8k4.row.col.f32.f16.f16.f16` — the
    fp16→fp32 mma we use.
  - `mma.sync.aligned.m8n8k8.row.col.f16.f16.f16.f16` — DO NOT USE,
    fp16 accumulator.
- FlashAttention 2 paper: Dao 2023, "FlashAttention-2: Faster Attention
  with Better Parallelism and Work Partitioning." Section 2.3 on
  work partitioning is directly applicable.
- Turing whitepaper: NVIDIA 2018, "NVIDIA Turing GPU Architecture."
  Section on "Turing's 4th Gen Tensor Core" for the sm_75 mma shape
  inventory.
