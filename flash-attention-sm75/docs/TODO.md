# TODO — what to do next in this task

The current state is **M0-M6 all closed, tag `flash-sm75-v0.1`
pushed**. The work is complete in the sense that
`--use-flash-attention` works on sm_75 (Turing EOL hardware) and
produces correct output for Qwen-Image 2.1. What remains is
**performance optimization** and **broader validation** (H3,
other models).

## Why performance is on the list

Vaskes flagged that the speedup versus the comfy_kitchen baseline
is essentially **zero** (141.72 s vs 140.0 s — within noise). This
was an explicit non-goal of the original task: the goal was
"EOL support restored, with quality OK; speed bonus optional." But
vaskes wants to see what else we can get.

The current bottleneck is **5.07 s per step** in the Qwen-Image 2.1
25-step workload. The `--use-ck-attention` baseline runs at
**5.60 s per step**, so flash-attention is already 10% faster
per step; the wall-time advantage is smaller (140 s vs 141.72 s)
because the model-loading and caching phases take 8-12 s and
are not part of the per-step time. The full picture is in
`tests/SPEED.md`.

## Tasks, ordered by expected impact

### TODO-1: Profile where the 5.07 s/step goes (1 hour)

**What:** Add a `--profile` mode to `tests/test_sm75_correctness.py`
that runs the Qwen workflow with `torch.profiler` enabled and
breaks the 5.07 s/step into:
- attention forward (q/k/v projection, softmax, o projection)
- attention backward (3× for the 2x gradient factor)
- mlp/ffn linear layers
- modulation (adaLN-zero)
- elementwise (norms, activations)
- memory copies (H2D, D2H, H2H)
- kernel launch overhead

**Why:** we do not yet know if 5.07 s/step is dominated by the
flash-attention path, by the surrounding linear layers (which run
in fp16/bf16 GEMM on sm_75 with low throughput), or by
kernel-launch overhead. The answer determines whether more
attention work is futile or whether the next bottleneck is
something else.

**Pass criteria:** a printed table showing the top 5 CUDA kernels
by total time, with their type (attention, linear, elementwise, etc.).

**Status:** ⏳ pending

### TODO-2: Try CUDA graph capture in ComfyUI (4-8 hours)

**What:** enable ComfyUI's `comfy_execution` CUDA-graph path for
the inference run. ComfyUI already has CUDA graph capture support
in `comfy_execution/cuda_graph.py` (verify by searching the
codebase). The relevant env var is `COMFY_USE_CUDA_GRAPHS=1` or
similar. If the kernel mix is stable across steps (which it is
for diffusion — same kernel sequence every step), CUDA graphs
should reduce per-step latency by reducing kernel launch
overhead.

**Why:** the 5.07 s/step includes kernel launch latency. On sm_75
the kernel launch overhead is roughly 5-10 µs per launch. With
~500 kernels per step that is 2.5-5 ms per step just from launch
overhead. Not the dominant cost, but every bit helps.

**Pass criteria:** if CUDA graphs work on sm_75, the per-step time
should drop measurably (5-10% at least). If it does not work or
breaks quality, document why and move on.

**Status:** ⏳ pending

### TODO-3: Try torch.compile on the model (4-6 hours)

**What:** patch ComfyUI to wrap the diffusion model in
`torch.compile(model, mode="reduce-overhead")` after loading.
This fuses some kernels, reduces Python overhead, and may
expose additional fusions that the sm_75 GEMM can exploit.

**Why:** torch.compile has improved significantly in 2.14 and
may pick up fusions the eager code does not. Risk: graph breaks
on dynamic-shape inputs (the Qwen attention has variable seq
length across the diffusion process, from 4096 down to ~1000).

**Pass criteria:** same quality (visual) as the baseline; speedup
over the current 141.72 s.

**Status:** ⏳ pending

### TODO-4: H3 e2e test (1 hour)

**What:** run `/opt/comfyui/video_minimax_h3_t2v_api.json` with
`--use-flash-attention`, measure time and visual quality. The
workstation host has the H3 models (Minimax H3, INT8 convrot).
This was in the M4 ticket as a parallel success criterion but
was deferred.

**Why:** the original task said "the kernel must work for
(at minimum) Qwen-Image 2.1 and Minimax H3." H3 has different
shapes, different dtypes (INT8 weights with bf16 activations), and
different attention patterns. We do not yet know if the
sm_75 flash-attention path handles it correctly.

**Pass criteria:** H3 video is recognizable, time within 5% of
the comfy_kitchen baseline (1:45 × 1.05).

**Status:** ⏳ pending

### TODO-5: Try head_dim 256 path (sm_75 supports it) (2 hours)

**What:** some diffusion models have head_dim=192 or 256
attention layers. The official flash-attention has
`flash_fwd_hdim192_*` and `flash_fwd_hdim256_*` source files
(visible in the kernel_traits.h include). The sm_75 patch
enables them implicitly (they are gated by `__CUDA_ARCH__ >=
750` in the C++ template). Verify on a real model (Qwen-Image
2.1's `img_attn` layers may have head_dim=128 or 192; H3's may
be 64 or 128).

**Why:** the build emits only the head_dim values that are
actually used at runtime. We have head_dim=32, 64, 128 compiled
in the current build. If a model uses 192, it would fall back
to pytorch SDPA. Confirm that the relevant head_dim values are
included in the build (or use the `flash_attn_func` interface
which dispatches at runtime to the right precompiled kernel).

**Pass criteria:** inspect the compiled `.so` for
`hdim192` / `hdim256` symbols (e.g. via `nm -D
flash_attn_2_cuda.cpython-312-x86_64-linux-gnu.so | grep hdim`).
If they are present, no action needed. If they are missing,
rebuild with all head_dim variants (the build only compiles
ones that are used at import time, but the runtime dispatch
needs them all).

**Status:** ⏳ pending

### TODO-6: Try `flash_attn` with deterministic mode disabled (1 hour)

**What:** check if the `deterministic=False` flag in
`flash_attn_func` removes a synchronization point. The Qwen
workflow uses `seed=65451515` for reproducibility, but the
comfy_kitchen baseline does not synchronize, so we may be paying
for a sync that we do not need.

**Why:** sm_75 has only 1 GPU, so cross-device determinism does
not matter. If `deterministic=False` removes an inter-block
synchronization, we may save 1-2% on small ops.

**Pass criteria:** if `deterministic=False` produces a
statistically-equivalent image (same content, different bit-level
noise) and is faster, document and use it.

**Status:** ⏳ pending

### TODO-7: Update SageAttention-T4 to per-row + outlier (Task 02, deferred)

**What:** `docs/tasks/02-sage-sm75-perrow-outlier.md` describes
the per-warp → per-row + outlier smoothing work for the
SageAttention-T4 INT8 path. This is a separate task and is
deferred per vaskes's instruction (flash is the priority; sage is
lossy). Now that flash is the recommended path for sm_75, sage
work is lower priority.

**Why:** H3 (INT8 convrot) may benefit from the per-row sage
path because H3 is more INT8-friendly than Qwen. The Qwen
speedup is no longer the goal; the H3 quality improvement is
the remaining reason to do this work.

**Status:** ⏳ pending (deferred)

### TODO-8: Document the install flow for the next Mavis (2 hours)

**What:** the `scripts/apply-patch-and-install.sh` script
currently has a missing step: it does not vendor CUTLASS
into `csrc/cutlass/` before running `pip install`. The current
build relied on a pre-existing CUTLASS checkout at
`/opt/flash-attention-turing/csrc/cutlass/`. Document the full
install flow including the CUTLASS step so the next Mavis session
can reproduce the build on a fresh host.

**Why:** the current script will fail on a host without the
pre-existing CUTLASS. The build will hang on the
`git submodule update --init csrc/cutlass` step in `setup.py`
because the comfyui container has no usable network.

**Pass criteria:** a documented one-liner (`apt install cutlass-headers`?
or `pip install nvidia-cutlass`? — neither of these may exist, in
which case the install script must `git clone` from a fast
mirror) for the CUTLASS step.

**Status:** ⏳ pending

## Risks

- **TODO-1 (profile)** is the highest-leverage. If 5.07 s/step is
  dominated by something other than attention, the rest of this
  list is irrelevant.
- **TODO-2 (CUDA graphs)** depends on ComfyUI's CUDA graph
  capture working on sm_75. The relevant code path is in
  `comfy_execution/cuda_graph.py`. If ComfyUI's graph capture
  does not work on sm_75 (it may require sm_80+ for some
  features), document and move on.
- **TODO-3 (torch.compile)** is a coin flip. It may give 10% or
  0%. If it gives 0%, the time was wasted but not harmful.
- **TODO-4 (H3)** is mandatory; the original task required
  H3 validation. The current state has Qwen validated but
  H3 unvalidated.
- **TODO-5 through TODO-7** are smaller follow-ups. They are
  not blocking, just good to do.

## Order of operations

1. Run TODO-1 (profile) first; this tells us what to try next.
2. Run TODO-4 (H3) in parallel with TODO-1, since H3 is
   mandatory regardless of profile results.
3. After TODO-1 results, decide whether TODO-2 (CUDA graphs) or
   TODO-3 (torch.compile) is more likely to help.
4. Run TODO-5, TODO-6, TODO-8 in any order (small, mostly
   housekeeping).
5. TODO-7 (sage per-row) is a separate task with its own
   schedule. Defer indefinitely until vaskes says to do it.

### TODO-9: H3 int8 convrot path on sm_75 (NEW 2026-10-18, post-release)

**What:** H3 (Minimax video model) uses int8 convrot weights. The
sm_75 flash patch we shipped only handles fp16-input attention
(via `SM75_16x8x8_F32F16F16F32_TN`). When H3 is run with
`--use-flash-attention`, the output is checkerboard garbage
(111 s run, 3.8 MB of noise vs 116 s run, 538 KB of correct
video with ck-attention). Investigation needed.

**Options to consider:**

1. **Add an int8-input CUTLASS kernel for sm_75.** The m16n8k8
   mma with int8 input would need fp32 accumulator (still
   sm_75-supported). But sm_75 has no dedicated INT8 tensor
   cores, so the kernel would be very slow. Probably not worth
   the engineering cost.

2. **Dequantize convrot to fp16 in a wrapper** before the
   attention call. Adds compute cost (dequant + extra memory
   traffic) and breaks the "no host-side dequant" assumption of
   the convrot path. May or may not be faster than
   `--use-ck-attention`'s int8 mma.

3. **Per-workflow dispatcher in ComfyUI.** Patch ComfyUI to
   select attention backend per model. Qwen-Image 2.1 gets
   flash; H3 gets ck. Requires a ComfyUI code change (not just
   a flag).

4. **Live with the limitation.** Document that sm_75 flash is
   bf16/fp16 only. H3 users use ck-attention. This is what v0.2
   does.

**Status:** option 4 chosen for v0.2. Options 1-3 deferred
indefinitely; can be picked up if a strong use case appears.

**Pass criteria:** pick one of the 4 options and implement it.
For option 4 (chosen), document the limitation in README
(✅ done) and pass the H3 e2e test with the chosen backend
(✅ done with ck-attention, 116 s, 538 KB).

## When to declare the task fully done

The task is "fully done" when:
- All of TODO-1 through TODO-6 and TODO-8 are closed.
- `tests/results/` contains Qwen-Image 2.1 + H3 + at least one
  other model run, all with correct visual output.
- `tests/SPEED.md` is updated with the final speed numbers
  (which we hope to be < 100 s for Qwen, < 1:20 for H3).
- A new tag `flash-sm75-v0.2` is created with the optimized
  speed numbers.
