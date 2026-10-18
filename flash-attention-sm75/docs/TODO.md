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

### TODO-9: sm_75 mma kernel port (BLOCKING v0.3, NEW 2026-10-18)

**What**: write sm_75-specific `.cu` files using
`SM75_16x8x8_F32F16F16F32_TN` (m16n8k8 fp32-accumulator, fp16-input)
CUTLASS atom, plus dispatch logic to select these paths on
`cc_major == 7 && cc_minor == 5`. Also bypass the shim that shadows
our sm_75 build at runtime.

**Why**: v0.2.1 ships with the limitation that
`--use-flash-attention` works only for bf16/fp16 weight models
(Qwen-Image 2.1 etc.). For H3 and any other int8 convrot model,
the runtime routes through `flash_attn_turing` (ssiu community fork
with m8n8k8 fp16-accumulator) which is precision-insufficient. The
v0.2.1 patch in `Dao-AILab/flash-attention:setup.py` enables the sm_75
build but the shim shadows it. See `docs/H3_DIAGNOSTIC.md` for the
full analysis.

**Three sub-tasks**, in dependency order:

#### TODO-9.1 — Bypass the shim (4-8h)

The current shim at
`/opt/venv/lib/python3.12/site-packages/flash_attn/__init__.py`
routes to `flash_attn_turing`. To make our sm_75 build actually run:

- **Option A (preferred):** patch the ComfyUI image's Dockerfile to
  not install the shim when a real `flash_attn` package is present.
  The shim's own docstring says "If the real flash_attn package is
  installed, this shim is shadowed" — but site-packages is searched
  BEFORE the editable install path, so this assumption is wrong.
  The fix is either to remove the shim, or to make it auto-detect
  the real package and defer to it.
- **Option B (workaround):** keep the shim, but make our sm_75 build
  shadow it via a `.pth` file in site-packages (adds
  `/tmp/flash-attention` to sys.path BEFORE site-packages).
- **Option C (workaround):** override the shim in compose.yaml via
  bind mount (already tested; works for our build but is fragile).

TODO-9.1 deliverable: a stable install path where
`import flash_attn` resolves to the Dao-AILab sm_75 build on sm_75
hosts and to the standard wheel on sm_80+ hosts.

#### TODO-9.2 — sm_75 fp16-input kernel files (1-2 days)

Add new `.cu` source files alongside the existing
`flash_fwd_hdim*_sm80.cu` family:

- `flash_fwd_hdim32_fp16_sm75.cu`
- `flash_fwd_hdim64_fp16_sm75.cu`
- `flash_fwd_hdim128_fp16_sm75.cu`
- `flash_fwd_hdim128_fp16_causal_sm75.cu`
- (similar for bf16, and for bwd if needed)

Each file instantiates `run_mha_fwd_hdimN<half_t, false>` (and
causal) using the sm_75 mma atom via the existing `kernel_traits.h`
template specialization. The CUTLASS `SM75_16x8x8_F32F16F16F32_TN`
atom is already in `kernel_traits.h` from upstream; we just need to
instantiate it with the same templates that the sm_80 files use.

The pattern: copy `flash_fwd_hdim128_fp16_sm80.cu` → rename to
`flash_fwd_hdim128_fp16_sm75.cu`, change the `__CUDA_ARCH__` guard
in `flash_fwd_launch_template.h` to `>= 750`, and verify the build.

**Pass criteria:**
- `python3 -c "from flash_attn import flash_attn_func; ..."` works on
  sm_75 with no FATAL messages in stderr
- 6/6 unit tests pass with the new build
- E2E Qwen-Image 2.1 produces correct output (compared to v0.2.1
  baseline)
- Speed is within 5% of v0.2.1 (the sm_75 mma throughput is the
  same, just different atom)

#### TODO-9.3 — sm_75 int8 convrot input (2-4 days, optional)

H3 uses int8 convrot weights. The flash kernel needs to accept
values that are out-of-range for fp16 (e.g. dequantized rotated
weights can have magnitude 5-10x normal fp16). Two options:

- **Option A:** add a CUTLASS `SM75_16x8x8_INT8INT8INT32_TN`
  instantiation. Sm_75 has no dedicated INT8 tensor cores, so
  throughput will be 1/4 of fp16 path. Probably not worth it.
- **Option B:** dequantize convrot to fp16 in a wrapper before the
  attention call (in `comfy_kitchen` or a custom node). This
  preserves flash kernel design but adds compute overhead. May
  not be faster than ck-attention for H3.

**Recommendation:** defer TODO-9.3 indefinitely. Document H3 as
"requires ck-attention, not flash-attention" in README. If H3
becomes a critical use case, revisit with a dedicated kernel port.

**Effort estimate:**
- TODO-9.1: 4-8h
- TODO-9.2: 1-2 days
- TODO-9.3: 2-4 days (deferred)

**Total for v0.3:** ~1-2 working days of focused kernel work
(TODO-9.1 + TODO-9.2). Test + cleanup ~half a day.

**Status:** ⏳ pending vaskes approval. Plan written 2026-10-18
(see `docs/tasks/03-sm75-mma-kernel-port.md` when port starts).

## When to declare the task fully done

The task is "fully done" when:
- All of TODO-1 through TODO-6 and TODO-8 are closed.
- `tests/results/` contains Qwen-Image 2.1 + H3 + at least one
  other model run, all with correct visual output.
- `tests/SPEED.md` is updated with the final speed numbers
  (which we hope to be < 100 s for Qwen, < 1:20 for H3).
- A new tag `flash-sm75-v0.2` is created with the optimized
  speed numbers.
