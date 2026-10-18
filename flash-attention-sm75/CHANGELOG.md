# Changelog

## v0.2.1 — 2026-10-18

**Status:** Released. Scope: bf16/fp16 weight models (Qwen-Image 2.1 etc.)

### What works
- `--use-flash-attention` for Qwen-Image 2.1 (bf16/fp16): **9.5% per-step
  speedup** over `--use-ck-attention` (141.72s vs 140.00s wall time for
  25 steps; 5.07 vs 5.6 s/it).
- 6/6 unit tests pass against pytorch SDPA reference (T1–T5 max_diff <
  0.0001, T6 stress max_diff = 0.0010).
- Install script works end-to-end on fresh host (3 critical quoting bugs
  found by Mavis review all fixed).
- E2E Qwen-Image 2.1 output is recognizable (woman in polka-dot шуба,
  identical scene to ck baseline, 2.29 MB PNG).

### Known issues
- **`--use-flash-attention` for H3 (Minimax video, int8 convrot weights)
  produces checkerboard garbage.** Root cause: a shim package in the
  ComfyUI image shadows our sm_75 patch and routes to
  `flash_attn_turing` (ssiu community fork with m8n8k8 fp16-accumulator),
  which has insufficient precision for int8 convrot. H3 with
  `--use-ck-attention` is the correct path (116s, 538 KB correct).
  See `docs/H3_DIAGNOSTIC.md` for full analysis.
- Recommended default in `compose.yaml` is `--use-ck-attention` (works
  for both Qwen and H3). Switch to `--use-flash-attention` per-workflow
  for Qwen/bf16/fp16 to get the speedup.

### Roadmap
- v0.3 will require writing sm_75-specific CUTLASS dispatch
  (m16n8k8 fp32-accumulator, fp16-input) for the actual flash kernel
  to execute on sm_75. Multi-day kernel port. See `docs/TODO.md`
  §TODO-9 and `docs/tasks/03-sm75-mma-kernel-port.md` (the latter
  is created when the port starts).

---

## v0.2.0 — 2026-10-18 (Mavis review fixes)

**Status:** Superseded by v0.2.1. Included Mavis review action items
A1–A6, A8, A11.

### Changed
- **A1 (CRITICAL):** install script — `bash -c "..."` quoting fixed
  (single-quote outer string, positional args for host vars). The
  script now actually works end-to-end on a fresh checkout.
- **A2:** `tests/SPEED.md` — unfilled template placeholders replaced
  with real measurement data.
- **A3:** `README.md` — Validation status table updated to show all
  milestones as done; File layout fixed (removed non-existent
  `test_sm75_e2e.py`).
- **A4:** install script — CUTLASS vendoring step (clone on host
  where network works, tar-pipe into container).
- **A5:** `tests/test_sm75_correctness.py` — added T6 stress test
  (B=1, H=8, L=4096, D=128, scale=1.0). max_diff = 0.0010 PASS.
- **A6:** documented that torch 2.14 does not expose `backend=` kwarg
  on `scaled_dot_product_attention`; SDPA on sm_75 defaults to math
  backend.
- **A8:** `docs/VERSION_PINS.md` — added Build environment variables
  and Build flags tables.
- **A11:** `docs/upstream-analysis.md` — removed false claim about
  `philox.cuh` sm_75 guard.

### Reviews
- Qwen (code review agent): APPROVE
- Mavis (implementation session): REQUEST CHANGES → all A-items
  applied → effectively APPROVE

---

## v0.1 — 2026-10-17 (initial)

**Status:** Superseded.

The 5-line patch (`patches/0001-add-sm75-arch.patch`) modifying
`Dao-AILab/flash-attention:setup.py` to add sm_75 to the build matrix.
Includes 5 unit tests against pytorch SDPA reference (all pass with
max_diff < 0.0001). Documentation in `docs/`.

The sm_75 build produced by this patch is **not actually used at
runtime** due to the shim package shadowing it (see v0.2.1 known
issues). The 9.5% Qwen speedup observed with v0.2.0 was provided
by the existing `flash_attn_turing` community fork, not by our
patch.
