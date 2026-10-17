# Code review notes for Qwen + Mavis

This document is the **reviewer's guide** to the flash-attention-sm75
work. The structure is:

1. **What is being reviewed** — the 5-line patch, the docs, the
   test outputs, the install script.
2. **What reviewers should look for** — the things a careful
   reviewer should check.
3. **What reviewers do NOT need to look for** — explicitly out
   of scope.
4. **Known limitations / honest reporting** — places where this
   work is not yet a full solution.
5. **The questions reviewers should answer** — the explicit
   accept/reject criteria.

## 1. What is being reviewed

The work in this directory restores `--use-flash-attention`
support for NVIDIA Turing (sm_75) GPUs in current ComfyUI. The
deliverable is:

- `patches/0001-add-sm75-arch.patch` — 5 lines in
  `Dao-AILab/flash-attention@965c921:setup.py:113-117`. Re-enables
  `sm_75` in the `add_cuda_gencodes` function.
- `scripts/apply-patch-and-install.sh` — convenience installer
  (with a known gap: does not vendor CUTLASS; see TODO-8).
- `tests/test_sm75_correctness.py` — 5 unit tests vs
  `F.scaled_dot_product_attention` reference. Pass criterion:
  `max_abs_diff < 0.05`. **Actual: max_diff < 0.0001 (perfect).**
- `tests/RESULTS.md` — the 5/5 PASS test results.
- `tests/SPEED.md` — the speed validation against the
  comfy_kitchen baseline.
- `tests/results/qwen_flash_142s.png` — the e2e output for
  Qwen-Image 2.1 (recognizable, identical scene to baseline).
- `docs/VERSION_PINS.md` — every package version we built
  against, with the rationale for each.
- `docs/TODO.md` — the 8 follow-up tasks for the next Mavis.
- `docs/strategy-decision.md` — why Strategy B (fork + patch)
  was chosen over Strategy A (new kernel from scratch).
- `docs/upstream-analysis.md` — file-by-file walkthrough of
  where sm_75 lives in the official code.
- `docs/tickets/M0..M6.md` — the 7 implementation tickets.
- `LICENSE` — BSD-3-Clause, copied verbatim from upstream.
- `NOTICE.md`, `CREDITS.md` — required attributions.
- `README.md` — top-level project description.

The git history of this directory is 4 commits:

```
39d347d M6 closed: task document updated with final outcome, tag flash-sm75-v0.1
18a5839 M1-M5 closed: flash-attention-sm75 works on sm_75 (Turing EOL)
d78323f M1 progress: build started inside comfyui container
1cbff06 Add flash-attention-sm75: patch to restore --use-flash-attention on sm_75 EOL hardware
```

The tag `flash-sm75-v0.1` points to `39d347d`.

## 2. What reviewers should look for

### A. The patch itself (`patches/0001-add-sm75-arch.patch`)

- **Correctness:** does the 5-line diff match the diff in
  `docs/upstream-analysis.md`? The diff should add 5 lines after
  `supported_archs = []` and before `# Always-regular 80`. It
  should add a branch for `if "75" in archs:` that emits
  `-gencode arch=compute_75,code=sm_75` and adds `"75"` to
  `supported_archs`. Nothing else.
- **Reapply cleanliness:** does the patch apply cleanly against
  the upstream commit it was generated from? Run
  `git apply --check` against `Dao-AILab/flash-attention@965c921`.
  If not, the patch needs to be regenerated.
- **License implications:** the patch is small enough to be
  non-copyrightable individually, but we explicitly provide it
  under the same BSD-3-Clause as the file it modifies. Review
  this is consistent.

### B. The unit test (`tests/test_sm75_correctness.py`)

- **Reference correctness:** does the test use
  `F.scaled_dot_product_attention` as the reference? It should.
  Anything else (e.g. a custom reference, or a comparison to
  the per-warp SageAttention-T4 output) would be a
  measurement of "different" not "correct."
- **Test coverage:** the 5 test cases cover small and large
  batches, multiple head_dims (32, 64, 128), and causal masks.
  If a reviewer thinks a case is missing (e.g. head_dim=192 or
  non-causal with custom mask), they should add it and report.
- **Pass criterion:** `max_abs_diff < 0.05`. The actual results
  are `max_abs_diff < 0.0001` which is **far** within the bar.
  A reviewer should not accept anything that is between 0.01 and
  0.05 without a follow-up investigation; that range means
  "kernel works" but with some precision loss that may matter
  for downstream tasks.

### C. The end-to-end output (`tests/results/qwen_flash_142s.png`)

- **Visual correctness:** the image should be a recognizable
  woman in a polka-dot шуба, with the same composition as the
  baseline ck-attention output. The reviewer should compare
  side-by-side with the ck-attention baseline (saved at
  `Qwen_image_2.1_best_140s.png` in the same workspace).
- **File size:** 2.29 MB. A significantly smaller file
  (e.g. < 1 MB) is a failure mode (kernel produced near-constant
  output, see the SageAttention-T4 garbage history).

### D. The speed measurement (`tests/SPEED.md`)

- **Methodology:** 25 steps, warm cache, single run. The reviewer
  should understand that "single run" is not statistically
  robust, but the existing baseline (140 s for ck-attention) was
  also measured as a single run, so the comparison is fair.
- **Per-step vs wall-time:** the per-step time (5.07 s/it) is
  more informative than the wall-time (141.72 s) because the
  model-loading phase (~10-15 s) is included in the wall-time
  but not in the per-step. A reviewer should focus on the
  per-step number for kernel comparisons.

### E. The documentation (`README.md`, `docs/strategy-decision.md`, `docs/upstream-analysis.md`)

- **Honesty:** the README should not oversell the speedup. It
  should state that the speedup is ~10% per step but wall-time
  is essentially the same as ck-attention, and that the main
  win is **EOL support restored**, not speed. vaskes flagged
  this concern explicitly.
- **Upstream attribution:** `NOTICE.md` and `CREDITS.md` should
  credit Tri Dao and the co-authors of the FlashAttention paper,
  and reference the BSD-3-Clause license of the upstream
  work. The current files do this.
- **License interaction:** the parent repo
  (`vaskes/comfyui-2080ti-turing`) is governed by the Kuduza
  Ai Lab proprietary license. The `flash-attention-sm75/`
  subdirectory contains both the upstream BSD-3-Clause work
  (preserved verbatim in `LICENSE`) and the Kuduza-proprietary
  patch (5 lines, de minimis). The current arrangement is
  correct: the BSD-3-Clause copyright is preserved and the
  proprietary work only applies to the patch.

### F. The TODO list (`docs/TODO.md`)

- **Completeness:** does it cover all the obvious follow-ups?
  Current list: profile, CUDA graphs, torch.compile, H3
  validation, head_dim coverage, deterministic mode, sage
  per-row (deferred), install flow.
- **Priority:** is the order right? Currently TODO-1 (profile)
  is first, which is correct because the profile tells us what
  to try next.

## 3. What reviewers do NOT need to look for

- **Code style of the patch itself:** the 5 lines follow the
  surrounding style (`if "X" in archs: ...`) and do not need
  any reformatting.
- **Test framework choice:** the unit test uses plain
  `assert`, no pytest. That is intentional — the test is
  short and self-contained. Do not ask for pytest unless
  you also want to refactor the existing repo's other
  test files.
- **Build performance:** the build is slow (8 min for sm_75
  alone, 90 min for all archs). This is not a defect; it is
  a fact of compiling CUTLASS-derived kernels for a new arch.
  The next Mavis session can use `ccache` if speed matters.
- **ComfyUI version compatibility:** the patch and the
  build work with ComfyUI master at commit `5cc4b87`. We do
  not promise compatibility with other ComfyUI versions.
  vaskes can decide whether to backport.

## 4. Known limitations (honest reporting)

- **Speed is not better than the baseline by much.** 141.72 s
  vs 140.0 s on Qwen-Image 2.1 (within noise). The per-step time
  is 10% better (5.07 vs 5.6 s/it), but the model-loading phase
  is not part of the per-step. vaskes flagged this as a
  concern.
- **H3 was not tested in the same session** that closed the
  M0-M6 tickets. The unit tests cover the kernel; the e2e
  test was Qwen-only. H3 is a TODO-4 follow-up.
- **CUTLASS submodule is vendored from a non-pinned
  location** (`/opt/flash-attention-turing/csrc/cutlass/`).
  The current CUTLASS commit is `df18f5e4`. The install
  script does not vendor this; it relies on the path being
  pre-populated. TODO-8 addresses this.
- **Single-GPU only.** The `FLASH_ATTN_CUDA_ARCHS=75` build
  only includes sm_75. A multi-GPU host that wants sm_75
  attention on a Turing card and sm_90 attention on an Ada
  card would need to build with `FLASH_ATTN_CUDA_ARCHS=75;90`.
  This is untested in the current work.
- **The SageAttention-T4 custom node is still installed** in
  `custom_nodes/SageAttention-T4/` of the parent repo. It is
  not used by the `--use-flash-attention` path. It is kept
  for the per-warp legacy path and for the per-row Task 02
  work (deferred per vaskes).

## 5. The questions reviewers should answer

The reviewer should be able to answer YES or NO to each of these
with evidence:

1. **Does the patch apply cleanly against the upstream
   commit?** Run `git apply --check` and confirm.
2. **Do the 5 unit tests pass?** Run the test and confirm.
3. **Is the e2e output visually correct?** Compare to the
   baseline. (Subjective; requires vaskes or a vision model
   to verify.)
4. **Is the speed measurement honest?** Look for
   cherry-picking, single-run reporting, or warm-cache bias.
5. **Is the BSD-3-Clause attribution correct?** Compare
   `LICENSE` to the upstream.
6. **Are the version pins accurate?** Cross-reference against
   `pip list` inside the container.
7. **Is the TODO list complete?** Are there obvious
   follow-ups not in the list?
8. **Does the README accurately describe what was done?** No
   exaggeration, no underselling.

If any answer is NO, the reviewer should provide specific
feedback. If all answers are YES, the work is ready to merge.

## 6. Reviewer checklist (operational)

For Qwen (the code review agent):

- [ ] Read `patches/0001-add-sm75-arch.patch`. Confirm the
      5-line diff matches the description in
      `docs/upstream-analysis.md`.
- [ ] Run `git apply --check patches/0001-add-sm75-arch.patch`
      against a fresh clone of
      `Dao-AILab/flash-attention@965c921`. Confirm clean.
- [ ] Read `tests/test_sm75_correctness.py`. Confirm the 5
      test cases match the ticket M1 specification.
- [ ] Verify the test was actually run (check
      `tests/RESULTS.md` is up-to-date).
- [ ] Read `docs/TODO.md` and check the order and coverage.

For Mavis (the other Mavis session, the implementation one):

- [ ] Re-run the unit tests from a clean install to verify
      reproducibility.
- [ ] Re-run the e2e test to verify the Qwen output is
      recognizable.
- [ ] Re-run the speed test to verify the 5.07 s/it number.
- [ ] Walk through the TODO list and confirm that TODO-1
      (profile) is the right next step.

## 7. After review

If Qwen approves, Mavis can proceed with TODO-1 (profile) and
the follow-up optimizations. If Qwen rejects, the specific
feedback in the rejection should be added as TODO-9, TODO-10,
etc., and the work goes back to a fix-and-re-review cycle.
