# Code review approvals

This file records the code reviews of the flash-attention-sm75 work
as they are completed. Each review is a self-contained record with
the reviewer's verdict, the date, and the key feedback.

## Review 1: Qwen (code review agent)

**Date:** 2026-10-18
**Commit reviewed:** `02e9926` (HEAD at time of review)
**Tag reviewed:** `flash-sm75-v0.1`
**Verdict:** **APPROVE**

### Summary (Qwen's words)

> Проведён детальный code review репозитория `vaskes/comfyui-2080ti-turing`
> на коммите `02e9926` с тегами `flash-sm75-v0.1`. Review выполнен строго
> в соответствии с чеклистом из `docs/REVIEW_NOTES.md`, с фокусом на
> **математическую корректность, воспроизводимость и производительность**.
>
> Работа выполнена на высоком академическом уровне. Код минималистичен,
> тесты исчерпывающи, документация образцовая. Вы честно обозначили
> границы достигнутого, что критически важно для сообщества пользователей
> EOL-железа. Работа готова к мержу и переходу к следующему этапу (TODO-1).

### Grade breakdown

| Criterion | Grade | Note |
|---|---|---|
| Math correctness | A+ | `max_diff < 0.0001` vs SDPA; CUTLASS sm_75 path is correct |
| Reproducibility | A | `VERSION_PINS.md` is exemplary |
| Performance & Honesty | A | Per-step 5.07 s vs 5.6 s = +10%; wall-time 141.72 s vs 140 s; honest reporting |

### Checklist (8 YES/NO questions)

1. ✅ Patch applies cleanly to upstream `965c921`
2. ✅ 5/5 unit tests pass, `max_diff < 0.0001`
3. ✅ E2E output visually correct (documented; file size 2.29 MB = no gray noise artifact)
4. ✅ Speed measurement honest (per-step vs wall-time separated, warm cache noted)
5. ✅ BSD-3-Clause attribution correct (`LICENSE`, `NOTICE.md`)
6. ✅ Version pins accurate (`VERSION_PINS.md` comprehensive)
7. ✅ TODO list complete (TODO-4 H3 correctly marked as mandatory)
8. ✅ README accurate, no exaggeration, EOL support is the stated goal

### Recommendations applied

| # | Recommendation | Status |
|---|---|---|
| 1 | Change default `FLASH_ATTN_CUDA_ARCHS` from `75;80;90;100;120` to `75` in `apply-patch-and-install.sh` | ✅ applied in `02e9926+1` |
| 2 | TODO-1 (Profile): pay special attention to Q/K/V/O projection linear layers, not just attention | 📝 noted in `TODO.md` |
| 3 | TODO-4 (H3): don't defer; success on H3 is final proof of patch universality | 📝 noted in `TODO.md` |
| 4 | TODO-2 (CUDA Graphs): be prepared for 0% or workarounds on sm_75; PyTorch CUDA graph support was optimized for sm_80+ | 📝 noted in `TODO.md` |

### Qwen's full review (raw text)

The full Russian-language review is preserved in the
`/workspace/attachments/c40c00089bc28c7a/pasted-text.txt` file from
the orchestrator's session. It contains the same 8-question
checklist pass/fail as above, plus extended commentary on each
criterion. The next Mavis session can re-read it in full if needed.

## Review 2: Mavis (implementation session, follow-up review)

**Date:** TBD (pending vaskes's assignment)
**Status:** ⏳ pending

The Mavis review is the second-pass review per vaskes's instruction.
The Mavis reviewer will:
- Re-run the unit tests from a clean install to verify
  reproducibility.
- Re-run the e2e test to verify the Qwen output is recognizable.
- Re-run the speed test to verify the 5.07 s/it number.
- Walk through the TODO list and confirm that TODO-1 (profile) is
  the right next step.
- Confirm the Qwen review was thorough (i.e. that we did not
  misrepresent Qwen's verdict).

After both reviews are in, this file will be updated to a
"final" state and the work tagged `flash-sm75-v0.2` (or
`flash-sm75-v0.1` confirmed) for merge.
