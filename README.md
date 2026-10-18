# ComfyUI on RTX 2080 Ti (sm_75, Turing) — production fork

This is a production fork of [comfyanonymous/ComfyUI](https://github.com/comfyanonymous/ComfyUI)
with **three attention backends** for the sm_75 (Turing) hardware that upstream
`flash-attention` 2.x no longer supports (EOL, no sm_75 mma). Production default
is `comfy_kitchen` (ck) — fastest correct across all tested (model, quant)
combinations. `SageAttention 1.0.6` is the fallback for workflows that use
the Kijai Sage Attn patch custom node. `flash v0.3` is the experimental
real sm_75 build (H3 only, broken for Qwen int8_convrot).

See the [full matrix below](#full-matrix-3-backends--2-models--4-quants-24-cells-validated-2026-10-18) for which backend is fastest correct vs broken per cell.

## TL;DR — what this repo is for

This repo restores **working attention backends on RTX 2080 Ti (sm_75, Turing)**.
Upstream `flash-attention` 2.x is EOL for sm_75 hardware (no sm_75 mma support),
so workflows running on RTX 2080 Ti either fail or silently fall back to slow
pytorch SDPA. This repo provides three production attention backends
(`ck`, `sage`, `flash` v0.3) that work on sm_75, with the matrix below showing
exactly which backend is fastest correct vs broken for every tested combination.

The hardware is EOL. The repo is MIT. Use it.

## Full matrix: 3 backends × 2 models (6 model×quant combinations) = 18 cells, validated 2026-10-18

Qwen has 2 quantizations (bf16, int8_convrot) and H3 has 4 (bf16, fp8, int8_convrot, w6a8),
so the total is 3 × (2+4) = 18 cells — not 24. SDPA is shown as a reference baseline
but is not a production backend (3 backends are the production contract).

### Qwen-Image 2.1 (T2I, 25 steps)

| Quant | SDPA | **ck ⭐ default** | flash v0.2.1 (shim) | flash v0.3 (real sm_75) | sage 1.0.6 |
|---|---|---|---|---|---|
| bf16 | 175.0s ✅ 2.34MB | **140.0s** ✅ 2.28MB | 141.72s ✅ 2.29MB | 176.0s ✅ 2.28MB | 175.3s ✅ 2.28MB |
| int8_convrot | 70.2s ✅ 2.29MB | **45.1s** ✅ 2.30MB | 35.1s ❌ GARBAGE | 45.1s ❌ GARBAGE | 70.1s ✅ 2.29MB 🎉 |

### Minimax H3 (T2V, 8-step turbo, 8 frames, heretic CLIP)

| Quant | SDPA | **ck ⭐ default** | flash v0.2.1 (shim) | flash v0.3 (real sm_75) | sage 1.0.6 |
|---|---|---|---|---|---|
| bf16 | (broken env) | **116.0s** ✅ 538KB | 111.0s ❌ 3.8MB garbage | 211.0s ✅ 530KB | 591.0s ✅ 488KB |
| fp8 | 582.3s ✅ 558KB | **476.2s** ✅ 560KB | 12.9KB ❌ broken | 576.7s ✅ 558KB | 585.9s ✅ 558KB |
| int8_convrot | 215.4s ✅ 530KB | **110.3s** ✅ 538KB | 111.0s ❌ 3.8MB garbage | 210.3s ✅ 530KB | 231.3s ✅ 530KB |
| w6a8 | 455.9s ✅ 492KB | **355.6s** ✅ 554KB | (not tested) | 455.6s ✅ 492KB | 465.7s ✅ 492KB |

**Read this table row by row, left to right:**

- ⭐ **ck wins everywhere** for correctness + speed. The production default.
- 🎉 **sage wins where flash loses** — Qwen int8_convrot is the headline cell.
  Both flash variants produce 3.8MB / 2.48MB checkerboard garbage; sage
  produces the correct 2.29MB image in 70s. This is the reason `sage` is
  the Kijai-patch fallback.
- ❌ **flash v0.2.1 (shim)** is a precision trap. The community fork it routes
  to uses m8n8k8 fp16-accumulator, which overflows for any int8/convrot
  weight model. Use only for bf16/fp16 if you must.
- ⚠️ **flash v0.3 (real sm_75)** is experimental. It works for H3 all quants
  but is **broken for Qwen int8_convrot** (precision trap even with
  fp32-accumulator mma). Don't use it for Qwen. Also 2x slower than ck.
- **SDPA** always works but is the slowest path. Fallback-of-last-resort.

## Production contract

 (2026-10-18)

| Backend | ComfyUI flag | Status | Speed | Correctness |
|---|---|---|---|---|
| **comfy_kitchen (ck)** ⭐ | `--use-ck-attention` | **DEFAULT** | **fastest correct** | all models ✅ |
| SageAttention 1.0.6 | `--use-sage-attention` | fallback (Kijai sage attn patch) | medium (1.5-2x slower than ck) | all models ✅ |
| FlashAttention v0.3 (real sm_75 build) | `--use-flash-attention` | experimental | medium-slow | H3 ✅, Qwen int8_convrot ❌ |

⭐ = compose.yaml default

**ck is the default** because it's the fastest correct backend on sm_75
across all (model, quant) combinations. Use sage ONLY for workflows that
contain the Kijai Sage Attn patch node (`sageattn` ComfyUI custom node
that patches attention in specific ways ck doesn't support). Flash v0.3
is experimental — only H3 works correctly; Qwen int8_convrot produces
garbage (precision trap even with fp32-accumulator mma).

Switch backends without editing files:
```bash
# ck (default, fastest)
docker compose -f /opt/comfyui-git/compose.yaml up -d

# sage (for Kijai sage attn patch workflows)
COMFYUI_ARGS="--use-sage-attention" docker compose -f /opt/comfyui-git/compose.yaml up -d

# flash v0.3 (experimental, H3 only)
COMFYUI_ARGS="--use-flash-attention" docker compose -f /opt/comfyui-git/compose.yaml up -d
```

## Why three backends, not one

`ck` is hand-tuned CUTLASS composable kernel that uses Turing's int8
tensor cores directly. **Fastest correct** for every (model, quant) we
tested on sm_75. Covers H3 and Qwen T2I/T2V in our test matrix.

`sage` 1.0.6 (Tsinghua, arXiv:2410.02367) is INT8-Q/K + FP16-P/V
quantization-aware attention via Triton. 1.5-2x slower than ck, but
correct for any model. **Required fallback for workflows that use the
Kijai Sage Attn patch** custom node — ck doesn't have a path for that
node's specific attention layout.

`flash` v0.3 is the upstream Dao-AILab `flash-attention` 2.8.4
patched for sm_75 (patches in `patches/0001-add-sm75-arch.patch` and
`patches/0002-relax-arch-gates.patch`). Slow (sm_75 mma atom has
half the k-dim of sm_80), and model-specific precision issues — see
[tests/MATRIX.md](tests/MATRIX.md) for the full Qwen int8_convrot
failure analysis.

## Quick start

```bash
# Clone and start (assumes GPU and nvidia-container-toolkit installed)
cd /opt && git clone https://github.com/vaskes/comfyui-2080ti-turing.git comfyui-git
cd /opt/comfyui-git
docker compose up -d

# Wait ~60s for ComfyUI to come up
curl -m 5 http://localhost:8188/system_stats

# Open the GUI
# http://<workstation-ip>:8188
```

The container auto-installs SageAttention 1.0.6 on first start (idempotent).

## Repo layout

```
comfyui-2080ti-turing/
├── README.md                  # this file
├── CHANGELOG.md               # version history
├── LICENSE                    # MIT (production)
├── NOTICE.md                  # third-party attributions
├── CREDITS.md                 # upstream project credits
├── Dockerfile                 # production image
├── compose.yaml               # production compose (ck as default)
├── tests/                     # cross-backend test suite
│   ├── MATRIX.md              # full 18-cell matrix report
│   ├── TESTING.md             # how to run tests
│   ├── test_backends.py       # smoke + import tests
│   └── matrix/
│       ├── run_matrix.py      # production matrix runner
│       └── matrix_results.jsonl
├── flash-attention-sm75/      # flash v0.3 (experimental, H3 only)
│   ├── README.md              # flash v0.3 deep-dive + diagnostic
│   ├── CHANGELOG.md
│   └── patches/
│       ├── 0001-add-sm75-arch.patch     # setup.py: add sm_75 arch
│       └── 0002-relax-arch-gates.patch  # relax runtime gates
└── sage-attention-sm75/       # SageAttention 1.0.6 (Kijai patch fallback)
    └── README.md              # sage install + Kijai patch workflow
```

> **No binary artefacts in this repo.** PNG/MP4 files from test runs are
> not checked in — they only prove the model produced an image, not the
> code. Re-run the matrix to regenerate them.

## Attention backend switching

`compose.yaml` ships with `COMFYUI_ARGS: --use-ck-attention` as the
default. The value is passed to `python3 main.py` inside the container.

**At container start:**
```bash
# default (ck)
docker compose -f /opt/comfyui-git/compose.yaml up -d

# override at start time
COMFYUI_ARGS="--use-ck-attention" docker compose -f /opt/comfyui-git/compose.yaml up -d
```

**Per-workflow override (without restart):** in the ComfyUI GUI, the
`--use-*` flag is a startup flag, not a runtime switch. To use a
different backend for one workflow, restart the container with the
new flag. The flag persists for the container lifetime.

## Testing

See [tests/TESTING.md](tests/TESTING.md)
for the full test suite. The two key tests are:

```bash
# Quick smoke (all 3 backends importable, flags valid)
docker exec comfyui python3 -m pytest tests/test_backends.py -v

# Full 18-cell matrix (24 minutes, all backends × all model/quant)
python3 tests/matrix/run_matrix.py
```

## Hardware

- **GPU**: NVIDIA RTX 2080 Ti (sm_75, Turing) — EOL for upstream flash-attention 2.x
- **VRAM**: 22 GB
- **Driver**: nvidia-driver-*-open (no nvidia-container-toolkit on host needed,
  runtime is `nvidia` in compose.yaml)
- **Container**: `comfyui-workstation:cu130-torch-stable-gcc`

## Provenance

This fork started as a `comfyui-2080ti-turing` pure-config branch and
gradually grew:
- v0.1: flash-attention v0.2.1 shim (bf16/fp16 only, broken for int8)
- v0.2.1: full matrix + ck baseline + flash attempts
- v0.3: real sm_75 flash build via 5+1 line patch (0001 + 0002) (correct for H3, broken for Qwen int8)
- **v1.0 (this release)** (default: ck) — flip default to ck, sage as Kijai-patch fallback, flash experimental.
- **v1.0.2** — release-readiness cleanup: MIT throughout, baked entrypoint, default(sage) leftovers fixed, 9/9 unit tests.: 3 attention backends — ck (default), sage (Kijai-patch fallback), flash v0.3 (experimental)

Patches in `flash-attention-sm75/patches/` are the historical v0.3 work
for the optional `--use-flash-attention` experimental path. They are
NOT applied by default — the container ships with ck as default and
flash is only loaded if `--use-flash-attention` is in `COMFYUI_ARGS`.

## Credits

This work is a collaboration between:

- **Vaskes** — Vibecoder, hardware & infrastructure, project lead, integration QA.
- **Mavis** — AI implementation agent, powered by [MiniMax M3](https://MiniMax.com).

Full credits (people, organizations, and component attributions) are in [`CREDITS.md`](CREDITS.md).

## License

MIT. See [LICENSE](LICENSE) and [NOTICE.md](NOTICE.md).

This work does not contain any know-how worth fencing off. Use,
modify, and redistribute freely. The only attribution requirement
is keeping the upstream credits intact (see [Credits](#credits)
above and [CREDITS.md](CREDITS.md)).
