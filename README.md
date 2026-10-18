# ComfyUI on RTX 2080 Ti (sm_75, Turing) — production fork

This fork of [comfyui-2080ti-turing](https://github.com/vaskes/comfyui-2080ti-turing)
ships **three attention backends** for the sm_75 (Turing) hardware, with
**SageAttention 1.0.6 as the production default** and `comfy_kitchen` (ck)
as the fastest correct alternative. The hardware is end-of-life for
upstream `flash-attention` 2.x (no sm_75 mma support), so this fork
provides working backends via three independent code paths.

## Production contract (2026-10-18)

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
├── LICENSE                    # BSD-3-Clause (production)
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
# default (sage)
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
- **VRAM**: 11 GB
- **Driver**: nvidia-driver-*-open (no nvidia-container-toolkit on host needed,
  runtime is `nvidia` in compose.yaml)
- **Container**: `comfyui-workstation:cu130-torch-stable-gcc`

## Provenance

This fork started as a `comfyui-2080ti-turing` pure-config branch and
gradually grew:
- v0.1: flash-attention v0.2.1 shim (bf16/fp16 only, broken for int8)
- v0.2.1: full matrix + ck baseline + flash attempts
- v0.3: real sm_75 flash build via 2-line patch (correct for H3, broken for Qwen int8)
- **v1.0 (this release)**: sage 1.0.6 + ck + flash v0.3, sage as default

Patches in `flash-attention-sm75/patches/` are the historical v0.3 work
for the optional `--use-flash-attention` experimental path. They are
NOT applied by default — the container ships with sage as default and
flash is only loaded if `--use-flash-attention` is in `COMFYUI_ARGS`.

## License

BSD-3-Clause. See [LICENSE](LICENSE) and [NOTICE.md](NOTICE.md).
Third-party attributions: comfyui, comfyui-manager, comfy_kitchen,
SageAttention (Tsinghua), Dao-AILab flash-attention.

## Contact

`vaskes` on GitHub. Issues and PRs welcome for the sm_75 attention
backends or the production compose setup.
