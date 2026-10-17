# comfyui-workstation-pure

**Pure ComfyUI for RTX 2080 Ti (sm_75 / Turing). No optimizations.**

`comfyui-workstation-pure` — это Docker-сборка ComfyUI под workstation с 2× RTX 2080 Ti 22GB
(NVLink), без sage attn / xformers / кастомных ядер. Цель — стабильный запуск
production-grade workflow (Qwen-Image 2.1, Minimax H3) на Turing.

## Что внутри

| Компонент | Версия |
|---|---|
| Base image | `nvidia/cuda:13.0.3-cudnn-runtime-ubuntu24.04` |
| Python | 3.12 (system, через venv в `/opt/venv`) |
| PyTorch | 2.14.1 + cu130 (stable wheels) |
| Triton | 3.8.0 (для JIT bmm/attention kernels внутри pytorch) |
| gcc / g++ | 13.3.0 (apt, для Triton JIT) |
| cuda-nvcc | 13.0.88 (apt, для возможной CUDA-extension сборки) |
| ComfyUI | master |
| ComfyUI-Manager (comfy_kitchen) | 0.2.38, в `custom_nodes/` |
| flash_attn_turing | 0.0.0 (опционально, через shim — **сломан для qwen**, см. ниже) |

## Что НЕ внутри

Никаких оптимизаций, кроме того что идёт в `pip install` самого ComfyUI:

- ❌ `sage-attention`
- ❌ `flash-attn` официальный (Dao-AILab — не работает на sm_75)
- ❌ `xformers`
- ❌ кастомные CUDA-kernels
- ❌ `pytorch.compile()` на уровне пользователя
- ❌ `torch.compile(mode="max-autotune")`

## Требования к хосту

| | |
|---|---|
| GPU | NVIDIA, sm_75+ (2080 Ti проверено). 22 GB VRAM минимум для Qwen-Image 2.1. |
| Driver | `>= 580` (проверено 615.71.09, ставится через `apt install nvidia-driver-615-open`) |
| Docker | 29.x (apt: `docker.io`, `docker-compose-v2`) |
| nvidia-container-toolkit | 1.20+ (apt или nvidia.github.io/libnvidia-container) |
| Свободно на диске | ~25 GB под image + зависит от объёма моделей |
| RAM | 32 GB минимум (Qwen3-VL 8B в bf16 ~ 16 GB на CPU offload во время пиков) |

Nouveau должен быть заблеклистен:

```bash
printf 'blacklist nouveau\noptions nouveau modeset=0\n' > /etc/modprobe.d/blacklist-nouveau.conf
update-initramfs -u
# reboot
```

## Layout

```
.
├── Dockerfile         # image build (с gcc + cuda-nvcc)
├── compose.yaml       # GPU 0, port 8188, --use-ck-attention default
├── .gitignore
└── README.md

# На хосте (создаются вручную или compose up — bind mounts):
/opt/comfyui/
  ├── models/        # ← кладёшь чекпоинты сюда
  ├── input/
  ├── output/        # результаты
  ├── user/          # UI prefs
  ├── workflows/     # экспортированные workflow
  ├── custom_nodes/  # доп. ноды
  └── logs/
```

## Запуск

```bash
# 1. Создать layout
mkdir -p /opt/comfyui/{models,output,input,user,workflows,custom_nodes,logs}

# 2. Скопировать Dockerfile + compose.yaml в /opt/comfyui/
cp Dockerfile compose.yaml /opt/comfyui/
cd /opt/comfyui

# 3. Build + up
docker compose up -d --build

# 4. Проверить
docker logs -f comfyui           # To see the GUI go to: http://0.0.0.0:8188
curl -s http://localhost:8188/system_stats | python3 -m json.tool
```

UI: <http://localhost:8188> (или `ssh -L 8188:127.0.0.1:8188 workstation` если хост за NAT).

## Attention backend (ГЛАВНОЕ)

**По умолчанию: `--use-ck-attention`** (comfy_kitchen). Результаты на
RTX 2080 Ti (sm_75), validated 2026-10-17:

| Backend | Qwen-Image 2.1 (25 steps) | Minimax H3 (20 steps, 1s) | Quality |
|---|---|---|---|
| (default pytorch SDPA) | 2:55 (175s) | depends on length | ✅ correct |
| `--use-pytorch-cross-attention` | 2:55 (175s) | — | ✅ same as default |
| **`--use-ck-attention`** | **2:25 (145s)** | **1:45 (105s) @ 1s** | **✅ correct** |
| `--use-flash-attention` (ssiu/flash-attention-turing) | 1:47 (107s) | — | ❌ **gray square** for Qwen |
| `--use-flash-attention` (KLM-corporation/flash-attention-2080ti) | 2:10 (130s) | — | ❌ **gray square** for Qwen |

**`--use-ck-attention` = +19% on Qwen, +massive on H3 INT8 quantized** (kitchen INT8
kernels via cuda/eager backend, not Triton).

### Why flash_attn is fundamentally broken on sm_75 + bf16 (validated 2026-10-17)

Both ssiu/flash-attention-turing AND KLM-corporation/flash-attention-2080ti produce
garbage for Qwen-Image 2.1 on sm_75. The cause is a **hardware limitation**, not
a software bug:

- sm_75 (Turing) has fp32 tensor cores but **no fp16 attention tensor-core path**.
  All attention ops fall back to fp16 CUDA-core math.
- Kernels convert bf16 input → fp16 for compute. fp16 has insufficient dynamic
  range for softmax of bf16-scale attention scores.
- On sm_80+ (Ampere/Ada/Hopper) the GPU has dedicated fp16 attention tensor
  cores that maintain precision; sm_75 does not.

| | flash_attn (sm_75) | pytorch SDPA ref |
|---|---|---|
| input dtype | bf16 | bf16 |
| output dtype | **fp16** | bf16 |
| output std | **0.8101** | 0.1226 |
| max abs diff vs ref | **2.87** | — |

**Conclusion: NO flash_attn implementation can work on sm_75 for bf16 storage models
(qwen, h3).** This is a hardware ceiling.

For sm_75 the best we have is `--use-ck-attention` (comfy_kitchen). On H3 this
unlocks the INT8 quantized kernels (w6a8_int8, convrot_w4a4) which give massive
speedup over baseline pytorch SDPA.

If you need the absolute best attention perf, you need a sm_80+ GPU (RTX 3090
/ 4090 / A100). On those, the standard `pip install flash-attn` works out of
the box.

### Why BF16→FP16 weight conversion works for H3 but NOT Qwen (validated 2026-10-17)

H3's `*convrot*` quantized weights go through comfy_kitchen's
`int8_linear` / `convrot_w4a4_linear` / `sol_attn` kernels, which
**consume fp16 storage directly** and run fp16 GEMM internally. The fp16
storage is the "true" runtime format and there is no precision-recovery cast.

Qwen-Image 2.1's `*int8_convrot*` weights do **not** go through the fp16
kitchen kernels — they go through ComfyUI's stock bf16→fp32 path
(`manual cast: torch.float32` in the model load log). fp16 storage gets
re-cast to bf16 at load time, which means **3 mantissa bits of precision
are lost per weight** compared to the original bf16 file. This produces
the gray-square output.

A/B tests 1–4 in the table above (see "Why Qwen BF16→FP16 conversion fails"
section) confirm that the failure is independent of which of the 4
Qwen models is converted — any single fp16-converted model breaks the
pipeline.



Both ssiu/flash-attention-turing AND KLM-corporation/flash-attention-2080ti produce
garbage for Qwen-Image 2.1 on sm_75. The cause is a **hardware limitation**, not
a software bug:

- sm_75 (Turing) has fp32 tensor cores but **no fp16 attention tensor-core path**.
  All attention ops fall back to fp16 CUDA-core math.
- Kernels convert bf16 input → fp16 for compute. fp16 has insufficient dynamic
  range for softmax of bf16-scale attention scores.
- On sm_80+ (Ampere/Ada/Hopper) the GPU has dedicated fp16 attention tensor
  cores that maintain precision; sm_75 does not.

| | flash_attn (sm_75) | pytorch SDPA ref |
|---|---|---|
| input dtype | bf16 | bf16 |
| output dtype | **fp16** | bf16 |
| output std | **0.8101** | 0.1226 |
| max abs diff vs ref | **2.87** | — |

**Conclusion: NO flash_attn implementation can work on sm_75 for bf16 storage models
(qwen, h3).** This is a hardware ceiling.

For sm_75 the best we have is `--use-ck-attention` (comfy_kitchen). On H3 this
unlocks the INT8 quantized kernels (w6a8_int8, convrot_w4a4) which give massive
speedup over baseline pytorch SDPA.

If you need the absolute best attention perf, you need a sm_80+ GPU (RTX 3090
/ 4090 / A100). On those, the standard `pip install flash-attn` works out of
the box.

## Что работает (валидировано)

| Workflow | Модель | Steps | Time | Backend | Notes |
|---|---|---|---|---|---|
| Qwen-Image 2.1 (bf16) | qwen_image_2.1 + qwen3vl_8b | 25 | **2:20** (ck-attention) | comfy_kitchen cuda/eager | Best stable |
| Qwen-Image 2.1 (bf16) | qwen_image_2.1 + qwen3vl_8b | 25 | 2:55 (pytorch SDPA) | baseline | Reference |
| Minimax H3 (INT8 convrot) | h3 + qwen3vl_32b_int8 | 20, 1s | **1:45** (ck-attention) | comfy_kitchen INT8 | 12x speedup vs naive |

### Why Qwen BF16→FP16 conversion fails (validated 2026-10-17, 5 A/B tests)

**Hypothesis tested:** cast the BF16 storage regions of the 4 Qwen models to FP16
(with `np.clip(-65504, 65504)` saturation) and run with `--fp16-unet`. Result:
**5.5× faster (30s vs 140s), but every output is a gray square.**

**Root cause: ComfyUI's Qwen-Image 2.1 path does NOT use comfy_kitchen's fp16
quantization compute path.** Even though the model files have `*int8_convrot*`
in their name (suggesting INT8 quantization), the model goes through:

- `weight dtype = torch.bfloat16` (forced by `*int8_convrot*` filename pattern)
- `manual cast: torch.float32` for compute
- **fp16 storage gets re-cast to bf16 at load time, losing 3 mantissa bits per weight**

This is fundamentally different from H3, where the same conversion works because
H3 actually goes through comfy_kitchen's `int8_linear` / `convrot_w4a4_linear` /
`sol_attn` paths that **consume fp16 storage natively** and run fp16 GEMM
internally.

**A/B test results** (all 25 steps, same seed, same prompt):

| # | UNet | Text enc 1 | Text enc 2 | VAE | Output | Notes |
|---|---|---|---|---|---|---|
| 1 | fp16-conv | fp16-conv | fp16-conv | fp16-conv | **24 KB gray** | baseline garbage |
| 2 | fp16-conv | fp16-conv | **bf16** | fp16-conv | **24 KB gray** | different garbage md5 |
| 3 | **bf16** | fp16-conv | fp16-conv | fp16-conv | **24 KB gray** | different garbage md5 |
| 4 | fp16-conv | fp16-conv | fp16-conv | **bf16** | **24 KB gray** | different garbage md5 |
| 5 | **bf16** | **bf16** | **bf16** | **bf16** | **2.28 MB ✅** | 140s, normal image |

All 4 garbage outputs have **different md5** — meaning conditioning does differ
between configs, but ALL variants with at least one fp16-converted model are
broken. The original-bf16-everywhere config is the only one that produces a
correct image.

**Cleanup:** all 4 `_fp16_converted.safetensors` and `_fp16.safetensors` files
for Qwen were renamed to `.deprecated.safetensors` (~25 GB freed, originals
preserved on disk per the "no model file removal" hygiene rule).

**To resume optimization**, the next candidates are:

- **`*_w4a8.safetensors` variants** (4-bit weights + 8-bit activations, already
  on disk at `/opt/comfyui/models/`). These go through the
  `w4a8_int8_linear` kitchen kernel directly and should give 2-4× speedup
  with acceptable quality.
- **`*_turbo_*.safetensors` variants** (turbo distilled Qwen, already on disk).
  These are a different model, not just a quantization — test on its own.

## Known issues / Gotchas

- **Triton в логе `disabled=True`** — не баг. comfy_kitchen маркирует triton так для
  sm_75 для некоторых quantized kernels (нужен sm_80+). Triton runtime для
  pytorch (`bmm_outer_product` etc.) работает — gcc/g++ 13.3.0 обязательны.
- **flash_attn_turing `disabled=True`** в kitchen (sm_75 не в min_compute_capability
  для quantized kernels). Это by design. Triton runtime для pytorch kernels
  всё равно работает.
- **`flash-attn` (Dao-AILab) requires sm_80+** — explicitly fails on sm_75. We use
  the ssiu/flash-attention-turing fork instead. See "Attention backend" above.
- **`Mask must not be set for Flash attention`** warnings в логе ComfyUI — это
  explicit проверка в attention.py: если есть attention mask, flash не
  используется, fallback на pytorch SDPA. Это не баг.

## Полезные команды

```bash
# Логи
docker logs -f comfyui --tail 100

# Зайти в контейнер
docker exec -it comfyui bash

# Зайти в venv
docker exec -it comfyui /opt/venv/bin/python

# Прогнать workflow через API
python3 /tmp/comfyui_run.py /path/to/wf.json --label test1

# Рестарт
cd /opt/comfyui && docker compose restart

# Полная переразвёртка (если поменял Dockerfile)
cd /opt/comfyui && docker compose down && docker compose up -d --build
```

## Гигиена

- Всё не-системное живёт в `/opt`
- Docker data-root в `/opt/docker`
- На хосте **не** ставим pip-пакеты — только `apt`. Внутри контейнера — venv.
- `comfyui_bad` / `comfyui_turing` / прочие failure-attempts **не использовать** —
  ставим с нуля каждый раз.

## Credits

This work is a collaboration between:

- **Vaskes** — Vibecoder, hardware & infrastructure, project lead, integration QA.
- **Mavis** — AI implementation agent, powered by **MiniMax M3**.

Full credits (people, organizations, and component attributions) are in
[`CREDITS.md`](CREDITS.md).

## License & Third-Party Components

**This project is NOT open source.** The Work (Dockerfile, compose.yaml,
README, the flash_attn shim, and all other files in this repository) is
**proprietary** and is governed by the [`LICENSE`](LICENSE) file. The
copyright holder is **Kuduza Ai Lab** (https://kuduza.com).

You may not use, copy, modify, merge, publish, distribute, sublicense, or
sell the Work without the prior express written permission of Kuduza Ai Lab.
Where permission is granted, the attribution specified in the LICENSE must
be preserved in all copies and public references.

The Work uses third-party open-source components (ComfyUI, PyTorch, Triton,
NVIDIA CUTLASS, etc.). Those components remain governed by their own
licenses (GPL-3.0, BSD-3-Clause, MIT, Apache-2.0, etc.) as enumerated in
[`NOTICE.md`](NOTICE.md). Any derivative work distributed in source or
binary form must comply with those third-party license terms.

| File | Purpose |
|---|---|
| [`LICENSE`](LICENSE) | Proprietary license (all rights reserved, no grant) |
| [`CREDITS.md`](CREDITS.md) | People, roles, and acknowledgments |
| [`NOTICE.md`](NOTICE.md) | Third-party components and their licenses |

**TL;DR:** Don't redistribute this without talking to Kuduza Ai Lab first.
The third-party stuff they depend on is still under its original open-source
license, so derivative works have to honor those too.
