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
| `--use-flash-attention --force-upcast-attention` | 1:47 (107s) | — | ❌ still gray |

**`--use-ck-attention` = +19% on Qwen, +massive on H3 INT8 quantized** (kitchen INT8
kernels via cuda/eager backend, not Triton).

**`flash_attn_turing` is broken for sm_75 + bf16 qwen text encoder.** Even with
upcast to fp32 the output is uniform gray. We don't recommend it for now. See
[flash-attn-turing](#flash-attn-turing) below.

If you want the absolute fastest no-flash option: stick with `--use-ck-attention`.

### flash-attn-turing

Status: **installed in image, but disabled by default because it produces garbage for Qwen.**

The shim at `/opt/venv/lib/python3.12/site-packages/flash_attn/__init__.py` routes
the standard `flash_attn` API to the Turing build from
[https://github.com/ssiu/flash-attention-turing](https://github.com/ssiu/flash-attention-turing)
(sm_75-only, fp16-only, no dropout, no mask).

To enable manually for testing:
1. Edit `compose.yaml`: change `COMFYUI_ARGS` to add `--use-flash-attention`
2. `docker compose down && docker compose up -d`

**Caveat**: For Qwen-Image 2.1 on sm_75, the result is a uniform gray image —
the Turing-compiled .cu kernels are numerically broken for bf16 + sm_75. The
official issue is at ssiu/flash-attention-turing (no fix at the time of
writing). For H3, not validated — likely similar issue since h3 text encoder
is also bf16.

We're tracking alternative forks (e.g. `KLM-corporation/flash-attention-2080ti`
which advertises CUDA 13.0 / PyTorch 2.9.0 support) but none has been tested yet.

## Что работает (валидировано)

| Workflow | Модель | Steps | Time | Backend | Notes |
|---|---|---|---|---|---|
| Qwen-Image 2.1 (bf16) | qwen_image_2.1 + qwen3vl_8b | 25 | **2:25** (ck-attention) | comfy_kitchen cuda/eager | Best stable |
| Qwen-Image 2.1 (bf16) | qwen_image_2.1 + qwen3vl_8b | 25 | 2:55 (pytorch SDPA) | baseline | Reference |
| Minimax H3 (INT8 convrot) | h3 + qwen3vl_32b_int8 | 20, 1s | **1:45** (ck-attention) | comfy_kitchen INT8 | 12x speedup vs naive |

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

## Лицензия

MIT (или как решит vaskes).
