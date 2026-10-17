# comfyui-workstation-pure

**Pure ComfyUI for RTX 2080 Ti (sm_75 / Turing). No optimizations.**

`comfyui-workstation-pure` — это Docker-сборка ComfyUI под workstation с 2× RTX 2080 Ti 22GB
(NVLink), без sage attn / flash attn / xformers / кастомных ядер. Цель — стабильный запуск
production-grade workflow (Qwen-Image 2.1, hunyuan-style, INT8 quant) на Turing без трюков.

## Что внутри

| Компонент | Версия |
|---|---|
| Base image | `nvidia/cuda:13.0.3-cudnn-runtime-ubuntu24.04` |
| Python | 3.12 (system, через venv в `/opt/venv`) |
| PyTorch | 2.14.1 + cu130 (stable wheels) |
| Triton | 3.8.0 (для JIT bmm/attention kernels внутри pytorch) |
| gcc / g++ | 13.3.0 (apt, для Triton JIT) |
| ComfyUI | master (закреплено в Dockerfile как `master` branch) |
| ComfyUI-Manager | он же `comfy_kitchen` 0.2.38, в `custom_nodes/` |

## Что НЕ внутри

Никаких оптимизаций, кроме того что идёт в `pip install` самого ComfyUI:

- ❌ `sage-attention`
- ❌ `flash-attn` (не работает на sm_75)
- ❌ `xformers`
- ❌ кастомные CUDA-kernels
- ❌ `pytorch.compile()` на уровне пользователя
- ❌ `torch.compile(mode="max-autotune")`

Если тебе нужны оптимизации — это другая репа. Здесь — **stable only**.

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
├── Dockerfile         # image build
├── compose.yaml       # GPU 0, port 8188, bind mounts
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

# 3. (опционально) Положить токены в .env
cat > .env <<EOF
HF_TOKEN=hf_...
EOF
chmod 600 .env

# 4. Build + up
docker compose up -d --build

# 5. Проверить
docker logs -f comfyui           # To see the GUI go to: http://0.0.0.0:8188
curl -s http://localhost:8188/system_stats | python3 -m json.tool
```

UI: <http://localhost:8188> (или `ssh -L 8188:127.0.0.1:8188 workstation` если хост за NAT).

## Что работает (валидировано)

| Workflow | Модель | VRAM peak | Время (1 sample) |
|---|---|---|---|
| Qwen-Image 2.1 (25 шагов, bf16) | qwen_image_2.1 + qwen3vl_8b | ~30 GB staged | ~2:21 |
| Hunyuan-style H3 (20 шагов, INT8 convrot) | h3 + qwen3vl_32b_int8_convrot | ~46 GB staged | ~21:00 |

Скорости — **не оптимизированные**. pytorch attention + cuda/eager kitchen backends. Triton
доступен (как backend), но kitchen видит его `available=True, disabled=True` (sm_75 не
в `min_compute_capability` для нескольких quantized kernels — это by design comfy_kitchen).
Работает через `cuda` и `eager` бэкенды.

## Known issues

- **Triton в логе `disabled=True`** — это не баг. comfy_kitchen маркирует triton так для
  sm_75 для некоторых quantized kernels (нужен sm_80+). Но Triton как runtime для pytorch
  (`bmm_outer_product` etc.) работает — gcc/g++ 13.3.0 в образе обязательны.
- **Slow generation** — 5–60 s/iter это baseline на 2080 Ti без оптимизаций. На Blackwell
  (sm_100) в 5-10× быстрее. Здесь — accept it или форкай с оптимизациями.
- **AudioVAE + VideoVAE одновременно** — peak VRAM ~46 GB. 22 GB карты не хватит, нужно
  либо 2×22 (44 GB) с NVLink либо CPU offload.

## Полезные команды

```bash
# Логи
docker logs -f comfyui --tail 100

# Зайти в контейнер
docker exec -it comfyui bash

# Зайти в venv
docker exec -it comfyui /opt/venv/bin/python

# Проверить kitchen backends в живом контейнере
docker exec comfyui python3 -c "
from comfy_kitchen.registry import registry
for n in ['cuda', 'eager', 'triton']:
    s = registry.list_backends().get(n, {})
    print(n, 'avail=', s.get('available'), 'disabled=', s.get('disabled'))
"

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
