# ComfyUI production image for RTX 2080 Ti (sm_75, Turing) — workstation
# Production fork of upstream ComfyUI with three attention backends:
#   - comfy_kitchen (ck)   — DEFAULT, fastest correct across all models
#   - SageAttention 1.0.6  — Kijai-patch fallback, auto-installed on first run
#   - flash-attn v0.3      — experimental real sm_75 build, bind-mounted from host
# Base: nvidia/cuda:13.0.3-cudnn-runtime-ubuntu24.04
# PyTorch: stable cu130 wheels. Host driver 615.71.09.
# Entrypoint: docker/entrypoint.sh (baked into the image).

FROM nvidia/cuda:13.0.3-cudnn-runtime-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONHASHSEED=random

# --- system deps (только apt; на хосте ничего не pip-им) ---
# python3.12 + venv + git + минимальные либы для opencv-headless / pillow.
# libgl1, libglib2.0-0 — нужны для некоторых custom_nodes (cv2 headless).
# libsndfile1 — safetensors audio.
# gcc/g++ — нужны для Triton JIT (компилит .so для bmm/attention kernels внутри pytorch).
# Без них pytorch упадёт: "RuntimeError: Failed to find C compiler".
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        ca-certificates curl wget git \
        python3.12 python3.12-venv python3.12-dev \
        gcc g++ \
        libgl1 libglib2.0-0 \
        libsndfile1 libjpeg-dev libpng-dev \
        && \
    rm -rf /var/lib/apt/lists/* && \
    update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.12 1 && \
    python3 -V && python3 -m venv --help >/dev/null

# --- venv ---
ENV VENV=/opt/venv
ENV PATH="$VENV/bin:$PATH"
RUN python3 -m venv "$VENV" && \
    pip install --upgrade pip wheel setuptools

# --- pytorch (stable cu130). torch 2.7+ поддерживает sm_75 (Turing). ---
# Если cu130 wheel не найдётся для python 3.12 — fallback cu128 (тоже стабильный).
RUN pip install \
        torch torchvision torchaudio \
        --index-url https://download.pytorch.org/whl/cu130 \
    || pip install \
        torch torchvision torchaudio \
        --index-url https://download.pytorch.org/whl/cu128

# --- ComfyUI master ---
# ComfyUI pinned to a specific commit (2026-10-18) for reproducible builds.
# Update by `git pull` in /opt/ComfyUI after `docker compose up -d`.
ARG COMFYUI_REF=5cc4b87
WORKDIR /opt
# retry на early EOF от github: clone падает если сервер дропает TCP посреди pack transfer.
# http.version=HTTP/1.1 (не HTTP/2) + увеличенный postBuffer + retry-loop.
RUN for i in 1 2 3 4 5; do \
      git -c http.version=HTTP/1.1 -c http.postBuffer=524288000 \
          clone --depth 1 --single-branch --branch ${COMFYUI_REF} \
          https://github.com/comfyanonymous/ComfyUI.git /opt/ComfyUI \
          && break; \
      echo "git clone failed (attempt $i), retrying in 10s..."; \
      rm -rf /opt/ComfyUI; sleep 10; \
    done && \
    cd /opt/ComfyUI && \
    pip install -r requirements.txt

# --- ComfyUI-Manager (он же "comfy kitchen") в custom_nodes ---
# ComfyUI-Manager — main branch at build time.
# Pin to a specific commit hash for fully reproducible builds.
ARG COMFY_MANAGER_REF=main
WORKDIR /opt/ComfyUI/custom_nodes
RUN git clone --depth 1 --branch ${COMFY_MANAGER_REF} https://github.com/ltdrdata/ComfyUI-Manager.git && \
    pip install -r ComfyUI-Manager/requirements.txt

# --- Triton (нужен для современных attention kernels, не оптимизация, а зависимость) ---
# Ставим triton pin к версии, совместимой с pytorch. Берем просто stable.
RUN pip install triton

# --- runtime defaults ---
ENV NVIDIA_VISIBLE_DEVICES=0 \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility \
    PYTHONPATH=/opt/ComfyUI:/opt/ComfyUI/custom_nodes/ComfyUI-Manager \
    COMFYUI_ARGS="--listen 0.0.0.0 --port 8188 --disable-auto-launch"

# --- Entrypoint (auto-installs sage + pytest on first use) ---
# This is the script that runs inside the container at startup. It is
# baked into the image at /usr/local/bin/comfyui-entrypoint.sh, so a
# fresh `docker compose up -d` works without any host-side setup.
COPY docker/entrypoint.sh /usr/local/bin/comfyui-entrypoint.sh
RUN chmod +x /usr/local/bin/comfyui-entrypoint.sh

WORKDIR /opt/ComfyUI

# Healthcheck через /system_stats (ComfyUI API). Не требует модели.
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=5 \
    CMD python3 -c "import urllib.request,sys; r=urllib.request.urlopen('http://127.0.0.1:8188/system_stats', timeout=5); sys.exit(0 if r.status==200 else 1)" \
    || exit 1

# ENTRYPOINT is the baked script. It auto-installs sage if missing, then
# execs python3 main.py with $COMFYUI_ARGS (default flags from env, or
# overridden via COMFYUI_ARGS=... docker compose up -d).
ENTRYPOINT ["/usr/local/bin/comfyui-entrypoint.sh"]
