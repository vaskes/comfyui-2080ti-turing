#!/bin/bash
# GreenBoost install for THIS box: 2x RTX 2080 Ti (sm_75, 22 GB), 64 GB RAM,
# i7-12700K, driver 615.71.09. Upstream defaults target an RTX 5070 in a 64 GB
# desktop and are wrong here on physical_vram_gb, virtual_vram_gb and
# nvme_swap_gb -- so all of them are passed explicitly.
#
# insmod, not dkms: the module is gone after every reboot. Re-run this script.
set -euo pipefail

SRC="${GREENBOOST_SRC:-/opt/greenboost/src}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PATCH="$HERE/patches/0001-shim-uva-and-ctx.patch"

PHYS_GB="${PHYS_GB:-22}"          # RTX 2080 Ti, not the upstream 12 (RTX 5070)
VIRT_GB="${VIRT_GB:-40}"          # 64 GB total, minus what the Win11 VM needs
RESERVE_GB="${RESERVE_GB:-6}"
PCORES_MAX="${PCORES_MAX:-15}"    # i7-12700K: P-cores are CPU 0-15

say() { printf '\n\033[1;34m== %s\033[0m\n' "$*"; }

# ---------------------------------------------------------------- preflight
say "preflight"
[ "$(id -u)" -eq 0 ] || { echo "must run as root"; exit 1; }
[ -f "$SRC/greenboost.c" ] || { echo "GreenBoost source not at $SRC"; exit 1; }
[ -f "$SRC/greenboost_cuda.map" ] || { cp "$HERE/probe/greenboost_cuda.map" "$SRC/" 2>/dev/null || true; }
[ -f "$PATCH" ] || { echo "patch missing: $PATCH"; exit 1; }

KVER="$(uname -r)"
echo "kernel      : $KVER"
echo "vram        : $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1)"
echo "driver      : $(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1)"
echo "free RAM    : $(free -g | awk 'NR==2{print $7}') GB of $(free -g | awk 'NR==2{print $2}') GB"

if [ "$(free -g | awk 'NR==2{print $7}')" -lt 20 ]; then
    echo "WARNING: less than 20 GB free RAM. The Win11 VM eats ~15 GB."
    echo "         Stop it first, or lower VIRT_GB."
fi
echo "modules     : $(lsmod | grep -c greenboost) greenboost loaded"

# ---------------------------------------------------------------- kernel module
say "build kernel module"
cd "$SRC"
make clean >/dev/null 2>&1 || true
make module 2>&1 | tail -5
[ -f greenboost.ko ] || { echo "greenboost.ko not built"; exit 1; }
echo "vermagic: $(modinfo greenboost.ko | grep vermagic)"

# ---------------------------------------------------------------- shim patch
say "apply shim patch"
cp greenboost_cuda_shim.c "greenboost_cuda_shim.c.orig"
if patch -p1 --dry-run -i "$PATCH" >/dev/null 2>&1; then
    patch -p1 -i "$PATCH"
else
    echo "patch already applied or conflicting -- checking"
    grep -q "using UVA host pointer" greenboost_cuda_shim.c \
        && echo "already patched, continuing" \
        || { echo "patch does not apply and source is not patched"; exit 1; }
fi

say "build shim"
make shim 2>&1 | tail -3
[ -f libgreenboost_cuda.so ] || { echo "shim not built"; exit 1; }
cp libgreenboost_cuda.so /usr/local/lib/
echo "installed: /usr/local/lib/libgreenboost_cuda.so ($(stat -c%s libgreenboost_cuda.so) bytes)"

# ---------------------------------------------------------------- load
say "load module"
lsmod | grep -q '^greenboost' && { rmmod greenboost; echo "unloaded previous"; }
insmod greenboost.ko \
    physical_vram_gb="$PHYS_GB" \
    virtual_vram_gb="$VIRT_GB" \
    safety_reserve_gb="$RESERVE_GB" \
    nvme_swap_gb=0 \
    nvme_pool_gb=0 \
    pcores_only=1 \
    pcores_max_cpu="$PCORES_MAX" \
    use_hugepages=0
sleep 1

lsmod | grep greenboost || { echo "module did not load"; exit 1; }
[ -c /dev/greenboost ] || { echo "/dev/greenboost missing"; exit 1; }

say "pool state"
sed -n '1,12p' /sys/class/greenboost/greenboost/pool_info

cat <<EOF

Installed. Next:

  bare-metal bandwidth check:
    gcc -O2 -o /tmp/probe $HERE/probe/probe.c -lcuda && /tmp/probe 512

  shim overflow check (needs the test instance):
    cd $SRC && GREENBOOST_ACTIVE=1 GREENBOOST_DEBUG=1 \\
      GREENBOOST_VRAM_HEADROOM_MB=2048 \\
      LD_PRELOAD=/usr/local/lib/libgreenboost_cuda.so ./probe_shim 20480

  test ComfyUI instance (port 8190, GPU 0):
    cd $(dirname "$HERE") && docker compose -f compose-greenboost.yaml up -d

Read LIMITATIONS.md before pointing this at anything you care about.
EOF