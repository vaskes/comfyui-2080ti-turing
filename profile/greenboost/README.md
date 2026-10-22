# GreenBoost — Tier-2 VRAM overflow pool for RTX 2080 Ti

vaskes-specific. The main repo stays universal; everything here is tuned to
**this** box: 2x RTX 2080 Ti (sm_75, 22 GB each), 64 GB RAM, i7-12700K,
driver 615.71.09, CUDA UMD 13.4, kernel 7.0.0-38-generic.

Upstream: <https://gitlab.com/IsolatedOctopi/nvidia_greenboost> (Ferran Duarri).
Upstream defaults are written for an RTX 5070 (12 GB) in a 64 GB desktop —
**they are wrong for this machine on every axis**, which is why the install
script below passes explicit parameters for all of them.

---

## What it does

Three tiers, one virtual VRAM address space:

| Tier | Where | Speed | This box |
|---|---|---|---|
| T1 | GPU VRAM | ~500 GB/s | 22 GB |
| T2 | pinned system RAM, DMA-BUF export | ~11 GB/s | 40 GB |
| T3 | NVMe swap | ~1.8 GB/s | disabled |

A `LD_PRELOAD` shim inflates what `cuMemGetInfo` reports, so the application
packs more into VRAM than physically exists. Every `cudaMalloc` that does not
fit in real VRAM is redirected into T2 as pinned, DMA-BUF-exported memory the
GPU can reach directly over PCIe.

## Status: works, for weights only

**Works.** Path A (kernel module + DMA-BUF + pinned host registration) runs at
full PCIe 3.0 x16 speed, on bare metal and in Docker:

| Path | Throughput |
|---|---|
| VRAM → T2 (H2D) | **11.06 – 11.40 GB/s** |
| T2 → host (D2H) | 13.8 – 15.1 GB/s |
| T2 read from GPU side | **10.63 – 11.21 GB/s** |
| control: read from VRAM | ~490 – 505 GB/s |
| Path C (UVM) baseline, before the patch | 3.92 GB/s |

Flat across 2 / 4 / 8 / 20 GB. That is **2.9x over UVM**.

Capacity roughly doubles for the H3 32B model set:
```
before:  21 604 MB usable, full load: False  (model entirely offloaded)
after:   43 486 MB usable, full load: True   (model resident, hot part in VRAM)
```

**The rule that makes it usable: `--lowvram`.** Weights may live in T2;
activations must not. See `LIMITATIONS.md` for the A/B that proves it and the
config that gets it right.

With `--reserve-vram 6144` + `GREENBOOST_VRAM_HEADROOM_MB=2048`, H3 32B ran
**1 MP / 226 frames / 8 steps in 1:04:55**, steps flat at 453–459 s, 194 W,
`utilization.memory` 12.6%. Output validated: h264 + aac, 226/226 frames @ 24 fps,
9.417 s, 7.89 MB, no artefacts. That is a size that **OOMs natively** (the pure
VRAM ceiling on this box is 192 frames).

---

## Verdict: viable on Gen5, marginal on Gen3

The whole cost model is the PCIe link. Measured here: **11.4 GB/s = 71% of the
Gen3 x16 ceiling**, and the slot is physically Gen3 — no setting recovers more
than ~12%.

| bus | T2 speed | penalty vs native | what it is |
|---|---|---|---|
| Gen3 x16 | 11.4 GB/s | **x2.1** | emergency exit — runs what would OOM |
| Gen5 x16 | ~50 GB/s | **x1.26** | normal working mode |

Threshold is roughly **x1.5**. Below it the tool is a crash-harness, above it a
mode. On this box it is the first; on a 5080/5090 it is the second.

**Do not tune the pool expecting a different answer. Pool size is not the
constraint — bus bandwidth is.**

---

## Install

```bash
./install.sh
```

Builds `greenboost.ko` against the running kernel, applies the shim patch,
compiles the shim, and `insmod`s it. **Not persistent** — `insmod`, not `dkms`,
so it is gone after a reboot. Re-run `install.sh`.

Parameters this box needs (upstream defaults are an RTX 5070):

| Parameter | Default | Here | Why |
|---|---|---|---|
| `physical_vram_gb` | 12 | **22** | 2080 Ti, not a 5070 |
| `virtual_vram_gb` | 51 | **40** | 64 GB total less what the Win11 VM needs |
| `safety_reserve_gb` | 12 | **6** | |
| `nvme_swap_gb` | 64 | **0** | T3 is a footgun, see LIMITATIONS.md |
| `pcores_only` | 1 | 1 | i7-12700K P-cores are CPU 0-15, matches upstream |
| `use_hugepages` | 1 | **0** | no hugepages reserved on this host |

Pool state: `cat /sys/class/greenboost/greenboost/pool_info`

To remove: `./uninstall.sh`

---

## The patch

`patches/0001-shim-uva-and-ctx.patch` — 59 lines, two fixes. Both are real
bugs that hit anyone who `LD_PRELOAD`s this shim, not just us.

1. **`cuMemGetInfo_v2` returns `INVALID_CONTEXT` before any context exists.**
   `libcudart` initialises the primary context and *then* calls
   `cuMemGetInfo_v2`. Our interposition routes that internal call straight
   into `libcuda`, bypassing libcudart's init. ComfyUI calls this at import
   time and **dies before the server ever starts**. Fix: retain the primary
   context and retry.

2. **`cuMemHostGetDevicePointer` returns `INVALID_CONTEXT` on driver 615.71.09**
   for every size (64 MB – 20 GB), every flag (`DEVICEMAP` / `0` / `PORTABLE`),
   with a valid context. This made Path A and Path B unreachable — the shim
   threw a good buffer away and dropped to UVM. On UVA systems a
   `DEVICEMAP`-registered region is mapped at the same virtual address, so the
   host pointer **is** the device pointer. Fix: use it.

Measured, not assumed: a standalone binary calling the identical sequence gets
`devptr == hostptr` at every size and flag on this driver.

---

## Docker

A bind-mount of the device node is **not enough** — the device cgroup still
denies it (`Operation not permitted`). Use the `devices:` key:

```yaml
devices:
  - /dev/greenboost:/dev/greenboost
volumes:
  - /sys/class/greenboost:/sys/class/greenboost:ro
```

Also required in the container: `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:False`.
PyTorch's expandable segments use the VMM API (`cuMemCreate` / `cuMemMap` /
`cuMemAddressReserve`), which the shim does **not** intercept — with it on, the
shim inflates the reported total and the driver still OOMs.

`compose-greenboost.yaml` is the test instance (`comfyui2080_test`, port 8190,
GPU 0 = the x16 slot; GPU 1 is only x4 and would be worse than UVM).

---

## Probes

Standalone C, build against the host:

```bash
gcc -O2 -o /tmp/probe probe/probe.c -lcuda
/tmp/probe 512        # bandwidth by size
GB_NOBW=1 GB_COLD=1 GB_PIN=1 /tmp/probe 20480   # register only, no VRAM scratch
```

`probe.c` measures Tier1 vs Tier2 read bandwidth. `probe_shim.c` forces the
shim's overflow path and asks the kernel module where the buffer actually
landed.

**Both probes live in a normal executable on purpose.** The same CUDA call
sequence returns 201 from inside the shim and 0 from the main program — see
LIMITATIONS.md.