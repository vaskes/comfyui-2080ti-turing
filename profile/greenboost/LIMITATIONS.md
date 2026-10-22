# LIMITATIONS — measured, not theorised

Everything here was measured on this box (RTX 2080 Ti, driver 615.71.09,
CUDA UMD 13.4, kernel 7.0.0-38). Numbers, not expectations.

---

## 1. `--highvram` breaks it. `--lowvram` is mandatory.

**The rule: spill weights, never activations.**

Weights stream once per layer. Activations stream dozens of times per step.
T2 gives 11 GB/s; VRAM gives ~580 GB/s for the same data. That is a **52x
penalty on exactly the data that needs to be hot.**

The shim inflates what `cuMemGetInfo` reports, so ComfyUI believes it has
72.5 GB when it has 22 GB (`Initial CUDA memory: 72.12GB free / 72.50GB total`).
It cannot tell the difference, and `--highvram` is executed literally: it packs
whatever it thinks is VRAM until nothing is left, and the hot data is what gets
pushed out.

### A/B, H3 32B, 1 MP, 124 frames, 4 steps — one flag changed

| | B `--lowvram` | A `--highvram` |
|---|---|---|
| s/it | **172.89 / 175.15 / 176.25 / 176.98** | **zero steps in 60 minutes** |
| UNET | `loaded partially; 21604.54 MB offloaded` | `loaded completely; 21604.54 MB loaded` |
| `real_free` | 19 499 MB | **3 MB** |
| T2 available | has room | **0 MB** |
| `utilization.memory` | 12.6% avg | **1% avg** |
| power while working | 194 W | 123 W |

In A the card is filled to within 3 MB of empty, activations have nowhere real
to go, and the hot path drops onto PCIe. The GPU reports 100% utilisation the
whole time and draws 123 W — **it is computing against memory it cannot reach
at speed.** `utilization.gpu` lies here; `utilization.memory` does not.

No OOM, no exception, nothing in the log. It just quietly never finishes a step.

**`--highvram` and `--gpu-only` must not be used under any VRAM-inflating shim.**
If you see one in a profile, that is a landmine, not an optimisation.

### What works

`--reserve-vram 6144` + `GREENBOOST_VRAM_HEADROOM_MB=2048` + `--lowvram`:

```
--lowvram --reserve-vram 6144 --disable-dynamic-vram --disable-cuda-malloc
--disable-smart-memory --use-ck-attention
```

H3 32B, 1 MP, 226 frames, 8 steps → **1:04:55**, steps 453–459 s flat, 194 W.
h264 + aac, 226/226 frames @ 24 fps, 9.417 s, 7.89 MB, frames clean.
That length **OOMs without the pool** — pure VRAM caps at 192 frames.

`--lowvram` is not a precaution here. It loads CLIP into VRAM once, runs it,
and offloads it to CPU — that part is normal and correct. What it also does is
keep the UNET resident in T2, which is what leaves real VRAM for activations.

### Gotcha: `--reserve-vram` is in inflated units

`--reserve-vram 6144` means 6144 MB **of the 72.5 GB ComfyUI thinks it has**, not
6 GB of real VRAM. It does not control real memory at all. The only knob that
does is `GREENBOOST_VRAM_HEADROOM_MB` on the shim side.

---

## 1b. Viability depends on bus generation, not pool size

Measured H2D 11.4 GB/s = 71% of the Gen3 x16 ceiling. `lspci` reports
`LnkSta 2.5GT/s` at idle — that is ASPM power saving, not degradation: 11.4 GB/s
is physically above the Gen1 x16 ceiling, so the link trains to Gen3 under load.
The slot is physically Gen3.

Tier-2 overhead scales roughly as 1/bandwidth:

| bus | T2 | overhead vs native | verdict |
|---|---|---|---|
| Gen3 x16 | 11.4 GB/s | x2.1 | emergency exit |
| Gen5 x16 | ~50 GB/s | x1.26 | normal mode |

**Enlarging the pool on a slow link does not help.** If you take this to a 5080,
that is the configuration it was designed for.

---

## 2. Path C (UVM) took the host down

A 28 GB UVM allocation while the Win11 VM was running:

```
NVRM: GPU0 Out of memory [NV_ERR_NO_MEMORY] @ _memdescAllocInternal
DMAR: [DMA Write] Request device [01:00.0] fault ... PTE Write access is not set
dmar_fault: 416 callbacks suppressed
```

→ host reboot.

The IOMMU was behaving correctly — it refused DMA the driver attempted after it
had lost track of the memory. `intel_iommu` is on via BIOS with no kernel
parameter, which means `Default domain type: Translated` (strict), already the
strictest mode. `iommu=pt` would only silence the warnings, not fix the
underlying OOM.

**T3 is disabled in `install.sh` for the same reason: it is the tier that
reaches for swap under pressure.**

Path A does not do this — pages are explicitly allocated and accounted, and the
module releases everything on process exit. Verified: 35 554 MB / 33 live
DMA-BUF objects → 0 after teardown.

---

## 3. The 201 happens inside the `.so` only

`cuMemHostGetDevicePointer` returns `INVALID_CONTEXT`:

- from the shim, at **every** size (64 MB – 20 GB), every flag, top-level or
  nested inside the `cuMemAlloc_v2` hook
- from a normal executable on the same machine, same driver, same memory,
  same sequence: **SUCCESS**

Six variants were ruled out: buffer size, page residency (`MAP_POPULATE` +
`memset` vs cold), kernel pinning via the ioctl, fresh primary context,
`dlsym` pointer vs PLT call, and disabling the shim's `dlsym`/`dlopen`
interposition. A seventh — a byte-for-byte copy of the working code compiled
into the shim — also failed. Root cause not established.

The UVA fallback sidesteps it entirely, so this is documented rather than
worked around by understanding.

**Consequence for anyone debugging CUDA:** do not use code inside an
`LD_PRELOAD` library as a reference. A call that works from the main program
can fail from the library, and the reverse has not been disproven.

---

## 4. Other things that do not work

- **No PCIe byte counter on this driver.** `nvidia-smi --query-gpu=utilization.pcie`
  → `Field "utilization.pcie" is not a valid field to query`; `dmon` has no PCIe
  columns; `pmon -s t` returns nothing. GeForce on 615.71.09 simply does not
  expose them. Use `utilization.gpu` + `utilization.memory` + `power.draw` at 1 Hz
  instead. **`utilization.memory` is the metric that distinguishes "working" from
  "spinning"** — `utilization.gpu` reads 100% in both cases and lies.
- `cuImportExternalMemory` + `cuExternalMemoryGetMappedBuffer` from the DMA-BUF
  fd → `999 CUDA_ERROR_UNKNOWN`. The shim resolves both symbols (lines
  736-740) and **never calls them** — dead code in upstream.
- `expandable_segments:True` is incompatible: VMM API is not intercepted, so
  PyTorch allocates through it while the shim inflates the reported total, and
  the driver OOMs with tens of GB free.
- The shim is not persistent across reboot (`insmod`, not `dkms`).

---

## 5. Scope

Single GPU assumed. GPU 1 is a PCIe x4 slot — slower than UVM, so tests use
GPU 0 (x16). Two instances running GreenBoost against a shared pool is not
tested and the pool has no per-process accounting for it.