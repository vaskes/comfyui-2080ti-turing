# LIMITATIONS — measured, not theorised

Everything here was measured on this box (RTX 2080 Ti, driver 615.71.09,
CUDA UMD 13.4, kernel 7.0.0-38). Numbers, not expectations.

---

## 1. Offloading activations is not viable

**The rule: spill weights, never activations.**

Weights stream once per layer. Activations stream dozens of times per step.
T2 gives 11 GB/s; VRAM gives ~580 GB/s for the same data. That is a **52x
penalty on exactly the data that needs to be hot.**

Measured on H3 32B, 1 MP, 226 frames, `GREENBOOST_VRAM_HEADROOM_MB=2048`:

```
loaded completely; 43486.75 MB usable, 21604.54 MB loaded, full load: True
```

The model loads fully — capacity went from 21.6 GB usable / fully offloaded to
43.5 GB / fully resident. And then:

```
0%|          | 0/8 [00:00<?, ?it/s]     ← after 35 minutes
```

GPU pinned at 100% / 128 W the whole time, so it was computing, not hung. One
denoise step did not finish in 35 minutes. GPU memory held steady, so no leak.

Cause: with headroom at 2048 MB, activations spilled to T2 as well as weights.
VRAM was full of model weights (`full load: True`), leaving no room, so every
activation buffer went over PCIe.

**Next thing to try, not yet done:** raise `GREENBOOST_VRAM_HEADROOM_MB` so
ComfyUI's own `--reserve-vram` leaves headroom for activations, keeping hot data
in VRAM and only the cold weight layers in T2. That is the configuration that
would actually make this worth deploying — test it on a config that previously
OOM'd.

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