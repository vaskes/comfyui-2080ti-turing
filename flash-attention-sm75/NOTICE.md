# Third-Party Notices

This directory (`flash-attention-sm75/`) contains a **patch** to
[`Dao-AILab/flash-attention`](https://github.com/Dao-AILab/flash-attention)
that re-enables support for NVIDIA Turing (sm_75) GPUs in the build
matrix. The patch is 5 lines in `setup.py`; the rest of the source code
is upstream.

## Upstream: flash-attention

- **Repository:** https://github.com/Dao-AILab/flash-attention
- **License:** BSD 3-Clause ("New" or "Revised" License)
- **Copyright:** © 2023, Tri Dao.
- **Upstream LICENSE file preserved verbatim in this directory as `LICENSE`.**

### BSD 3-Clause License summary

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice,
   this list of conditions and the following disclaimer.
2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.
3. Neither the name of the copyright holder nor the names of its
   contributors may be used to endorse or promote products derived from
   this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF
THE POSSIBILITY OF SUCH DAMAGE.

The full license text is in `LICENSE` in this directory.

### Upstream contributors

The upstream `AUTHORS` file lists contributors to flash-attention. As of
the commit this patch was generated against (`965c921`):

- Tri Dao (primary author and maintainer)
- Daniel Y. Fu (early contributor, FA paper co-author)
- Stefano Ermon (advisor, FA paper co-author)
- Atri Rudra (advisor, FA paper co-author)
- Christopher Ré (advisor, FA paper co-author)

Plus many community contributors acknowledged in the upstream `AUTHORS`
file. See https://github.com/Dao-AILab/flash-attention/blob/main/AUTHORS
for the full list.

## What this patch does (and does not)

This patch re-enables the sm_75 (Turing) compute capability in the
build matrix of flash-attention. The patch:

- **Adds 5 lines to `setup.py`** to include `sm_75` in the list of
  supported architectures when the user requests it via the
  `FLASH_ATTN_CUDA_ARCHS` environment variable.
- **Does not modify** any CUDA source code in `csrc/`. The sm_75 code
  paths in `csrc/flash_attn/src/kernel_traits.h` (CUTLASS
  `SM75_16x8x8_F32F16F16F32_TN` mma atom, `SM75_U32x4_LDSM_N` and
  `SM75_U16x8_LDSM_T` ldmatrix wrappers) are upstream code, unchanged.

The kernel code is governed by the upstream BSD-3-Clause license. The
patch itself (5 lines) is too small to have a meaningful copyright
status, but is provided under the same BSD-3-Clause license for
consistency with the file it modifies.

## How to obtain the upstream source

```bash
git clone https://github.com/Dao-AILab/flash-attention.git
cd flash-attention
git checkout 965c921   # commit this patch was generated against
```

Or use the `apply-patch-and-install.sh` script in the `scripts/`
subdirectory of this patch.
