# flash-attention-sm75 unit test results

**Date:** 2026-10-17 (initial 5/5), 2026-10-18 (added T6 stress → 6/6)
**Torch:** 2.14.1+cu130
**CUDA:** 13.0
**GPU:** NVIDIA RTX 2080 Ti (sm_75, Turing), 22 GB VRAM
**Workstation host:** 85.136.112.144:12008
**flash_attn version:** 2.8.4 (Dao-AILab/flash-attention@965c921 + 5-line patch)

## Results: 6/6 PASS

```
  [PASS] T1: B=1 H=4 L=128 D=64: causal=False scale=0.1 max_diff=0.0000 mean_diff=0.0000
  [PASS] T2: B=1 H=4 L=128 D=128: causal=False scale=0.1 max_diff=0.0000 mean_diff=0.0000
  [PASS] T3: B=2 H=8 L=256 D=128: causal=False scale=0.1 max_diff=0.0001 mean_diff=0.0000
  [PASS] T4: causal D=128: B=1 H=4 L=128 D=128 causal=True scale=0.1 max_diff=0.0000 mean_diff=0.0000
  [PASS] T5: causal D=64: B=1 H=4 L=128 D=64 causal=True scale=0.1 max_diff=0.0000 mean_diff=0.0000
  [PASS] T6: stress B=1 H=8 L=4096 D=128 (Qwen-like): causal=False scale=1.0 max_diff=0.0010 mean_diff=0.0000

ALL PASS
```

## What this means

**T1–T5 (easy mode, scale=0.1):** max_diff ≤ 0.0001. These are
sanity checks that confirm the kernel runs on sm_75 and the API
surface is correct. With scale=0.1, Q@K^T scores are in
`[-0.05, 0.05]`, softmax is near-uniform, and any reasonable
kernel produces essentially the same output.

**T6 (stress mode, scale=1.0, L=4096):** max_diff = 0.0010.
This is the regime where the fp32-accumulator design choice
actually matters. With scale=1.0 and D=128, Q@K^T scores are in
`[-5, 5]`, the dynamic range of softmax is meaningful, and
fp16-accumulator kernels (like the community `ssiu` and `KLM-corporation`
forks) lose precision. The official upstream's fp32-accumulator
m16n8k8 design holds to `max_diff = 0.0010` at this scale,
which is the design's intended precision floor.

The shape (B=1, H=8, L=4096, D=128) was chosen to match
Qwen-Image 2.1's attention pattern at peak sequence length.

## How to reproduce

```
docker exec comfyui /opt/venv/bin/python3 \
    /opt/comfyui-git/flash-attention-sm75/tests/test_sm75_correctness.py
```

(Or copy the test to `/tmp/flash-sm75-tests/` inside the
container if the `/opt/comfyui-git/` path is not bind-mounted.)
