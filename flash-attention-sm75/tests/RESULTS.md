=== flash-attention sm_75 unit test results (2026-10-18) ===

torch=2.14.1+cu130 cuda=13.0 dev0=(7, 5)

[PASS] T1: B=1 H=4 L=128 D=64       causal=False  max_diff=0.0000 mean_diff=0.0000
[PASS] T2: B=1 H=4 L=128 D=128      causal=False  max_diff=0.0000 mean_diff=0.0000
[PASS] T3: B=2 H=8 L=256 D=128      causal=False  max_diff=0.0001 mean_diff=0.0000
[PASS] T4: causal mask D=128         causal=True   max_diff=0.0000 mean_diff=0.0000
[PASS] T5: causal mask D=64          causal=True   max_diff=0.0000 mean_diff=0.0000

ALL PASS — 5/5 tests pass with max_diff < 0.05 (actual < 0.0001)
