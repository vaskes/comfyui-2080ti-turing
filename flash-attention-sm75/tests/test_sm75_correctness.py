"""Smoke test: does the patched flash-attention produce correct output
on sm_75? Run inside the comfyui container after applying the patch
and installing the package.

Usage (inside the comfyui container):
    python3 /opt/comfyui-git/flash-attention-sm75/tests/test_sm75_correctness.py

Pass criterion: all 6 test cases produce max_abs_diff below threshold
(0.05 for T1-T5 easy mode, 0.10 for T6 stress mode) against
torch.nn.functional.scaled_dot_product_attention (math backend
on sm_75 — pytorch's only option for fp16 attention on Turing).

Note: T6 (L=4096, scale=1.0) is the regime where the fp32
accumulator design choice in the m16n8k8 mma actually matters.
T1-T5 (scale=0.1) are smoke tests that confirm the kernel
runs on sm_75 and the API surface is correct; they don't
exercise precision because Q@K^T scores are too small to
trigger fp16-accumulator issues.
"""
import sys
import torch
import torch.nn.functional as F

# Import the patched flash_attn (must be installed in the venv)
try:
    from flash_attn import flash_attn_func
except ImportError as e:
    print(f"FATAL: flash_attn not installed in this Python env: {e}")
    print("Run scripts/apply-patch-and-install.sh first.")
    sys.exit(1)


def test_case(B, H, L, D, causal=False, label="", scale=0.1):
    """Run one correctness test case.

    Args:
        scale: multiplier for the randn input. Default 0.1 keeps
            Q@K^T scores in [-0.05, 0.05] (easy mode — softmax
            ~uniform, precision barely matters). For T6 stress we
            use scale=1.0, which gives scores in [-5, 5] (the regime
            where fp32-accumulator design actually matters).
    """
    torch.manual_seed(42)
    q = torch.randn(B, H, L, D, dtype=torch.float16, device="cuda") * scale
    k = torch.randn(B, H, L, D, dtype=torch.float16, device="cuda") * scale
    v = torch.randn(B, H, L, D, dtype=torch.float16, device="cuda") * scale

    # Reference: pytorch SDPA. On sm_75 the math backend is the
    # only SDPA backend that works for fp16/bf16 attention (no
    # flash, no mem-efficient kernel for attention on Turing
    # because sm_75 lacks the bf16 tensor-core mma for attention).
    # pytorch 2.14 does not expose the backend= kwarg on
    # scaled_dot_product_attention (added in 2.5+ but only via the
    # torch.nn.attention module); we rely on pytorch's default
    # backend selection, which is math on sm_75.
    out_ref = F.scaled_dot_product_attention(
        q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2),
        is_causal=causal,
    ).transpose(1, 2)

    # Test: flash_attn
    out_flash = flash_attn_func(q, k, v, causal=causal)

    diff = (out_flash.float() - out_ref.float()).abs()
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()
    # Pass criterion: max_diff < 0.05 for the easy mode (T1-T5);
    # max_diff < 0.10 for the stress mode (T6) where Q@K^T scores
    # are in [-5, 5] and precision differences are amplified. The
    # math-backend SDPA reference also has its own fp32 limitations
    # at this scale, so a tighter bar is not meaningful.
    threshold = 0.10 if scale > 0.5 else 0.05
    status = "PASS" if max_diff < threshold else "FAIL"
    print(
        f"  [{status}] {label}: B={B} H={H} L={L} D={D} causal={causal} "
        f"scale={scale} max_diff={max_diff:.4f} mean_diff={mean_diff:.4f}"
    )
    return max_diff < threshold


def main():
    print(f"torch={torch.__version__} cuda={torch.version.cuda} dev0={torch.cuda.get_device_capability(0)}")
    print()
    print("=== flash-attention sm_75 correctness tests ===")
    results = []
    results.append(test_case(1, 4, 128, 64, label="T1: B=1 H=4 L=128 D=64"))
    results.append(test_case(1, 4, 128, 128, label="T2: B=1 H=4 L=128 D=128"))
    results.append(test_case(2, 8, 256, 128, label="T3: B=2 H=8 L=256 D=128"))
    results.append(test_case(1, 4, 128, 128, causal=True, label="T4: causal D=128"))
    results.append(test_case(1, 4, 128, 64, causal=True, label="T5: causal D=64"))
    # T6: stress test — no easy-mode scaling, long sequence (matches
    # Qwen-Image 2.1 attention's actual shape). This is the regime
    # where the fp32 accumulator design choice actually matters.
    results.append(test_case(
        1, 8, 4096, 128, label="T6: stress B=1 H=8 L=4096 D=128 (Qwen-like)",
        scale=1.0,
    ))
    print()
    if all(results):
        print("ALL PASS")
        sys.exit(0)
    else:
        print(f"FAIL: {sum(1 for r in results if not r)}/6 failed")
        sys.exit(1)


if __name__ == "__main__":
    main()
