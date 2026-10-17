"""Smoke test: does the patched flash-attention produce correct output
on sm_75? Run inside the comfyui container after applying the patch
and installing the package.

Usage (inside the comfyui container):
    python3 /opt/comfyui-git/flash-attention-sm75/tests/test_sm75_correctness.py

Pass criterion: all 5 test cases produce max_abs_diff < 0.05
against torch.nn.functional.scaled_dot_product_attention reference.
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


def test_case(B, H, L, D, causal=False, label=""):
    torch.manual_seed(42)
    q = torch.randn(B, H, L, D, dtype=torch.float16, device="cuda") * 0.1
    k = torch.randn(B, H, L, D, dtype=torch.float16, device="cuda") * 0.1
    v = torch.randn(B, H, L, D, dtype=torch.float16, device="cuda") * 0.1

    # Reference: pytorch SDPA
    out_ref = F.scaled_dot_product_attention(
        q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), is_causal=causal
    ).transpose(1, 2)

    # Test: flash_attn
    out_flash = flash_attn_func(q, k, v, causal=causal)

    diff = (out_flash.float() - out_ref.float()).abs()
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()
    status = "PASS" if max_diff < 0.05 else "FAIL"
    print(
        f"  [{status}] {label}: B={B} H={H} L={L} D={D} causal={causal} "
        f"max_diff={max_diff:.4f} mean_diff={mean_diff:.4f}"
    )
    return max_diff < 0.05


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
    print()
    if all(results):
        print("ALL PASS")
        sys.exit(0)
    else:
        print(f"FAIL: {sum(1 for r in results if not r)}/5 failed")
        sys.exit(1)


if __name__ == "__main__":
    main()
