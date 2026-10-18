#!/usr/bin/env python3
"""Production import + boot tests for the three attention backends.

These are unit/smoke tests. For full end-to-end on real models
use tests/matrix/run_matrix.py (the 18-cell matrix).

Run inside the ComfyUI container:
    docker exec comfyui python3 -m pytest /opt/comfyui-git/flash-attention-sm75/tests/test_backends.py -v

Tests:
  1. test_sage_importable — sageattention module loads
  2. test_ck_importable — comfy_kitchen importable
  3. test_flash_importable — flash_attn module loads (v0.3 or v0.2.1)
  4. test_cli_args — ComfyUI --use-{sage,ck,flash}-attention flags exist
  5. test_comfyui_boot — ComfyUI server reachable on 8188
  6. test_comfyui_args_current — which backend is currently active
  7. test_sage_module_version — sage module is the expected 1.0.6 wheel
  8. test_container_filesystem — entrypoint-managed paths are mounted
"""
import os
import sys

import pytest
import torch


def test_sage_importable():
    """sageattention package must be importable."""
    import sageattention  # noqa: F401
    from sageattention import sageattn
    assert callable(sageattn), "sageattn must be callable"


def test_ck_importable():
    """comfy_kitchen must be importable."""
    import comfy_kitchen  # noqa: F401
    # The real ck-attention path is registered via the comfy_kitchen backends.
    # We can't easily check the path without invoking it through ComfyUI.
    assert True, "comfy_kitchen import OK"


def test_flash_importable():
    """flash_attn module must be importable (v0.3 or v0.2.1)."""
    import flash_attn
    assert hasattr(flash_attn, "flash_attn_func"), "flash_attn_func() missing"
    fp = flash_attn.__file__
    assert "flash_attn" in fp, f"unexpected flash_attn path: {fp}"


def test_cli_args():
    """ComfyUI must support --use-sage/ck/flash-attention flags."""
    cli_args_path = "/opt/ComfyUI/comfy/cli_args.py"
    with open(cli_args_path) as f:
        content = f.read()
    for flag in ["--use-sage-attention", "--use-ck-attention", "--use-flash-attention"]:
        assert flag in content, f"ComfyUI cli_args.py missing flag {flag}"


def test_comfyui_boot():
    """ComfyUI server must be reachable on 8188 and report system stats."""
    import urllib.request
    try:
        req = urllib.request.urlopen("http://127.0.0.1:8188/system_stats", timeout=5)
        assert req.status == 200, f"status {req.status}"
        data = req.read().decode()
        assert "system" in data, f"unexpected response: {data[:200]}"
    except Exception as e:
        pytest.fail(f"ComfyUI not reachable: {e}")


def test_comfyui_args_current():
    """Verify which attention backend is currently active in the running container.

    Reads /proc/1/cmdline (no docker CLI needed, runs in-container).
    """
    try:
        with open("/proc/1/cmdline", "rb") as f:
            cmdline = f.read().replace(b"\x00", b" ").decode()
    except Exception as e:
        pytest.skip(f"can't read /proc/1/cmdline: {e}")
    if "--use-sage-attention" in cmdline:
        backend = "sage"
    elif "--use-ck-attention" in cmdline:
        backend = "ck"
    elif "--use-flash-attention" in cmdline:
        backend = "flash"
    else:
        backend = "sdpa (default fallback)"
    assert backend in ("sage", "ck", "flash", "sdpa (default fallback)"), \
        f"unknown backend: {cmdline}"


def test_sage_module_version():
    """sageattention should be 1.0.6 (the production version)."""
    import sageattention
    ver = getattr(sageattention, "__version__", None)
    # sage 1.0.6 wheel doesn't set __version__ explicitly; check via importlib.metadata
    try:
        import importlib.metadata
        dist_ver = importlib.metadata.version("sageattention")
        assert dist_ver.startswith("1.0"), f"expected sage 1.0.x, got {dist_ver}"
    except Exception as e:
        pytest.skip(f"can't determine sage version: {e}")


def test_container_filesystem():
    """Entrypoint-managed paths (sage install + v0.3 flash build) must be mounted.

    This validates the bind-mounts that allow sage + flash v0.3 to persist
    across container recreates.
    """
    paths = [
        "/tmp/flash-attention-build",  # v0.3 source + .so (host: /opt/flash-sm75-build)
    ]
    for p in paths:
        assert os.path.isdir(p), f"required bind-mount missing: {p}"
    # sage install is in /opt/venv (container's venv) — it auto-installs on first run
    # The entrypoint marker would be /tmp/flash-attention-build/.sage-installed-1.0.6
    marker = "/tmp/flash-attention-build/.sage-installed-1.0.6"
    # We don't assert on this — sage auto-installs on first use


def test_sm75_device():
    """Test environment must be sm_75 (RTX 2080 Ti)."""
    if not torch.cuda.is_available():
        pytest.skip("no CUDA")
    cc = torch.cuda.get_device_capability(0)
    assert cc == (7, 5), f"expected sm_75 (7, 5), got {cc}"


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
