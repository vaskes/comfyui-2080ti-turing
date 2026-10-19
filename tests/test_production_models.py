#!/usr/bin/env python3
"""Production model manifest + integrity tests.

These are the actual models used in production for H3 32B heretic workflows
on RTX 2080 Ti (sm_75). The heretic CLIP is not the official Qwen3-VL release
(it is the uncensored variant), so byte-level verification prevents
accidental model swaps that would silently change output.

Run inside the ComfyUI container:
    docker exec comfyui python3 -m pytest /opt/comfyui-git/tests/test_production_models.py -v

Why this exists:
  - Production models are immutable per vaskes rule (changing models
    silently breaks pipeline output).
  - The heretic CLIP name has "heretic" in it for a reason — anyone
    running the matrix tests needs to know they're using the uncensored
    variant, not the official Qwen3-VL.

Production workflow used: /opt/comfyui/video_minimax_h3_t2v_10sec.json
  - Node 140:127 UNETLoader   -> unet_name
  - Node 140:128 CLIPLoader   -> clip_name  (HERETIC)
  - Node 140:119 VAELoader    -> vae_name (video)
  - Node 140:120 VAELoader    -> vae_name (audio)
  - Node 140:134 LoraLoaderModelOnly -> lora_name
"""
import os
import sys
import hashlib

import pytest


# Production model manifest. Bytes are EXACT and verified on 2026-10-19.
# Do NOT add models that are not actually used by the production workflow.
PRODUCTION_MODELS = {
    "diffusion_models/minimax_h3_fl2va_pruned_int8_convrot.safetensors": {
        "role": "UNET (main diffusion, H3 32B heretic int8_convrot)",
        "size_bytes": 20970379616,
        "size_human": "19,998.91 MB",
    },
    "text_encoders/qwen3vl_32b_h3_ultra_uncensored_heretic_int8_convrot.safetensors": {
        "role": "CLIP (text encoder, HERETIC uncensored variant, NOT official Qwen3-VL)",
        "size_bytes": 26363476151,
        "size_human": "25,142.17 MB",
    },
    "vae/minimax_h3_video_vae_fp16.safetensors": {
        "role": "VAE (video, fp16)",
        "size_bytes": 5207808496,
        "size_human": "4,966.55 MB",
    },
    "vae/minimax_h3_audio_vae_fp32.safetensors": {
        "role": "VAE (audio, fp32)",
        "size_bytes": 605254808,
        "size_human": "577.22 MB",
    },
    "loras/minimax_h3_fl2v_turbo_8step_v1.0_comfyui_bf16.safetensors": {
        "role": "LoRA (8-step turbo, bf16)",
        "size_bytes": 1956193000,
        "size_human": "1,865.36 MB",
    },
}

# Models root inside the container (bind-mounted from host /opt/comfyui/models)
MODELS_ROOT = "/opt/ComfyUI/models"


def _model_path(rel_path: str) -> str:
    return os.path.join(MODELS_ROOT, rel_path)


@pytest.mark.parametrize("rel_path,info", list(PRODUCTION_MODELS.items()))
def test_model_exists(rel_path, info):
    """Each production model must exist at the expected path."""
    full = _model_path(rel_path)
    assert os.path.isfile(full), f"production model missing: {full}"


@pytest.mark.parametrize("rel_path,info", list(PRODUCTION_MODELS.items()))
def test_model_size_exact(rel_path, info):
    """Each production model must be EXACTLY the expected size in bytes.

    This catches accidental model swaps (e.g. downloading a different quant,
    a different uncensored variant, or a corrupted download). The heretic
    CLIP especially — different heretic variants can produce different
    output, so size equality is a sanity check.
    """
    full = _model_path(rel_path)
    if not os.path.isfile(full):
        pytest.skip(f"model not present: {full}")
    actual = os.path.getsize(full)
    expected = info["size_bytes"]
    assert actual == expected, (
        f"production model size mismatch for {rel_path}:\n"
        f"  expected: {expected} bytes ({info['size_human']})\n"
        f"  actual:   {actual} bytes\n"
        f"  role:     {info['role']}\n"
        f"This usually means a different model variant was downloaded.\n"
        f"Check: ls -la {full}"
    )


def test_production_models_complete():
    """The manifest must be complete (5 models)."""
    assert len(PRODUCTION_MODELS) == 5, (
        f"expected 5 production models, got {len(PRODUCTION_MODELS)}"
    )
    # Sanity: total size ~52 GB
    total = sum(m["size_bytes"] for m in PRODUCTION_MODELS.values())
    assert 50 * 1024**3 < total < 60 * 1024**3, (
        f"total size {total} bytes not in expected 50-60 GB range"
    )


def test_heretic_clip_marked():
    """The heretic CLIP must be marked in its filename AND in the manifest role.

    The word 'heretic' identifies the uncensored variant. Anyone debugging
    the matrix tests needs to see this clearly.
    """
    heretic = [
        p for p, info in PRODUCTION_MODELS.items()
        if "heretic" in p.lower() and info["role"].startswith("CLIP")
    ]
    assert len(heretic) == 1, (
        f"expected exactly 1 heretic CLIP model, found {len(heretic)}: {heretic}"
    )
    # The role string must contain HERETIC (uppercase) to make the warning
    # visible in test output.
    assert "HERETIC" in PRODUCTION_MODELS[heretic[0]]["role"], (
        "heretic CLIP role must contain 'HERETIC' uppercase as a visual warning"
    )


def test_workflow_uses_production_models():
    """The /opt/comfyui/video_minimax_h3_t2v_10sec.json workflow must reference
    the same models listed in PRODUCTION_MODELS.

    This catches the case where the workflow gets modified to use a
    different model without updating this manifest.
    """
    # Workflow lives in /opt/ComfyUI/user/workflows (host: /opt/comfyui/user/workflows).
    # Try a few common locations.
    wf_path = None
    for cand in [
        "/opt/ComfyUI/user/workflows/video_minimax_h3_t2v_10sec.json",
        "/opt/ComfyUI/workflows/video_minimax_h3_t2v_10sec.json",
    ]:
        if os.path.isfile(cand):
            wf_path = cand
            break
    if wf_path is None:
        pytest.skip("video_minimax_h3_t2v_10sec.json not found in any workflow dir")
    if not os.path.isfile(wf_path):
        pytest.skip(f"workflow not present: {wf_path}")
    import json
    with open(wf_path) as f:
        wf = json.load(f)
    used = set()
    for nid, node in wf.items():
        if not isinstance(node, dict):
            continue
        for inp_val in (node.get("inputs") or {}).values():
            if isinstance(inp_val, str) and inp_val.endswith(".safetensors"):
                used.add(inp_val)
    manifest_basenames = {
        os.path.basename(p) for p in PRODUCTION_MODELS
    }
    missing_in_manifest = used - manifest_basenames
    missing_in_workflow = manifest_basenames - used
    assert not missing_in_manifest, (
        f"workflow uses models not in manifest: {missing_in_manifest}"
    )
    assert not missing_in_workflow, (
        f"manifest has models not used by workflow: {missing_in_workflow}"
    )


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
