#!/usr/bin/env python3
"""Matrix test runner for Qwen-Image 2.1 + H3 across attention backends.

Supports both v0.2.1 (shim) and v0.3 (real sm_75) flash builds via
--flash-mode flag. The v0.3 setup is re-applied after every container
recreation triggered by switch_attention().
"""
import argparse
import json
import os
import re
import subprocess
import time
from pathlib import Path

REPO = Path("/opt/comfyui-git/flash-attention-sm75/tests/matrix")
QWEN_WF_SRC = "/opt/comfyui/image_qwen_image_2_1_t2i_api.json"
H3_WF_SRC = "/opt/comfyui/video_minimax_h3_t2v_api.json"
COMFY_URL = "http://127.0.0.1:8188"
RESULT_LOG = REPO / "matrix_results.jsonl"

ATTN_BACKENDS = {
    "sdpa": "",
    "ck":   "--use-ck-attention",
    "flash": "--use-flash-attention",
    "sage": "--use-sage-attention",
}

QWEN_MODELS = {
    "bf16": "qwen_image_2.1/qwen_image_2.1_bf16.safetensors",
    "int8_convrot": "qwen_image_2.1/qwen_image_2.1_int8_convrot.safetensors",
}

H3_MODELS = {
    "bf16":         "minimax_h3_fl2va_pruned_bf16.safetensors",
    "fp8":          "minimax_h3_fl2va_pruned_fp8_scaled.safetensors",
    "int8_convrot": "minimax_h3_fl2va_pruned_int8_convrot.safetensors",
    "w6a8":         "minimax_h3_fl2va_pruned_w6a8.safetensors",
}

H3_CLIP = "qwen3vl_32b_h3_ultra_uncensored_heretic_int8_convrot.safetensors"
FLASH_MODE = "v0.3"  # set by main() via --flash-mode


def log(msg):
    t = time.strftime("%H:%M:%S")
    print("[" + t + "] " + str(msg), flush=True)


def run(cmd, timeout=180, check=True):
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=timeout)
    if check and r.returncode != 0:
        log("CMD FAIL: " + cmd)
        log("stdout: " + r.stdout)
        log("stderr: " + r.stderr)
        raise RuntimeError("Command failed: " + cmd)
    return r.returncode, r.stdout, r.stderr


def setup_v0_3_in_container():
    """Set up v0.3 (real sm_75) flash attention in the running container.
    Idempotent. Source + .so are bind-mounted at /tmp/flash-attention-build.
    """
    log("setup_v0_3: configuring real sm_75 build")
    script_lines = [
        "#!/bin/bash",
        "set -e",
        "SHIM=/opt/venv/lib/python3.12/site-packages/flash_attn",
        "SHIM_BAK=/opt/venv/lib/python3.12/site-packages/flash_attn_shim_backup",
        "PTH=/opt/venv/lib/python3.12/site-packages/zz_flash_attn_sm75.pth",
        "if [ -d \"$SHIM\" ] && [ ! -d \"$SHIM_BAK\" ]; then",
        "  mv \"$SHIM\" \"$SHIM_BAK\"",
        "fi",
        "echo /tmp/flash-attention-build > \"$PTH\"",
        "python3 -c 'import flash_attn; print(\"v0.3 active:\", flash_attn.__file__)'",
    ]
    script = "\n".join(script_lines) + "\n"
    host_script = "/tmp/setup-v0.3-in-container.sh"
    with open(host_script, "w") as f:
        f.write(script)
    run("docker cp " + host_script + " comfyui:/tmp/setup-v0.3.sh")
    run("docker exec comfyui bash /tmp/setup-v0.3.sh")

def setup_v0_2_in_container():
    """Restore v0.2.1 (shim)."""
    log("setup_v0_2: restoring shim")
    script_lines = [
        "#!/bin/bash",
        "set -e",
        "SHIM=/opt/venv/lib/python3.12/site-packages/flash_attn",
        "SHIM_BAK=/opt/venv/lib/python3.12/site-packages/flash_attn_shim_backup",
        "PTH=/opt/venv/lib/python3.12/site-packages/zz_flash_attn_sm75.pth",
        "rm -f \"$PTH\"",
        "if [ -d \"$SHIM_BAK\" ] && [ ! -d \"$SHIM\" ]; then",
        "  mv \"$SHIM_BAK\" \"$SHIM\"",
        "fi",
        "python3 -c 'import flash_attn; print(\"v0.2.1 active:\", flash_attn.__file__)'",
    ]
    script = "\n".join(script_lines) + "\n"
    host_script = "/tmp/setup-v0.2-in-container.sh"
    with open(host_script, "w") as f:
        f.write(script)
    run("docker cp " + host_script + " comfyui:/tmp/setup-v0.2.sh")
    run("docker exec comfyui bash /tmp/setup-v0.2.sh")

def switch_attention(backend_name):
    flag = ATTN_BACKENDS[backend_name]
    new_args = "--listen 0.0.0.0 --port 8188 --disable-auto-launch"
    if flag:
        new_args += " " + flag
    log("switching attn to " + backend_name + ": " + new_args)
    run("docker stop comfyui")
    env_arg = "COMFYUI_ARGS='" + new_args + "'"
    run(env_arg + " docker compose -f /opt/comfyui-git/compose.yaml up -d")
    for i in range(60):
        time.sleep(2)
        try:
            r = subprocess.run(["curl", "-m", "3", "-s", COMFY_URL + "/system_stats"],
                              capture_output=True, text=True, timeout=5)
            if r.returncode == 0 and "system" in r.stdout:
                log("ComfyUI ready after " + str((i+1)*2) + "s")
                return
        except Exception:
            pass
    raise RuntimeError("ComfyUI did not become ready")


def build_qwen_workflow(model_path, label):
    wf = json.load(open(QWEN_WF_SRC))
    for nid, node in wf.items():
        ctype = node.get("class_type", "")
        inputs = node.get("inputs", {})
        if ctype == "UNETLoader":
            inputs["unet_name"] = model_path
        if ctype == "SaveImageAdvanced":
            inputs["filename_prefix"] = "qwen_" + label
    return wf


def build_h3_workflow(model_path, clip_path, label):
    wf = json.load(open(H3_WF_SRC))
    for nid, node in wf.items():
        ctype = node.get("class_type", "")
        inputs = node.get("inputs", {})
        if ctype == "UNETLoader":
            inputs["unet_name"] = model_path
        if ctype == "CLIPLoader" and inputs.get("type") == "minimax":
            inputs["clip_name"] = clip_path
        if ctype == "SaveVideo":
            inputs["filename_prefix"] = "h3_" + label
    return wf


def submit_and_wait(workflow, timeout_sec=1800):
    body = {"prompt": workflow, "client_id": "matrix-runner"}
    r = subprocess.run(["curl", "-m", "30", "-s", "-X", "POST", COMFY_URL + "/prompt",
                       "-H", "Content-Type: application/json", "-d", json.dumps(body)],
                      capture_output=True, text=True, timeout=60)
    if r.returncode != 0:
        return False, 0, "curl rc=" + str(r.returncode) + ": " + r.stderr
    try:
        resp = json.loads(r.stdout)
    except Exception as e:
        return False, 0, "json parse: " + str(e) + " stdout=" + r.stdout[:500]
    if "error" in resp and resp["error"]:
        return False, 0, "submit error: " + str(resp["error"])
    prompt_id = resp.get("prompt_id", "")
    if not prompt_id:
        return False, 0, "no prompt_id: " + r.stdout[:500]
    start = time.time()
    while True:
        elapsed = time.time() - start
        if elapsed > timeout_sec:
            return False, elapsed, "TIMEOUT"
        time.sleep(5)
        try:
            r = subprocess.run(["curl", "-m", "5", "-s", COMFY_URL + "/history/" + prompt_id],
                              capture_output=True, text=True, timeout=10)
            data = json.loads(r.stdout)
            entry = data.get(prompt_id, {})
            status = entry.get("status", {})
            if status.get("completed"):
                total = time.time() - start
                if status.get("error"):
                    return False, total, "workflow error: " + str(status.get("error"))
                return True, total, ""
            if status.get("error"):
                return False, time.time() - start, "workflow error: " + str(status.get("error"))
        except Exception:
            pass


def find_outputs(workflow, output_dir="/opt/comfyui/output"):
    prefix = None
    for nid, node in workflow.items():
        ctype = node.get("class_type", "")
        if ctype in ("SaveImage", "SaveImageAdvanced", "SaveVideo"):
            prefix = node.get("inputs", {}).get("filename_prefix", "")
            break
    if not prefix:
        return []
    files = []
    for p in Path(output_dir).iterdir():
        if p.name.startswith(prefix):
            files.append((p, p.stat().st_mtime))
    files.sort(key=lambda x: x[1], reverse=True)
    return [str(p) for p, _ in files[:3]]


def run_test(model_kind, quant, attn):
    label = model_kind + "_" + quant + "_" + attn
    log("====== TEST START: " + label + " ======")
    switch_attention(attn)
    if attn == "flash":
        if FLASH_MODE == "v0.3":
            setup_v0_3_in_container()
        else:
            setup_v0_2_in_container()
    if model_kind == "qwen":
        wf = build_qwen_workflow(QWEN_MODELS[quant], label)
    elif model_kind == "h3":
        wf = build_h3_workflow(H3_MODELS[quant], H3_CLIP, label)
    else:
        raise ValueError("unknown model_kind " + model_kind)
    success, total, err = submit_and_wait(wf)
    outputs = find_outputs(wf) if success else []
    result = {
        "label": label,
        "model_kind": model_kind,
        "quant": quant,
        "attn": attn,
        "flash_mode": FLASH_MODE if attn == "flash" else None,
        "success": success,
        "total_seconds": round(total, 1),
        "error": err,
        "outputs": outputs,
        "output_sizes": [
            os.path.getsize(p) if os.path.isfile(p) else
            sum(f.stat().st_size for f in Path(p).iterdir() if f.is_file())
            for p in outputs
        ],
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    log("====== TEST END: " + label + " success=" + str(success) + " time=" + str(round(total, 1)) + "s err=" + err + " ======")
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", help="Run only matching tests (regex on label)")
    parser.add_argument("--flash-mode", default="v0.3", choices=["v0.2.1", "v0.3"],
                       help="Which flash build to use (default: v0.3)")
    args = parser.parse_args()
    global FLASH_MODE
    FLASH_MODE = args.flash_mode
    tests = []
    for quant in QWEN_MODELS:
        for attn in ATTN_BACKENDS:
            tests.append(("qwen", quant, attn))
    for quant in H3_MODELS:
        for attn in ATTN_BACKENDS:
            tests.append(("h3", quant, attn))
    if args.only:
        pattern = re.compile(args.only)
        tests = [(mk, q, a) for mk, q, a in tests if pattern.search(mk + "_" + q + "_" + a)]
    log("Running " + str(len(tests)) + " tests, flash_mode=" + FLASH_MODE)
    for mk, q, a in tests:
        result = run_test(mk, q, a)
        with open(RESULT_LOG, "a") as f:
            f.write(json.dumps(result) + "\n")
    log("All tests done. Results in matrix_results.jsonl")


if __name__ == "__main__":
    main()
