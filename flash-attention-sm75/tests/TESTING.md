# Testing

Production test suite for the three attention backends on sm_75.

## Quick smoke (all 3 backends importable + flags valid)

```bash
# Inside the running container
docker exec comfyui python3 -m pytest /opt/comfyui-git/flash-attention-sm75/tests/test_backends.py -v
```

Expected: 4 tests pass, no warnings.

## Full matrix (18 cells: 3 backends × 2 models × 4 quants)

Each cell runs an actual ComfyUI workflow end-to-end with a real
diffusion model. Total time ~24 minutes for the focused matrix
(Qwen bf16/int8_convrot + H3 bf16/fp8/int8_convrot/w6a8 = 6 model×quant
combinations × 3 backends = 18 cells).

```bash
cd /opt/comfyui-git
python3 flash-attention-sm75/tests/matrix/run_matrix.py
```

The script:
1. For each (model, quant, backend) combination:
   - Stops the container, restarts with the backend flag
   - For flash backend: re-applies v0.3 setup (sm_75 real build)
   - Submits the workflow via `/prompt` API
   - Waits for completion (or 30 min timeout)
   - Records time + output file size to `matrix_results.jsonl`
2. Results are written incrementally so partial failures are kept.

### Run only specific tests

```bash
# Only Qwen + sage
python3 run_matrix.py --only 'qwen_.*_sage'

# Only H3 + ck (fastest subset)
python3 run_matrix.py --only 'h3_.*_ck'

# All flash (v0.3) tests
python3 run_matrix.py --only '.*_flash' --flash-mode v0.3
```

## Manual verification

For visual confirmation, compare the generated outputs to the
archived artefacts in `tests/results/`:

```bash
# Qwen bf16 + ck should match tests/results/matrix_2026-10-18/qwen_qwen_bf16_ck_00001.png
# H3 int8_convrot + sage should match tests/results/sage_2026-10-18/h3_h3_int8_convrot_sage_00001_.mp4
```

A correct output:
- Qwen PNG: 2.2-2.5 MB, recognizable woman in polka-dot outfit
- H3 MP4: 480-560 KB, recognizable rooftop scene with characters

A broken output (precision trap):
- Qwen PNG: 2.4 MB but pattern noise, no recognizable content
- H3 MP4: 3.8 MB (v0.2.1 garbage) or 12.9 KB (workflow error)

## What the matrix does NOT cover

- **Real-time perf under load** (multiple workflows in parallel).
  Run your actual workflow and compare.
- **VRAM usage profiling**. Use `nvidia-smi` while workflows run.
- **Long-context correctness** (>4096 tokens). Sage triton kernel
  was tested up to 4096 seq length in isolation; longer sequences
  may hit different code paths.
- **Other models** (Wan, LTX, etc.). The matrix covers Qwen-Image 2.1
  and Minimax H3. Other diffusion models on sm_75 should also work
  with sage, but verify before deploying.

## Adding a new model/quant to the matrix

1. Edit `tests/matrix/run_matrix.py`:
   ```python
   H3_MODELS = {
       "bf16": "...",
       "new_quant": "minimax_h3_fl2va_pruned_NEWQUANT.safetensors",  # add
   }
   ```
2. Make sure the model file exists in `/opt/comfyui/models/diffusion_models/`.
3. Run: `python3 run_matrix.py --only '.*_newquant_.*'`

## CI integration (recommended)

```yaml
# .github/workflows/test.yml
- name: Run smoke tests
  run: |
    docker exec comfyui python3 -m pytest /opt/comfyui-git/flash-attention-sm75/tests/test_backends.py -v
```
