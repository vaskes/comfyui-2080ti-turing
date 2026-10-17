#!/bin/bash
# Run the H3 video workflow via ComfyUI HTTP API and time it.
# Usage: bash run_h3.sh <label>
set -e
LABEL="${1:-h3-run}"

# 1. Create unique output dir for this run
TS=$(date +%Y%m%d_%H%M%S)
OUTDIR="/opt/comfyui/output/h3_${LABEL}_${TS}"
mkdir -p "$OUTDIR"
echo "[h3] output dir: $OUTDIR"

# 2. Get current ComfyUI args (so we know what attention is active)
echo "[h3] current COMFYUI_ARGS:"
docker exec comfyui bash -c 'env | grep COMFYUI_ARGS' | sed 's/^/  /'
echo

# 3. Submit workflow via API
echo "[h3] submitting H3 workflow to /prompt..."
RESP=$(curl -m 30 -s -X POST http://127.0.0.1:8188/prompt \
    -H "Content-Type: application/json" \
    -d "$(python3 -c '
import json
wf = json.load(open("/opt/comfyui/video_minimax_h3_t2v_api.json"))
# Inject output dir override
for nid, node in wf.items():
    ctype = node.get("class_type", "")
    if ctype == "SaveVideo":
        node["inputs"]["filename_prefix"] = f"h3_'"$LABEL"'_'"$TS"'"
        break
body = {"prompt": wf, "client_id": "h3-runner"}
print(json.dumps(body))
')")
PROMPT_ID=$(echo "$RESP" | python3 -c 'import json,sys; d=json.load(sys.stdin); print(d.get("prompt_id",""))')
ERR=$(echo "$RESP" | python3 -c 'import json,sys; d=json.load(sys.stdin); print(d.get("error","none"))')
echo "[h3] prompt_id: $PROMPT_ID"
echo "[h3] error: $ERR"
if [ -z "$PROMPT_ID" ] || [ "$ERR" != "none" ]; then
    echo "ERROR submitting prompt. Full response:"
    echo "$RESP" | python3 -m json.tool 2>&1 | head -30
    exit 1
fi

# 4. Wait for completion (poll /history)
START=$(date +%s)
echo "[h3] waiting for completion (timeout 30 min)..."
while true; do
    ELAPSED=$(($(date +%s) - START))
    if [ $ELAPSED -gt 1800 ]; then
        echo "TIMEOUT after 30 min"
        exit 1
    fi
    STATUS=$(curl -m 5 -s "http://127.0.0.1:8188/history/$PROMPT_ID" | python3 -c '
import json,sys
try:
    d=json.load(sys.stdin)
    p=d.get("'"$PROMPT_ID"'",{})
    s=p.get("status",{})
    if s.get("completed", False): print("done")
    elif s.get("error"): print("error")
    else: print("running")
except: print("running")
')
    if [ "$STATUS" = "done" ] || [ "$STATUS" = "error" ]; then
        echo "[h3] status: $STATUS after $ELAPSED s"
        break
    fi
    sleep 5
done
END=$(date +%s)
TOTAL=$((END - START))
echo "[h3] total wall time: ${TOTAL}s"

# 5. Save result summary
RESULT_FILE="$OUTDIR/RESULT.txt"
{
    echo "label: $LABEL"
    echo "started: $(date -d @$START)"
    echo "ended: $(date -d @$END)"
    echo "total_seconds: $TOTAL"
    echo "prompt_id: $PROMPT_ID"
    echo
    echo "=== attention ==="
    docker exec comfyui bash -c 'env | grep COMFYUI_ARGS'
    echo
    echo "=== outputs ==="
    ls -la "$OUTDIR"
    echo
    echo "=== full history (last status) ==="
    curl -m 5 -s "http://127.0.0.1:8188/history/$PROMPT_ID" | python3 -m json.tool 2>&1 | head -50
} > "$RESULT_FILE"

echo
echo "=== RESULT saved to $RESULT_FILE ==="
cat "$RESULT_FILE"
