#!/usr/bin/env bash
# e20 headless smoke test — start the server on CPU, exercise EVERY endpoint with assertions
# and wall-clock timings, then shut down. Results land in outputs/smoke_results.txt.
set -Eeuo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/../../../.." && pwd)"
PY="${PY:-$ROOT/.venv-gcp/bin/python}"
PORT="${APP_PORT:-7893}"
OUT="$HERE/../outputs"
LOG="$OUT/smoke_server.log"
mkdir -p "$OUT"
export CUDA_VISIBLE_DEVICES=""

"$PY" "$HERE/app.py" "$ROOT/stage2_data/public_test.csv" \
  --model-dir "${MODEL_DIR:-$ROOT/asya/s2_exp/e15_student/outputs_frozen_v2/ckpt_gf2_berturk_6ep_bf16}" \
  --device cpu --host 127.0.0.1 --port "$PORT" > "$LOG" 2>&1 &
SRV=$!
trap 'kill "$SRV" 2>/dev/null || true; wait "$SRV" 2>/dev/null || true' EXIT

"$PY" "$HERE/smoke_client.py" "http://127.0.0.1:$PORT" | tee "$OUT/smoke_results.txt"
echo "server log: $LOG"
