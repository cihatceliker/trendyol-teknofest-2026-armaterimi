#!/usr/bin/env bash
# e20 dev server — run the champion explainability app locally on CPU.
# Usage: ./dev_server.sh [extra app.py args…]     (APP_PORT / DEVICE / MODEL_DIR env overrides)
set -Eeuo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/../../../.." && pwd)"                      # trendyol-teknofest-work
PY="${PY:-$ROOT/.venv-gcp/bin/python}"
[ -x "$PY" ] || PY=python3
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES-}"       # CPU by default; GPU is training

exec "$PY" "$HERE/app.py" \
  "$ROOT/stage2_data/public_test.csv" \
  --model-dir "${MODEL_DIR:-$ROOT/asya/s2_exp/e15_student/outputs_frozen_v2/ckpt_gf2_berturk_6ep_bf16}" \
  --device "${DEVICE:-cpu}" \
  --host 127.0.0.1 \
  --port "${APP_PORT:-7862}" \
  "$@"
