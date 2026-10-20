#!/usr/bin/env bash
# Asure-12B zero-shot hardness scores for the CURRENT selected_v5 (gate e). The output name is
# bound to the selection's sha256 so a regenerated selection can never inherit positional
# scores from an older one (review 2026-08-22 pt 9). Uses the m6a3 frozen prompt/tokens via
# e05_score/scripts/score_asure.py (vLLM venv). ~67 rows/s on the A100.
set -euo pipefail
W=/home/jupyter/trendyol-teknofest-work
E17=$W/asya/s2_exp/e17_mine_v5
VPY=$W/asya/gcp_jobs/m6a0_asure_zeroshot_v1/.venv/bin/python
SEL=${1:-$E17/outputs/selected_v5.parquet}
SHA=$(sha256sum "$SEL" | cut -c1-12)
OUT=$E17/outputs/asure_scores_v5_$SHA
echo "[score_v5] selection $SEL sha12=$SHA -> $OUT.parquet"
# the scorer needs (query, item_key) + recipe; pass the selection itself (contrast rows included, harmless)
cd $W/asya/s2_exp/e05_score && CAND="$SEL" OUT="$OUT" CHUNK=20000 $VPY scripts/score_asure.py
echo "[score_v5] done"
