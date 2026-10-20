#!/usr/bin/env bash
# Deliverable-class candidate sweep on the current best irrelevant set (v4 clean).
# Asure is reference-only; every model here is a shippable size.
set -uo pipefail
W=/home/jupyter/trendyol-teknofest-work
E15=$W/asya/s2_exp/e15_student
D=$E15/outputs_clean/student_train.parquet
PY=$W/.venv-gcp/bin/python
log(){ echo "[$(date +%H:%M:%S)] ===== $* ====="; }

log "C1 TY-ecomm-embed 306M (in-domain Trendyol e-commerce encoder)"
TAG=c1_tyecomm MODEL=Trendyol/TY-ecomm-embed-multilingual-base-v1.2.0 TRUST=1 \
  BS=32 LR=2e-5 DATA=$D OUT=$E15/outputs_cand $PY $E15/scripts/train_student.py 2>&1 | tail -3
TRUST=1 CKPT=$E15/outputs_cand/ckpt_c1_tyecomm OUT=$E15/outputs_cand \
  $PY $E15/scripts/eval_student.py 2>&1 | tail -4

log "C2 XLM-R-large 560M (capacity reference at encoder speed)"
TAG=c2_xlmrl MODEL=FacebookAI/xlm-roberta-large BS=32 LR=1e-5 \
  DATA=$D OUT=$E15/outputs_cand $PY $E15/scripts/train_student.py 2>&1 | tail -3
CKPT=$E15/outputs_cand/ckpt_c2_xlmrl OUT=$E15/outputs_cand \
  $PY $E15/scripts/eval_student.py 2>&1 | tail -4

log "CANDIDATES DONE"
