#!/usr/bin/env bash
# Direct-training ladder, day 1: hard labels only, no teacher anywhere.
# D1a: full organizer R/S (498k) + matched judged-I (47.6k)   [data already built]
# D1b: full organizer R/S + ALL 112k judged-I from the consolidated label store
set -uo pipefail
W=/home/jupyter/trendyol-teknofest-work
E15=$W/asya/s2_exp/e15_student
PY=$W/.venv-gcp/bin/python
log(){ echo "[$(date +%H:%M:%S)] ===== $* ====="; }

log "waiting for GPU"
until [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)" -lt 2000 ]; do sleep 20; done

log "D1a train (545,944 rows, 2ep, BERTurk-128k, class-weighted CE)"
cd $E15 && TAG=d1a $PY scripts/train_student.py 2>&1 | tail -5
log "D1a eval"
CKPT=$E15/outputs/ckpt_d1a $PY scripts/eval_student.py 2>&1 | tail -8

log "D1b build (label store, 112k judged-I)"
LABELS=$W/asya/s2_exp/e16_mine_v4/outputs/label_store.parquet OUT=$E15/outputs_d1b \
  $PY scripts/build_student_data.py 2>&1 | tail -4
log "D1b train"
TAG=d1b DATA=$E15/outputs_d1b/student_train.parquet OUT=$E15/outputs_d1b \
  $PY scripts/train_student.py 2>&1 | tail -5
log "D1b eval"
CKPT=$E15/outputs_d1b/ckpt_d1b OUT=$E15/outputs_d1b $PY scripts/eval_student.py 2>&1 | tail -8
log "D1 LADDER DONE"
