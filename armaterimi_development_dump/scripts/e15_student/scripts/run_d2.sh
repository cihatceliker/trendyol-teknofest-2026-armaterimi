#!/usr/bin/env bash
# D2 rungs: (a) D1a data + 3 epochs; (b) larger backbone mE5-large, same data, 2 epochs.
set -uo pipefail
W=/home/jupyter/trendyol-teknofest-work
E15=$W/asya/s2_exp/e15_student
PY=$W/.venv-gcp/bin/python
log(){ echo "[$(date +%H:%M:%S)] ===== $* ====="; }
until [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)" -lt 2000 ]; do sleep 15; done
log "D2a train (D1a data, EPOCHS=3)"
cd $E15 && TAG=d2a EPOCHS=3 $PY scripts/train_student.py 2>&1 | tail -4
log "D2a eval"
CKPT=$E15/outputs/ckpt_d2a $PY scripts/eval_student.py 2>&1 | tail -6
log "D2b train (multilingual-e5-large, D1a data, 2ep, BS 32)"
TAG=d2b MODEL=intfloat/multilingual-e5-large BS=32 LR=1.5e-5 $PY scripts/train_student.py 2>&1 | tail -4
log "D2b eval"
CKPT=$E15/outputs/ckpt_d2b $PY scripts/eval_student.py 2>&1 | tail -6
log "D2 DONE"
