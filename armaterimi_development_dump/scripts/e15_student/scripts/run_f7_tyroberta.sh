#!/usr/bin/env bash
# F7: tyroberta on v2 text (2 ep) — pairs with F5 BERTurk-v2. Waits for the F5/F6 chain.
set -uo pipefail
W=/home/jupyter/trendyol-teknofest-work; E15=$W/asya/s2_exp/e15_student; PY=$W/.venv-gcp/bin/python
O=$E15/outputs_frozen_v2; S=$O/F7_SUMMARY.txt; : > $S
log(){ echo "[$(date +%H:%M:%S)] $*" | tee -a $S; }
cd $E15
log "waiting for F56 chain"
until grep -q "F56 DONE" $O/F56_SUMMARY.txt 2>/dev/null; do pgrep -f "run_f5_f6.sh" >/dev/null || break; sleep 60; done
log "F7 tyroberta v2 text (2 ep, BS 64, LR 3e-5)"
TAG=f7_tyroberta_v2 MODEL=Trendyol/tyroberta ALPHA=1 BETA=0 DATA=$O/student_train.parquet OUT=$O $PY scripts/train_student.py > $O/train_f7.log 2>&1
tail -1 $O/train_f7.log | cut -c1-160 | tee -a $S
ATTR_CLEAN=v2 CKPT=$O/ckpt_f7_tyroberta_v2 OUT=$O $PY scripts/eval_student.py 2>&1 | grep -E "2-boundary|AUC" | tee -a $S
log "F7 DONE"
