#!/usr/bin/env bash
# V-chain: give BERTurk the SAME data the TY-ecomm champion got (813,269 rows incl. the old store).
# c1_berturk_3ep was trained on 678,549 rows, so the head-to-head so far was not apples-to-apples.
set -u
W=/home/jupyter/trendyol-teknofest-work; E15=$W/asya/s2_exp/e15_student; PY=$W/.venv-gcp/bin/python
O=$E15/outputs_frozen_v2; S=$O/V_SUMMARY.txt; : > $S
log(){ echo "[$(date +%H:%M:%S)] $*" | tee -a $S; }
cd $E15
run(){ n=$(ps -eo cmd | grep -c "[t]rain_student.py"); [ "$n" -gt 0 ] && { log "GUARD"; exit 1; }
  log "== $1 (ep=$3 lr=$4)"
  if TAG=$1 MODEL=$2 BS=64 LR=$4 EPOCHS=$3 ALPHA=1 BETA=0 DATA=$O/build_oldstore/student_train.parquet \
     OUT=$O $PY scripts/train_student.py > $O/train_$1.log 2>&1; then
    ATTR_CLEAN=v2 CKPT=$O/ckpt_$1 OUT=$O $PY scripts/eval_student.py 2>&1 | grep -E "2-boundary|AUC" | tee -a $S
  else log "!! $1 FAILED"; tail -4 $O/train_$1.log | tee -a $S; fi; }
log "V-CHAIN — BERTurk on the FULL champion recipe (813,269 rows). Prior: 0.8205 on 678,549 rows."
run v1_berturk_full dbmdz/bert-base-turkish-128k-cased 3 3e-5
run v2_berturk_full_4ep dbmdz/bert-base-turkish-128k-cased 4 2e-5
log "V-CHAIN DONE"
