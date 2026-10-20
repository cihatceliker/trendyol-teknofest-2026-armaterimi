#!/usr/bin/env bash
# DECIDER, resumed. The first launch's train_student.py was orphaned when I killed only its parent
# bash (kill <pid> does not touch the child; kill -- -<pgid> would have). That orphan IS the run we
# want -- TAG=d2_tiny2_6ep, turkish-medium, EPOCHS=6, started 10:59 -- so it is left to finish and
# only its EVAL is missing. The relaunched wrapper was stuck waiting behind it and would have
# started a SECOND identical d2 on top of the first; it has been stopped.
# This script: wait out the orphan -> eval d2 -> then run D1 (BERTurk seed 43, BS 64 as original).
set -u
W=/home/jupyter/trendyol-teknofest-work; E15=$W/asya/s2_exp/e15_student
E23=$W/asya/s2_exp/e23_regime_cuts; PY=$W/.venv-gcp/bin/python; O=$E15/outputs_frozen_v2
DATA=$O/build_oldstore/student_train.parquet
S=$O/DECIDE_SUMMARY.txt
log(){ echo "[$(date +%H:%M:%S)] $*" | tee -a $S; }
cd $E15; : > $S
log "waiting for the orphaned d2_tiny2_6ep training to finish"
while ps -eo cmd | grep -qE "[t]rain_student.py"; do sleep 60; done
sleep 10
if [ -d "$O/ckpt_d2_tiny2_6ep" ]; then
  log "== d2_tiny2_6ep (turkish-medium 8L/512, 6 epochs vs the standard 3)"
  ATTR_CLEAN=v2 TRUST=1 CKPT=$O/ckpt_d2_tiny2_6ep OUT=$O $PY scripts/eval_student.py 2>&1 \
    | grep -E "2-boundary|AUC" | sed 's/^/   /' | tee -a $S
  cd $E23 && $PY scripts/fit_regime_cuts.py --probs $O/dev_probs_d2_tiny2_6ep.parquet 2>&1 \
    | grep -E "global|regime" | sed 's/^/   /' | tee -a $S; cd $E15
else log "!! d2 checkpoint missing - the orphan did not finish cleanly"; tail -4 $O/train_d2_tiny2_6ep.log | sed 's/^/   /' | tee -a $S; fi

log "== d1_berturk_s43 (second seed of the hedge candidate; BS 64 = its original recipe)"
if TAG=d1_berturk_s43 MODEL=dbmdz/bert-base-turkish-128k-cased TRUST=1 BS=64 LR=3e-5 SEED=43 \
   EPOCHS=3 ALPHA=1 BETA=0 DATA=$DATA OUT=$O $PY scripts/train_student.py > $O/train_d1_berturk_s43.log 2>&1; then
  ATTR_CLEAN=v2 TRUST=1 CKPT=$O/ckpt_d1_berturk_s43 OUT=$O $PY scripts/eval_student.py 2>&1 \
    | grep -E "2-boundary|AUC" | sed 's/^/   /' | tee -a $S
  cd $E23 && $PY scripts/fit_regime_cuts.py --probs $O/dev_probs_d1_berturk_s43.parquet 2>&1 \
    | grep -E "global|regime" | sed 's/^/   /' | tee -a $S; cd $E15
else log "!! d1 FAILED"; tail -4 $O/train_d1_berturk_s43.log | sed 's/^/   /' | tee -a $S; fi
log "DECIDER DONE"
