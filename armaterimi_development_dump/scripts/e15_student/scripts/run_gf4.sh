#!/usr/bin/env bash
# gf4 — TY-ecomm 12L on the gender-fixed data. Without this the comparison is CONFOUNDED: the two
# genderfix models each gained ~+0.012, so a table that pits them against non-fixed TY numbers
# rewards "got the fix", not "is the better model". Same error as comparing tiny2@6ep against
# BERTurk@3ep this morning. Every candidate must sit on the same data.
# 3 epochs / BS32 / LR2e-5 = the champion's own recipe, so the ONLY variable is the artifact.
set -u
W=/home/jupyter/trendyol-teknofest-work; E15=$W/asya/s2_exp/e15_student
E23=$W/asya/s2_exp/e23_regime_cuts; PY=$W/.venv-gcp/bin/python; O=$E15/outputs_frozen_v2
DATA=$O/build_oldstore/student_train_genderfix.parquet
S=$O/GENDERFIX_SUMMARY.txt
log(){ echo "[$(date +%H:%M:%S)] $*" | tee -a $S; }
cd $E15
while ps -eo cmd | grep -qE "[t]rain_student.py|[e]val_student.py"; do sleep 60; done
sleep 15
n=$(ps -eo cmd | grep -cE "[t]rain_student.py"); [ "$n" -gt 0 ] && { log "GUARD"; exit 1; }
log "== gf4_tyecomm_12L -- champion recipe on gender-fixed data (removes the last confound)"
if TAG=gf4_tyecomm_12L MODEL=Trendyol/TY-ecomm-embed-multilingual-base-v1.2.0 TRUST=1 BS=32 LR=2e-5 \
   SEED=42 EPOCHS=3 ALPHA=1 BETA=0 DATA=$DATA OUT=$O $PY scripts/train_student.py > $O/train_gf4_tyecomm_12L.log 2>&1; then
  ATTR_CLEAN=v2 TRUST=1 CKPT=$O/ckpt_gf4_tyecomm_12L OUT=$O $PY scripts/eval_student.py 2>&1 \
    | grep -E "2-boundary|AUC" | sed 's/^/   /' | tee -a $S
  cd $E23 && $PY scripts/fit_regime_cuts.py --probs $O/dev_probs_gf4_tyecomm_12L.parquet 2>&1 \
    | grep -E "global|regime" | sed 's/^/   /' | tee -a $S; cd $E15
else log "!! gf4 FAILED"; tail -4 $O/train_gf4_tyecomm_12L.log | sed 's/^/   /' | tee -a $S; fi
log "GF4 DONE — all four candidates now on identical data"
