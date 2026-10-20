#!/usr/bin/env bash
# P-chain (goal item 4): make the champion better / prove it robust. Waits for the m2 retry.
# p1: seed-43 replicate of the EXACT freeze recipe (R1 data, 3ep) — robustness pair for the freeze.
# p2: continued fine-tune of ckpt_r1_oldstore, 1 extra epoch at LR 5e-6 — cheap squeeze attempt.
set -u
W=/home/jupyter/trendyol-teknofest-work; E15=$W/asya/s2_exp/e15_student; PY=$W/.venv-gcp/bin/python
O=$E15/outputs_frozen_v2; DATA=$O/build_oldstore/student_train.parquet
S=$O/P_SUMMARY.txt; : > $S
log(){ echo "[$(date +%H:%M:%S)] $*" | tee -a $S; }
cd $E15
log "waiting for m2 retry"
until grep -q "M2 RETRY DONE" $O/M_SUMMARY.txt 2>/dev/null; do
  n=$(ps -eo cmd | grep -c "[r]un_m2_retry.sh"); [ "$n" -eq 0 ] && break; sleep 60
done
TY=Trendyol/TY-ecomm-embed-multilingual-base-v1.2.0
run(){ # tag model epochs lr extra
  n=$(ps -eo cmd | grep -c "[t]rain_student.py"); [ "$n" -gt 0 ] && { log "GUARD"; exit 1; }
  log "== $1 ($2 ep=$3 lr=$4 ${5:-})"
  if env ${5:-} TAG=$1 MODEL=$2 TRUST=1 BS=32 LR=$4 EPOCHS=$3 ALPHA=1 BETA=0 DATA=$DATA OUT=$O $PY scripts/train_student.py > $O/train_$1.log 2>&1; then
    ATTR_CLEAN=v2 TRUST=1 CKPT=$O/ckpt_$1 OUT=$O $PY scripts/eval_student.py 2>&1 | grep -E "2-boundary|AUC" | tee -a $S
  else
    log "!! $1 FAILED"; tail -3 $O/train_$1.log | tee -a $S
  fi
}
run p1_seed43_r1 $TY 3 2e-5 "SEED=43"
run p2_continue1ep $O/ckpt_r1_oldstore 1 5e-6 ""
log "P DONE — refs: r1_oldstore 0.8553 (seed42)"
