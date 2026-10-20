#!/usr/bin/env bash
# C-chain 2026-08-24: sequential, ONE training at a time, on the champion data (678,549 rows,
# organizer + all-judged-I + contrast + extras, ATTR_CLEAN=v2). Reference: TY-ecomm 2ep = DEV 0.8441.
set -u
W=/home/jupyter/trendyol-teknofest-work; E15=$W/asya/s2_exp/e15_student; PY=$W/.venv-gcp/bin/python
O=$E15/outputs_frozen_v2; DATA=$O/build_g2_extras/student_train.parquet
S=$O/C_SUMMARY.txt; : > $S
log(){ echo "[$(date +%H:%M:%S)] $*" | tee -a $S; }
cd $E15
guard(){ n=$(ps -eo cmd | grep -c "[t]rain_student.py"); if [ "$n" -gt 0 ]; then log "GUARD: $n training(s) already running — refusing"; exit 1; fi; }
run(){ # tag model epochs bs lr
  guard
  log "== $1 ($2 ep=$3 bs=$4 lr=$5)"
  if TAG=$1 MODEL=$2 TRUST=1 BS=$4 LR=$5 EPOCHS=$3 ALPHA=1 BETA=0 DATA=$DATA OUT=$O $PY scripts/train_student.py > $O/train_$1.log 2>&1; then
    ATTR_CLEAN=v2 TRUST=1 CKPT=$O/ckpt_$1 OUT=$O $PY scripts/eval_student.py 2>&1 | grep -E "2-boundary" | tee -a $S
  else
    log "!! $1 FAILED"; tail -3 $O/train_$1.log | tee -a $S
  fi
}
log "C-CHAIN START — champion data 678,549 rows | ref TY-ecomm 2ep DEV 0.8441 / HOLDOUT 0.8385"
run c1_berturk_3ep dbmdz/bert-base-turkish-128k-cased 3 64 3e-5
run c2_tyecomm_3ep Trendyol/TY-ecomm-embed-multilingual-base-v1.2.0 3 32 2e-5
log "C-CHAIN DONE"
