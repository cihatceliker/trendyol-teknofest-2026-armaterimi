#!/usr/bin/env bash
# Q-chain: quality experiments aimed at the semi/irrelevant boundary (B2 §1.2: 2.2x the headroom
# of the R boundary; ~0.87 macro-F1 per AUC point). Champion = TY-ecomm 3ep, DEV 0.8529, AUC S|I 0.8964.
set -u
W=/home/jupyter/trendyol-teknofest-work; E15=$W/asya/s2_exp/e15_student; PY=$W/.venv-gcp/bin/python
O=$E15/outputs_frozen_v2; DATA=$O/build_g2_extras/student_train.parquet
S=$O/Q_SUMMARY.txt; : > $S
log(){ echo "[$(date +%H:%M:%S)] $*" | tee -a $S; }
cd $E15
TY=Trendyol/TY-ecomm-embed-multilingual-base-v1.2.0
run(){ # tag epochs extra_env
  n=$(ps -eo cmd | grep -c "[t]rain_student.py"); [ "$n" -gt 0 ] && { log "GUARD: training already running"; exit 1; }
  log "== $1 (ep=$2 ${3:-})"
  if env ${3:-} TAG=$1 MODEL=$TY TRUST=1 BS=32 LR=2e-5 EPOCHS=$2 ALPHA=1 BETA=0 DATA=$DATA OUT=$O $PY scripts/train_student.py > $O/train_$1.log 2>&1; then
    ATTR_CLEAN=v2 TRUST=1 CKPT=$O/ckpt_$1 OUT=$O $PY scripts/eval_student.py 2>&1 | grep -E "2-boundary|AUC" | tee -a $S
  else
    log "!! $1 FAILED"; tail -3 $O/train_$1.log | tee -a $S
  fi
}
log "Q-CHAIN START — ref c2_tyecomm_3ep DEV 0.8529, AUC(S|I) 0.8964"
run q1_aux_si05  3 "AUX_SI=0.5"      # explicit semi-vs-irrelevant margin
run q2_seed43    3 "SEED=43"         # seed variance + ensemble partner
run q3_aux_si10  3 "AUX_SI=1.0"      # stronger S|I weight (only informative if q1 moves)
log "Q-CHAIN DONE"
