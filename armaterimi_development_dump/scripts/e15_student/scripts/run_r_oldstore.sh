#!/usr/bin/env bash
# R1: does the 136,489-pair PRE-v5 label store help? (old hard-band+matched judged pairs, all 3 classes.)
# Paired vs c2_tyecomm_3ep (0.8529). Waits for the Q-chain, then runs alone.
set -u
W=/home/jupyter/trendyol-teknofest-work; E15=$W/asya/s2_exp/e15_student; E17=$W/asya/s2_exp/e17_mine_v5; PY=$W/.venv-gcp/bin/python
O=$E15/outputs_frozen_v2; S=$O/R_SUMMARY.txt; : > $S
log(){ echo "[$(date +%H:%M:%S)] $*" | tee -a $S; }
cd $E15
log "waiting for Q-chain"
until grep -q "Q-CHAIN DONE" $O/Q_SUMMARY.txt 2>/dev/null; do
  n=$(ps -eo cmd | grep -c "[r]un_q_chain.sh"); [ "$n" -eq 0 ] && break; sleep 60
done
n=$(ps -eo cmd | grep -c "[t]rain_student.py"); [ "$n" -gt 0 ] && { log "GUARD: training still running"; exit 1; }
log "build champion+oldstore set"
ATTR_CLEAN=v2 LABELS=$E17/outputs/labels_allI_plus_old.parquet CONTRAST=$E17/outputs/labels_extras_plus_old.parquet \
  OUT=$O/build_oldstore $PY scripts/build_student_data.py 2>&1 | grep -E "wrote|judged-I|contrast" | cut -c1-170 | tee -a $S
log "R1 TY-ecomm 3ep + old store (ref 0.8529)"
if TAG=r1_oldstore MODEL=Trendyol/TY-ecomm-embed-multilingual-base-v1.2.0 TRUST=1 BS=32 LR=2e-5 EPOCHS=3 ALPHA=1 BETA=0 \
   DATA=$O/build_oldstore/student_train.parquet OUT=$O $PY scripts/train_student.py > $O/train_r1_oldstore.log 2>&1; then
  ATTR_CLEAN=v2 TRUST=1 CKPT=$O/ckpt_r1_oldstore OUT=$O $PY scripts/eval_student.py 2>&1 | grep -E "2-boundary|AUC" | tee -a $S
else
  log "!! r1 FAILED"; tail -3 $O/train_r1_oldstore.log | tee -a $S
fi
log "R DONE"
