#!/usr/bin/env bash
# F5: BERTurk-128k on v2 text (2 ep, pairs with F1 0.7920). F6: TY-ecomm-embed on v2 text, 3 epochs (pairs with F4 0.8149).
set -uo pipefail
W=/home/jupyter/trendyol-teknofest-work; E15=$W/asya/s2_exp/e15_student; PY=$W/.venv-gcp/bin/python
O=$E15/outputs_frozen_v2; S=$O/F56_SUMMARY.txt; : > $S
log(){ echo "[$(date +%H:%M:%S)] $*" | tee -a $S; }
cd $E15
log "F5 BERTurk-128k v2 text (2 ep, BS 64, LR 3e-5)"
TAG=f5_berturk_v2 ALPHA=1 BETA=0 DATA=$O/student_train.parquet OUT=$O $PY scripts/train_student.py > $O/train_f5.log 2>&1
tail -1 $O/train_f5.log | cut -c1-160 | tee -a $S
ATTR_CLEAN=v2 CKPT=$O/ckpt_f5_berturk_v2 OUT=$O $PY scripts/eval_student.py 2>&1 | grep -E "2-boundary|AUC" | tee -a $S
log "F6 TY-ecomm-embed v2 text, 3 epochs (BS 32, LR 2e-5)"
TAG=f6_tyecomm_v2_3ep MODEL=Trendyol/TY-ecomm-embed-multilingual-base-v1.2.0 TRUST=1 BS=32 LR=2e-5 EPOCHS=3 ALPHA=1 BETA=0 DATA=$O/student_train.parquet OUT=$O $PY scripts/train_student.py > $O/train_f6.log 2>&1
tail -1 $O/train_f6.log | cut -c1-160 | tee -a $S
ATTR_CLEAN=v2 TRUST=1 CKPT=$O/ckpt_f6_tyecomm_v2_3ep OUT=$O $PY scripts/eval_student.py 2>&1 | grep -E "2-boundary|AUC" | tee -a $S
log "references: F1 BERTurk v1 0.7920 | F4 TY-ecomm v2 2ep 0.8149 | F56 DONE"
