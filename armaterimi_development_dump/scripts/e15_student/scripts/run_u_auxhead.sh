#!/usr/bin/env bash
# U1: 3-way softmax (the shipping architecture) PLUS a SEPARATE auxiliary S|I output head.
# Rationale: the two-head runs proved a dedicated S/I output reaches AUC(S|I) 0.9126 vs 0.9037,
# but paid for it by losing the 3-way contrast on R|S. This keeps the 3-way head intact (so R|S is
# untouched) and adds a 4th logit trained only on stage-2 non-relevant rows. At inference the aux
# logit is IGNORED — the shipped scalar/cuts read the first three logits exactly as today, so this
# can only help via better representations, and serving is byte-identical in cost.
set -u
W=/home/jupyter/trendyol-teknofest-work; E15=$W/asya/s2_exp/e15_student; PY=$W/.venv-gcp/bin/python
O=$E15/outputs_frozen_v2; S=$O/U_SUMMARY.txt; : > $S
log(){ echo "[$(date +%H:%M:%S)] $*" | tee -a $S; }
cd $E15
n=$(ps -eo cmd | grep -c "[t]rain_student.py"); [ "$n" -gt 0 ] && { log "GUARD"; exit 1; }
log "U1 3-way + aux S|I output head (ref 3-class 0.8553 / R|S 0.9663 / S|I 0.9037)"
if AUX_SI_HEAD=0.5 TAG=u1_auxhead MODEL=Trendyol/TY-ecomm-embed-multilingual-base-v1.2.0 TRUST=1 \
   BS=32 LR=2e-5 EPOCHS=3 ALPHA=1 BETA=0 DATA=$O/build_oldstore/student_train.parquet OUT=$O \
   $PY scripts/train_student.py > $O/train_u1_auxhead.log 2>&1; then
  ATTR_CLEAN=v2 TRUST=1 CKPT=$O/ckpt_u1_auxhead OUT=$O $PY scripts/eval_student.py 2>&1 | grep -E "2-boundary|AUC" | tee -a $S
else log "!! u1 FAILED"; tail -4 $O/train_u1_auxhead.log | tee -a $S; fi
log "U DONE"
