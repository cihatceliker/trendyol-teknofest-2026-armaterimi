#!/usr/bin/env bash
# Third seed of the freeze recipe (variance documentation for the report; champion switch only if >0.012).
set -u
W=/home/jupyter/trendyol-teknofest-work; E15=$W/asya/s2_exp/e15_student; PY=$W/.venv-gcp/bin/python
O=$E15/outputs_frozen_v2; S=$O/S3_SUMMARY.txt; : > $S
log(){ echo "[$(date +%H:%M:%S)] $*" | tee -a $S; }
cd $E15
n=$(ps -eo cmd | grep -c "[t]rain_student.py"); [ "$n" -gt 0 ] && { log GUARD; exit 1; }
log "== s3_seed44 (freeze recipe, SEED=44; refs seed42 0.8553, seed43 0.8514)"
if SEED=44 TAG=s3_seed44 MODEL=Trendyol/TY-ecomm-embed-multilingual-base-v1.2.0 TRUST=1 BS=32 LR=2e-5 EPOCHS=3 ALPHA=1 BETA=0 \
   DATA=$O/build_oldstore/student_train.parquet OUT=$O $PY scripts/train_student.py > $O/train_s3_seed44.log 2>&1; then
  ATTR_CLEAN=v2 TRUST=1 CKPT=$O/ckpt_s3_seed44 OUT=$O $PY scripts/eval_student.py 2>&1 | grep -E "2-boundary|AUC" | tee -a $S
else
  log "!! s3 FAILED"; tail -3 $O/train_s3_seed44.log | tee -a $S
fi
log "S3 DONE"
