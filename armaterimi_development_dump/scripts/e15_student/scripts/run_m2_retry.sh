#!/usr/bin/env bash
set -u
W=/home/jupyter/trendyol-teknofest-work; E15=$W/asya/s2_exp/e15_student; E12=$W/asya/s2_exp/e12_lean_infer; PY=$W/.venv-gcp/bin/python
O=$E15/outputs_frozen_v2; S=$O/M_SUMMARY.txt
log(){ echo "[$(date +%H:%M:%S)] $*" | tee -a $S; }
cd $E15
n=$(ps -eo cmd | grep -c "[t]rain_student.py"); [ "$n" -gt 0 ] && { log "GUARD"; exit 1; }
log "== m2_retry modernbert-tr-reranker 3ep (head-swap fix)"
if TAG=m2_mbert_rr MODEL=ytu-ce-cosmos/modernbert-tr-reranker TRUST=0 BS=64 LR=3e-5 EPOCHS=3 ALPHA=1 BETA=0 \
   DATA=$O/build_oldstore/student_train.parquet OUT=$O $PY scripts/train_student.py > $O/train_m2_mbert_rr.log 2>&1; then
  ATTR_CLEAN=v2 CKPT=$O/ckpt_m2_mbert_rr OUT=$O $PY scripts/eval_student.py 2>&1 | grep -E "2-boundary|AUC" | tee -a $S
  cd $E12 && BENCH_SCRIPT=serve_hf.py $PY scripts/bench_e2e.py --model-dir=$O/ckpt_m2_mbert_rr --runs=1 2>&1 | grep -E "projected" | tee -a $S; cd $E15
else
  log "!! m2_retry FAILED"; tail -3 $O/train_m2_mbert_rr.log | tee -a $S
fi
log "M2 RETRY DONE"
