#!/usr/bin/env bash
# L-chain: depth/speed tradeoff. 8-layer and 6-layer TY-ecomm (bottom-K of the pretrained model,
# same recipe: champion data, 3ep) + end-to-end SPS bench per model. Ref: 12L = DEV 0.8529 @ 1,758 SPS.
set -u
W=/home/jupyter/trendyol-teknofest-work; E15=$W/asya/s2_exp/e15_student; E12=$W/asya/s2_exp/e12_lean_infer; PY=$W/.venv-gcp/bin/python
O=$E15/outputs_frozen_v2; DATA=$O/build_g2_extras/student_train.parquet
S=$O/L_SUMMARY.txt; : > $S
log(){ echo "[$(date +%H:%M:%S)] $*" | tee -a $S; }
cd $E15
log "waiting for R chain (old-store run)"
until grep -q "R DONE" $O/R_SUMMARY.txt 2>/dev/null; do
  n=$(ps -eo cmd | grep -c "[r]un_r_oldstore.sh"); [ "$n" -eq 0 ] && break; sleep 60
done
for K in 8 6; do
  n=$(ps -eo cmd | grep -c "[t]rain_student.py"); [ "$n" -gt 0 ] && { log "GUARD: training running"; exit 1; }
  log "== l${K}_tyecomm (bottom-$K layers, 3ep)"
  if TAG=l${K}_tyecomm MODEL=$E15/backbones/tyecomm_${K}L TRUST=1 BS=32 LR=2e-5 EPOCHS=3 ALPHA=1 BETA=0 \
     DATA=$DATA OUT=$O $PY scripts/train_student.py > $O/train_l${K}_tyecomm.log 2>&1; then
    ATTR_CLEAN=v2 TRUST=1 CKPT=$O/ckpt_l${K}_tyecomm OUT=$O $PY scripts/eval_student.py 2>&1 | grep -E "2-boundary|AUC" | tee -a $S
    log "bench l${K}"
    cd $E12 && BENCH_SCRIPT=serve_hf.py $PY scripts/bench_e2e.py --model-dir=$O/ckpt_l${K}_tyecomm --runs=1 2>&1 | grep -E "projected" | tee -a $S; cd $E15
  else
    log "!! l${K} FAILED"; tail -3 $O/train_l${K}_tyecomm.log | tee -a $S
  fi
done
log "L DONE"
