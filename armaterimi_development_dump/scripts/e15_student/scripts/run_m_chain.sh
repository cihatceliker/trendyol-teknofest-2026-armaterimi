#!/usr/bin/env bash
# M-chain (goal items 1+3): truncation speed/quality sweep on the best model, then the Turkish
# reranker candidate. Waits for the L-chain. Ref: r1_oldstore DEV 0.8553 @ MAXLEN 256.
set -u
W=/home/jupyter/trendyol-teknofest-work; E15=$W/asya/s2_exp/e15_student; E12=$W/asya/s2_exp/e12_lean_infer; PY=$W/.venv-gcp/bin/python
O=$E15/outputs_frozen_v2; DATA=$O/build_oldstore/student_train.parquet
S=$O/M_SUMMARY.txt; : > $S
log(){ echo "[$(date +%H:%M:%S)] $*" | tee -a $S; }
cd $E15
log "waiting for L chain"
until grep -q "L DONE" $O/L_SUMMARY.txt 2>/dev/null; do
  n=$(ps -eo cmd | grep -c "[r]un_l_layers.sh"); [ "$n" -eq 0 ] && break; sleep 60
done
n=$(ps -eo cmd | grep -c "[t]rain_student.py"); [ "$n" -gt 0 ] && { log "GUARD"; exit 1; }
# m1: eval-only truncation sweep on the best checkpoint (trained @256) + SPS bench per length
for ML in 192 160 128; do
  log "== m1 maxlen $ML eval (ckpt_r1_oldstore)"
  ATTR_CLEAN=v2 TRUST=1 MAXLEN=$ML CKPT=$O/ckpt_r1_oldstore OUT=$O TAG=r1_ml$ML $PY scripts/eval_student.py 2>&1 | grep -E "2-boundary" | tee -a $S
  cd $E12 && BENCH_SCRIPT=serve_hf.py MAXLEN=$ML $PY scripts/bench_e2e.py --model-dir=$O/ckpt_r1_oldstore --runs=1 2>&1 | grep -E "projected" | tee -a $S; cd $E15
done
# m2: Turkish reranker (ModernBERT 149M, distilled from Qwen3-Reranker-8B) on the full R1 data
n=$(ps -eo cmd | grep -c "[t]rain_student.py"); [ "$n" -gt 0 ] && { log "GUARD"; exit 1; }
log "== m2 modernbert-tr-reranker 3ep (149M, ModernBERT)"
if TAG=m2_mbert_rr MODEL=ytu-ce-cosmos/modernbert-tr-reranker TRUST=0 BS=64 LR=3e-5 EPOCHS=3 ALPHA=1 BETA=0 \
   DATA=$DATA OUT=$O $PY scripts/train_student.py > $O/train_m2_mbert_rr.log 2>&1; then
  ATTR_CLEAN=v2 CKPT=$O/ckpt_m2_mbert_rr OUT=$O $PY scripts/eval_student.py 2>&1 | grep -E "2-boundary|AUC" | tee -a $S
  cd $E12 && BENCH_SCRIPT=serve_hf.py $PY scripts/bench_e2e.py --model-dir=$O/ckpt_m2_mbert_rr --runs=1 2>&1 | grep -E "projected" | tee -a $S; cd $E15
else
  log "!! m2 FAILED"; tail -3 $O/train_m2_mbert_rr.log | tee -a $S
fi
log "M DONE"
