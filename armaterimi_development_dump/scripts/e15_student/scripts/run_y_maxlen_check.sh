#!/usr/bin/env bash
# Y: before adopting MAXLEN=128 for the shipped image (+7.7% SPS), verify accuracy AT that length with
# the regime cuts REFIT at 128 — the earlier 128 numbers reused cuts fitted at 256.
set -u
W=/home/jupyter/trendyol-teknofest-work; E15=$W/asya/s2_exp/e15_student; E23=$W/asya/s2_exp/e23_regime_cuts
PY=$W/.venv-gcp/bin/python; O=$E15/outputs_frozen_v2; S=$O/Y_SUMMARY.txt; : > $S
log(){ echo "[$(date +%H:%M:%S)] $*" | tee -a $S; }
cd $E15
log "waiting for the X chain"
until grep -q "X-CHAIN DONE" $O/X_SUMMARY.txt 2>/dev/null; do
  ps -eo cmd | grep -q "[r]un_x_queue.sh" || break; sleep 60; done
for ML in 256 128; do
  log "== champion eval at MAXLEN=$ML (global + regime cuts refit at this length)"
  ATTR_CLEAN=v2 TRUST=1 MAXLEN=$ML TAG=r1_ml${ML}_chk CKPT=$O/ckpt_r1_oldstore OUT=$O $PY scripts/eval_student.py 2>&1 | grep -E "2-boundary|AUC" | tee -a $S
  cd $E23 && $PY scripts/fit_regime_cuts.py --probs $O/dev_probs_r1_ml${ML}_chk.parquet 2>&1 | grep -E "global|regime" | tee -a $S; cd $E15
done
log "Y DONE"
