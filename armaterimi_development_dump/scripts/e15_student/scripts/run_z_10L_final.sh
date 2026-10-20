#!/usr/bin/env bash
# Z: finish the 10-layer candidate properly once the queue drains.
#   z1 MAXLEN-128 accuracy check ON THE 10L MODEL with regime cuts refit at 128 (the +7.7% guard)
#   z2 90k bench of 10L at the best config (batch 256 + whichever maxlen z1 validates)
set -u
W=/home/jupyter/trendyol-teknofest-work; E15=$W/asya/s2_exp/e15_student; E12=$W/asya/s2_exp/e12_lean_infer
E23=$W/asya/s2_exp/e23_regime_cuts; PY=$W/.venv-gcp/bin/python; O=$E15/outputs_frozen_v2
SC=/var/tmp/claude-1000/-home-jupyter/483abbaf-6d7f-4b27-838c-53d3d34a821c/scratchpad
S=$O/Z_SUMMARY.txt; : > $S
log(){ echo "[$(date +%H:%M:%S)] $*" | tee -a $S; }
cd $E15
log "waiting for Y (which itself waits for X)"
until grep -q "Y DONE" $O/Y_SUMMARY.txt 2>/dev/null; do
  ps -eo cmd | grep -qE "[r]un_x_queue.sh|[r]un_y_maxlen_check.sh" || break; sleep 60; done
for ML in 256 128; do
  log "== 10L eval at MAXLEN=$ML (cuts refit at this length)"
  ATTR_CLEAN=v2 TRUST=1 MAXLEN=$ML TAG=x1_10L_ml${ML} CKPT=$O/ckpt_x1_tyecomm_10L OUT=$O $PY scripts/eval_student.py 2>&1 | grep -E "2-boundary" | tee -a $S
  cd $E23 && $PY scripts/fit_regime_cuts.py --probs $O/dev_probs_x1_10L_ml${ML}.parquet 2>&1 | grep -E "global|regime" | tee -a $S; cd $E15
  log "   90k bench maxlen=$ML batch=256"
  cd $E12 && BATCH_ROWS=256 TOKEN_BUDGET=65536 MAXLEN=$ML $PY scripts/serve_gte.py $SC/synth90k.csv $SC/z_10L_$ML.csv --model-dir=$O/ckpt_x1_tyecomm_10L 2>&1 | grep STAGE | sed 's/.*infer=/   infer=/' | cut -c1-80 | tee -a $S; cd $E15
done
log "Z DONE"
