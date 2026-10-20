#!/usr/bin/env bash
# TINY: is anything FASTER than BERTurk able to clear the 0.80 speed-points floor?
# Purpose-built small Turkish BERTs (uncased — our text is already lowercased by normalize.py),
# trained on the EXACT champion recipe (813,269 rows, v2 text) and benchmarked at true 90k.
# Reference: BERTurk-128k 12L/768 = 0.8373 DEV @ 5,073 SPS ; TY-ecomm 10L = 0.8571 @ 3,340.
set -u
W=/home/jupyter/trendyol-teknofest-work; E15=$W/asya/s2_exp/e15_student; E12=$W/asya/s2_exp/e12_lean_infer
E23=$W/asya/s2_exp/e23_regime_cuts; PY=$W/.venv-gcp/bin/python; O=$E15/outputs_frozen_v2
SC=/var/tmp/claude-1000/-home-jupyter/483abbaf-6d7f-4b27-838c-53d3d34a821c/scratchpad
S=$O/TINY_SUMMARY.txt; : > $S
log(){ echo "[$(date +%H:%M:%S)] $*" | tee -a $S; }
cd $E15
log "waiting for the aux run to finish and the GPU to free"
until ! ps -eo cmd | grep -q "[t]rain_student.py"; do sleep 60; done
until [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)" -lt 1500 ]; do sleep 30; done
run(){ # tag model lr
  n=$(ps -eo cmd | grep -c "[t]rain_student.py"); [ "$n" -gt 0 ] && { log "GUARD"; exit 1; }
  log "== $1 ($2)"
  if TAG=$1 MODEL=$2 BS=64 LR=$3 EPOCHS=3 ALPHA=1 BETA=0 DATA=$O/build_oldstore/student_train.parquet \
     OUT=$O $PY scripts/train_student.py > $O/train_$1.log 2>&1; then
    ATTR_CLEAN=v2 CKPT=$O/ckpt_$1 OUT=$O $PY scripts/eval_student.py 2>&1 | grep -E "2-boundary" | sed 's/^/   /' | tee -a $S
    cd $E23 && $PY scripts/fit_regime_cuts.py --probs $O/dev_probs_$1.parquet 2>&1 | grep -E "global|regime" | sed 's/^/   /' | tee -a $S; cd $E15
    cd $E12 && BATCH_ROWS=256 TOKEN_BUDGET=65536 $PY scripts/lean_infer.py $SC/synth90k.csv $SC/tiny_$1.csv --model-dir=$O/ckpt_$1 2>&1 | grep STAGE | sed 's/.*infer=/   90k infer=/' | cut -c1-80 | tee -a $S; cd $E15
  else log "!! $1 FAILED"; tail -3 $O/train_$1.log | tee -a $S; fi; }
log "TINY-CHAIN — target: beat BERTurk's 5,073 SPS while staying above 0.80"
run tiny1_distilbert_tr dbmdz/distilbert-base-turkish-cased 5e-5
run tiny2_medium_8L512  ytu-ce-cosmos/turkish-medium-bert-uncased 5e-5
run tiny3_small_4L512   ytu-ce-cosmos/turkish-small-bert-uncased 7e-5
run tiny4_mini_4L256    ytu-ce-cosmos/turkish-mini-bert-uncased 1e-4
log "TINY-CHAIN DONE"
