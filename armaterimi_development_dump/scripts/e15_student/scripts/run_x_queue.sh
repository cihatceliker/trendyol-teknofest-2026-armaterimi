#!/usr/bin/env bash
# X-chain: the 2-3h queue. Runs strictly after the W speed sweep so benchmarks are uncontended.
#   x1 TY-ecomm 10-layer  — the untested midpoint (12L 0.8553 @2,892 SPS ; 8L 0.8393 @1,938 old path).
#      If it holds ~0.85 at ~3,400 SPS it dominates both current finalists.
#   x2 aux S|I output head (3-way softmax kept intact; 4th logit stripped at save) — last quality idea.
set -u
W=/home/jupyter/trendyol-teknofest-work; E15=$W/asya/s2_exp/e15_student; E12=$W/asya/s2_exp/e12_lean_infer
PY=$W/.venv-gcp/bin/python; O=$E15/outputs_frozen_v2
SC=/var/tmp/claude-1000/-home-jupyter/483abbaf-6d7f-4b27-838c-53d3d34a821c/scratchpad
S=$O/X_SUMMARY.txt; : > $S
log(){ echo "[$(date +%H:%M:%S)] $*" | tee -a $S; }
cd $E15
log "waiting for the W sweep to finish"
until grep -q "W DONE" $O/W_SUMMARY.txt 2>/dev/null; do
  ps -eo cmd | grep -q "[r]un_w_speed_sweep.sh" || break; sleep 60; done
run(){ n=$(ps -eo cmd | grep -c "[t]rain_student.py"); [ "$n" -gt 0 ] && { log "GUARD"; exit 1; }
  log "== $1"
  if env ${4:-FOO=bar} TAG=$1 MODEL=$2 TRUST=1 BS=32 LR=2e-5 EPOCHS=3 ALPHA=1 BETA=0 \
     DATA=$O/build_oldstore/student_train.parquet OUT=$O $PY scripts/train_student.py > $O/train_$1.log 2>&1; then
    env ${4:-FOO=bar} ATTR_CLEAN=v2 TRUST=1 CKPT=$O/ckpt_$1 OUT=$O $PY scripts/eval_student.py 2>&1 | grep -E "2-boundary|AUC" | tee -a $S
    [ -n "${3:-}" ] && { cd $E12; BENCH=$($PY scripts/serve_gte.py $SC/synth90k.csv $SC/x_$1.csv --model-dir=$O/ckpt_$1 2>&1 | grep STAGE); log "   90k: $(echo $BENCH | sed 's/.*infer=/infer=/' | cut -c1-70)"; cd $E15; }
  else log "!! $1 FAILED"; tail -4 $O/train_$1.log | tee -a $S; fi; }
log "X-CHAIN — champion 0.8587 DEV / 0.8646 HOLDOUT @ 2,892 SPS ; BERTurk (full recipe) pending"
run x1_tyecomm_10L $E15/backbones/tyecomm_10L bench
run x2_auxhead     Trendyol/TY-ecomm-embed-multilingual-base-v1.2.0 "" "AUX_SI_HEAD=0.5"
log "X-CHAIN DONE"
