#!/usr/bin/env bash
# DECIDER — two runs that between them settle the ship choice.
#
# In-container frontier (jury clock, all fixes in):
#   TY-12L 0.8587/2652(1 seed)  TY-8L 0.8474/3351(3)  BERTurk 0.8373/4401(1)  tiny2 0.8205/5625(2)
# The archetype math puts tiny2 ahead on expected value and BERTurk close behind with roughly half
# the cliff risk (margin over the 0.80 speed gate: tiny2 +0.020, BERTurk +0.037). Two unknowns
# decide between them, and both are cheap:
#
#   D1  BERTurk seed 43. It is a SINGLE-SEED number. TY-8L's replication showed the best of three
#       sat +0.0056 above the mean, so 0.8373 may be a lucky draw and its real edge over tiny2
#       could be half what it looks.
#   D2  tiny2 at 6 epochs. Every model here trained 3 epochs, a schedule tuned on the 306M
#       champion. A 42M model on 813k rows is the one most likely to be undertrained, and tiny2 is
#       the candidate whose ONLY weakness is F1. If 6 epochs buys +0.01 it clears the cliff worry
#       and the decision stops being close.
set -u
W=/home/jupyter/trendyol-teknofest-work; E15=$W/asya/s2_exp/e15_student
E23=$W/asya/s2_exp/e23_regime_cuts; PY=$W/.venv-gcp/bin/python; O=$E15/outputs_frozen_v2
DATA=$O/build_oldstore/student_train.parquet
S=$O/DECIDE_SUMMARY.txt
log(){ echo "[$(date +%H:%M:%S)] $*" | tee -a $S; }
cd $E15; : > $S
while ps -eo cmd | grep -qE "[t]rain_student.py|[e]val_student.py"; do sleep 30; done
log "DECIDER starting"

run(){ # tag model bs lr seed epochs
  local n; n=$(ps -eo cmd | grep -cE "[t]rain_student.py"); [ "$n" -gt 0 ] && { log "GUARD"; exit 1; }
  log "== $1 (seed $5, ${6}ep)"
  if TAG=$1 MODEL=$2 TRUST=1 BS=$3 LR=$4 SEED=$5 EPOCHS=$6 ALPHA=1 BETA=0 DATA=$DATA OUT=$O \
     $PY scripts/train_student.py > $O/train_$1.log 2>&1; then
    ATTR_CLEAN=v2 TRUST=1 CKPT=$O/ckpt_$1 OUT=$O $PY scripts/eval_student.py 2>&1 \
      | grep -E "2-boundary|AUC" | sed 's/^/   /' | tee -a $S
    cd $E23 && $PY scripts/fit_regime_cuts.py --probs $O/dev_probs_$1.parquet 2>&1 \
      | grep -E "global|regime" | sed 's/^/   /' | tee -a $S; cd $E15
  else log "!! $1 FAILED"; tail -4 $O/train_$1.log | sed 's/^/   /' | tee -a $S; fi
}

run d2_tiny2_6ep    ytu-ce-cosmos/turkish-medium-bert-uncased 64 5e-5 42 6
run d1_berturk_s43  dbmdz/bert-base-turkish-128k-cased        64 3e-5 43 3
log "DECIDER DONE"
