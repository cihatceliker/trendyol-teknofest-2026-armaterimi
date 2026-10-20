#!/usr/bin/env bash
# FAIRNESS + REPLICATION. Two problems with the table as it stands:
#
#  1. The comparison is CONFOUNDED. tiny2 got 6 epochs; BERTurk got 3. The +0.0052 that 6 epochs
#     bought tiny2 may be available to BERTurk too, and giving one candidate a better schedule
#     than the other is not a model comparison, it is a schedule comparison. BERTurk gets 6 too.
#  2. tiny2-6ep (0.8257) is a SINGLE SEED, and it is currently the leading candidate. Everything
#     else on the table that mattered got replicated; so does this.
#
# Seed evidence so far -- replication is not optional, and it does not always say the same thing:
#     TY-8L    0.8530 / 0.8436 / 0.8455  -> mean 0.8474, spread 0.0094 (best was +0.0056 lucky)
#     BERTurk  0.8373 / 0.8370           -> mean 0.8372, spread 0.0003 (rock solid)
#     tiny2/3ep 0.8218 / 0.8191          -> mean 0.8205, spread 0.0027
set -u
W=/home/jupyter/trendyol-teknofest-work; E15=$W/asya/s2_exp/e15_student
E23=$W/asya/s2_exp/e23_regime_cuts; PY=$W/.venv-gcp/bin/python; O=$E15/outputs_frozen_v2
DATA=$O/build_oldstore/student_train.parquet
S=$O/FAIR_SUMMARY.txt
log(){ echo "[$(date +%H:%M:%S)] $*" | tee -a $S; }
cd $E15; : > $S
while ps -eo cmd | grep -qE "[t]rain_student.py|[e]val_student.py"; do sleep 30; done
log "FAIR/REPLICATION starting"
run(){ # tag model bs lr seed epochs note
  local n; n=$(ps -eo cmd | grep -cE "[t]rain_student.py"); [ "$n" -gt 0 ] && { log "GUARD"; exit 1; }
  log "== $1 -- $7"
  if TAG=$1 MODEL=$2 TRUST=1 BS=$3 LR=$4 SEED=$5 EPOCHS=$6 ALPHA=1 BETA=0 DATA=$DATA OUT=$O \
     $PY scripts/train_student.py > $O/train_$1.log 2>&1; then
    ATTR_CLEAN=v2 TRUST=1 CKPT=$O/ckpt_$1 OUT=$O $PY scripts/eval_student.py 2>&1 \
      | grep -E "2-boundary|AUC" | sed 's/^/   /' | tee -a $S
    cd $E23 && $PY scripts/fit_regime_cuts.py --probs $O/dev_probs_$1.parquet 2>&1 \
      | grep -E "global|regime" | sed 's/^/   /' | tee -a $S; cd $E15
  else log "!! $1 FAILED"; tail -4 $O/train_$1.log | sed 's/^/   /' | tee -a $S; fi
}
run f1_berturk_6ep  dbmdz/bert-base-turkish-128k-cased        64 3e-5 42 6 "BERTurk with tiny2's schedule - removes the confound"
run f2_tiny2_6ep_s43 ytu-ce-cosmos/turkish-medium-bert-uncased 64 5e-5 43 6 "second seed of the leading candidate"
log "FAIR DONE"
