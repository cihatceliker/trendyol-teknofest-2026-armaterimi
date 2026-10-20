#!/usr/bin/env bash
# NIGHT PHASE 2 — seed replication of the two most likely finalists.
#
# CLAUDE.md: "no overfit, no cherry-picking, robustness is extremely important." Every candidate
# in the table so far is a SINGLE SEED. Picking the best of eight single-seed numbers and shipping
# it is textbook cherry-picking: the winner is partly winning on seed luck, and the private 90k
# will not reproduce that luck. The recipe mean matters, not the lucky draw.
#
# The two most likely finalists, replicated at seed 43:
#   TY-ecomm 8L  -- the quality end. TY's depth curve is nearly flat (12L->10L cost 0.0016), so 8L
#                   is the likeliest Pareto winner; trained fresh tonight in phase 1.
#   tiny2 8L/512 -- the speed end. The only tiny model over the 0.80 floor, and its margin (+0.022)
#                   is under 2x the 0.012 noise floor, so a second seed is not optional: if seed 43
#                   lands near 0.80 the whole fast line is a gamble, and we need to know that BEFORE
#                   choosing, not after shipping.
# A separate script from phase 1 ON PURPOSE: bash reads a script incrementally, so appending to a
# running one can corrupt execution.
set -u
W=/home/jupyter/trendyol-teknofest-work; E15=$W/asya/s2_exp/e15_student; E12=$W/asya/s2_exp/e12_lean_infer
E23=$W/asya/s2_exp/e23_regime_cuts; PY=$W/.venv-gcp/bin/python; O=$E15/outputs_frozen_v2
DATA=$O/build_oldstore/student_train.parquet
S=$O/NIGHT_SUMMARY.txt
log(){ echo "[$(date +%H:%M:%S)] $*" | tee -a $S; }
cd $E15
log "PHASE 2 (seeds) — waiting for phase 1"
while ps -eo cmd | grep -qE "[r]un_night_frontier.sh|[r]un_t3_grow.sh|[t]rain_student.py|[e]val_student.py"; do sleep 60; done
sleep 20
log "== PHASE 2: seed replication"

seed(){ # tag backbone bs lr seed
  local n; n=$(ps -eo cmd | grep -cE "[t]rain_student.py"); [ "$n" -gt 0 ] && { log "GUARD"; exit 1; }
  log "== $1 (seed $5)"
  if TAG=$1 MODEL=$2 TRUST=1 BS=$3 LR=$4 SEED=$5 EPOCHS=3 ALPHA=1 BETA=0 DATA=$DATA OUT=$O \
     $PY scripts/train_student.py > $O/train_$1.log 2>&1; then
    ATTR_CLEAN=v2 TRUST=1 CKPT=$O/ckpt_$1 OUT=$O $PY scripts/eval_student.py 2>&1 \
      | grep -E "2-boundary|AUC" | sed 's/^/   /' | tee -a $S
    cd $E23 && $PY scripts/fit_regime_cuts.py --probs $O/dev_probs_$1.parquet 2>&1 \
      | grep -E "global|regime" | sed 's/^/   /' | tee -a $S; cd $E15
  else log "!! $1 FAILED"; tail -4 $O/train_$1.log | sed 's/^/   /' | tee -a $S; fi
}

# tiny2 first: it is the cheaper run AND the higher-variance question (its floor margin is thin)
seed tiny2_seed43   ytu-ce-cosmos/turkish-medium-bert-uncased 64 5e-5 43
seed l8b_seed43     $E15/backbones/tyecomm_8L                 32 2e-5 43
log "PHASE 2 DONE — all night work complete"
