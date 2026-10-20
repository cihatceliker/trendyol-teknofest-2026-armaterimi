#!/usr/bin/env bash
# GENDER-FIX RETRAIN. The organizer's train export makes gender=='çocuk' a PERFECT predictor of
# semi-relevant (16,634 rows, zero relevant). At test those rows are 44% irrelevant / 29% relevant
# / 27% semi, and the model — believing the training signal — scores 69.0% there vs 88.1%
# elsewhere on 9.7% of rows. Same disease as `unisex`, which T4 already neutralised; we fixed one
# value of that field and never audited the rest.
#
# Rewriting çocuk->unknown at INFERENCE ONLY, no retraining, already measured on DEV:
#     TY-ecomm 12L  0.8633 -> 0.8702   (+0.0069)
#     BERTurk 6ep   0.8324 -> 0.8444   (+0.0120)
# That is a patch: the model still spent training capacity on a misleading feature. Retraining on
# data where the artifact never existed should do at least as well and frees that capacity, so the
# model must lean on age_group/title instead — both of which the audit showed are clean
# (age_group's worst train-vs-test shift is 0.10, versus 0.51 for gender çocuk).
#
# Order = the three live frontier candidates, cheapest decisive one first.
set -u
W=/home/jupyter/trendyol-teknofest-work; E15=$W/asya/s2_exp/e15_student
E23=$W/asya/s2_exp/e23_regime_cuts; PY=$W/.venv-gcp/bin/python; O=$E15/outputs_frozen_v2
DATA=$O/build_oldstore/student_train_genderfix.parquet
S=$O/GENDERFIX_SUMMARY.txt
log(){ echo "[$(date +%H:%M:%S)] $*" | tee -a $S; }
cd $E15; : > $S
while ps -eo cmd | grep -qE "[t]rain_student.py|[e]val_student.py"; do sleep 30; done
log "GENDER-FIX retrain — data: $(basename $DATA)"
run(){ # tag model bs lr epochs note
  local n; n=$(ps -eo cmd | grep -cE "[t]rain_student.py"); [ "$n" -gt 0 ] && { log "GUARD"; exit 1; }
  log "== $1 -- $6"
  if TAG=$1 MODEL=$2 TRUST=1 BS=$3 LR=$4 SEED=42 EPOCHS=$5 ALPHA=1 BETA=0 DATA=$DATA OUT=$O \
     $PY scripts/train_student.py > $O/train_$1.log 2>&1; then
    ATTR_CLEAN=v2 TRUST=1 CKPT=$O/ckpt_$1 OUT=$O $PY scripts/eval_student.py 2>&1 \
      | grep -E "2-boundary|AUC" | sed 's/^/   /' | tee -a $S
    cd $E23 && $PY scripts/fit_regime_cuts.py --probs $O/dev_probs_$1.parquet 2>&1 \
      | grep -E "global|regime" | sed 's/^/   /' | tee -a $S; cd $E15
  else log "!! $1 FAILED"; tail -4 $O/train_$1.log | sed 's/^/   /' | tee -a $S; fi
}
run gf1_tiny2_6ep    ytu-ce-cosmos/turkish-medium-bert-uncased 64 5e-5 6 "speed candidate, ~52 min"
run gf2_berturk_6ep  dbmdz/bert-base-turkish-128k-cased        64 3e-5 6 "balanced candidate, ~86 min"
run gf3_tyecomm_8L   $E15/backbones/tyecomm_8L                 32 2e-5 3 "quality candidate, ~71 min"
log "GENDER-FIX DONE"
