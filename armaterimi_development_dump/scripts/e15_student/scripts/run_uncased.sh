#!/usr/bin/env bash
# BERTurk 128k UNCASED — straight A/B against the frozen pick. EVERYTHING identical except the
# backbone: same data (student_train_genderfix.parquet), same 6 epochs, BS 64, LR 3e-5, seed 42,
# max_len 256, no glue augmentation. Only MODEL changes.
#
# Rationale: the organizer's data contains ZERO uppercase characters in any of the seven fields,
# so we have been fine-tuning a CASED checkpoint on entirely lowercase text. An uncased backbone
# was pretrained on text that looks like ours. The tokenisation comparison mildly favours cased
# (81.6 vs 83.3 mean tokens on our real 90k, i.e. uncased costs ~2% more compute), but 2% of
# inference is not a reason to skip testing a better-matched pretraining distribution.
#
# Never previously tested: our original backbone probe compared VOCAB SIZES
# (BERTurk-128k 0.7358 > BERTurk-cased-32k 0.7249), never cased-vs-uncased at 128k.
#
# Baseline to beat: 0.8546 DEV OOF / 0.8579 public. Fallback stays frozen and packaged.
set -u
W=/home/jupyter/trendyol-teknofest-work; E15=$W/asya/s2_exp/e15_student
E23=$W/asya/s2_exp/e23_regime_cuts; PY=$W/.venv-gcp/bin/python; O=$E15/outputs_frozen_v2
DATA=$O/build_oldstore/student_train_genderfix.parquet
S=$O/UNCASED_SUMMARY.txt
log(){ echo "[$(date +%H:%M:%S)] $*" | tee -a $S; }
cd $E15; : > $S
log "waiting for the glue run to finish"
while ps -eo cmd | grep -qE "[t]rain_student.py|[e]val_student.py|[r]un_glue.sh"; do sleep 60; done
sleep 15
log "== u1_berturk_uncased : dbmdz/bert-base-turkish-128k-uncased, otherwise identical to the frozen pick"
if TAG=u1_berturk_uncased MODEL=dbmdz/bert-base-turkish-128k-uncased TRUST=1 BS=64 LR=3e-5 SEED=42 \
   EPOCHS=6 ALPHA=1 BETA=0 DATA=$DATA OUT=$O $PY scripts/train_student.py > $O/train_u1.log 2>&1; then
  ATTR_CLEAN=v2 TRUST=1 CKPT=$O/ckpt_u1_berturk_uncased OUT=$O $PY scripts/eval_student.py 2>&1 \
    | grep -E "2-boundary|AUC" | sed 's/^/   /' | tee -a $S
  cd $E23 && $PY scripts/fit_regime_cuts.py --probs $O/dev_probs_u1_berturk_uncased.parquet \
    --emit $O/ckpt_u1_berturk_uncased/decision.json 2>&1 | grep -E "global|regime" | sed 's/^/   /' | tee -a $S
  cd $E15
  log "   BASELINE: frozen BERTurk-128k-CASED = 0.8546 DEV OOF / 0.8579 public / ~5,030 SPS"
else log "!! FAILED"; tail -5 $O/train_u1.log | sed 's/^/   /' | tee -a $S; fi
log "UNCASED DONE"
