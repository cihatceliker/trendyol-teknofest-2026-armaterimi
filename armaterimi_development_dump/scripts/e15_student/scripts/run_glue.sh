#!/usr/bin/env bash
# SPACE-REMOVAL AUGMENTATION. Turkish shoppers type compounds glued together; the model has an
# 11-point hole there.
#
# MEASURED on the shipped model, public_test:
#   glued queries (no space, >=11 chars)  n=39     74.4% accuracy
#   everything else                       n=7,461  85.8%
#   misses: minibuzdolabı->irrelevant (truth relevant), celikbileklik->semi (truth relevant),
#           bebekislakmendil, yuztemizleyici, plastation4, stadivaryus
# The slice is 0.52% of rows (~468 of the private 90k), so the DIRECTLY measurable ceiling is only
# ~+0.0006 macro-F1. The reason to do it anyway is that this is a general robustness property, not
# a 39-row patch: 77% of training queries are 2-3 words and Turkish users glue them freely.
#
# p=3% on multi-word queries only, RESAMPLED EVERY EPOCH inside the trainer (GLUE_P env).
# Gluing once into the data file would be a fixed corrupted subset: the same 2.7% of rows glued in
# all 6 epochs, memorisable, and 97.3% of rows never glued even once. Redrawing per epoch means a
# row is seen BOTH glued and spaced across training, which is what teaches the invariance rather
# than the strings. Measured: ~24.4k rows glued per epoch, 16.7% glued at least once, 0% glued in
# every epoch, epoch-to-epoch overlap 728 rows (= chance). Deterministic in SEED.
# TRAIN-ONLY: inference does nothing, so there is no serving change and no train/serve divergence.
# Fallback: ckpt_gf2_berturk_6ep_bf16 stays frozen and packaged; this ships only if it wins.
set -u
W=/home/jupyter/trendyol-teknofest-work; E15=$W/asya/s2_exp/e15_student
E23=$W/asya/s2_exp/e23_regime_cuts; PY=$W/.venv-gcp/bin/python; O=$E15/outputs_frozen_v2
DATA=$O/build_oldstore/student_train_genderfix.parquet   # clean data; glue happens in the trainer
S=$O/GLUE_SUMMARY.txt
log(){ echo "[$(date +%H:%M:%S)] $*" | tee -a $S; }
cd $E15; : > $S
while ps -eo cmd | grep -qE "[t]rain_student.py|[e]val_student.py"; do sleep 30; done
log "== g1_berturk_glue03 : identical recipe to the frozen pick, only 3% query-glue augmentation"
if TAG=g1_berturk_glue03 MODEL=dbmdz/bert-base-turkish-128k-cased TRUST=1 BS=64 LR=3e-5 SEED=42 \
   EPOCHS=6 ALPHA=1 BETA=0 GLUE_P=0.03 DATA=$DATA OUT=$O $PY scripts/train_student.py > $O/train_g1.log 2>&1; then
  ATTR_CLEAN=v2 TRUST=1 CKPT=$O/ckpt_g1_berturk_glue03 OUT=$O $PY scripts/eval_student.py 2>&1 \
    | grep -E "2-boundary|AUC" | sed 's/^/   /' | tee -a $S
  cd $E23 && $PY scripts/fit_regime_cuts.py --probs $O/dev_probs_g1_berturk_glue03.parquet \
    --emit $O/ckpt_g1_berturk_glue03/decision.json 2>&1 | grep -E "global|regime" | sed 's/^/   /' | tee -a $S
  cd $E15
  log "   BASELINE: frozen BERTurk 0.8546 DEV OOF / 0.8579 public / 74.4% on the glued slice"
else log "!! FAILED"; tail -4 $O/train_g1.log | sed 's/^/   /' | tee -a $S; fi
log "GLUE DONE"
