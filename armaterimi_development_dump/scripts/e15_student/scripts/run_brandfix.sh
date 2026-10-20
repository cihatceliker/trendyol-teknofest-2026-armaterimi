#!/usr/bin/env bash
# BRAND-SINK retrain. `brand == 'genel markalar'` is the same two-export artifact as gender=çocuk:
# in the 3-class training frame it sits at P(relevant)=0.038 against an overall 0.326, and a
# category-MATCHED control on public_test features showed the model really reads it (p_rel is
# 0.077-0.101 below same-category non-GM rows across three checkpoints).
#
# Why this needs a RETRAIN rather than an inference rewrite, unlike gender:
#   gender worked because `unknown` was already a 334,767-row sink at P(semi)=0.572, so folding
#   16,634 rows in moved it 0.572 -> 0.593 and the artifact vanished. Brand has NO sink: zero
#   empty and zero 'unknown' across 56,085 values. Mapping GM->unknown in training would just
#   rename the artifact; mapping it at inference only feeds a token that appears in no training
#   row at all. So we CREATE the sink with 15% brand dropout, then fold GM into it.
# Fallback: ckpt_gf2_berturk_6ep_bf16 stays frozen and packaged. This only ships if it wins.
set -u
W=/home/jupyter/trendyol-teknofest-work; E15=$W/asya/s2_exp/e15_student
E23=$W/asya/s2_exp/e23_regime_cuts; PY=$W/.venv-gcp/bin/python; O=$E15/outputs_frozen_v2
DATA=$O/build_oldstore/student_train_genderfix_brandsink.parquet
S=$O/BRANDFIX_SUMMARY.txt
log(){ echo "[$(date +%H:%M:%S)] $*" | tee -a $S; }
cd $E15; : > $S
while ps -eo cmd | grep -qE "[t]rain_student.py|[e]val_student.py"; do sleep 30; done
log "== bf1_berturk_brandsink (same recipe as the frozen pick, only the brand artifact differs)"
if TAG=bf1_berturk_brandsink MODEL=dbmdz/bert-base-turkish-128k-cased TRUST=1 BS=64 LR=3e-5 SEED=42 \
   EPOCHS=6 ALPHA=1 BETA=0 DATA=$DATA OUT=$O $PY scripts/train_student.py > $O/train_bf1.log 2>&1; then
  ATTR_CLEAN=v2 TRUST=1 CKPT=$O/ckpt_bf1_berturk_brandsink OUT=$O $PY scripts/eval_student.py 2>&1 \
    | grep -E "2-boundary|AUC" | sed 's/^/   /' | tee -a $S
  cd $E23 && $PY scripts/fit_regime_cuts.py --probs $O/dev_probs_bf1_berturk_brandsink.parquet \
    --emit $O/ckpt_bf1_berturk_brandsink/decision.json 2>&1 | grep -E "global|regime" | sed 's/^/   /' | tee -a $S
else log "!! FAILED"; tail -4 $O/train_bf1.log | sed 's/^/   /' | tee -a $S; fi
log "BRANDFIX DONE"
