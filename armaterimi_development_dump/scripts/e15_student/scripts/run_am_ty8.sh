#!/usr/bin/env bash
# MORNING — tighten the two numbers the ship decision now hangs on.
#
# TY-ecomm 8L came out of the night as the best quality/speed compromise (0.8530 @ 4,112 bare),
# but its two seeds were 0.8530 / 0.8436 -- a 0.0094 spread, nearly the whole 0.012 noise floor.
# 0.8530 was the lucky draw; the recipe mean is 0.8483 on two samples and that estimate is soft.
# A third seed is not optional before this becomes the shipped model.
# Also: TY-8L at MAXLEN=128 was never measured (12L lost 0.0066, 10L lost 0.0142 -- the cost grows
# as depth falls, so 8L needs its own number rather than an interpolation).
set -u
W=/home/jupyter/trendyol-teknofest-work; E15=$W/asya/s2_exp/e15_student; E12=$W/asya/s2_exp/e12_lean_infer
E23=$W/asya/s2_exp/e23_regime_cuts; PY=$W/.venv-gcp/bin/python; O=$E15/outputs_frozen_v2
SC=/var/tmp/claude-1000/-home-jupyter/483abbaf-6d7f-4b27-838c-53d3d34a821c/scratchpad
DATA=$O/build_oldstore/student_train.parquet
S=$O/AM_SUMMARY.txt
log(){ echo "[$(date +%H:%M:%S)] $*" | tee -a $S; }
cd $E15; : > $S
n=$(ps -eo cmd | grep -cE "[t]rain_student.py"); [ "$n" -gt 0 ] && { log "GUARD: training alive"; exit 1; }

log "== A) TY-8L @ MAXLEN=128, cuts REFIT at 128"
MAXLEN=128 ATTR_CLEAN=v2 TRUST=1 CKPT=$O/ckpt_l8b_tyecomm_8L OUT=$O TAG=l8b_tyecomm_8L_ml128 \
  $PY scripts/eval_student.py 2>&1 | grep -E "2-boundary" | sed 's/^/   /' | tee -a $S
for r in 1 2; do
  out=$(cd $E12 && BATCH_ROWS=256 TOKEN_BUDGET=65536 MAXLEN=128 timeout 900 $PY scripts/serve_gte.py \
          $SC/synth90k.csv $SC/am_ty8_128.csv --model-dir=$O/ckpt_l8b_tyecomm_8L 2>&1)
  s=$(grep "^STAGE" <<<"$out" | sed 's/.*sps=//')
  [ -z "$s" ] && { log "   !! bench FAILED:"; tail -3 <<<"$out" | sed 's/^/      /' | tee -a $S; break; }
  log "   BENCH TY-8L maxlen=128 rep$r  $s SPS"
done

log "== B) TY-8L seed 44 (third sample of the recipe mean)"
if TAG=l8b_seed44 MODEL=$E15/backbones/tyecomm_8L TRUST=1 BS=32 LR=2e-5 SEED=44 EPOCHS=3 \
   ALPHA=1 BETA=0 DATA=$DATA OUT=$O $PY scripts/train_student.py > $O/train_l8b_seed44.log 2>&1; then
  ATTR_CLEAN=v2 TRUST=1 CKPT=$O/ckpt_l8b_seed44 OUT=$O $PY scripts/eval_student.py 2>&1 \
    | grep -E "2-boundary|AUC" | sed 's/^/   /' | tee -a $S
  cd $E23 && $PY scripts/fit_regime_cuts.py --probs $O/dev_probs_l8b_seed44.parquet 2>&1 \
    | grep -E "global|regime" | sed 's/^/   /' | tee -a $S; cd $E15
else log "!! seed44 FAILED"; tail -4 $O/train_l8b_seed44.log | sed 's/^/   /' | tee -a $S; fi
log "AM DONE"
