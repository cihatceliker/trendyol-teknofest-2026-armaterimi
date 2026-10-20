#!/usr/bin/env bash
# NIGHT QUEUE — fill in the quality/speed frontier with COMPARABLE numbers, so tomorrow's ship
# decision is made on measurements instead of extrapolations.
#
# What we know going in (DEV OOF / in-container 90k SPS):
#   TY-ecomm 12L 0.8587 / 2,967      TY-ecomm 10L 0.8571 / 3,340
#   BERTurk  12L 0.8373 / 5,073      tiny2 8L/512 0.8218 / ~7,250 (EXTRAPOLATED, not measured)
# The gap in the middle is TY-ecomm at 8 and 6 layers. Backbones were prepared long ago
# (backbones/tyecomm_{8,6}L); l8 was trained once but on the OLD g2_extras recipe (0.8393) and
# l6 was killed before it finished. Both are retrained here on the CHAMPION recipe so every
# number in the table comes from one data pipeline.
#
# Depth is nearly free on this architecture -- 12L->10L cost 0.0016 -- so 8L and 6L are the most
# likely Pareto winners we have not measured.
set -u
W=/home/jupyter/trendyol-teknofest-work; E15=$W/asya/s2_exp/e15_student; E12=$W/asya/s2_exp/e12_lean_infer
E23=$W/asya/s2_exp/e23_regime_cuts; PY=$W/.venv-gcp/bin/python; O=$E15/outputs_frozen_v2
SC=/var/tmp/claude-1000/-home-jupyter/483abbaf-6d7f-4b27-838c-53d3d34a821c/scratchpad
DATA=$O/build_oldstore/student_train.parquet      # the CHAMPION recipe, 813,269 rows
S=$O/NIGHT_SUMMARY.txt
log(){ echo "[$(date +%H:%M:%S)] $*" | tee -a $S; }
cd $E15
: > $S
log "NIGHT QUEUE — waiting for T3 (grown 10L + turkish-base) to finish"
while ps -eo cmd | grep -qE "[r]un_t3_grow.sh|[t]rain_student.py|[e]val_student.py"; do sleep 60; done
sleep 20
log "T3 done — starting the frontier fill"

# ---------------------------------------------------------------- helpers
port_for(){ grep -q 'NewForSequence' "$1/config.json" 2>/dev/null && echo scripts/serve_gte.py || echo scripts/lean_infer.py; }

bench(){ # label ckpt maxlen -> prints best-of-2 SPS
  local lbl=$1 dir=$2 ml=${3:-256} best=0 reps="" sc; sc=$(port_for "$dir")
  [ -d "$dir" ] || { log "   SKIP bench $lbl (no ckpt)"; return; }
  for r in 1 2; do
    local out sps
    out=$(cd $E12 && BATCH_ROWS=256 TOKEN_BUDGET=65536 MAXLEN=$ml timeout 900 $PY $sc \
            $SC/synth90k.csv $SC/n_${lbl}_${ml}.csv --model-dir=$dir 2>&1)
    sps=$(grep "^STAGE" <<<"$out" | sed 's/.*sps=//')
    if [ -z "$sps" ]; then log "   !! bench $lbl (maxlen $ml) FAILED:"; tail -3 <<<"$out" | sed 's/^/      /' | tee -a $S; return; fi
    reps="$reps $sps"; awk -v a="$sps" -v b="$best" 'BEGIN{exit !(a>b)}' && best=$sps
  done
  log "   BENCH $(printf '%-22s' "$lbl") maxlen=$ml  best ${best} SPS  (reps:${reps})"
}

train_eval(){ # tag backbone bs lr
  local n; n=$(ps -eo cmd | grep -cE "[t]rain_student.py"); [ "$n" -gt 0 ] && { log "GUARD: training alive"; exit 1; }
  log "== $1"
  if TAG=$1 MODEL=$2 TRUST=1 BS=$3 LR=$4 EPOCHS=3 ALPHA=1 BETA=0 DATA=$DATA OUT=$O \
     $PY scripts/train_student.py > $O/train_$1.log 2>&1; then
    ATTR_CLEAN=v2 TRUST=1 CKPT=$O/ckpt_$1 OUT=$O $PY scripts/eval_student.py 2>&1 \
      | grep -E "2-boundary|AUC" | sed 's/^/   /' | tee -a $S
    cd $E23 && $PY scripts/fit_regime_cuts.py --probs $O/dev_probs_$1.parquet 2>&1 \
      | grep -E "global|regime" | sed 's/^/   /' | tee -a $S; cd $E15
    bench "$1" "$O/ckpt_$1" 256
  else log "!! $1 TRAINING FAILED"; tail -4 $O/train_$1.log | sed 's/^/   /' | tee -a $S; fi
}

# ---------------------------------------------------------------- N1/N2: the missing TY depths
train_eval l8b_tyecomm_8L $E15/backbones/tyecomm_8L 32 2e-5
train_eval l6b_tyecomm_6L $E15/backbones/tyecomm_6L 32 2e-5

# ---------------------------------------------------------------- N3: one comparable speed table
log "== N3 clean bench, every finalist, identical conditions, maxlen 256"
bench tyecomm_12L        $O/ckpt_r1_oldstore
bench tyecomm_10L        $O/ckpt_x1_tyecomm_10L
bench tyecomm_8L         $O/ckpt_l8b_tyecomm_8L
bench tyecomm_6L         $O/ckpt_l6b_tyecomm_6L
bench berturk_12L        $O/ckpt_v1_berturk_full
bench tiny2_medium_8L    $O/ckpt_tiny2_medium_8L512
bench t3a_grown_10L      $O/ckpt_t3a_medium_grown_10L
bench t3b_base_12L       $O/ckpt_t3b_base_uncased_12L

# ---------------------------------------------------------------- N4: is MAXLEN=128 bankable?
# LOG.md: batch 256 + MAXLEN 128 measured +7.7% SPS, but on cuts fitted at 256 -- so the speed was
# never bankable. Here the cuts are REFIT at 128 (eval_student.py refits inside its own folds), so
# the accuracy number is honest. If it holds within the 0.012 noise floor it is free speed for
# whichever model we ship.
log "== N4 MAXLEN=128 -- accuracy WITH CUTS REFIT AT 128, then speed"
for pair in "tyecomm_12L:ckpt_r1_oldstore" "tyecomm_10L:ckpt_x1_tyecomm_10L" \
            "berturk_12L:ckpt_v1_berturk_full" "tiny2_medium_8L:ckpt_tiny2_medium_8L512" \
            "t3a_grown_10L:ckpt_t3a_medium_grown_10L"; do
  lbl=${pair%%:*}; ck=$O/${pair##*:}
  [ -d "$ck" ] || { log "   SKIP $lbl (no ckpt)"; continue; }
  log "   -- $lbl @128"
  # TAG must be set explicitly: eval_student.py defaults TAG to the ckpt basename, so without
  # this the 128 run would OVERWRITE eval_<tag>.json and dev_probs_<tag>.parquet from the 256 run
  # -- silently destroying the champion's DEV probabilities.
  tag128="${pair##*:}"; tag128="${tag128#ckpt_}_ml128"
  MAXLEN=128 ATTR_CLEAN=v2 TRUST=1 CKPT=$ck OUT=$O TAG=$tag128 $PY scripts/eval_student.py 2>&1 \
    | grep -E "2-boundary" | sed 's/^/      /' | tee -a $S
  bench "$lbl" "$ck" 128
done

log "NIGHT QUEUE DONE"
