#!/usr/bin/env bash
# OVERNIGHT 2026-08-27 -> 28.  Runs after u1 (128k-uncased) finishes.
#
# WHY THIS ORDER. Every comparison made today was a SINGLE SEED against a SINGLE-SEED baseline
# (0.8546), and every delta we measured -- brand 5% -0.0041, brand 10% +0.0008, glue -0.0002 --
# is the same size as BERTurk's seed spread. So the highest-value GPU hours tonight are not more
# variants: they are SEEDS OF THE BASELINE, because they retroactively tell us whether anything
# we saw today was real. Two extra seeds turn "0.8546" into "0.8546 +/- s", and every one of
# today's runs becomes interpretable.
#
# Then the grid, in evidence order:
#   9 epochs   -- 3->6 epochs gave a MEASURED +0.0056 (BERTurk 0.8372 -> 0.8428 pre-genderfix).
#                 That is the only hyperparameter with a demonstrated gradient. Untested whether
#                 it has plateaued.
#   LR 2e-5    -- lower LR with the longer schedule; standard pairing with more epochs.
#   LR 5e-5    -- the other direction, to bracket rather than assume 3e-5 sits on a slope.
#
# BACKBONE: whichever of cased/uncased won. u1 must beat the baseline by more than the seed noise
# we are about to measure, so the bar is +0.002; below that the packaged cased model stays.
set -u
W=/home/jupyter/trendyol-teknofest-work; E15=$W/asya/s2_exp/e15_student
E23=$W/asya/s2_exp/e23_regime_cuts; PY=$W/.venv-gcp/bin/python; O=$E15/outputs_frozen_v2
DATA=$O/build_oldstore/student_train_genderfix.parquet
S=$O/NIGHT2_SUMMARY.txt
BASE=0.8546
log(){ echo "[$(date +%H:%M:%S)] $*" | tee -a $S; }
cd $E15; : > $S

log "waiting for u1 (uncased A/B) to finish"
while ps -eo cmd | grep -qE "[t]rain_student.py|[e]val_student.py|[r]un_uncased.sh"; do sleep 60; done
sleep 20

# ---- decide the backbone from u1's measured result -------------------------------------------
U=$(grep -oE "2-boundary OOF 0\.[0-9]+" $O/UNCASED_SUMMARY.txt 2>/dev/null | tail -1 | grep -oE "0\.[0-9]+")
MODEL=dbmdz/bert-base-turkish-128k-cased; TAGB=cased
if [ -n "${U:-}" ]; then
  if awk -v u="$U" -v b="$BASE" 'BEGIN{exit !(u > b+0.002)}'; then
    MODEL=dbmdz/bert-base-turkish-128k-uncased; TAGB=uncased
    log "u1 = $U  vs baseline $BASE  -> UNCASED WINS by >0.002, using it for the grid"
  else
    log "u1 = $U  vs baseline $BASE  -> inside noise, keeping CASED (the packaged model)"
  fi
else
  log "u1 result unreadable; defaulting to CASED"
fi
log "backbone for tonight: $MODEL"
log ""
log "  run                        DEV OOF   vs $BASE"

run(){ # tag seed epochs lr note [extra-env]
  local n; n=$(ps -eo cmd | grep -cE "[t]rain_student.py"); [ "$n" -gt 0 ] && { log "GUARD: training alive"; exit 1; }
  local T="n2_${TAGB}_$1"
  if env TAG=$T MODEL=$MODEL TRUST=1 BS=64 LR=$4 SEED=$2 EPOCHS=$3 ALPHA=1 BETA=0 DATA=$DATA OUT=$O \
     ${6:-IGNORE=1} $PY scripts/train_student.py > $O/train_$T.log 2>&1; then
    local line=$(ATTR_CLEAN=v2 TRUST=1 CKPT=$O/ckpt_$T OUT=$O $PY scripts/eval_student.py 2>&1 | grep -E "2-boundary" | head -1)
    local f=$(grep -oE "2-boundary OOF 0\.[0-9]+" <<<"$line" | grep -oE "0\.[0-9]+")
    local d=$(awk -v a="$f" -v b="$BASE" 'BEGIN{printf "%+.4f", a-b}')
    log "  $(printf '%-26s' "$5")  $f   $d"
    cd $E23 && $PY scripts/fit_regime_cuts.py --probs $O/dev_probs_$T.parquet \
      --emit $O/ckpt_$T/decision.json >/dev/null 2>&1; cd $E15
  else log "  $(printf '%-26s' "$5")  FAILED"; tail -3 $O/train_$T.log | sed 's/^/       /' | tee -a $S; fi
}

# --- the seeds first: they make everything else (and everything from today) interpretable ---
run seed43   43 6 3e-5 "baseline recipe, seed 43"
run seed44   44 6 3e-5 "baseline recipe, seed 44"
# --- then the grid, in order of evidence ---
run ep9      42 9 3e-5 "9 epochs (3->6 gave +0.0056)"
run lr2e5    42 6 2e-5 "LR 2e-5"
# LR 5e-5 dropped in favour of the augmentation with the largest MEASURED deficit. On 1,472 real
# public rows the frozen model flips 33.4% of labels when one query character is deleted and 41.0%
# when two are swapped -- against 1.2% for attribute-order shuffling. Character robustness is the
# hole; the product side already has the invariance.
run typo03   42 6 3e-5 "query typo aug p=0.03" TYPO_P=0.03

log ""
log "READ THIS IN THE MORNING:"
log "  the two seed rows give the spread s. Any delta from today smaller than ~2s is noise:"
log "    brand 5%  -0.0041 | brand 10% +0.0008 | glue -0.0002 | uncased ${U:-?}"
log "  measured invariance deficits (frozen model, 1,472 rows): attr-order 1.2% | attr-drop 1.4%"
log "    | query word order 12.0% | delete 1 char 33.4% | swap 2 chars 41.0%"
log "  the packaged submission is BERTurk-128k-cased 6ep genderfix, public 0.857943456149,"
log "  ~5,030 SPS, and is NOT changed by anything here unless a run clearly beats the spread."
log "NIGHT2 DONE"
