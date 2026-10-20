#!/usr/bin/env bash
# T3 — two candidates that both target the same gap: tiny2 (8L/512, 0.8218) sits 0.0155 under
# BERTurk (12L/768, 0.8373) while being much faster. Can we keep the speed and close the gap?
#
#   t3a  turkish-medium GROWN to 10L/512 (progressive stacking: the top 2 pretrained layers are
#        duplicated). No pretrained 10L/512 Turkish BERT exists, so we build one. The drift probe
#        (expand_layers.py --probe) showed BERT's post-LayerNorm means NO init preserves the
#        function -- dup/zero/rand all land in the same range -- so we use the established method
#        (Gong et al. 2019) rather than burning a second 33-min slot on a rand ablation.
#   t3b  ytu-ce-cosmos/turkish-base-bert-uncased, 12L/768, 32k vocab. Same FLOPs as BERTurk
#        (vocab affects only the embedding lookup, which is free) but UNCASED -- and normalize.py
#        already lowercases every input, so BERTurk-cased spends capacity on case we destroyed.
#        Untested and the cheapest shot at beating 0.8373 at BERTurk speed.
#
# Identical recipe to the whole tiny chain: 813,269 rows, v2 text, 3 epochs, BS 64, MAXLEN 256.
set -u
W=/home/jupyter/trendyol-teknofest-work; E15=$W/asya/s2_exp/e15_student; E12=$W/asya/s2_exp/e12_lean_infer
E23=$W/asya/s2_exp/e23_regime_cuts; PY=$W/.venv-gcp/bin/python; O=$E15/outputs_frozen_v2
SC=/var/tmp/claude-1000/-home-jupyter/483abbaf-6d7f-4b27-838c-53d3d34a821c/scratchpad
S=$O/GROW_SUMMARY.txt
log(){ echo "[$(date +%H:%M:%S)] $*" | tee -a $S; }
cd $E15
: > $S
log "waiting for the tiny chain AND the clean bench to finish"
while ps -eo cmd | grep -qE "[r]un_t2_tiny.sh|[b]ench_tiny_clean.sh|[t]rain_student.py|[e]val_student.py"; do sleep 30; done
sleep 20
log "GPU free — starting T3"

run(){ # tag model lr
  n=$(ps -eo cmd | grep -cE "[t]rain_student.py"); [ "$n" -gt 0 ] && { log "GUARD: a training is already alive"; exit 1; }
  log "== $1 ($2)"
  if TAG=$1 MODEL=$2 BS=64 LR=$3 EPOCHS=3 ALPHA=1 BETA=0 DATA=$O/build_oldstore/student_train.parquet \
     OUT=$O $PY scripts/train_student.py > $O/train_$1.log 2>&1; then
    ATTR_CLEAN=v2 CKPT=$O/ckpt_$1 OUT=$O $PY scripts/eval_student.py 2>&1 | grep -E "2-boundary" | sed 's/^/   /' | tee -a $S
    cd $E23 && $PY scripts/fit_regime_cuts.py --probs $O/dev_probs_$1.parquet 2>&1 | grep -E "global|regime" | sed 's/^/   /' | tee -a $S; cd $E15
    # bench, and say so LOUDLY when it dies (the tiny chain silently swallowed tiny1's KeyError)
    b=$(cd $E12 && BATCH_ROWS=256 TOKEN_BUDGET=65536 $PY scripts/lean_infer.py $SC/synth90k.csv $SC/grow_$1.csv --model-dir=$O/ckpt_$1 2>&1)
    if grep -q "^STAGE" <<<"$b"; then grep "^STAGE" <<<"$b" | sed 's/.*infer=/   90k infer=/' | cut -c1-80 | tee -a $S
    else log "   !! 90k bench FAILED for $1:"; tail -3 <<<"$b" | sed 's/^/      /' | tee -a $S; fi
  else log "!! $1 TRAINING FAILED"; tail -3 $O/train_$1.log | tee -a $S; fi; }

run t3a_medium_grown_10L $E15/models_local/turkish-medium-10L 5e-5
run t3b_base_uncased_12L ytu-ce-cosmos/turkish-base-bert-uncased 3e-5
log "T3 DONE"
