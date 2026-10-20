#!/usr/bin/env bash
# EVENING QUEUE — runs after b05. Two questions, both worth the GPU either way b05 lands.
#
# E1  brand dropout DOSE-RESPONSE at 10%. At 5% the sink is still 32% GM and P(rel) only recovers
#     to 0.241 (overall 0.326); at 10% it is 19% GM and 0.279. If 5% helped, this says whether it
#     was under-dosed. If 5% did nothing, this says whether the dose was the reason.
# E2  SECOND SEED of the frozen pick itself. ckpt_gf2_berturk_6ep is a SINGLE seed at 0.8546 DEV
#     OOF, and it is the number every comparison today is measured against. BERTurk at 3 epochs
#     replicated to 0.0003, so a second 6-epoch seed either confirms the baseline is that tight or
#     tells us the deltas we have been reading are inside seed noise. Useful whatever b05 does.
set -u
W=/home/jupyter/trendyol-teknofest-work; E15=$W/asya/s2_exp/e15_student
E23=$W/asya/s2_exp/e23_regime_cuts; PY=$W/.venv-gcp/bin/python; O=$E15/outputs_frozen_v2
S=$O/EVENING_SUMMARY.txt
log(){ echo "[$(date +%H:%M:%S)] $*" | tee -a $S; }
cd $E15; : > $S
log "waiting for b05 to finish"
while ps -eo cmd | grep -qE "[t]rain_student.py|[e]val_student.py|[r]un_brand05.sh"; do sleep 60; done
sleep 15
run(){ # tag data seed t5 note
  local n; n=$(ps -eo cmd | grep -cE "[t]rain_student.py"); [ "$n" -gt 0 ] && { log "GUARD"; exit 1; }
  log "== $1 -- $5"
  if TAG=$1 MODEL=dbmdz/bert-base-turkish-128k-cased TRUST=1 BS=64 LR=3e-5 SEED=$3 EPOCHS=6 \
     ALPHA=1 BETA=0 DATA=$2 OUT=$O $PY scripts/train_student.py > $O/train_$1.log 2>&1; then
    T5_BRAND_SINK=$4 ATTR_CLEAN=v2 TRUST=1 CKPT=$O/ckpt_$1 OUT=$O $PY scripts/eval_student.py 2>&1 \
      | grep -E "2-boundary|AUC" | sed 's/^/   /' | tee -a $S
    cd $E23 && $PY scripts/fit_regime_cuts.py --probs $O/dev_probs_$1.parquet \
      --emit $O/ckpt_$1/decision.json 2>&1 | grep -E "global|regime" | sed 's/^/   /' | tee -a $S; cd $E15
    [ "$4" = "1" ] && $PY -c "
import json,pathlib
p=pathlib.Path('$O/ckpt_$1/decision.json'); d=json.loads(p.read_text()); d['brand_sink']=True
p.write_text(json.dumps(d,indent=1,ensure_ascii=False)+chr(10))
print('   decision.json: brand_sink=True')" | tee -a $S
  else log "!! $1 FAILED"; tail -4 $O/train_$1.log | sed 's/^/   /' | tee -a $S; fi
}
run e1_brand10      $O/build_oldstore/student_train_gf_brand10.parquet     42 1 "brand dropout 10% (dose-response vs the 5% run)"
run e2_frozen_seed43 $O/build_oldstore/student_train_genderfix.parquet     43 0 "second seed of the FROZEN pick — tightens the 0.8546 baseline"
log "BASELINES: frozen BERTurk 0.8546 DEV OOF / 0.8579 public / 4,506 SPS"
log "EVENING DONE"
