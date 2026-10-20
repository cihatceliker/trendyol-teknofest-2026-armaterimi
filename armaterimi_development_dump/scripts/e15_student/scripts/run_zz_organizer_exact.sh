#!/usr/bin/env bash
# ZZ: measure EXACTLY what the organizer measures.
#  1. the CONTAINER (jury flags) on a real 90,000-row input, timed run_solution.sh -> file written.
#     Everything so far extrapolated container overhead from a 7.5k run; this is the real SPS.
#  2. deliverable-style F1: cuts fitted on DEV, container run on public_test.csv, scored with the
#     ORGANIZER's evaluate.py, reported on the untouched HOLDOUT half (the geometry of the private set).
set -u
W=/home/jupyter/trendyol-teknofest-work; E15=$W/asya/s2_exp/e15_student; E19=$W/asya/s2_exp/e19_docker_ty
PY=$W/.venv-gcp/bin/python; O=$E15/outputs_frozen_v2
SC=/var/tmp/claude-1000/-home-jupyter/483abbaf-6d7f-4b27-838c-53d3d34a821c/scratchpad
S=$O/ZZ_SUMMARY.txt; : > $S
log(){ echo "[$(date +%H:%M:%S)] $*" | tee -a $S; }
log "waiting for Z"
until grep -q "Z DONE" $O/Z_SUMMARY.txt 2>/dev/null; do
  ps -eo cmd | grep -qE "[r]un_x_queue|[r]un_y_maxlen|[r]un_z_10L" || break; sleep 60; done
until [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)" -lt 1500 ]; do sleep 30; done
mkdir -p $SC/jury90k
log "=== TRUE 90k CONTAINER RUN (jury flags, timed exactly as the organizer times it)"
T0=$(date +%s.%N)
docker run --rm --gpus all --read-only --cap-drop ALL --network none \
  --tmpfs /tmp:rw,noexec,nosuid,size=8g,mode=1777 --user 1000:1001 \
  -v $SC/synth90k.csv:/input/input.csv:ro -v $SC/jury90k:/output \
  armaterimi_solution /input/input.csv /output/submission.csv > $SC/jury90k/run.log 2>&1
RC=$?; T1=$(date +%s.%N)
WALL=$($PY -c "print(f'{$T1-$T0:.2f}')")
log "container exit=$RC  WALL=${WALL}s for 90,000 rows  -> SPS = $($PY -c "print(f'{90000/($T1-$T0):.0f}')")"
grep STAGE $SC/jury90k/run.log | sed 's/^/   /' | cut -c1-150 | tee -a $S
log "   rows written: $(($(wc -l < $SC/jury90k/submission.csv)-1))"
log "=== deliverable F1 the organizer way (their evaluate.py, HOLDOUT half)"
cd $E19 && bash test_like_jury.sh 2>&1 | grep -E "macro_f1|PASS" | sed 's/^/   /' | tee -a $S
$PY - <<'PY' 2>&1 | tee -a $S
import sys, pandas as pd
sys.path.insert(0,'/home/jupyter/trendyol-teknofest-work/asya/s2_exp/common'); import harness as H
from sklearn.metrics import f1_score
W='/home/jupyter/trendyol-teknofest-work'
sub=pd.read_csv(f'{W}/asya/s2_exp/e19_docker_ty/jury_out/submission.csv')
sol=pd.read_csv(f'{W}/stage2_data/public_solution.csv')
m=sub.merge(sol,on='row_id',suffixes=('_p','_t')); L=['irrelevant','semi-relevant','relevant']
pc=[c for c in m.columns if c.endswith('_p')][0]; tc=[c for c in m.columns if c.endswith('_t')][0]
h=set(H.load_public_split('holdout').row_id); hh=m[m.row_id.isin(h)]
print(f"   HOLDOUT half (untouched, n={len(hh):,}): {f1_score(hh[tc],hh[pc],average='macro',labels=L):.4f}  <- the deliverable number")
PY
log "ZZ DONE"
