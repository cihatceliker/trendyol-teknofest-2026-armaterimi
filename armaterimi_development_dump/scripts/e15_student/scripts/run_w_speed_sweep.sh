#!/usr/bin/env bash
# W: squeeze inference on BOTH finalists. Batching and MAXLEN only; every config byte-compared.
# MAXLEN 128 was measured accuracy-neutral on DEV (2b 0.8521 / scalar 0.8569 vs 256's 0.8553/0.8543).
set -u
W=/home/jupyter/trendyol-teknofest-work; E12=$W/asya/s2_exp/e12_lean_infer; E15=$W/asya/s2_exp/e15_student
PY=$W/.venv-gcp/bin/python; O=$E15/outputs_frozen_v2
SC=/var/tmp/claude-1000/-home-jupyter/483abbaf-6d7f-4b27-838c-53d3d34a821c/scratchpad
S=$O/W_SUMMARY.txt; : > $S
log(){ echo "[$(date +%H:%M:%S)] $*" | tee -a $S; }
log "waiting for GPU"
until ! ps -eo cmd | grep -q "[t]rain_student.py"; do sleep 60; done
until [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)" -lt 1500 ]; do sleep 30; done
cd $E12
one(){ # tag script ckpt rows budget maxlen
  o=$SC/w_$1_$4_$6.csv
  r=$(BATCH_ROWS=$4 TOKEN_BUDGET=$5 MAXLEN=$6 $PY scripts/$2 $SC/synth90k.csv $o --model-dir=$3 2>&1 | grep STAGE)
  log "  $1 rows=$4 budget=$5 maxlen=$6 :: $(echo $r | sed 's/.*infer=/infer=/' | cut -c1-60)"
}
log "=== TY-ecomm (serve_gte)"
for cfg in "128 32768 256" "256 65536 256" "512 131072 256" "256 65536 192" "256 65536 128"; do
  set -- $cfg; one ty serve_gte.py $O/ckpt_r1_oldstore $1 $2 $3; done
log "=== BERTurk (lean_infer)"
for cfg in "128 32768 256" "256 65536 256" "512 131072 256" "256 65536 128"; do
  set -- $cfg; one bt lean_infer.py $O/ckpt_c1_berturk_3ep $1 $2 $3; done
log "W DONE"
