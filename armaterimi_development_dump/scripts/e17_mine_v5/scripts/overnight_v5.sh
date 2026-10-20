#!/usr/bin/env bash
# Overnight chain (2026-08-22/23): everything up to, and NOT including, paid judging.
set -uo pipefail
W=/home/jupyter/trendyol-teknofest-work; E17=$W/asya/s2_exp/e17_mine_v5; PY=$W/.venv-gcp/bin/python
S=$E17/outputs/OVERNIGHT_SUMMARY.txt
log(){ echo "[$(date +%H:%M:%S)] $*" | tee -a $S; }
: > $S
log "waiting for Asure scoring (pid $(pgrep -f score_asure.py | head -1))"
while pgrep -f score_asure.py >/dev/null; do sleep 30; done
grep -E "SCORE COMPLETE|rows/s" $E17/outputs/score_v5.log | tail -2 | tee -a $S
log "gates_v5 with scores"
cd $E17 && $PY scripts/gates_v5.py > outputs/gates_v5_final_pre.log 2>&1
grep -A12 "CHECKS" outputs/gates_v5_final_pre.log | tee -a $S
grep -E "margin|frac_gt0" outputs/gates_v5_final_pre.log | tee -a $S
ALL=$($PY -c "import json;print(json.load(open('$E17/outputs/gates_v5.json'))['all_passed'])")
log "ALL_PASSED=$ALL"
if [ "$ALL" = "True" ]; then
  log "judge split (no money)"
  $PY scripts/judge_v5.py --stage split 2>&1 | grep -E "canary|remainder|covered|est_cost|uncovered" | tee -a $S
else
  log "gates NOT passed -> no split; see outputs/gates_v5.json"
fi
log "pilot-transfer test (v5 negatives, rule labels, e07 protocol)"
$PY scripts/pilot_v5.py 2>&1 | tail -1 | tee -a $S
O=$E17/outputs/pilotV5 $PY $W/asya/s2_exp/e07_pilot/scripts/train.py > $E17/outputs/pilotV5/train.log 2>&1
$PY -c "import json;d=json.load(open('$E17/outputs/pilotV5/pilot_results.json'));print(json.dumps(d,indent=1))" | tee -a $S
log "reference: v4 clean pilot 0.7708 DEV / DEV-I 0.695 ; matched 0.7358 / 0.605"
log "OVERNIGHT DONE — next step is PAID: judge_v5.py --stage canary (needs explicit go)"
