#!/usr/bin/env bash
# Morning chain 2026-08-23: resume scoring (pid-based wait), gates, split (no money), pilot-transfer.
set -uo pipefail
W=/home/jupyter/trendyol-teknofest-work; E17=$W/asya/s2_exp/e17_mine_v5; PY=$W/.venv-gcp/bin/python
S=$E17/outputs/OVERNIGHT_SUMMARY.txt
log(){ echo "[$(date +%H:%M:%S)] $*" | tee -a $S; }
: > $S
log "resuming Asure scoring (mask has $($PY -c "import numpy as np;print(int(np.load('$E17/outputs/asure_scores_v5_feefd2813f20.done.npy').sum()))") rows done)"
cd $E17 && setsid scripts/score_v5.sh > outputs/score_v5_resume.log 2>&1 < /dev/null &
SP=$!
while kill -0 $SP 2>/dev/null; do sleep 20; done
grep -E "SCORE COMPLETE|rows/s" outputs/score_v5_resume.log | tail -2 | tee -a $S
if ! grep -q "SCORE COMPLETE" outputs/score_v5_resume.log; then log "SCORING FAILED — see score_v5_resume.log"; exit 1; fi
log "gates_v5 with scores"
$PY scripts/gates_v5.py > outputs/gates_v5_final_pre.log 2>&1
grep -E "adversarial AUC|BASELINE" outputs/gates_v5_final_pre.log | tee -a $S
grep -A9 "^CHECKS" outputs/gates_v5_final_pre.log | tee -a $S
$PY - <<PYEOF | tee -a $S
import json; g=json.load(open('$E17/outputs/gates_v5.json')); m=g['margin_stub']
print('margin: source', m.get('source'), '| coverage', m['coverage_frac'], '| frac>0', m['selected_overlap']['frac_gt0'], 'vs DEV-I', m['dev_I_reference']['frac_gt0'], '| median', m['selected_overlap']['median'], 'vs', m['dev_I_reference']['median'], '| passed', m.get('passed'))
print('ALL_PASSED', g['all_passed'])
PYEOF
ALL=$($PY -c "import json;print(json.load(open('$E17/outputs/gates_v5.json'))['all_passed'])")
if [ "$ALL" = "True" ]; then
  log "judge split (no money)"
  $PY scripts/judge_v5.py --stage split 2>&1 | grep -E '"(selected|covered_by_store|uncovered|canary|remainder|est_cost_canary|est_cost_remainder)"' | tee -a $S
else
  log "gates NOT passed -> no split"
fi
log "pilot-transfer test (v5 negatives, rule labels, e07 protocol)"
$PY scripts/pilot_v5.py 2>&1 | tail -1 | tee -a $S
O=$E17/outputs/pilotV5 $PY $W/asya/s2_exp/e07_pilot/scripts/train.py > $E17/outputs/pilotV5/train.log 2>&1
$PY -c "import json;d=json.load(open('$E17/outputs/pilotV5/pilot_results.json'));print(json.dumps(d,indent=1))" | tee -a $S
log "reference: v4 clean pilot 0.7708 DEV / DEV-I 0.695 ; matched 0.7358 / 0.605"
log "CHAIN DONE — next step is PAID: judge_v5.py --stage canary (needs explicit go)"
