#!/usr/bin/env bash
set -uo pipefail
W=/home/jupyter/trendyol-teknofest-work; E17=$W/asya/s2_exp/e17_mine_v5; PY=$W/.venv-gcp/bin/python
S=$E17/outputs/CANARY_SUMMARY.txt; : > $S
log(){ echo "[$(date +%H:%M:%S)] $*" | tee -a $S; }
cd $E17
log "gates (pre-judge semantics)"; $PY scripts/gates_v5.py > outputs/gates_v5_prejudge.log 2>&1
grep -E "ALL_PASSED" outputs/gates_v5_prejudge.log | tee -a $S
log "split"; $PY scripts/judge_v5.py --stage split 2>&1 | grep -E '"(selected|covered_by_store|uncovered|canary|remainder|est_cost_canary|est_cost_remainder)"' | tee -a $S
log "CANARY (paid) conc 10 cap 9.6"; $PY scripts/judge_v5.py --stage canary --conc 10 --cap 9.6 2>&1 | tee outputs/judge_canary.log | grep -E "coverage|keep_rate|passed|CANARY|labelled|cost|REFUSE" | tee -a $S
log "CANARY STAGE DONE"
