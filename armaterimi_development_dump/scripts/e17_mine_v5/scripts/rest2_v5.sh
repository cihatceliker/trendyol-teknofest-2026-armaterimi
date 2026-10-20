#!/usr/bin/env bash
set -uo pipefail
W=/home/jupyter/trendyol-teknofest-work; E17=$W/asya/s2_exp/e17_mine_v5; PY=$W/.venv-gcp/bin/python
S=$E17/outputs/REST_SUMMARY.txt
log(){ echo "[$(date +%H:%M:%S)] $*" | tee -a $S; }
cd $E17
log "waiting for the top-up run to finish"
while pgrep -f "rest_v5.sh" >/dev/null; do sleep 15; done
grep -E "labelled|cost" outputs/judge_topup.log | tail -1 | tee -a $S
log "REMAINDER (paid) conc 10 cap 56"
$PY scripts/judge_v5.py --stage rest --conc 10 --cap 56 2>&1 | tee outputs/judge_rest.log | grep -E "coverage|labelled|cost|REFUSE|failed" | tee -a $S
log "MERGE"
$PY scripts/judge_v5.py --stage merge 2>&1 | tail -3 | tee -a $S
log "REST STAGE DONE"
