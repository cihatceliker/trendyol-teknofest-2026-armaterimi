#!/usr/bin/env bash
set -uo pipefail
W=/home/jupyter/trendyol-teknofest-work; E17=$W/asya/s2_exp/e17_mine_v5; PY=$W/.venv-gcp/bin/python
S=$E17/outputs/REST_SUMMARY.txt; : > $S
log(){ echo "[$(date +%H:%M:%S)] $*" | tee -a $S; }
cd $E17
log "REMAINDER (paid) conc 10 cap 56"
$PY scripts/judge_v5.py --stage rest --conc 10 --cap 56 2>&1 | tee outputs/judge_rest.log | grep -E "coverage|labelled|cost|REFUSE|failed" | tee -a $S
log "TOP-UP (paid) conc 10 cap 6"
$PY scripts/judge_v5.py --stage topup --conc 10 --cap 6 2>&1 | tee outputs/judge_topup.log | grep -E "coverage|labelled|cost|REFUSE|failed" | tee -a $S
log "MERGE"
$PY scripts/judge_v5.py --stage merge 2>&1 | tail -3 | tee -a $S
log "REST STAGE DONE"
