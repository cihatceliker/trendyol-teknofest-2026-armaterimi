#!/usr/bin/env bash
# t1 (two-head, NO stage-1) is not a ship candidate: if stage-1 doesn't help we fall back to the
# existing 3-class single head (r1_oldstore, DEV 0.8553). Kill the chain the moment t1 starts.
O=/home/jupyter/trendyol-teknofest-work/asya/s2_exp/e15_student/outputs_frozen_v2
until grep -q "== t1_twohead" $O/T_SUMMARY.txt 2>/dev/null; do
  ps -eo cmd | grep -q "[r]un_t_twohead.sh" || exit 0
  sleep 20
done
sleep 3
for p in $(ps -eo pid,cmd | grep "[r]un_t_twohead.sh" | awk '{print $1}'); do kill $p; done
for p in $(ps -eo pid,cmd | grep "[t]rain_student.py" | awk '{print $1}'); do kill $p; done
echo "[$(date +%H:%M:%S)] t1 (no-stage-1 control) SKIPPED — fallback is the existing 3-class single head 0.8553" >> $O/T_SUMMARY.txt
echo "[$(date +%H:%M:%S)] T-CHAIN DONE" >> $O/T_SUMMARY.txt
