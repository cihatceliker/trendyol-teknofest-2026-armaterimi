#!/usr/bin/env bash
# v2 (BERTurk 4ep) is a near-certain negative (4ep measured 0.808 vs 3ep 0.8205); skip it so the
# 10-layer run gets the GPU instead.
O=/home/jupyter/trendyol-teknofest-work/asya/s2_exp/e15_student/outputs_frozen_v2
until grep -q "== v2_berturk_full_4ep" $O/V_SUMMARY.txt 2>/dev/null; do
  ps -eo cmd | grep -q "[r]un_v_berturk_max.sh" || exit 0; sleep 20; done
sleep 3
for p in $(ps -eo pid,cmd | grep "[r]un_v_berturk_max.sh" | awk '{print $1}'); do kill $p; done
for p in $(ps -eo pid,cmd | grep "[t]rain_student.py" | awk '{print $1}'); do kill $p; done
echo "[$(date +%H:%M:%S)] v2 (4ep) SKIPPED — 4 epochs measured worse; GPU handed to the 10-layer run" >> $O/V_SUMMARY.txt
echo "[$(date +%H:%M:%S)] V-CHAIN DONE" >> $O/V_SUMMARY.txt
