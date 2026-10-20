#!/usr/bin/env bash
# q1 showed AUX_SI=0.5 is flat-to-negative -> q3 (AUX_SI=1.0) is not worth 80 GPU-min.
# Watch Q_SUMMARY; the moment rung q3 starts, kill the chain + its just-launched training so
# the R1 old-store run (waiting on the chain) starts immediately. q2 (seed variance) is kept.
O=/home/jupyter/trendyol-teknofest-work/asya/s2_exp/e15_student/outputs_frozen_v2
until grep -q "== q3_aux_si10" $O/Q_SUMMARY.txt 2>/dev/null; do
  n=$(ps -eo cmd | grep -c "[r]un_q_chain.sh"); [ "$n" -eq 0 ] && exit 0
  sleep 20
done
sleep 5
for p in $(ps -eo pid,cmd | grep "[r]un_q_chain.sh" | awk '{print $1}'); do kill $p; done
for p in $(ps -eo pid,cmd | grep "[t]rain_student.py" | awk '{print $1}'); do kill $p; done
echo "[$(date +%H:%M:%S)] q3 skipped by decision: q1 (AUX_SI=0.5) flat -> stronger weight not informative" >> $O/Q_SUMMARY.txt
