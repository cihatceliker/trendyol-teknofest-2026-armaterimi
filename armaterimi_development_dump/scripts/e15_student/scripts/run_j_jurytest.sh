#!/usr/bin/env bash
# J: timed jury-style docker dress rehearsal (current image, ckpt r1_oldstore inside).
# Runs ONLY when the P-chain is done and the GPU is idle. Gives the REAL SPS number incl. container start.
set -u
W=/home/jupyter/trendyol-teknofest-work; E15=$W/asya/s2_exp/e15_student; E19=$W/asya/s2_exp/e19_docker_ty
O=$E15/outputs_frozen_v2; S=$O/J_SUMMARY.txt; : > $S
log(){ echo "[$(date +%H:%M:%S)] $*" | tee -a $S; }
log "waiting for P chain"
until grep -q "P DONE" $O/P_SUMMARY.txt 2>/dev/null; do
  n=$(ps -eo cmd | grep -cE "[r]un_p_champion.sh|[r]un_m2_retry.sh"); [ "$n" -eq 0 ] && break; sleep 120
done
until [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)" -lt 1500 ]; do sleep 30; done
log "JURY-STYLE TIMED RUN (image armaterimi_solution, public 7.5k)"
cd $E19 && bash test_like_jury.sh 2>&1 | tail -25 | tee -a $S
log "J DONE"
