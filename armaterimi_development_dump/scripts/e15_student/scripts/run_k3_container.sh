#!/usr/bin/env bash
# K3: container-overhead study — the official clock includes docker start (~10s measured cold).
# Run the jury test twice back-to-back (cold-ish vs warm page cache) to split the overhead.
set -u
W=/home/jupyter/trendyol-teknofest-work; E15=$W/asya/s2_exp/e15_student; E19=$W/asya/s2_exp/e19_docker_ty
O=$E15/outputs_frozen_v2; S=$O/K_SUMMARY.txt
log(){ echo "[$(date +%H:%M:%S)] $*" | tee -a $S; }
until grep -q "K2 DONE" $O/K_SUMMARY.txt 2>/dev/null; do ps -eo cmd | grep -q "[r]un_k2_gte.sh" || break; sleep 60; done
until [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)" -lt 1500 ]; do sleep 30; done
for i in 1 2; do
  log "K3 jury run $i (same image, back-to-back)"
  cd $E19 && bash test_like_jury.sh 2>&1 | grep -E "wall=|STAGE|PASS" | tee -a $S
done
log "K3 DONE"
