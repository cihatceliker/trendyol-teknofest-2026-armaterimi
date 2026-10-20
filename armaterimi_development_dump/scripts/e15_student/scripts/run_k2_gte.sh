#!/usr/bin/env bash
# K2: GPU bench of serve_gte (no-transformers champion path) after K.
set -u
W=/home/jupyter/trendyol-teknofest-work; E15=$W/asya/s2_exp/e15_student; E12=$W/asya/s2_exp/e12_lean_infer; PY=$W/.venv-gcp/bin/python
O=$E15/outputs_frozen_v2; S=$O/K_SUMMARY.txt
log(){ echo "[$(date +%H:%M:%S)] $*" | tee -a $S; }
until grep -q "K DONE" $O/K_SUMMARY.txt 2>/dev/null; do
  ps -eo cmd | grep -qE "[r]un_k_bench.sh|[r]un_j_jurytest.sh|[r]un_p_champion.sh|[r]un_m2_retry.sh" || break; sleep 120
done
until [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)" -lt 1500 ]; do sleep 30; done
log "K2: GPU bench serve_gte (champion, no-transformers)"
cd $E12 && BENCH_SCRIPT=serve_gte.py $PY scripts/bench_e2e.py --model-dir=$O/ckpt_r1_oldstore --runs=2 2>&1 | grep -E "projected|run " | tee -a $S
log "K2 DONE"
