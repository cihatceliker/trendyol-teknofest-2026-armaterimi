#!/usr/bin/env bash
# K: after the jury dress rehearsal (J) — GPU A/B benches of the NEW serving paths, then rebuild
# the docker image with the fast scripts and re-run the timed jury test (before/after on the official clock).
set -u
W=/home/jupyter/trendyol-teknofest-work; E15=$W/asya/s2_exp/e15_student; E12=$W/asya/s2_exp/e12_lean_infer; E19=$W/asya/s2_exp/e19_docker_ty; PY=$W/.venv-gcp/bin/python
O=$E15/outputs_frozen_v2; S=$O/K_SUMMARY.txt; : > $S
log(){ echo "[$(date +%H:%M:%S)] $*" | tee -a $S; }
log "waiting for J (first jury run)"
until grep -q "J DONE" $O/J_SUMMARY.txt 2>/dev/null; do
  ps -eo cmd | grep -qE "[r]un_j_jurytest.sh|[r]un_p_champion.sh|[r]un_m2_retry.sh" || break; sleep 120
done
until [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)" -lt 1500 ]; do sleep 30; done
CK=$O/ckpt_r1_oldstore
log "GPU bench: serve_hf NEW (champion)"
cd $E12 && BENCH_SCRIPT=serve_hf.py $PY scripts/bench_e2e.py --model-dir=$CK --runs=2 2>&1 | grep -E "projected|run " | tee -a $S
log "GPU bench: serve_hf NEW, serial-read A/B"
LEAN_SERIAL_READ=1 BENCH_SCRIPT=serve_hf.py $PY scripts/bench_e2e.py --model-dir=$CK --runs=1 2>&1 | grep -E "projected" | tee -a $S
log "GPU bench: lean NEW (berturk c1)"
BENCH_SCRIPT=lean_infer.py $PY scripts/bench_e2e.py --model-dir=$O/ckpt_c1_berturk_3ep --runs=1 2>&1 | grep -E "projected" | tee -a $S
log "rebuild image with fast scripts + champion ckpt"
cd $E19 && MODEL_SRC=$CK bash prepare_context.sh >> $S 2>&1 && bash build_docker.sh 2>&1 | tail -3 | tee -a $S
log "second jury-timed run (fast path)"
bash test_like_jury.sh 2>&1 | tail -20 | tee -a $S
log "K DONE"
