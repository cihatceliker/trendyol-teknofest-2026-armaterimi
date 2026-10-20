#!/usr/bin/env bash
# IN-CONTAINER SPS — the only speed number the jury actually scores.
#
# Why now: the image was still shipping BATCH_ROWS=128 TOKEN_BUDGET=32768, smaller than anything
# we ever benched, so every in-container figure on record (TY-12L 2,967 / TY-10L 3,340 /
# BERTurk 5,073) was measured on an UNTUNED config and understates the models. The Dockerfile now
# carries the measured-best bit-exact settings (256/65536 + OMP/RAYON pinned + 8 workers), so all
# four candidates need re-measuring before any of them can be quoted.
#
# SPS = 90000 / (run_solution.sh start -> submission fully written) via the frozen
# test_like_jury.sh, i.e. container start + imports + model load + data path + inference + write.
# Bare benches do not include that window; these numbers do.
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"
W=/home/jupyter/trendyol-teknofest-work
O=$W/asya/s2_exp/e15_student/outputs_frozen_v2
SC=/var/tmp/claude-1000/-home-jupyter/483abbaf-6d7f-4b27-838c-53d3d34a821c/scratchpad
S=$O/INCONTAINER.txt
log(){ echo "[$(date +%H:%M:%S)] $*" | tee -a "$S"; }
cd "$HERE"; : > "$S"
log "in-container re-measurement, tuned image (bit-exact config)"

# test_like_jury.sh refuses to time a run while anything else holds the GPU -- correctly, since a
# contended number is worthless. The explainability UI is a CUDA process, so park it for the
# duration and bring it back at the end.
# the UI's cmdline is ".../python scripts/app.py <csv>" -- it does NOT contain the directory,
# so match on the script plus the public_test argument instead.
UI_PID=$(ps -eo pid,cmd | grep "[s]cripts/app.py" | grep public_test | awk '{print $1}' | head -1)
if [ -n "${UI_PID:-}" ]; then log "parking the explainability UI (pid $UI_PID) so the GPU is clean"; kill "$UI_PID" 2>/dev/null; sleep 5; fi
restore_ui(){
  [ -n "${UI_PID:-}" ] || return 0
  cd $W/asya/s2_exp/e20_explain_ty
  setsid nohup $W/.venv-gcp/bin/python scripts/app.py $W/stage2_data/public_test.csv \
    --model-dir $O/ckpt_r1_oldstore --host 127.0.0.1 --port 7862 --device cuda \
    > outputs/ui_server.log 2>&1 < /dev/null &
  sleep 12; cd "$HERE"
  log "explainability UI back up: $(curl -s --max-time 5 http://127.0.0.1:7862/readyz | head -c 60)"
}
trap restore_ui EXIT

one(){ # label  ckpt_dir  old_number
  local lbl=$1 ck=$2 old=$3
  [ -d "$ck" ] || { log "  SKIP $lbl (no checkpoint)"; return; }
  log "== $lbl"
  MODEL_SRC="$ck" bash prepare_context.sh >/dev/null 2>&1 || { log "   !! prepare_context failed"; return; }
  if ! CONTAINER_NAME=armaterimi_bench bash build_docker.sh > "$SC/build_$lbl.log" 2>&1; then
    log "   !! build failed:"; tail -4 "$SC/build_$lbl.log" | sed 's/^/      /' | tee -a "$S"; return
  fi
  local best=0 reps=""
  for r in 1 2; do
    # Time the official window MYSELF: test_like_jury prints only whole seconds (wall=NNs), which
    # is +/-4% on a ~20s run and, worse, my earlier regex missed that format entirely and latched
    # onto an unrelated float -- that is where "782,608 SPS" came from. run_solution.sh is the
    # frozen jury entrypoint; wrapping it in date +%s.%N measures exactly the scored window.
    rm -f "$SC/ic_$lbl.csv"
    local t0 t1 sec rows
    t0=$(date +%s.%N)
    CONTAINER_NAME=armaterimi_bench bash "$HERE/run_solution.sh" "$SC/synth90k.csv" "$SC/ic_$lbl.csv" \
      > "$SC/run_${lbl}_$r.log" 2>&1
    local rc=$?
    t1=$(date +%s.%N)
    if [ $rc -ne 0 ]; then log "   !! run exit=$rc:"; tail -4 "$SC/run_${lbl}_$r.log" | sed 's/^/      /' | tee -a "$S"; return; fi
    rows=$(( $(wc -l < "$SC/ic_$lbl.csv") - 1 ))
    if [ "$rows" -ne 90000 ]; then log "   !! wrote $rows rows, expected 90000"; return; fi
    sec=$(awk -v a="$t0" -v b="$t1" 'BEGIN{printf "%.2f", b-a}')
    local sps; sps=$(awk -v s="$sec" 'BEGIN{printf "%.1f", 90000/s}')
    reps="$reps ${sps}(${sec}s)"; awk -v a="$sps" -v b="$best" 'BEGIN{exit !(a>b)}' && best=$sps
  done
  log "   $(printf '%-14s' "$lbl") in-container best ${best} SPS   (was ${old})   reps:${reps}"
}

one tyecomm_8L  $O/ckpt_l8b_tyecomm_8L      "never measured"
one berturk_12L $O/ckpt_v1_berturk_full     5073
one tiny2_8L512 $O/ckpt_tiny2_medium_8L512  "never measured"
one tyecomm_12L $O/ckpt_r1_oldstore         2967

log "restoring the deliverable image from the champion checkpoint"
MODEL_SRC=$O/ckpt_r1_oldstore bash prepare_context.sh >/dev/null 2>&1
CONTAINER_NAME=armaterimi_solution bash build_docker.sh > "$SC/build_restore.log" 2>&1 \
  && log "armaterimi_solution rebuilt on the champion" || log "!! restore build FAILED"
log "IN-CONTAINER DONE"
