#!/usr/bin/env bash
# CLEAN back-to-back 90k speed bench on an IDLE GPU.
#
# Why this exists: the in-chain benches fired while the NEXT model was already training, so they
# are contended and not comparable to the in-container numbers in LOG.md (2,967 / 3,340 / 5,073),
# which were measured the way the jury measures. This runs every candidate under identical,
# uncontended conditions, twice, and reports the BEST of the two (warm cache, no neighbour).
#
# Bare bench != jury SPS: the jury clock starts at run_solution.sh and includes container start,
# python import, model load and CSV write. This measures the MODEL, so use it for model-vs-model
# ratios only; the winner still needs a real in-container run before any number is banked.
set -u
W=/home/jupyter/trendyol-teknofest-work; E12=$W/asya/s2_exp/e12_lean_infer; PY=$W/.venv-gcp/bin/python
O=$W/asya/s2_exp/e15_student/outputs_frozen_v2
SC=/var/tmp/claude-1000/-home-jupyter/483abbaf-6d7f-4b27-838c-53d3d34a821c/scratchpad
OUT=$O/BENCH_CLEAN.txt
log(){ echo "[$(date +%H:%M:%S)] $*" | tee -a "$OUT"; }

: > "$OUT"
# Wait for the WHOLE CHAIN, not just the current training. Between tiny3's training ending and
# tiny4's beginning there is an eval window with no train_student.py alive; a naive guard fires
# there and contends with the very next model. Two conditions must BOTH hold:
#   * run_t2_tiny.sh has exited (or logged DONE), and
#   * no train_student.py / eval_student.py is alive.
# NOTE: no free-VRAM check here — the explainability UI legitimately holds ~1.5 GB on the GPU
# and would deadlock a "memory must be under X" gate forever. Its load is sporadic and tiny.
log "waiting for the tiny chain to finish (chain process + any train/eval)"
while ps -eo cmd | grep -q "[r]un_t2_tiny.sh" \
   || ps -eo cmd | grep -qE "[t]rain_student.py|[e]val_student.py"; do sleep 30; done
sleep 20   # let the last process release its CUDA context
log "chain finished — GPU state: $(nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader)"
log "GPU idle — starting clean bench (BATCH_ROWS=256 TOKEN_BUDGET=65536 MAXLEN=256, e12 lean path)"

bench(){ # label ckpt_dir
  local lbl=$1 dir=$2
  [ -d "$dir" ] || { log "  SKIP $lbl (no checkpoint at $dir)"; return; }
  local best=0 tot="" line=""
  for rep in 1 2; do
    # pick the right lean port: lean_infer.py is the BERT one, serve_gte.py the GTE "New" one
    # (TY-ecomm). Using the wrong port dies on a state-dict key, which is what happened to the
    # tyecomm rows in the 21:00 bench.
    local script=scripts/lean_infer.py
    grep -q '"New\|GteModel\|NewForSequence' "$dir/config.json" 2>/dev/null && script=scripts/serve_gte.py
    line=$(cd "$E12" && BATCH_ROWS=256 TOKEN_BUDGET=65536 timeout 900 "$PY" "$script" \
             "$SC/synth90k.csv" "$SC/bench_${lbl}.csv" --model-dir="$dir" 2>&1 | grep "^STAGE" || true)
    if [ -z "$line" ]; then
      # LOUD failure — the chain's `grep STAGE` swallowed a KeyError and logged a blank line.
      log "  !! $lbl bench FAILED on rep $rep (no STAGE line) — last error:"
      (cd "$E12" && BATCH_ROWS=256 TOKEN_BUDGET=65536 timeout 900 "$PY" "$script" \
         "$SC/synth90k.csv" "$SC/bench_${lbl}.csv" --model-dir="$dir" 2>&1 | tail -3 | sed 's/^/       /') \
        | tee -a "$OUT"
      return
    fi
    local sps; sps=$(sed 's/.*sps=//' <<<"$line")
    tot="$tot ${sps}"
    awk -v a="$sps" -v b="$best" 'BEGIN{exit !(a>b)}' && best=$sps
  done
  log "  $(printf '%-22s' "$lbl") best ${best} SPS   (reps:${tot})"
}

log "-- tiny candidates"
for t in tiny1_distilbert_tr tiny2_medium_8L512 tiny3_small_4L512 tiny4_mini_4L256; do
  bench "$t" "$O/ckpt_$t"
done
log "-- incumbents, SAME conditions (this is the comparable baseline)"
bench berturk_12L   "$O/ckpt_v1_berturk_full"
bench tyecomm_10L   "$O/ckpt_x1_tyecomm_10L"
bench tyecomm_12L   "$O/ckpt_r1_oldstore"
log "BENCH DONE — bare model speed only; jury SPS still needs an in-container run"
