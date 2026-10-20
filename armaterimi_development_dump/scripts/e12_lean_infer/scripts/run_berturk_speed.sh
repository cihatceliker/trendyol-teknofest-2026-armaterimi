#!/usr/bin/env bash
# BERTurk speed push. Baseline this morning: 5,557 SPS bare / 5,073 in-container, 0.8373 DEV.
#
# Already landed before this script (CPU-side, byte-identical output, verified on public 7.5k and
# the 90k synth): the 90k data path went 5.43s -> 3.98s by hoisting the folded constants out of
# _is_boiler (8.6M _fold calls -> ~1.4M) and narrowing the pool IPC to int32/int8.
#
# Budget for one 90k BERTurk run, measured: ~16.2s total = ~9.5s inference + ~0.1s write +
# ~6.6s pre-inference, where pre-inference is max(data path, torch import + CUDA context + load).
# Model load is only 0.30s, so the pre-inference floor is the torch/CUDA startup, now co-critical
# with the shortened data path. The largest remaining prize is overlapping the data path with
# INFERENCE (STREAM=1), which is worth up to ~4s of the 16.2s.
set -u
W=/home/jupyter/trendyol-teknofest-work; E12=$W/asya/s2_exp/e12_lean_infer; PY=$W/.venv-gcp/bin/python
O=$W/asya/s2_exp/e15_student/outputs_frozen_v2
SC=/var/tmp/claude-1000/-home-jupyter/483abbaf-6d7f-4b27-838c-53d3d34a821c/scratchpad
CK=$O/ckpt_v1_berturk_full
S=$O/BERTURK_SPEED.txt
log(){ echo "[$(date +%H:%M:%S)] $*" | tee -a $S; }
cd $E12; : > $S
log "waiting for the GPU (seed44)"
while ps -eo cmd | grep -qE "[t]rain_student.py|[e]val_student.py"; do sleep 60; done
sleep 20; log "GPU free"

run(){ # label  env-assignments...
  local lbl=$1; shift
  local best=0 reps=""
  for r in 1 2; do
    local out sps
    out=$(env "$@" timeout 900 $PY scripts/lean_infer.py $SC/synth90k.csv $SC/bs_${lbl}.csv --model-dir=$CK 2>&1)
    sps=$(grep "^STAGE" <<<"$out" | sed 's/.*sps=//')
    [ -z "$sps" ] && { log "   !! $lbl FAILED:"; tail -3 <<<"$out" | sed 's/^/      /' | tee -a $S; return; }
    [ "$r" = 1 ] && grep "^STAGE" <<<"$out" | sed 's/^/      /' | tee -a $S
    reps="$reps $sps"; awk -v a="$sps" -v b="$best" 'BEGIN{exit !(a>b)}' && best=$sps
  done
  log "   $(printf '%-34s' "$lbl") best ${best} SPS  (reps:${reps})"
}

log "== A) baseline with the faster data path"
run base_b256_t65536   BATCH_ROWS=256 TOKEN_BUDGET=65536 STREAM=0
log "== B) STREAM=1 — hide the data path behind inference"
run stream_b256_t65536 BATCH_ROWS=256 TOKEN_BUDGET=65536 STREAM=1
log "== C) batch / token-budget sweep"
run b384_t98304         BATCH_ROWS=384 TOKEN_BUDGET=98304  STREAM=0
run b512_t131072        BATCH_ROWS=512 TOKEN_BUDGET=131072 STREAM=0
run stream_b512_t131072 BATCH_ROWS=512 TOKEN_BUDGET=131072 STREAM=1
log "== D) CUDA_MODULE_LOADING=LAZY (cheaper context init)"
run lazy_b256           BATCH_ROWS=256 TOKEN_BUDGET=65536 STREAM=0 CUDA_MODULE_LOADING=LAZY
run lazy_stream_b512    BATCH_ROWS=512 TOKEN_BUDGET=131072 STREAM=1 CUDA_MODULE_LOADING=LAZY

log "== E) does STREAM=1 change any prediction at 90k? (this is why it was reverted)"
$PY - <<'PYEOF' 2>&1 | sed 's/^/   /' | tee -a $S
import csv, collections
SC="/var/tmp/claude-1000/-home-jupyter/483abbaf-6d7f-4b27-838c-53d3d34a821c/scratchpad"
def rd(p):
    with open(p, newline="", encoding="utf-8") as f:
        return {r["row_id"]: r["real_label"] for r in csv.DictReader(f)}
a, b = rd(f"{SC}/bs_base_b256_t65536.csv"), rd(f"{SC}/bs_stream_b256_t65536.csv")
d = [k for k in a if a[k] != b.get(k)]
print(f"STREAM=0 vs STREAM=1 over {len(a):,} rows: {len(d)} label differences ({100*len(d)/len(a):.3f}%)")
if d: print("  moves:", dict(collections.Counter((a[k], b[k]) for k in d).most_common(6)))
PYEOF
log "BERTURK SPEED DONE"
