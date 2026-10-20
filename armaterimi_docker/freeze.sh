#!/usr/bin/env bash
# FREEZE — one command, one candidate, every check. Written because today I rebuilt the image
# three times and twice verified the SOURCE tree instead of the artifact, and once shipped a
# change whose whole evidence base was 8 DEV predictions.
#
#   bash freeze.sh <ckpt_dir> [--holdout]
#
# Does, in order, and stops at the first failure:
#   1. stage the checkpoint + current serving code into the build context
#   2. build the image
#   3. prove the fix set is INSIDE the image (grep the container, never the source tree)
#   4. in-container 90k timed run with the jury's own run_solution.sh  -> the SPS we may quote
#   5. in-container DEV run -> macro-F1, the number that must match the offline eval
#   6. --holdout: the SINGLE holdout read. Refuses unless FREEZE_CONFIRM=yes, because holdout is
#      one-shot: once looked at, it is no longer an unbiased estimate of the private 90k.
set -Eeuo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"; W=/home/jupyter/trendyol-teknofest-work
PY=$W/.venv-gcp/bin/python; SC=/var/tmp/claude-1000/-home-jupyter/483abbaf-6d7f-4b27-838c-53d3d34a821c/scratchpad
CK="${1:?usage: freeze.sh <ckpt_dir> [--holdout]}"; HOLD="${2:-}"
S=$W/asya/s2_exp/e15_student/outputs_frozen_v2/FREEZE_$(basename "$CK").txt
log(){ echo "[$(date +%H:%M:%S)] $*" | tee -a "$S"; }
: > "$S"; cd "$HERE"
log "FREEZE candidate: $CK"

UI=$(ps -eo pid,cmd | grep "[s]cripts/app.py" | grep public_test | awk "{print \$1}" | head -1 || true)
# NOT `[ -n "$UI" ] && {...}` — under `set -e` that whole statement returns non-zero when no UI
# is running, which silently kills the script on its first line. Use a real if.
if [ -n "$UI" ]; then log "parking explainability UI (pid $UI) so the timed run is clean"; kill "$UI" || true; sleep 5; fi

log "1. staging"; MODEL_SRC="$CK" bash prepare_context.sh >/dev/null
log "2. building";  CONTAINER_NAME=armaterimi_solution bash build_docker.sh > "$SC/freeze_build.log" 2>&1
log "3. verifying the FIX SET from inside the image (not the source tree)"
for pat in 'çocuk' '_ordinal_scalar'; do
  n=$(docker run --rm --entrypoint sh armaterimi_solution -c "grep -rl '$pat' /app/src /models/current 2>/dev/null | wc -l")
  if [ "$n" -gt 0 ]; then log "   OK   '$pat' present in the image ($n file(s))"; else log "   FAIL '$pat' MISSING from the image"; exit 3; fi
done
docker run --rm --entrypoint python armaterimi_solution -c "
import json; c=json.load(open('/models/current/config.json'))
print('   model in image:', c.get('architectures'), c.get('num_hidden_layers'), 'layers')" | tee -a "$S"

log "4. in-container 90k timed run (jury clock)"
best=0
for r in 1 2; do
  t0=$(date +%s.%N); CONTAINER_NAME=armaterimi_solution bash run_solution.sh "$SC/synth90k.csv" "$SC/frz90k.csv" >/dev/null 2>&1; t1=$(date +%s.%N)
  rows=$(( $(wc -l < "$SC/frz90k.csv") - 1 )); [ "$rows" -ne 90000 ] && { log "   FAIL wrote $rows rows"; exit 4; }
  s=$(awk -v a=$t0 -v b=$t1 'BEGIN{printf "%.2f",b-a}'); sps=$(awk -v s=$s 'BEGIN{printf "%.0f",90000/s}')
  log "   rep$r  ${s}s  ${sps} SPS"; awk -v a=$sps -v b=$best 'BEGIN{exit !(a>b)}' && best=$sps
done
log "   IN-CONTAINER SPS = $best   (this is the number we may quote)"

log "5. in-container DEV score"
CONTAINER_NAME=armaterimi_solution bash run_solution.sh "$SC/dev_orig.csv" "$SC/frz_dev.csv" >/dev/null 2>&1
$PY - "$SC/frz_dev.csv" <<'PYEOF' | tee -a "$S"
import sys, pandas as pd
from sklearn.metrics import f1_score
sol=pd.read_csv('/home/jupyter/trendyol-teknofest-work/stage2_data/public_solution.csv',dtype=str).fillna('').rename(columns={'real_label':'truth'})
p=pd.read_csv(sys.argv[1],dtype=str).rename(columns={'real_label':'pred'}).merge(sol[['row_id','truth']],on='row_id')
print(f"   DEV macro-F1 (from the image) = {f1_score(p.truth,p.pred,average='macro'):.4f}  on {len(p)} rows")
PYEOF

if [ "$HOLD" = "--holdout" ]; then
  if [ "${FREEZE_CONFIRM:-}" != "yes" ]; then log "6. HOLDOUT read REFUSED — re-run with FREEZE_CONFIRM=yes."; log "   holdout is one-shot; after this read it is no longer unbiased."; exit 0; fi
  log "6. HOLDOUT — the single read"
  $PY - <<'PYEOF' | tee -a "$S"
import pandas as pd
pub=pd.read_csv('/home/jupyter/trendyol-teknofest-work/stage2_data/public_test.csv',dtype=str).fillna('')
spl=pd.read_csv('/home/jupyter/trendyol-teknofest-work/asya/s2_exp/common/public_split.csv',dtype=str)
h=pub.merge(spl[spl.half=='holdout'][['row_id']],on='row_id')
h.to_csv('/var/tmp/claude-1000/-home-jupyter/483abbaf-6d7f-4b27-838c-53d3d34a821c/scratchpad/holdout_in.csv',index=False)
print(f"   holdout rows: {len(h)}")
PYEOF
  CONTAINER_NAME=armaterimi_solution bash run_solution.sh "$SC/holdout_in.csv" "$SC/frz_hold.csv" >/dev/null 2>&1
  $PY - "$SC/frz_hold.csv" <<'PYEOF' | tee -a "$S"
import sys, pandas as pd
from sklearn.metrics import f1_score
sol=pd.read_csv('/home/jupyter/trendyol-teknofest-work/stage2_data/public_solution.csv',dtype=str).fillna('').rename(columns={'real_label':'truth'})
p=pd.read_csv(sys.argv[1],dtype=str).rename(columns={'real_label':'pred'}).merge(sol[['row_id','truth']],on='row_id')
print(f"   HOLDOUT macro-F1 = {f1_score(p.truth,p.pred,average='macro'):.4f}  on {len(p)} rows  <== THE HONEST NUMBER")
PYEOF
fi
log "FREEZE CHECKS DONE"
