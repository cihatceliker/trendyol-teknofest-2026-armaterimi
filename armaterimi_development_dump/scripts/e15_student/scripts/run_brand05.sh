#!/usr/bin/env bash
# BRAND ARTIFACT FIX — 'genel markalar' -> 'unknown' everywhere, plus 5% random brand->'unknown'
# augmentation at TRAIN time only.
#
# Train-only justification (no test data used to decide this): GM rows carry P(relevant)=0.038 in
# the training frame against an overall 0.326. But the `contrast` slice -- 1,990 GM rows from an
# independently built, deliberately adversarial mining process with R/S labels, never touched by
# the organizer's export split -- puts GM at **34.9% relevant, i.e. the base rate**. So the token
# carries no real relevance signal and the organizer's 0-of-11,243 is localised to their export,
# exactly like gender=çocuk.
#
# Why the augmentation is the load-bearing part: mapping GM to ANY token without it merely renames
# the artifact (the new token becomes the 100%-semi bucket), and brand has no natural sink -- 0
# empty and 0 'unknown' across 54,309 values. The 5% dropout creates one:
#     sink 57,420 rows, P(semi) 0.453 / P(rel) 0.241 against overall 0.349 / 0.326
#     P(relevant | GM-flavoured) 0.038 -> 0.241  (residual suppression 1.36x, was 8.5x)
# It should also regularise: the model must not lean on brand when it is absent.
#
# Serving applies T5 only when the CHECKPOINT declares brand_sink -- see lean_infer.configure_text.
# Fallback: ckpt_gf2_berturk_6ep_bf16 stays frozen. This ships only if it beats 0.8546 DEV OOF.
set -u
W=/home/jupyter/trendyol-teknofest-work; E15=$W/asya/s2_exp/e15_student
E23=$W/asya/s2_exp/e23_regime_cuts; PY=$W/.venv-gcp/bin/python; O=$E15/outputs_frozen_v2
DATA=$O/build_oldstore/student_train_gf_brand05.parquet
S=$O/BRAND05_SUMMARY.txt
log(){ echo "[$(date +%H:%M:%S)] $*" | tee -a $S; }
cd $E15; : > $S
while ps -eo cmd | grep -qE "[t]rain_student.py|[e]val_student.py"; do sleep 30; done
log "== b05_berturk : BERTurk 12L, 6 epochs, identical recipe to the frozen pick"
log "   only difference: GM->unknown + 5% brand dropout in the training text"
if TAG=b05_berturk MODEL=dbmdz/bert-base-turkish-128k-cased TRUST=1 BS=64 LR=3e-5 SEED=42 \
   EPOCHS=6 ALPHA=1 BETA=0 DATA=$DATA OUT=$O $PY scripts/train_student.py > $O/train_b05.log 2>&1; then
  # eval MUST apply the same T5 the model was trained with
  T5_BRAND_SINK=1 ATTR_CLEAN=v2 TRUST=1 CKPT=$O/ckpt_b05_berturk OUT=$O $PY scripts/eval_student.py 2>&1 \
    | grep -E "2-boundary|AUC" | sed 's/^/   /' | tee -a $S
  cd $E23 && $PY scripts/fit_regime_cuts.py --probs $O/dev_probs_b05_berturk.parquet \
    --emit $O/ckpt_b05_berturk/decision.json 2>&1 | grep -E "global|regime" | sed 's/^/   /' | tee -a $S
  cd $E15
  $PY - <<'PY2' | tee -a $S
import json,pathlib
p=pathlib.Path("outputs_frozen_v2/ckpt_b05_berturk/decision.json"); d=json.loads(p.read_text())
g,r=d["oof_global"],d["oof_regime"]
out={"head":"ordinal" if g>=r else "ordinal_regime",
     "labels":["irrelevant","semi-relevant","relevant"],
     "brand_sink":True,   # <- tells lean_infer to apply T5 for THIS checkpoint
     "provenance":{"fitted_on":"DEV only","oof_global":g,"oof_per_regime":r,
                   "brand_sink":"trained with GM->unknown + 5% brand dropout"}}
if out["head"]=="ordinal": out["cuts"]=d["global"]
else: out.update({k:d[k] for k in ("global","regimes","fashion_roots","technical_roots")})
p.write_text(json.dumps(out,indent=1,ensure_ascii=False)+"\n")
print(f"   decision.json: head={out['head']} brand_sink=True  (OOF global {g:.4f} / regime {r:.4f})")
PY2
  log "   BASELINE to beat: frozen BERTurk = 0.8546 DEV OOF / 0.8579 public"
else log "!! FAILED"; tail -4 $O/train_b05.log | sed 's/^/   /' | tee -a $S; fi
log "BRAND05 DONE"
