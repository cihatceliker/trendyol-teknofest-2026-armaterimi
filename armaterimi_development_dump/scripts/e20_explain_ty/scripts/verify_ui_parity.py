#!/usr/bin/env python
"""UI-vs-CONTAINER decision parity.

The organizer requires the explanation to reflect the model that actually predicts. Our container
decides with lean_infer.decide() (head=ordinal_regime: scalar p_semi+2*p_rel, cut pair chosen by
the row's OWN category root). The explainability UI decides with model_adapter.Adapter.decide().
Two implementations of one rule = a place for them to silently diverge; this asserts they do not.

Method: score N real public rows ONCE, then push the SAME logits through both deciders and require
identical labels. No HTTP, no container, CPU-only -> runs in a couple of minutes.

Usage: verify_ui_parity.py [--n 400] [--model-dir DIR] [--csv public_test.csv]
"""
import sys, os, argparse, csv as _csv
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = "/home/jupyter/trendyol-teknofest-work/asya/s2_exp"
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "e12_lean_infer", "scripts"))

ap = argparse.ArgumentParser()
ap.add_argument("--n", type=int, default=400)
ap.add_argument("--model-dir", default=os.path.join(ROOT, "e15_student/outputs_frozen_v2/ckpt_gf2_berturk_6ep_bf16"))
ap.add_argument("--csv", default="/home/jupyter/trendyol-teknofest-work/stage2_data/public_test.csv")
a = ap.parse_args()

os.environ.setdefault("DEVICE", "cpu")
import model_adapter as MA
import lean_infer as L

print(f"[parity] model_dir={a.model_dir}")
ad = MA.ChampionAdapter(a.model_dir, device=os.environ["DEVICE"])
print(f"[parity] head={ad.head} global_cuts={ad.cuts} regimes={sorted(ad.regimes)} "
      f"fashion_roots={len(ad.fashion_roots)} technical_roots={len(ad.technical_roots)}")

# --- rows: stratify across category roots so every regime is actually exercised -------------
rows = list(_csv.DictReader(open(a.csv, encoding="utf-8")))
by_reg = {}
for r in rows:
    by_reg.setdefault(ad.regime_of(r.get("category", "")), []).append(r)
per = max(1, a.n // max(1, len(by_reg)))
sample, seen = [], set()
for g in sorted(by_reg):
    take = by_reg[g][:per]
    print(f"[parity]   regime {g:<10} pool={len(by_reg[g]):>5}  taking {len(take)}")
    sample.extend(take)
sample = sample[: a.n]
print(f"[parity] {len(sample)} rows over {len(by_reg)} regimes")

# --- score once through the UI's own pipeline (same normalisation the container uses) -------
pairs, cats = [], []
import pandas as _pd
_raw = _pd.DataFrame([{c: str(r.get(c, "") or "") for c in MA.RAW_COLS} for r in sample], dtype=str)
_norm = MA.normalize_frame(_raw).to_dict("records")
for nf in _norm:
    nf = {c: str(nf[c]) for c in MA.RAW_COLS}
    pairs.append((nf["query"], MA.build_doc(nf)[0])); cats.append(nf["category"])
logits = ad.score_pairs(pairs)
logits = np.asarray(logits, dtype=np.float64)
print(f"[parity] scored {logits.shape}")

# --- decider A: the UI ----------------------------------------------------------------------
ui = [ad.decide(lg, category=c)["label"] for lg, c in zip(logits, cats)]

# --- decider B: the container. lean_infer.decide reads the category column from the CSV -----
tmp = os.path.join(os.environ.get("TMPDIR", "/tmp"), "parity_rows.csv")
with open(tmp, "w", newline="", encoding="utf-8") as f:
    w = _csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader()
    for r in sample: w.writerow(r)
class _Cfg: pass
cfg = _Cfg(); cfg.id2label = {0: "irrelevant", 1: "semi-relevant", 2: "relevant"}
srv = list(L.decide(logits, a.model_dir, cfg, input_csv=tmp))

# --- compare --------------------------------------------------------------------------------
bad = [(i, sample[i].get("row_id"), cats[i], ui[i], srv[i]) for i in range(len(ui)) if ui[i] != srv[i]]
from collections import Counter
print(f"[parity] UI  labels: {dict(Counter(ui))}")
print(f"[parity] SRV labels: {dict(Counter(srv))}")
if bad:
    print(f"[parity] FAIL — {len(bad)}/{len(ui)} disagree")
    for b in bad[:15]: print("   ", b)
    sys.exit(1)
# power check: how far off the pre-fix UI (plain argmax) would have been on these same rows
am = [cfg.id2label[int(i)] for i in np.argmax(logits, 1)]
n_am = sum(1 for i in range(len(ui)) if ui[i] != am[i])
print(f"[parity] argmax-vs-served differ on {n_am}/{len(ui)} rows ({100*n_am/len(ui):.1f}%) "
      f"— that is what a UI ignoring the regime cuts would have mislabelled")
print(f"[parity] PASS — {len(ui)}/{len(ui)} labels identical (UI == container)")
