#!/usr/bin/env python
"""TRAIN vs SERVE text parity — the check that matters after ANY edit to either normaliser.

Two independent implementations produce the (query, doc) strings the model sees:
  TRAIN : common/normalize.py (T1-T4) + e18 attr_clean.apply_v2, frame-at-a-time via pandas.
  SERVE : lean_infer._rows_to_pairs, row-at-a-time, inlined in the container's serving path.
They even fold differently on purpose -- attr_clean._fold is lower()+t1() while lean_infer._fold
applies Turkish diacritic folding and compares against pre-folded constants -- so agreement is a
property to be TESTED, never assumed. A silent divergence here is a train/serve skew: the model
scores text it was never trained on, and nothing else in the stack would notice.

Byte-identical on every row is the bar. CPU only.
Usage: verify_train_serve_parity.py [csv]
"""
import os, sys
os.environ["CUDA_VISIBLE_DEVICES"] = ""
W = "/home/jupyter/trendyol-teknofest-work"
HERE = os.path.dirname(os.path.abspath(__file__))
for p in (HERE, f"{W}/asya/s2_exp/e20_explain_ty/scripts"):
    if p not in sys.path: sys.path.insert(0, p)

CSV = sys.argv[1] if len(sys.argv) > 1 else f"{W}/stage2_data/public_test.csv"
import pandas as pd
import lean_infer as L
import model_adapter as MA          # imports common/normalize.py + e18 attr_clean (the TRAIN path)

# ---- SERVE side -----------------------------------------------------------------
ids_s, pairs_s = [], []
for rows, ix in L._iter_row_chunks(CSV, 4096):
    i, p = L._rows_to_pairs(rows, ix); ids_s += i; pairs_s += p

# ---- TRAIN side -----------------------------------------------------------------
raw = pd.read_csv(CSV, dtype=str, keep_default_na=False)
ids_t = raw["row_id"].tolist()
norm = MA.normalize_frame(raw[MA.RAW_COLS].astype(str)).to_dict("records")
pairs_t = [(str(r["query"]), MA.build_doc({c: str(r[c]) for c in MA.RAW_COLS})[0]) for r in norm]

# ---- compare --------------------------------------------------------------------
print(f"{CSV}\n  rows: serve {len(ids_s):,}  train {len(ids_t):,}")
assert ids_s == ids_t, "row_id order differs"
bad_q = [i for i in range(len(pairs_s)) if pairs_s[i][0] != pairs_t[i][0]]
bad_d = [i for i in range(len(pairs_s)) if pairs_s[i][1] != pairs_t[i][1]]
print(f"  query  mismatches: {len(bad_q)}")
print(f"  doc    mismatches: {len(bad_d)}")
for i in (bad_q + bad_d)[:3]:
    print(f"\n  row {ids_s[i]}\n    SERVE q={pairs_s[i][0]!r}\n    TRAIN q={pairs_t[i][0]!r}"
          f"\n    SERVE d={pairs_s[i][1]!r}\n    TRAIN d={pairs_t[i][1]!r}")
ok = not bad_q and not bad_d
print(f"\n  {'PASS — train and serve emit byte-identical text' if ok else 'FAIL — TRAIN/SERVE SKEW'}")
sys.exit(0 if ok else 1)
