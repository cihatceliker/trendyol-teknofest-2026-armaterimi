"""E23 — per-regime decision cuts.

The ordinal scalar x = p_semi + 2*p_rel is sliced by two cuts. A7 measured that the semi band is
~3x wider in compatibility verticals (electronics/auto) than in fashion, so ONE global pair of cuts
is a compromise. Cuts conditioned on the row's own category root are still a PER-ROW rule (no batch
statistics), so they stay legal under the "90k is not processed as one batch" constraint.

Regimes are PRE-REGISTERED (not fitted): fashion / technical / other. Fitting is query-disjoint OOF,
inside training folds only, with a per-fold fallback to the global cuts when a regime has < MIN_ROWS
training rows in that fold.

Usage: fit_regime_cuts.py --probs <dev_probs_TAG.parquet> [--emit outputs/cuts_TAG.json]
Prints global vs per-regime OOF macro-F1 and (with --emit) writes the shippable cut table fitted on
ALL rows of the given split.
"""
import argparse, json, os, sys
import numpy as np, pandas as pd

W = "/home/jupyter/trendyol-teknofest-work"
sys.path.insert(0, f"{W}/asya/s2_exp/common")
import harness as H

FASHION = {"giyim", "ayakkabı", "aksesuar"}
TECHNICAL = {"elektronik", "otomobil & motosiklet", "banyo yapı & hırdavat",
             "bahçe & elektrikli el aletleri"}
MIN_ROWS = 60
GRID = 48

def regime_of(cat: pd.Series) -> np.ndarray:
    root = cat.str.split("/").str[0].str.strip().str.lower()
    return np.where(root.isin(FASHION), "fashion",
                    np.where(root.isin(TECHNICAL), "technical", "other"))

def macro(yt, yp):
    f = 0.0
    for c in range(3):
        tp = ((yp == c) & (yt == c)).sum(); fp = ((yp == c) & (yt != c)).sum(); fn = ((yp != c) & (yt == c)).sum()
        f += 0.0 if tp == 0 else 2 * tp / (2 * tp + fp + fn)
    return f / 3

def fit(sv, yv, n=GRID):
    best = (-1.0, None)
    for c1 in np.quantile(sv, np.linspace(.08, .60, n)):
        lo = (sv >= c1).astype(int)
        for c2 in np.quantile(sv, np.linspace(.40, .95, n)):
            if c2 <= c1: continue
            v = macro(yv, lo + (sv >= c2))
            if v > best[0]: best = (v, (float(c1), float(c2)))
    return best[1]

def oof(s, yi, folds, reg=None):
    yp = np.zeros(len(s), int)
    for f in np.unique(folds):
        tr, te = folds != f, folds == f
        gl = fit(s[tr], yi[tr])
        groups = [("all", np.ones(len(s), bool))] if reg is None else [(g, reg == g) for g in np.unique(reg)]
        for g, gm in groups:
            a, b = tr & gm, te & gm
            if b.sum() == 0: continue
            c1, c2 = fit(s[a], yi[a]) if a.sum() >= MIN_ROWS else gl
            yp[b] = (s[b] >= c1).astype(int) + (s[b] >= c2)
    return macro(yi, yp), yp

ap = argparse.ArgumentParser()
ap.add_argument("--probs", required=True)
ap.add_argument("--half", default="dev")
ap.add_argument("--emit", default="")
a = ap.parse_args()

d = H.load_public_split(a.half)[["row_id", "query", "real_label", "category"]]
p = pd.read_parquet(a.probs).set_index("row_id").reindex(d.row_id)
s = (p.p_semi + 2 * p.p_rel).to_numpy()
yi = pd.Categorical(d.real_label, categories=["irrelevant", "semi-relevant", "relevant"]).codes
folds = H.query_folds(d["query"].values)
reg = regime_of(d.category)

g, _ = oof(s, yi, folds)
r, _ = oof(s, yi, folds, reg)
print(f"{os.path.basename(a.probs)}")
print(f"  global cuts  OOF macro-F1 : {g:.4f}")
print(f"  per-regime   OOF macro-F1 : {r:.4f}   ({r - g:+.4f})")
print(f"  regime sizes: {dict(pd.Series(reg).value_counts())}")
if a.emit:
    table = {"head": "ordinal_regime", "global": list(fit(s, yi)),
             "regimes": {g_: list(fit(s[reg == g_], yi[reg == g_])) for g_ in np.unique(reg)
                         if (reg == g_).sum() >= MIN_ROWS},
             "fashion_roots": sorted(FASHION), "technical_roots": sorted(TECHNICAL),
             "fitted_on": a.half, "n_rows": int(len(s)),
             "oof_global": round(float(g), 4), "oof_regime": round(float(r), 4)}
    os.makedirs(os.path.dirname(a.emit) or ".", exist_ok=True)
    json.dump(table, open(a.emit, "w"), indent=1, ensure_ascii=False)
    print(f"  wrote {a.emit}")
