"""Shared Stage-2 evaluation harness.

Everything downstream reports through this module so numbers stay comparable across
experiments. The metric is byte-equivalent to the organizer's stage2_data/evaluate.py.

Validation protocol (why it is built this way):
  Train / public / private are QUERY-DISJOINT (verified: 0 shared queries). public_test is the
  only in-distribution labelled data we have and it is also the only source of genuine
  `irrelevant` rows, so it has to serve as both fitting data and honest estimate. Every tunable
  (thresholds, calibration, blend weights) is therefore fitted inside query-disjoint folds and
  scored only out-of-fold. In-sample numbers are reported alongside, labelled as optimistic.
"""
from __future__ import annotations
import re
import numpy as np, pandas as pd

LABELS = ["irrelevant", "semi-relevant", "relevant"]
ORD = {l: i for i, l in enumerate(LABELS)}          # ordinal: I < S < R
S2 = "/home/jupyter/trendyol-teknofest-work/stage2_data"


# ---------------------------------------------------------------- Turkish-safe tokenisation
# BUG FOUND 2026-08-19 (A11, verified): 8.91% of DEV *queries* carry `i` + U+0307 COMBINING DOT
# ABOVE — the residue of lowercasing `İ` with a non-Turkish `.lower()`. ZERO of the 500k train
# queries do. `re.findall(r"\w+", "i̇phone")` yields ['i','phone'] because U+0307 is a combining
# mark and not a word character, so the token never matches the title. NFC does NOT fix it (there
# is no precomposed codepoint) — the mark must be stripped. Measured effect on affected rows:
# mean query-token coverage 0.5254 -> 0.6613 (+0.1359).
COMBINING_DOT = "\u0307"

def tr_norm(s: str) -> str:
    """NFC + strip the stray combining dot. Use before ANY tokenisation of these fields."""
    import unicodedata
    return unicodedata.normalize("NFC", str(s)).replace(COMBINING_DOT, "")

def toks(s: str) -> set:
    return set(re.findall(r"\w+", tr_norm(s).lower()))



# ---------------------------------------------------------------- metric
def macro_f1(y_true, y_pred) -> float:
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    return float(np.mean([_f1(y_true, y_pred, l) for l in LABELS]))

def per_class_f1(y_true, y_pred) -> dict:
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    return {l: _f1(y_true, y_pred, l) for l in LABELS}

def _f1(t, p, l):
    tp = int(((t == l) & (p == l)).sum()); fp = int(((t != l) & (p == l)).sum())
    fn = int(((t == l) & (p != l)).sum()); d = 2 * tp + fp + fn
    return 0.0 if d == 0 else 2 * tp / d

def confusion(y_true, y_pred) -> pd.DataFrame:
    return pd.crosstab(pd.Series(np.asarray(y_true), name="truth"),
                       pd.Series(np.asarray(y_pred), name="pred")).reindex(
                           index=LABELS, columns=LABELS).fillna(0).astype(int)


# ---------------------------------------------------------------- data + folds
def load_public() -> pd.DataFrame:
    pt = pd.read_csv(f"{S2}/public_test.csv", dtype=str, keep_default_na=False)
    sol = pd.read_csv(f"{S2}/public_solution.csv", dtype=str, keep_default_na=False)
    d = pt.merge(sol, on="row_id", validate="one_to_one")
    assert len(d) == 7500 and d.real_label.isin(LABELS).all()
    return d

def query_folds(queries, k: int = 5, seed: int = 20260819) -> np.ndarray:
    """Deterministic query-disjoint fold assignment. Same query -> same fold, always."""
    uq = sorted(set(map(str, queries)))
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(uq))
    m = {uq[p]: i % k for i, p in enumerate(perm)}
    return np.array([m[str(q)] for q in queries], dtype=np.int8)


# ---------------------------------------------------------------- ordinal 2-threshold fitting
def _grid(x, n=400):
    return np.unique(np.percentile(np.asarray(x, float), np.linspace(0.2, 99.8, n)))

def fit_cuts(x, y, grid=None):
    """Best (lo, hi) on one ordinal scalar: pred = R if x>hi, S if x>lo, else I.

    Vectorised: F1_irrelevant depends only on lo, F1_relevant only on hi, F1_semi on both,
    so the whole (lo, hi) surface is one outer-difference of cumulative class counts.
    """
    x = np.asarray(x, float); y = np.asarray(y)
    if grid is None: grid = _grid(x)
    o = np.argsort(x, kind="mergesort"); xs, ys = x[o], y[o]
    idx = np.searchsorted(xs, grid, side="right")
    cum = {l: np.concatenate([[0], np.cumsum(ys == l)])[idx].astype(float) for l in LABELS}
    cumall = idx.astype(float)
    N = {l: float((y == l).sum()) for l in LABELS}; n = float(len(x))
    tp_i = cum["irrelevant"]
    f1_i = 2 * tp_i / np.maximum(tp_i + cumall + N["irrelevant"] - tp_i, 1e-9)
    tp_r = N["relevant"] - cum["relevant"]
    f1_r = 2 * tp_r / np.maximum(tp_r + (n - cumall) + N["relevant"] - tp_r, 1e-9)
    tp_s = cum["semi-relevant"][None, :] - cum["semi-relevant"][:, None]
    pc_s = cumall[None, :] - cumall[:, None]
    f1_s = 2 * tp_s / np.maximum(tp_s + pc_s + N["semi-relevant"] - tp_s, 1e-9)
    M = (f1_i[:, None] + f1_s + f1_r[None, :]) / 3.0
    G = len(grid)
    M = np.where(np.arange(G)[:, None] < np.arange(G)[None, :], M, -1.0)
    k = np.unravel_index(np.argmax(M), M.shape)
    return float(M[k]), float(grid[k[0]]), float(grid[k[1]])

def apply_cuts(x, lo, hi):
    x = np.asarray(x, float)
    return np.where(x > hi, "relevant", np.where(x > lo, "semi-relevant", "irrelevant"))

def oof_scalar(x, y, folds, grid=None):
    """Honest estimate: cuts fitted on k-1 folds, applied to the held-out fold."""
    x = np.asarray(x, float); y = np.asarray(y); folds = np.asarray(folds)
    if grid is None: grid = _grid(x)
    pred = np.empty(len(x), dtype=object); cuts = []
    for f in np.unique(folds):
        tr, te = folds != f, folds == f
        _, lo, hi = fit_cuts(x[tr], y[tr], grid)
        pred[te] = apply_cuts(x[te], lo, hi); cuts.append((float(lo), float(hi)))
    ins, lo0, hi0 = fit_cuts(x, y, grid)
    return {"oof_macro_f1": macro_f1(y, pred), "oof_per_class": per_class_f1(y, pred),
            "in_sample_macro_f1": ins, "in_sample_cuts": (lo0, hi0),
            "fold_cuts": cuts, "oof_pred": pred}


# ---------------------------------------------------------------- diagnostics
def ordinal_auc(x, y) -> dict:
    """Pairwise AUCs for the boundaries that matter, plus Somers-style ordering checks."""
    x = np.asarray(x, float); y = np.asarray(y)
    def auc(pos, neg):
        a, b = x[np.isin(y, pos)], x[np.isin(y, neg)]
        if not len(a) or not len(b): return float("nan")
        r = pd.Series(np.concatenate([a, b])).rank().to_numpy()
        return float((r[:len(a)].sum() - len(a) * (len(a) + 1) / 2) / (len(a) * len(b)))
    return {"R_vs_rest": auc(["relevant"], ["semi-relevant", "irrelevant"]),
            "I_vs_rest": auc(["semi-relevant", "relevant"], ["irrelevant"]),
            "R_vs_S": auc(["relevant"], ["semi-relevant"]),
            "S_vs_I": auc(["semi-relevant"], ["irrelevant"]),
            "R_vs_I": auc(["relevant"], ["irrelevant"])}

def slice_report(d: pd.DataFrame, y_pred, min_n: int = 150) -> pd.DataFrame:
    """Per-slice macro-F1. Slices chosen from the measured train->test shifts."""
    d = d.copy(); d["_pred"] = np.asarray(y_pred)
    d["cat_root"] = d.category.str.split("/").str[0]
    d["qwords"] = d["query"].str.split().str.len().clip(1, 5)
    q = [toks(x) for x in d["query"]]
    t = [toks(a + " " + b + " " + c) for a, b, c in zip(d.title, d.category, d.brand)]
    d["cover"] = [len(a & b) / max(len(a), 1) for a, b in zip(q, t)]
    d["cover_band"] = pd.cut(d.cover, [-.01, .25, .5, .75, 1.01],
                             labels=["0-25%", "25-50%", "50-75%", "75-100%"])
    rows = []
    for col in ["cat_root", "qwords", "cover_band"]:
        for v, g in d.groupby(col, observed=True):
            if len(g) < min_n: continue
            rows.append({"slice": col, "value": str(v), "n": len(g),
                         "macro_f1": macro_f1(g.real_label, g._pred),
                         **{f"F1_{k[0].upper()}": v2 for k, v2 in
                            per_class_f1(g.real_label, g._pred).items()}})
    return pd.DataFrame(rows).sort_values(["slice", "macro_f1"])

def report(name, d, x, folds=None, verbose=True):
    """One-call summary for a single ordinal scalar."""
    y = d.real_label.values
    if folds is None: folds = query_folds(d["query"].values)
    r = oof_scalar(x, y, folds)
    a = ordinal_auc(x, y)
    if verbose:
        print(f"\n===== {name} =====")
        print(f"  OOF macro-F1 {r['oof_macro_f1']:.4f}   (in-sample {r['in_sample_macro_f1']:.4f}, optimistic)")
        print("  per-class F1: " + "  ".join(f"{k[0].upper()} {v:.4f}" for k, v in r["oof_per_class"].items()))
        print("  AUC: " + "  ".join(f"{k} {v:.4f}" for k, v in a.items()))
        print("  fold cuts: " + ", ".join(f"({lo:.2f},{hi:.2f})" for lo, hi in r["fold_cuts"]))
        print(confusion(y, r["oof_pred"]).to_string())
    return {"name": name, **{k: v for k, v in r.items() if k != "oof_pred"}, "auc": a,
            "oof_pred": r["oof_pred"]}


# ---------------------------------------------------------------- DEV / HOLDOUT discipline
SPLIT = "/home/jupyter/trendyol-teknofest-work/asya/s2_exp/common/public_split.csv"

def load_public_split(half: str | None = "dev") -> pd.DataFrame:
    """half='dev' (fit/select here) | 'holdout' (quarantined) | None (both, milestones only).

    public is 7,500 rows; private is 90,000. Anything chosen by looking at a public number is a
    chance to overfit 7,500 rows. DEV is where choices are made; HOLDOUT is the audit.
    """
    d = load_public().merge(pd.read_csv(SPLIT, dtype=str), on="row_id", validate="one_to_one")
    if half is not None:
        assert half in ("dev", "holdout")
        d = d[d.half == half].reset_index(drop=True)
    return d

def bootstrap_ci(y_true, y_pred, n_boot: int = 2000, seed: int = 0, groups=None):
    """Percentile CI for macro-F1. Resamples QUERIES when groups is given (rows within a query
    are not independent), else rows. Report this next to every headline number."""
    y_true = np.asarray(y_true); y_pred = np.asarray(y_pred)
    rng = np.random.default_rng(seed)
    if groups is not None:
        groups = np.asarray(groups); uq = np.unique(groups)
        idx_of = {g: np.flatnonzero(groups == g) for g in uq}
        stats = []
        for _ in range(n_boot):
            pick = rng.choice(uq, size=len(uq), replace=True)
            ii = np.concatenate([idx_of[g] for g in pick])
            stats.append(macro_f1(y_true[ii], y_pred[ii]))
    else:
        n = len(y_true)
        stats = [macro_f1(y_true[ii], y_pred[ii])
                 for ii in (rng.integers(0, n, n) for _ in range(n_boot))]
    s = np.sort(stats)
    return {"macro_f1": macro_f1(y_true, y_pred), "lo95": float(s[int(.025*n_boot)]),
            "hi95": float(s[int(.975*n_boot)]), "se": float(np.std(stats))}

def min_detectable_delta(n_rows: int = 3750) -> float:
    """Rough guard: differences smaller than this on a set of this size are noise, not signal."""
    return round(2 * 0.5 / np.sqrt(n_rows), 4)
