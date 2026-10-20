"""E17 v5-only recipe: M10 retrieval near-miss — mimic the organizer's RETRIEVE-then-JUDGE step.

Gate diagnosis 2026-08-22 (gates_v5_pre.log): the rule recipes put 55% of pairs at ZERO token
overlap with the item (organizer irrelevants: 19%) and 17% brand-in-query (organizer: 40%),
adversarial AUC 0.80. The organizer's irrelevants are lexically-near retrieved items that a
judge rejected. Since every v5 row is judged, a recipe's purity only affects cost, so this one
reproduces the retriever directly:
  * tf-idf (sublinear tf) over title + category + brand tokens of every Pool-A item;
  * per train query with a modal leaf: top-K by cosine, minus the query's own R/S rows, minus
    items in the query's FORBIDDEN leaves (own leaf, A4 family, pair rules, organizer guard —
    the same semi-avoidance every other recipe uses; guard-1 family extension applied);
  * emit the first `per_query` survivors in retrieval order (rank = retrieval rank).
Expected judge outcome: mostly irrelevant, a sizeable semi tail (quarantined), few relevant.
"""
import sys
import numpy as np, pandas as pd
RUBRIC = "/home/jupyter/trendyol-teknofest-work/asya/s2_exp/rubric_eda"
if RUBRIC not in sys.path:
    sys.path.insert(0, RUBRIC)
import A14_mine as M
from A14_mine import toks, _seed_queries

TOPK = 80
CHUNK = 256


def _index(R):
    idx = getattr(R, "_e17_tfidf", None)
    if idx is not None:
        return idx
    from sklearn.feature_extraction.text import TfidfVectorizer
    docs = [" ".join(toks(f"{t} {c} {b}")) for t, c, b in
            zip(R.items.title.values, R.items.category.values, R.items.brand.values)]
    vec = TfidfVectorizer(analyzer=str.split, sublinear_tf=True, min_df=2, dtype=np.float32)
    X = vec.fit_transform(docs).tocsr()
    R._e17_tfidf = (vec, X)
    return R._e17_tfidf


def mine_M10_retrieval(n=500, seed=0, per_query=20, R=None, verbose=False):
    R = R or M.res()
    vec, X = _index(R)
    XT = X.T.tocsc()
    lf = R.items.leaf.values

    def ok(q):
        return bool(R.qinfo_d.get(q, {}).get("modal_leaf"))

    qs = _seed_queries(R, ok, seed)
    out, total = [], 0
    for s in range(0, len(qs), CHUNK):
        chunk = qs[s:s + CHUNK]
        Q = vec.transform([" ".join(toks(q)) for q in chunk])
        S = (Q @ XT).toarray()                       # CHUNK x n_items, float32
        for qi, q in enumerate(chunk):
            sc = S[qi]
            if not sc.any():
                continue
            top = np.argpartition(-sc, TOPK)[:TOPK]
            top = top[np.argsort(-sc[top], kind="stable")]
            top = top[sc[top] > 0]
            used = set(R.q_rows.get(q, ()))
            L = R.qinfo_d[q]["modal_leaf"]
            bad = M._forbidden_leaves(R, L, q)
            keep = [int(i) for i in top if i not in used and lf[i] not in bad][:per_query]
            if not keep:
                continue
            out.append(M._emit(R, q, np.asarray(keep, int), "M10_retrieval_near",
                               f"tf-idf top-{TOPK} retrieval minus own R/S rows and forbidden leaves of {L}", L))
            total += len(out[-1])
        if total >= n:
            break
    return pd.concat(out, ignore_index=True).head(n) if out else pd.DataFrame(columns=M.OUT_COLS)
