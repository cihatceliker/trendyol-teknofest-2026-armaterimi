"""Freeze the public-set DEV / HOLDOUT split. Run ONCE. Never regenerate.

Why: public is 7,500 rows, private is 90,000. Every tunable fitted on public and every variant
chosen by looking at a public score is a chance to overfit 7,500 rows and miss on 90,000.
So public is split query-disjoint in half:
  DEV      - all fitting, all variant comparison, all threshold/blend selection
  HOLDOUT  - quarantined; read only at milestones, always with a bootstrap CI
Queries never span halves (train/public/private are query-disjoint by design; we preserve that).
"""
import numpy as np, pandas as pd, hashlib, json

S2 = "/home/jupyter/trendyol-teknofest-work/stage2_data"
OUT = "/home/jupyter/trendyol-teknofest-work/asya/s2_exp/common/public_split.csv"
SEED = 20260819

pt = pd.read_csv(f"{S2}/public_test.csv", dtype=str, keep_default_na=False)
sol = pd.read_csv(f"{S2}/public_solution.csv", dtype=str, keep_default_na=False)
d = pt.merge(sol, on="row_id", validate="one_to_one")
assert len(d) == 7500

# Greedy balanced assignment over queries: keeps both halves close in row count AND per-class
# count, without ever splitting a query. Deterministic given SEED.
qs = d.groupby("query").real_label.value_counts().unstack(fill_value=0)
for c in ["irrelevant", "semi-relevant", "relevant"]:
    if c not in qs: qs[c] = 0
qs = qs[["irrelevant", "semi-relevant", "relevant"]]
rng = np.random.default_rng(SEED)
order = rng.permutation(len(qs))
qs = qs.iloc[order]
tot = np.zeros((2, 3), dtype=int)
assign = {}
for q, row in qs.iterrows():
    v = row.to_numpy()
    # put this query in whichever half is currently further behind on the classes it carries
    cost = [np.sum((tot[h] + v - (tot.sum(0) + v) / 2) ** 2) for h in (0, 1)]
    h = int(np.argmin(cost))
    assign[q] = h; tot[h] += v

d["half"] = d["query"].map(assign).map({0: "dev", 1: "holdout"})
assert d.groupby("query").half.nunique().max() == 1, "a query spans both halves"
d[["row_id", "half"]].to_csv(OUT, index=False)

summ = d.groupby(["half", "real_label"]).size().unstack()
print(summ.to_string())
print("\nrows per half:", d.half.value_counts().to_dict())
print("queries per half:", d.groupby("half")['query'].nunique().to_dict())
print("split sha256:", hashlib.sha256(open(OUT, 'rb').read()).hexdigest()[:16])
