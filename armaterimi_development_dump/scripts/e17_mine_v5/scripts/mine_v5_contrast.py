"""E17 contrast rows for templated device-model queries (review 2026-08-22 pt 6).

mine_v5's templated M3b queries ("iphone 15 kılıf", ...) are synthetic: they have no organizer
R/S rows, so on their own they would teach "this query shape -> irrelevant". For every
templated query this emits up to PER_QUERY items from the SAME accessory leaves whose device
set CONTAINS the query's model (the right-model candidates). They are judged with everything
else and enter the set with whatever label the judge gives (expected relevant / semi).
Output: outputs/mined_v5_contrast.parquet  (query, item_key, recipe='M3b_device_match', rank,
        why, templated=True, seed_query_leaf, item_leaf)
"""
import os, sys, json, hashlib
import numpy as np, pandas as pd
W = "/home/jupyter/trendyol-teknofest-work"
E17 = f"{W}/asya/s2_exp/e17_mine_v5"
for p in (f"{W}/asya/s2_exp/rubric_eda", f"{W}/asya/s2_exp/e16_mine_v4/scripts"):
    sys.path.insert(0, p)
import A14_mine as M, a14_patches as P, templates as T, mine_v4 as V
PER_QUERY = 6
SEED = 20260823

R = M.res()
dev = T._device_index(R)
seeds = T.build_seeds(R)
rng = np.random.default_rng(SEED)
out = []
for q, m, cand_leaves, seed_leaf in seeds:
    keep = sorted({i for l in cand_leaves for i, dvs in dev[l].items() if m in dvs})
    if not keep:
        continue
    take = np.array(keep, int)
    if len(take) > PER_QUERY:
        take = np.sort(rng.choice(take, PER_QUERY, replace=False))
    df = M._emit(R, q, take, "M3b_device_match",
                 f"templated query; item carries the SAME host model {m} (contrast row)", seed_leaf)
    df["templated"] = True
    out.append(df)
con = pd.concat(out, ignore_index=True)
con["rank"] = con.groupby("query").cumcount().astype("int16")
con["item_key"] = V.item_key_of(con)
con = con.drop_duplicates(["query", "item_key"])
out_f = f"{E17}/outputs/mined_v5_contrast.parquet"
con[["query", "item_key", "recipe", "rank", "why", "templated", "seed_query_leaf", "item_leaf"]].to_parquet(out_f, index=False)
sha = hashlib.sha256(open(out_f, "rb").read()).hexdigest()
json.dump({"n": int(len(con)), "n_queries": int(con['query'].nunique()), "per_query": PER_QUERY,
           "seed": SEED, "sha256": sha}, open(out_f.replace(".parquet", ".manifest.json"), "w"), indent=1)
print(f"contrast rows: {len(con):,} over {con['query'].nunique():,} templated queries | sha {sha[:12]}")
print(con[["query", "title"]].head(8).to_string())
