"""E17 raw mining (v5) — ranked candidate lists per query, for the v5 selection step.

What changed vs e16 v4 (and why — see e17 README / LOG):
  * Generators emit a RANKED LIST per query (depth = PER_QUERY[recipe]) instead of a
    small fixed take. Selection walks each list rank-by-rank and skips items already used
    ITEM_CAP times, so item collisions cost nothing (v4 discarded 65k rows post hoc).
  * No gender/attribute preference weighting. The organizer's 62% gender-known rate is a
    consequence of its department profile (fashion items are 99% tagged); the department
    targets in select_v5 deliver it directly.
  * GLOBAL ITEM CAP AT GENERATION TIME: `_emit` is wrapped so an item can appear in at most
    ITEM_CAP rows across all recipes; generators search SEARCH_MULT x deeper than the emitted
    depth and walk past used-up items. Diagnosed 2026-08-22: the hardness ranker hands the same
    top items to every similar query, so a post-hoc cap threw away 41k fashion rows; capping at
    the source lets each recipe reach the alternatives it actually has. Scarce recipes run first.
  * Every recipe is run to seed exhaustion (n = NO_CAP). Per-query budgets are NOT applied
    here — raw is the supply, selection is the demand.
  * Public-query collisions (U+0307 normalisation) are dropped HERE as well as in selection.
Output schema (frozen): query, item_key, recipe, step, rank, why, templated,
                        seed_query_leaf, item_leaf
Env: MV5_SEED (20260823) | MV5_SMOKE=1 | MV5_RECIPES=csv of step names
"""
import hashlib, json, math, os, sys, time, traceback
import numpy as np, pandas as pd

W = "/home/jupyter/trendyol-teknofest-work"
E17 = f"{W}/asya/s2_exp/e17_mine_v5"
E16S = f"{W}/asya/s2_exp/e16_mine_v4/scripts"
RUBRIC = f"{W}/asya/s2_exp/rubric_eda"
COMMON = f"{W}/asya/s2_exp/common"
for p in (RUBRIC, E16S, COMMON):
    if p not in sys.path:
        sys.path.insert(0, p)
import A14_mine as M
import a14_patches as P            # importing applies the guard-1 A4-family extension
import templates as T
import mine_v4 as V                # item_key_of / verify_item_key (hash formula)
import recipes_v5 as RV            # M10 retrieval near-miss (v5-only)
import normalize as NZ

SEED = int(os.environ.get("MV5_SEED", "20260823"))
SMOKE = os.environ.get("MV5_SMOKE", "0") == "1"
ONLY = {s for s in os.environ.get("MV5_RECIPES", "").split(",") if s}
OUT_STEM = f"{E17}/outputs/mined_v5_raw"
NO_CAP = 2_000_000

# (step, short, fn, emitted depth, kwargs), in PRIORITY ORDER for the global item cap
# (scarce / high-value recipes claim items first; random-far last). Depth = rows kept per
# query after the cap; the generator is asked for SEARCH_MULT x depth candidates.
STEPS = [
    ("M1_brand_swap",      "M1",  M.mine_M1_brand_swap,   30, {}),
    ("M8_qualifier_strip", "M8",  P.mine_M8_relaxed,      15, {"min_viol": 1, "min_qual": 1}),
    ("M10_retrieval_near", "M10", RV.mine_M10_retrieval,  20, {}),
    ("M3a_color",          "M3a", M.mine_M3a_color,        6, {}),
    ("M3b_device_native",  "M3b", M.mine_M3b_device,      10, {}),
    ("M3c_numeric",        "M3c", M.mine_M3c_numeric,      6, {}),
    ("M5_wrong_entity",    "M5",  M.mine_M5_wrong_entity,  6, {}),
    ("M2_crosstype",       "M2",  P.mine_M2_patched,      20, {"prefer_meta": False}),
    ("M4_wrong_accessory", "M4",  P.mine_M4_patched,       3, {}),
    ("M6_lexical_trap",    "M6",  M.mine_M6_lexical_trap,  8, {}),
    ("M7_random_far",      "M7",  M.mine_M7_random_far,    2, {}),
]
M3B_TEMPLATED_PER_QUERY = 20
ITEM_CAP = 3
SEARCH_MULT = 3

# ---- global item cap at the emit point (one patch covers every recipe)
from collections import defaultdict
USAGE = defaultdict(int)
EMIT_DEPTH = {"M1_brand_swap": 30, "M8_qualifier_strip": 15, "M10_retrieval_near": 20, "M3a_color": 6, "M3b_device": 10,
              "M3c_numeric": 6, "M5_wrong_entity": 6, "M2_crosstype": 20, "M4_wrong_accessory": 3,
              "M6_lexical_trap": 8, "M7_random_far": 2}
_orig_emit = M._emit
CAP_STATS = defaultdict(lambda: [0, 0])          # recipe -> [offered, skipped_for_cap]

def _emit_capped(R, q, rows, recipe, why, qleaf):
    depth = EMIT_DEPTH.get(recipe, 30)
    keep = []
    st = CAP_STATS[recipe]
    for i in np.asarray(rows, dtype=int):
        st[0] += 1
        if USAGE[int(i)] >= ITEM_CAP:
            st[1] += 1
            continue
        keep.append(int(i))
        if len(keep) >= depth:
            break
    for i in keep:
        USAGE[i] += 1
    return _orig_emit(R, q, np.asarray(keep, dtype=int), recipe, why, qleaf)

M._emit = _emit_capped
P._emit = _emit_capped
T._emit = _emit_capped


def log(*a):
    print(f"[mine_v5 {time.strftime('%H:%M:%S')}]", *a, flush=True)


def gender_known(df):
    return round(float((df.gender.ne("unknown") & df.gender.ne("")).mean()), 3) if len(df) else 0.0


def main():
    t_start = time.time()
    kv = V.verify_item_key()
    log("item_key formula verified:", kv)
    R = M.res()
    pub = pd.read_parquet(f"{COMMON}/public_test_norm.parquet", columns=["query"])
    pubq = set(pub["query"].map(NZ.t1))

    frames, stats, failures = [], [], {}
    for step, short, fn, pq, kw in STEPS:
        if ONLY and step not in ONLY:
            continue
        n = 300 if SMOKE else NO_CAP
        t0 = time.time()
        try:
            df = fn(n, seed=SEED, per_query=pq * SEARCH_MULT, R=R, **kw)
        except Exception:
            failures[step] = traceback.format_exc()
            log(f"!! {step} FAILED\n{failures[step]}")
            continue
        dt = time.time() - t0
        df["templated"] = False
        df["step"] = step
        frames.append(df)
        stats.append({"step": step, "recipe": short, "depth": pq, "search_depth": pq * SEARCH_MULT,
                      "cap_offered": CAP_STATS[df.recipe.iloc[0]][0] if len(df) else 0,
                      "cap_skipped": CAP_STATS[df.recipe.iloc[0]][1] if len(df) else 0,
                      "generated": int(len(df)),
                      "n_queries": int(df["query"].nunique()) if len(df) else 0,
                      "gender_known": gender_known(df), "sec": round(dt, 1), "kwargs": kw})
        log(f"{step}: {len(df):,} rows over {df['query'].nunique():,} queries "
            f"(depth {pq}, gender_known {gender_known(df)}) in {dt:.0f}s")

    if not ONLY or "M3b_device_templated" in ONLY:
        t0 = time.time()
        try:
            df = T.mine_M3b_templated(300 if SMOKE else NO_CAP, seed=SEED,
                                      per_query=M3B_TEMPLATED_PER_QUERY * SEARCH_MULT, R=R)
            df["step"] = "M3b_device_templated"
            frames.append(df)
            stats.append({"step": "M3b_device_templated", "recipe": "M3b",
                          "depth": M3B_TEMPLATED_PER_QUERY, "generated": int(len(df)),
                          "n_queries": int(df["query"].nunique()) if len(df) else 0,
                          "gender_known": gender_known(df), "sec": round(time.time() - t0, 1)})
            log(f"M3b_device_templated: {len(df):,} rows over {df['query'].nunique():,} "
                f"synthetic queries in {time.time() - t0:.0f}s")
        except Exception:
            failures["M3b_device_templated"] = traceback.format_exc()
            log(f"!! M3b_device_templated FAILED\n{failures['M3b_device_templated']}")

    raw = pd.concat(frames, ignore_index=True)
    raw["templated"] = raw.templated.astype(bool)
    # rank = position in the generator's emit order within (query, step): for the ranked
    # recipes (M1/M2/M4/M6) that is hardness order; for the sampled ones it is arbitrary.
    raw["rank"] = raw.groupby(["query", "step"]).cumcount().astype("int16")
    raw["item_key"] = V.item_key_of(raw)
    n_gen = len(raw)

    # 1. never emit a pair that is R/S for that query in train
    tp = R.pairs[["query", "ikey"]].rename(columns={"ikey": "item_key"}).drop_duplicates()
    raw = raw.merge(tp.assign(_intrain=True), on=["query", "item_key"], how="left")
    n_train_drop = int(raw._intrain.notna().sum())
    raw = raw[raw._intrain.isna()].drop(columns="_intrain")
    # 2. public-query collisions after normalisation (78 strings; raw CSVs are disjoint)
    m = raw["query"].map(NZ.t1).isin(pubq)
    n_pubq = int(m.sum())
    raw = raw[~m]
    # 3. cross-recipe dedup keeping the RARER recipe (rank of the kept row is preserved)
    counts = raw.recipe.value_counts().to_dict()
    raw["_rarity"] = raw.recipe.map(counts)
    raw = raw.sort_values(["_rarity", "recipe", "query", "rank"], kind="stable")
    before = len(raw)
    raw = raw.drop_duplicates(["query", "item_key"], keep="first").drop(columns="_rarity")
    n_dedup = before - len(raw)
    raw = raw.sort_values(["recipe", "templated", "query", "rank"], kind="stable").reset_index(drop=True)
    for s in stats:
        s["kept"] = int((raw.step == s["step"]).sum())

    # samples, 25/step post-filter
    rng = np.random.default_rng(SEED)
    lines = [f"mined_v5 samples | seed {SEED} | {time.strftime('%Y-%m-%d %H:%M:%S')}",
             "25 rows per step, drawn AFTER train-drop + public-drop + dedup", ""]
    for step in raw.step.unique():
        g = raw[raw.step == step]
        take = g.iloc[np.sort(rng.choice(len(g), min(25, len(g)), replace=False))]
        lines.append(f"{'=' * 90}\n=== {step}  ({len(g):,} rows kept) ===\n{'=' * 90}")
        for _, r in take.iterrows():
            lines += [f"Q     : {r['query']}   [rank {r['rank']}]",
                      f"item  : {r.title}",
                      f"        cat={r.category} | brand={r.brand} | g={r.gender}/{r.age_group}",
                      f"        attrs={str(r.attributes)[:180]}",
                      f"why   : {str(r.why)[:220]}", "-" * 90]
        lines.append("")
    with open(f"{E17}/outputs/mined_v5_samples.txt", "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))

    out = raw[["query", "item_key", "recipe", "step", "rank", "why", "templated",
               "seed_query_leaf", "item_leaf"]].copy()
    out_f = f"{OUT_STEM}.parquet"
    out.to_parquet(out_f, index=False)
    sha = hashlib.sha256(open(out_f, "rb").read()).hexdigest()
    root = raw.category.map(M.root_of)
    manifest = {
        "script": "e17_mine_v5/scripts/mine_v5.py", "version": "v5",
        "created": time.strftime("%Y-%m-%d %H:%M:%S"), "seed": SEED, "smoke": SMOKE,
        "item_key_verification": kv,
        "design": {"ranked_lists": True, "preference_weighting": False,
                   "item_cap_at_generation": ITEM_CAP, "search_mult": SEARCH_MULT,
                   "priority_order": [st for st, *_ in STEPS] + ["M3b_device_templated"],
                   "max_item_reuse_in_raw": int(out.item_key.value_counts().max()),
                   "depth": {s: d for s, _, _, d, _ in STEPS} | {"M3b_device_templated": M3B_TEMPLATED_PER_QUERY}},
        "per_step": stats,
        "per_recipe_final": out.recipe.value_counts().to_dict(),
        "root_share_raw": root.value_counts(normalize=True).round(4).head(12).to_dict(),
        "gender_known_overall": gender_known(raw),
        "n_generated": n_gen, "n_train_pair_dropped": n_train_drop,
        "n_public_query_dropped": n_pubq, "n_cross_recipe_dedup": n_dedup,
        "n_final": len(out), "n_queries": int(out["query"].nunique()),
        "n_items": int(out.item_key.nunique()),
        "failures": failures, "runtime_sec": round(time.time() - t_start, 1), "sha256": sha,
        "schema": list(out.columns),
    }
    with open(f"{OUT_STEM}.manifest.json", "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=1, ensure_ascii=False)
    log(f"wrote {out_f}: {len(out):,} rows, {out['query'].nunique():,} queries, "
        f"{out.item_key.nunique():,} items | sha {sha[:12]} | {time.time() - t_start:.0f}s")
    log("per recipe:", json.dumps(manifest["per_recipe_final"]))
    log("root share:", json.dumps(manifest["root_share_raw"], ensure_ascii=False))
    if failures:
        log("FAILURES:", list(failures))


if __name__ == "__main__":
    main()
