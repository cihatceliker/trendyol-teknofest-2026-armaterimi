"""E15 build_student_data — assemble the student distillation set from up to three sources.

Sources (each independently switchable, so the S1/S2/S3 ablation ladder is one builder):
  (a) ORGANIZER=1  common/train_norm.parquet — all organizer R/S rows, hard labels.
  (b) JUDGED=1     e08_repair/outputs/matched_labels_v1.parquet, judge_label=='irrelevant'
                   only; item text materialized from e05_score/outputs/_pool_ext.parquet via
                   common/normalize.py normalize() + drop_keys.csv — the exact path of
                   e08_repair/scripts/build_data_matched.py. FAIL LOUDLY if missing.
  (c) TEACHER=...  comma-separated parquet globs of Asure-12B teacher probabilities.
                   Contract per row: float cols p_irr, p_semi, p_rel plus row identity —
                   EITHER an `item_key` column (text materialized from _pool_ext, e.g. the
                   mined candidates of e05_score/outputs/asure_scores_v32.parquet)
                   OR the full text columns (query,title,category,brand,gender,age_group,
                   attributes). Probs may cover any subset: rows matching an (a)/(b) row on
                   the FULL text identity (query + all 6 item fields — (query,title) alone is
                   unsafe: distinct item_keys can share a title while differing in brand or
                   attributes) get the probs attached; the rest become teacher-only rows with
                   real_label = "" (empty).

Query hygiene (reused from e11_si_lora/scripts/build_data.py): post-normalization train_norm
and public share 78 queries (1,626 rows — t1 collapses dotted-i variants), so every row whose
query appears in public_test_norm.parquet is dropped from ALL sources, and zero overlap is
asserted on the final set. public_test_norm is read test-side columns only, never labels.

Output OUT/student_train.parquet, columns:
  query, title, category, brand, gender, age_group, attributes,
  real_label ('' for teacher-only rows), p_irr/p_semi/p_rel (NaN where no teacher probs),
  source in {organizer, judged, teacher_only}.
Deterministic (stable sort -> seeded shuffle, seed 42) + OUT/student_train.manifest.json.

Env: ORGANIZER JUDGED TEACHER LABELS OUT ORG_N (0=all; per-class cap for quick ablations)
"""
import glob as _glob
import os, sys, json, time
import numpy as np, pandas as pd

W = "/home/jupyter/trendyol-teknofest-work"
sys.path.insert(0, f"{W}/asya/s2_exp/common")
sys.path.insert(0, f"{W}/asya/s2_exp/e11_si_lora/scripts")
import normalize as NZ
from si_common import log, sha256

ORGANIZER = os.environ.get("ORGANIZER", "1") == "1"
JUDGED    = os.environ.get("JUDGED", "1") == "1"
TEACHER   = os.environ.get("TEACHER", "")
LABELS    = os.environ.get("LABELS", f"{W}/asya/s2_exp/e08_repair/outputs/matched_labels_v1.parquet")
OUT       = os.environ.get("OUT", f"{W}/asya/s2_exp/e15_student/outputs")
ORG_N     = int(os.environ.get("ORG_N", 0))
TRAIN  = f"{W}/asya/s2_exp/common/train_norm.parquet"
POOL   = f"{W}/asya/s2_exp/e05_score/outputs/_pool_ext.parquet"
PUBLIC = f"{W}/asya/s2_exp/common/public_test_norm.parquet"
DROPK  = f"{W}/asya/s2_exp/common/drop_keys.csv"
SEED   = 42
TXT    = ["query", "title", "category", "brand", "gender", "age_group", "attributes"]
PCOLS  = ["p_irr", "p_semi", "p_rel"]
KEY_SEP = "\x1f"

assert ORGANIZER or JUDGED or TEACHER, "all three sources disabled — nothing to build"
pubq = set(pd.read_parquet(PUBLIC, columns=["query"])["query"])
drop_keys = pd.read_csv(DROPK).key.tolist()
inputs_used = [PUBLIC, DROPK]                       # every file actually read, for the manifest
_pool = None
def pool():
    """Normalized item pool, loaded once, indexed by item_key."""
    global _pool
    if _pool is None:
        _pool = NZ.normalize(pd.read_parquet(POOL), drop_keys).set_index("item_key")
        inputs_used.append(POOL)
    return _pool

def materialize(df, label, source):
    """(query,item_key) rows -> full text rows via the pool (e08 build_data_matched path)."""
    it = pool()
    missing = ~df.item_key.isin(it.index)
    assert not missing.any(), f"{int(missing.sum())} {source} item_keys not in _pool_ext"
    m = it.loc[df.item_key]
    out = pd.DataFrame({"query": df["query"].values, "title": m.title.values,
                        "category": m.category.values, "brand": m.brand.values,
                        "gender": m.gender.values, "age_group": m.age_group.values,
                        "attributes": m.attributes.values})
    out["real_label"], out["source"] = label, source
    return out

def rowkey(df):
    """Full-text row identity: query + the 6 item fields, \\x1f-joined (vectorized concat)."""
    k = df["query"].astype(str)
    for c in TXT[1:]:
        k = k + KEY_SEP + df[c].astype(str)
    return k

def drop_public(df, name, counters):
    n0 = len(df); df = df[~df["query"].isin(pubq)].reset_index(drop=True)
    counters[name] = n0 - len(df)
    return df

dropped, parts = {}, []

# ---------------------------------------------------------------- (a) organizer R/S
if ORGANIZER:
    tr = pd.read_parquet(TRAIN); inputs_used.append(TRAIN)
    tr = drop_public(tr, "organizer", dropped)
    if ORG_N:
        tr = (tr.sort_values(["real_label", "query", "title"], kind="mergesort")
                .groupby("real_label", group_keys=False)
                .apply(lambda g: g.sample(n=min(ORG_N, len(g)), random_state=SEED)))
    tr = tr.reset_index(drop=True); tr["source"] = "organizer"
    log(f"organizer: {len(tr):,} rows {tr.real_label.value_counts().to_dict()}")
    parts.append(tr)

# ---------------------------------------------------------------- (b) judged irrelevants
if JUDGED:
    if not os.path.exists(LABELS):
        sys.exit(f"FATAL: {LABELS} does not exist (e08 labeling not finished?). Refusing to "
                 f"build without judged irrelevants — do NOT substitute unjudged mined rows.")
    lab = pd.read_parquet(LABELS); inputs_used.append(LABELS)
    assert {"query", "item_key", "judge_label"} <= set(lab.columns), f"bad schema in {LABELS}"
    lab["query"] = lab["query"].map(NZ.t1)
    mi = (lab[lab.judge_label == "irrelevant"]
          .drop_duplicates(["query", "item_key"]).reset_index(drop=True))
    assert len(mi) > 0, "no judge_label=='irrelevant' rows in the labels file"
    I = materialize(mi, "irrelevant", "judged").drop_duplicates(["query", "title"])
    I = drop_public(I, "judged", dropped)
    log(f"judged-I: {len(I):,} rows (from {len(lab):,} labels)")
    parts.append(I)

# ---------------------------------------------------------------- (b3) stage-1 head-A rows (e21)
# Complementary labels: "relevant" / "not relevant" only. They train the relevant-vs-rest head and
# ABSTAIN on the semi-vs-irrelevant head (TWO_HEAD mode in train_student.py). Text already comes
# from the Stage-2 pool + v2 cleaning inside the e21 builder, so no Stage-1 catalog text enters.
STAGE1 = os.environ.get("STAGE1", "")
STAGE1_BANDS = os.environ.get("STAGE1_BANDS", "confident")     # confident | confident,uncertain
if STAGE1:
    s1 = pd.read_parquet(STAGE1); inputs_used.append(STAGE1)
    bands = [b.strip() for b in STAGE1_BANDS.split(",") if b.strip()]
    s1 = s1[s1.band.isin(bands)].copy()
    s1["real_label"] = np.where(s1.head_a_label == 1, "s1_rel", "s1_nonrel")
    s1["source"] = "stage1_headA"
    s1 = s1[TXT + ["real_label", "source"]].drop_duplicates(["query", "title"])
    s1 = drop_public(s1, "stage1", dropped)
    log(f"stage1 head-A: {len(s1):,} rows (bands={bands}) {s1.real_label.value_counts().to_dict()}")
    parts.append(s1)

# ---------------------------------------------------------------- (b2) contrast rows (e17): judged R/S/I, enter with the judge label
CONTRAST = os.environ.get("CONTRAST", "")
if CONTRAST:
    con = pd.read_parquet(CONTRAST); inputs_used.append(CONTRAST)
    con = con[con.judge_label.isin(["relevant", "semi-relevant", "irrelevant"])].copy()
    con["query"] = con["query"].map(NZ.t1)
    cparts = [materialize(g.drop_duplicates(["query", "item_key"]), lab_, "contrast")
              for lab_, g in con.groupby("judge_label")]
    C = pd.concat(cparts, ignore_index=True).drop_duplicates(["query", "title"])
    C = drop_public(C, "contrast", dropped)
    log(f"contrast: {len(C):,} rows {C.real_label.value_counts().to_dict()}")
    parts.append(C)

base = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=TXT + ["real_label", "source"])
# ---------------------------------------------------------------- attribute cleaning v2 (e18), opt-in, applied to EVERY source
ATTR_CLEAN = os.environ.get("ATTR_CLEAN", "v1")
if ATTR_CLEAN == "v2":
    sys.path.insert(0, f"{W}/asya/s2_exp/e18_attr_clean")
    import attr_clean as AC
    base = AC.apply_v2(base, f"{W}/asya/s2_exp/e18_attr_clean/drop_keys_v2.csv", drop_boilerplate=True)
    log(f"ATTR_CLEAN=v2 applied: mean attr chars {base.attributes.str.len().mean():.1f}")
for c in PCOLS: base[c] = np.nan

# ---------------------------------------------------------------- (c) teacher probabilities
teacher_files = sorted(f for pat in TEACHER.split(",") if pat.strip()
                       for f in _glob.glob(pat.strip()))
assert not TEACHER or teacher_files, f"TEACHER={TEACHER!r} matched no files"
n_attached = n_teacher_only = 0
if teacher_files:
    tps = []
    for f in teacher_files:
        t = pd.read_parquet(f)
        assert set(PCOLS) <= set(t.columns), f"{f} lacks {PCOLS}"
        t["query"] = t["query"].map(NZ.t1)
        if "item_key" in t.columns:
            keep = t[["query", "item_key"] + PCOLS].drop_duplicates(["query", "item_key"])
            m = materialize(keep, "", "teacher_only")
            m[PCOLS] = keep[PCOLS].to_numpy(np.float32)
            tps.append(m)
        else:
            assert set(TXT) <= set(t.columns), \
                f"{f} carries neither item_key nor the full text columns {TXT}"
            m = NZ.normalize(t[TXT], drop_keys)        # idempotent on already-normalized text
            m[PCOLS] = t[PCOLS].to_numpy(np.float32)
            m["real_label"], m["source"] = "", "teacher_only"
            tps.append(m)
    tp = pd.concat(tps, ignore_index=True)
    P = tp[PCOLS].to_numpy(np.float32)
    assert np.isfinite(P).all() and (P >= 0).all(), "teacher probs must be finite and >= 0"
    assert np.allclose(P.sum(1), 1.0, atol=1e-3), "teacher probs must sum to 1 (atol 1e-3)"
    tp[PCOLS] = P / P.sum(1, keepdims=True)            # exact renormalization
    # resolve duplicate identities: near-identical probs collapse to first, else fail
    tp["_k"] = rowkey(tp)
    dupm = tp["_k"].duplicated(keep=False)
    if dupm.any():                                     # spread check on the dup keys only
        g = tp.loc[dupm].groupby("_k")[PCOLS]
        spread = (g.max() - g.min()).to_numpy()
        assert spread.max() < 0.05, \
            f"conflicting teacher probs for one row identity: max spread {spread.max():.3f}"
    n_dup = int(tp["_k"].duplicated().sum())
    tp = (tp.sort_values(["_k"] + PCOLS, kind="mergesort")
            .drop_duplicates("_k", keep="first").reset_index(drop=True))
    tp = drop_public(tp, "teacher", dropped)
    # attach to base rows on the full text identity; leftovers become teacher-only rows
    bk = rowkey(base)
    probs = tp.set_index("_k")[PCOLS]
    hit = bk.isin(probs.index)
    base.loc[hit, PCOLS] = probs.loc[bk[hit]].to_numpy(np.float32)
    n_attached = int(hit.sum())
    only = tp[~tp["_k"].isin(set(bk))].drop(columns="_k").reset_index(drop=True)
    n_teacher_only = len(only)
    log(f"teacher: {len(tp):,} scored rows from {len(teacher_files)} file(s) "
        f"({n_dup} dup keys collapsed) -> {n_attached:,} attached, {n_teacher_only:,} teacher-only")
    base = only if base.empty else pd.concat([base, only], ignore_index=True)

# ---------------------------------------------------------------- finalize (deterministic)
cols = TXT + ["real_label"] + PCOLS + ["source"]
out = (base[cols]
       .sort_values(["source", "real_label", "query", "title"], kind="mergesort")
       .sample(frac=1, random_state=SEED).reset_index(drop=True))
VALID_LABELS = ["irrelevant", "semi-relevant", "relevant", "s1_rel", "s1_nonrel"]  # s1_* = head-A-only (e21)
has_hard = out.real_label.isin(VALID_LABELS)
has_teacher = out.p_irr.notna()
assert (has_hard | has_teacher).all(), "row with neither hard label nor teacher probs"
assert (out.real_label.isin(VALID_LABELS + [""])).all()
assert out.title.notna().all() and (out.title != "").all()
assert not (set(out["query"]) & pubq), "public query leaked into the student set"
pt = out[PCOLS].to_numpy(np.float32)[has_teacher.to_numpy()]
assert len(pt) == 0 or np.allclose(pt.sum(1), 1.0, atol=1e-4)

os.makedirs(OUT, exist_ok=True)
out_p = f"{OUT}/student_train.parquet"
out.to_parquet(out_p, index=False)
manifest = {
    "rows": int(len(out)), "seed": SEED,
    "attr_clean": ATTR_CLEAN, "stage1": STAGE1, "stage1_bands": STAGE1_BANDS,
    "sources": {"organizer": ORGANIZER, "judged": JUDGED,
                "teacher_files": teacher_files, "org_n": ORG_N},
    "source_counts": out.source.value_counts().to_dict(),
    "class_counts": out[has_hard].real_label.value_counts().to_dict(),
    "teacher_rows": int(has_teacher.sum()),
    "teacher_attached_to_labeled": n_attached, "teacher_only_rows": n_teacher_only,
    "dropped_public_query_rows": dropped,
    "inputs_sha256": {os.path.basename(p): sha256(p)
                      for p in sorted(set(inputs_used + teacher_files))},
    "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
}
json.dump(manifest, open(f"{out_p.removesuffix('.parquet')}.manifest.json", "w"), indent=1)
log(f"wrote {out_p}: {len(out):,} rows | sources {manifest['source_counts']} | "
    f"hard {manifest['class_counts']} | teacher-covered {manifest['teacher_rows']:,}")
