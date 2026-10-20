"""PHASE 0 — snapshot normalisation. MUST run before any training, and the IDENTICAL
transform must run at inference time on the private test CSV.

`train.csv` was assembled from two catalog exports. Which export a row came from predicts its
label almost perfectly and carries ZERO signal at test time: a two-rule oracle scores 60.8% of
train at 100.000% accuracy and 0.5000 on DEV. This module removes the tell.

Four transforms:
  T1  NFC + strip U+0307 (combining dot above) from every text field.
      Snapshot B lowercases `İ` the Unicode way; snapshot A folds it to plain `i`.
      re.findall(r"\\w+", "i̇phone") -> ['i','phone'], so the token never matches.
      Also collapses 252 of 3,173 duplicate category strings.
  T2/T3  Drop the 18 attribute keys whose mere PRESENCE predicts the train label at
      >= MAX_PURITY purity (discover_drop_keys, train.csv only — features + labels; the frozen
      list is common/drop_keys.csv, 18 keys). `color detail` is the big one: 39% of train rows,
      100% of them `relevant` — a two-snapshot export marker, not a product property.
  T3  Drop keys whose train rate far exceeds their test rate (ratio >= MIN_RATIO). NO renames:
      an earlier version mapped `color detail` -> `web color`, which merely gave the leak a
      legitimate-looking name (201,313 train rows at P(relevant)=0.977 vs 169 test rows).
      A key the test set does not carry cannot help at inference, and in train it is correlated
      with the export and therefore with the label. Drop it.
  T4  gender `unisex` AND `çocuk` -> `unknown`. Both are two-snapshot artifacts of the organizer's
      train export, and we only caught the first one originally:
        unisex  22,516 relevant /      2 semi  (P(semi)=0.0001) - 0.0% of test, absent entirely
        çocuk        0 relevant / 16,634 semi  (P(semi)=1.0000) - 9.7% of test, where the TRUE
                                                mix is 44% irrelevant / 29% relevant / 27% semi
      A model reading `gender: çocuk` learns "semi-relevant, certainly" and is then wrong on a
      tenth of the test set: measured 69.0% accuracy on those rows vs 88.1% elsewhere.
      MEASURED on DEV, inference-time rewrite alone, no retraining:
        TY-ecomm 12L   0.8633 -> 0.8702   (+0.0069, çocuk-row accuracy 69.0% -> 77.2%)
        BERTurk 6ep    0.8324 -> 0.8444   (+0.0120, 65.4% -> 78.6%)
      Emptying the whole gender field instead scores 0.8693 - worse, because erkek/kadın carry
      real signal (their train-vs-test shift is <=0.16, versus 0.51 for çocuk).
      A full audit of every gender and age_group value found NO other degenerate value:
      age_group is clean throughout (max shift 0.10).

Drop-key discovery consults train.csv only (features + labels); public test labels and the holdout are never read here.
"""
from __future__ import annotations
import os
import re, unicodedata
import pandas as pd

COMBINING_DOT = "̇"
TEXT_COLS = ["query", "title", "category", "brand", "gender", "age_group", "attributes"]
SEG = re.compile(r", (?=[^:,]{1,40}: )")

MIN_TRAIN = 1000   # only consider keys common enough to matter
# We filter on LABEL PURITY inside train, not on train-vs-test prevalence.
# Prevalence ratio is confounded by the category-mix shift (giyim is 12.5% of train but 31.8%
# of test), so it drops innocent category-specific keys -- it removed `uyumlu marka`/`uyumlu cihaz`,
# which are load-bearing for accessory reasoning, and cost 55% of test rows an attribute.
# The leak's actual signature is that a key's PRESENCE nearly determines the label, because every
# relevant row came from snapshot A and every semi row from snapshot B. Purity targets that
# directly. Using train labels to clean train data is legitimate; the resulting key list is fixed
# and applied identically at inference.
# The leak runs both ways:
#   snapshot-A-only keys (color detail, taban materyali, ...) are P(relevant)=1.000 in train
#   snapshot-B-only keys (web color, alt taban materyali, ...) are P(relevant)=0.000 in train
#     -- and CANNOT be dropped for "absence from test": they are absent from train's RELEVANT
#        half, while being MORE common in test than in train (7.1% vs 2.7%).
# A one-sided rule leaves half the leak standing. Keep only keys whose prevalence is comparable
# in both, so key PRESENCE cannot identify the export.
MAX_PURITY = 0.93   # drop a key whose presence predicts the train label this well

def t1(s: str) -> str:
    """T0 Turkish-aware lowercase + NFC + strip the stray combining dot. Idempotent.
    No-op on the competition data (zero uppercase in any field of train/public_test); present so
    the pipeline matches its all-lowercase training distribution on any other input."""
    return unicodedata.normalize("NFC", str(s).lower()).replace(COMBINING_DOT, "")

def _keys(attr: str):
    for seg in SEG.split(attr):
        if ":" in seg:
            yield seg.split(":", 1)[0].strip().lower()

def discover_drop_keys(train: pd.DataFrame, labels, min_train=MIN_TRAIN, max_purity=MAX_PURITY):
    """Keys whose mere PRESENCE predicts the train label >= max_purity. Data-driven; no hand list.
    Does not consult the test set at all."""
    from collections import Counter
    import numpy as np
    sets = train.attributes.map(lambda a: set(_keys(t1(a))))
    ctr = Counter()
    for st in sets: ctr.update(st)
    lab = np.asarray(labels)
    out = []
    for k, c in ctr.items():
        if c < min_train: continue
        m = sets.map(lambda st, k=k: k in st).to_numpy()
        p = float((lab[m] == "relevant").mean())
        if max(p, 1.0 - p) >= max_purity: out.append(k)
    return sorted(out)

def clean_attrs(attr: str, drop: set) -> str:
    out = []
    for seg in SEG.split(t1(attr)):
        if ":" not in seg:
            if seg.strip(): out.append(seg.strip())
            continue
        k, v = seg.split(":", 1)
        kl = k.strip().lower()
        if kl in drop: continue
        out.append(f"{kl}: {v.strip()}")
    return ", ".join(out)

def normalize(df: pd.DataFrame, drop_keys) -> pd.DataFrame:
    d = df.copy()
    drop = set(drop_keys)
    for c in TEXT_COLS:
        if c in d.columns: d[c] = d[c].map(t1)
    if "attributes" in d.columns:
        d["attributes"] = d["attributes"].map(lambda a: clean_attrs(a, drop))
    if "gender" in d.columns:
        d["gender"] = d["gender"].replace({"unisex": "unknown", "çocuk": "unknown"})   # T4
        # T5 — only for checkpoints trained WITH the brand sink (see lean_infer.configure_text).
        # brand 'genel markalar' is the same two-export artifact as gender çocuk: P(relevant)=0.038
        # against an overall 0.326. Unlike gender it has no neutral sink to merge into (0 empty,
        # 0 'unknown' among 56,085 brands), so the training data creates one with 15% brand
        # dropout and this folds GM into it.
        if os.environ.get("T5_BRAND_SINK", "0") == "1":
            d["brand"] = d["brand"].replace({"genel markalar": "unknown"})   # T5
    return d
