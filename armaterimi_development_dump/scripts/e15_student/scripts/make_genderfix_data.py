#!/usr/bin/env python3
"""Regenerate the CHAMPION training file: student_train_genderfix.parquet.

Why this file exists at all
---------------------------
The organizer's Stage-2 `train.csv` export makes `gender == "çocuk"` a near-perfect predictor
of `semi-relevant` (16,634 organizer rows, zero `relevant`). On public test those same rows are
44% irrelevant / 29% relevant / 27% semi. The value is an EXPORT ARTIFACT, not a signal - the
same disease `unisex` has, which the T4 rule in common/normalize.py already neutralises by
mapping it to "unknown". Patching it at inference time only was measured on DEV at
+0.0069 (TY-ecomm 12L) and +0.0120 (BERTurk); retraining on data where the artifact never
existed is the clean fix and is what the shipped checkpoint (ckpt_gf2_berturk_6ep) was trained on.

What it does
------------
Exactly one thing: `gender: "çocuk" -> "unknown"` on the frozen student mix. No row is added,
dropped or reordered and no other column is touched. Verified against the shipped artifact:
27,289 of 813,269 rows change, in the `gender` column only.

Usage
-----
    python make_genderfix_data.py                       # write next to the input
    python make_genderfix_data.py --verify              # rebuild in memory and diff vs the
                                                        # shipped parquet, write nothing
    python make_genderfix_data.py --in A.parquet --out B.parquet
"""
import argparse
import sys

import pandas as pd

W = "/home/jupyter/trendyol-teknofest-work"
DEFAULT_IN = f"{W}/asya/s2_exp/e15_student/outputs_frozen_v2/build_oldstore/student_train.parquet"
DEFAULT_OUT = f"{W}/asya/s2_exp/e15_student/outputs_frozen_v2/build_oldstore/student_train_genderfix.parquet"

# The T4 rule of common/normalize.py already maps `unisex` here; `çocuk` is the value the
# 2026-08-26 audit found we had missed.
ARTIFACT_VALUE = "çocuk"
NEUTRAL_VALUE = "unknown"


def genderfix(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["gender"] = out["gender"].where(out["gender"] != ARTIFACT_VALUE, NEUTRAL_VALUE)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="src", default=DEFAULT_IN)
    ap.add_argument("--out", dest="dst", default=DEFAULT_OUT)
    ap.add_argument("--verify", action="store_true",
                    help="do not write; diff the rebuild against an existing --out")
    a = ap.parse_args()

    src = pd.read_parquet(a.src)
    fixed = genderfix(src)
    n = int((src["gender"] == ARTIFACT_VALUE).sum())
    print(f"rows {len(src):,} | '{ARTIFACT_VALUE}' -> '{NEUTRAL_VALUE}': {n:,}")
    print(f"gender before: {src['gender'].value_counts().to_dict()}")
    print(f"gender after : {fixed['gender'].value_counts().to_dict()}")

    if a.verify:
        have = pd.read_parquet(a.dst)
        if list(have.columns) != list(fixed.columns) or len(have) != len(fixed):
            print(f"MISMATCH: shape/columns differ ({have.shape} vs {fixed.shape})", file=sys.stderr)
            return 1
        bad = [c for c in fixed.columns if not fixed[c].equals(have[c])]
        if bad:
            print(f"MISMATCH: columns differ: {bad}", file=sys.stderr)
            return 1
        print(f"VERIFIED: rebuild is identical to {a.dst}")
        return 0

    fixed.to_parquet(a.dst, index=False)
    print(f"wrote {a.dst}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
