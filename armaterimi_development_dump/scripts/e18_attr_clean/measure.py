"""E18 measure — what v2 attribute cleaning does to the model input length. CPU only, no GPU.

    CUDA_VISIBLE_DEVICES= HF_HUB_OFFLINE=1 \
      /home/jupyter/trendyol-teknofest-work/.venv-gcp/bin/python asya/s2_exp/e18_attr_clean/measure.py

Data: a deterministic 20,000-row sample of common/train_norm.parquet (random_state=42) and ALL of
common/public_test_norm.parquet (7,500 rows; attributes only — no labels are read or used).
Tokenizers: dbmdz/bert-base-turkish-128k-cased (HF cache, local_files_only) and the Qwen3
tokenizer at e15_student/backbones/vision_flash_text. Nothing is downloaded.

Model input, two encodings of the same text:
  pair    tok(query, doc)        — exactly what e15 train_student.py / eval_student.py do
  single  tok(query + " | " + doc)
  with doc = title | brand | category | gender age_group | attributes  (e07 template, verbatim).
Lengths are untruncated; "trunc256" = fraction of rows whose untruncated length exceeds 256
(MAXLEN default), i.e. rows that training/inference would cut.

Prints a compact table and writes measure.json next to this file.
"""
from __future__ import annotations
import os, sys, json, time
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
import numpy as np, pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import attr_clean as AC

W = "/home/jupyter/trendyol-teknofest-work"
S2 = f"{W}/asya/s2_exp"
TRAIN_NORM = f"{S2}/common/train_norm.parquet"
PUBLIC_NORM = f"{S2}/common/public_test_norm.parquet"
QWEN_DIR = f"{S2}/e15_student/backbones/vision_flash_text"
BERT_ID = "dbmdz/bert-base-turkish-128k-cased"
OUT = os.path.join(HERE, "measure.json")
N_TRAIN, SEED, MAXLEN = 20_000, 42, 256
TXT = ["query", "title", "category", "brand", "gender", "age_group", "attributes"]


def doc(d: pd.DataFrame) -> list[str]:
    return (d.title + " | " + d.brand + " | " + d.category + " | "
            + d.gender + " " + d.age_group + " | " + d.attributes).tolist()


def lengths(tok, d: pd.DataFrame, encoding: str, chunk: int = 4096) -> np.ndarray:
    A, B = d["query"].tolist(), doc(d)
    out = []
    for s in range(0, len(A), chunk):
        if encoding == "pair":
            e = tok(A[s:s + chunk], B[s:s + chunk], truncation=False)
        else:
            e = tok([a + " | " + b for a, b in zip(A[s:s + chunk], B[s:s + chunk])], truncation=False)
        out += [len(x) for x in e["input_ids"]]
    return np.asarray(out)


def n_pairs(s: str) -> int:
    return sum(1 for seg in AC.SEG.split(s) if ":" in seg)


def attr_stats(before: pd.Series, after: pd.Series) -> dict:
    emp_b = before.isin(["", AC.EMPTY_SENTINEL])
    emp_a = after == AC.EMPTY_SENTINEL
    return {
        "rows": int(len(before)),
        "empty_frac_before": float(emp_b.mean()),        # "" or "unknown" in the normalized data
        "empty_frac_after": float(emp_a.mean()),         # sentinel after v2
        "became_empty_frac": float((emp_a & ~emp_b).mean()),
        "pairs_per_row_before": float(before.map(n_pairs).mean()),
        "pairs_per_row_after": float(after.map(n_pairs).mean()),
        "attr_chars_before": float(before.str.len().mean()),
        "attr_chars_after": float(after.str.len().mean()),
    }


def main():
    t0 = time.time()
    import transformers
    transformers.logging.set_verbosity_error()
    from transformers import AutoTokenizer
    toks = {
        "berturk": AutoTokenizer.from_pretrained(BERT_ID, local_files_only=True),
        "qwen3": AutoTokenizer.from_pretrained(QWEN_DIR, local_files_only=True),
    }
    tr = pd.read_parquet(TRAIN_NORM, columns=TXT).sample(n=N_TRAIN, random_state=SEED).reset_index(drop=True)
    pu = pd.read_parquet(PUBLIC_NORM, columns=TXT).reset_index(drop=True)
    drop = AC.load_drop_keys(AC.DROP_KEYS_V2)
    splits = {"train20k": tr, "public": pu}
    res = {"config": {"n_train_sample": N_TRAIN, "seed": SEED, "maxlen": MAXLEN,
                      "drop_keys_v2": sorted(drop), "drop_boilerplate": True,
                      "tokenizers": {"berturk": BERT_ID, "qwen3": QWEN_DIR}},
           "attributes": {}, "tokens": {}}
    rows = []
    for sname, before in splits.items():
        after = AC.apply_v2(before, AC.DROP_KEYS_V2, drop_boilerplate=True)
        res["attributes"][sname] = attr_stats(before.attributes, after.attributes)
        for tname, tok in toks.items():
            for enc in ("pair", "single"):
                lb, la = lengths(tok, before, enc), lengths(tok, after, enc)
                r = {"split": sname, "tokenizer": tname, "encoding": enc, "rows": int(len(lb)),
                     "mean_before": float(lb.mean()), "mean_after": float(la.mean()),
                     "delta_tokens": float(la.mean() - lb.mean()),
                     "delta_pct": float(100 * (la.mean() - lb.mean()) / lb.mean()),
                     "p50_before": float(np.median(lb)), "p50_after": float(np.median(la)),
                     "p95_before": float(np.percentile(lb, 95)), "p95_after": float(np.percentile(la, 95)),
                     "max_before": int(lb.max()), "max_after": int(la.max()),
                     f"trunc{MAXLEN}_before": float((lb > MAXLEN).mean()),
                     f"trunc{MAXLEN}_after": float((la > MAXLEN).mean()),
                     "longer_after_rows": int((la > lb).sum())}
                rows.append(r)
                res["tokens"][f"{sname}/{tname}/{enc}"] = r

    # ------------------------------------------------------------------ print
    print(f"v2 attribute cleaning — {len(drop)} drop keys + boilerplate values; sentinel '{AC.EMPTY_SENTINEL}'")
    print(f"{'split':9s} {'tokenizer':9s} {'enc':6s} {'rows':>6s} {'mean_before':>11s} {'mean_after':>10s} "
          f"{'Δtok':>7s} {'Δ%':>7s} {'p95 b→a':>11s} {'trunc256 b→a':>16s}")
    for r in rows:
        print(f"{r['split']:9s} {r['tokenizer']:9s} {r['encoding']:6s} {r['rows']:6d} {r['mean_before']:11.1f} "
              f"{r['mean_after']:10.1f} {r['delta_tokens']:+7.1f} {r['delta_pct']:+6.1f}% "
              f"{r['p95_before']:5.0f}→{r['p95_after']:<5.0f} "
              f"{100 * r['trunc256_before']:5.2f}%→{100 * r['trunc256_after']:.2f}%")
    print()
    print(f"{'split':9s} {'empty_before':>12s} {'empty_after':>11s} {'became_empty':>12s} "
          f"{'pairs/row b→a':>14s} {'attr chars b→a':>16s}")
    for sname, a in res["attributes"].items():
        print(f"{sname:9s} {100 * a['empty_frac_before']:11.2f}% {100 * a['empty_frac_after']:10.2f}% "
              f"{100 * a['became_empty_frac']:11.2f}% {a['pairs_per_row_before']:6.2f}→{a['pairs_per_row_after']:<6.2f} "
              f"{a['attr_chars_before']:7.1f}→{a['attr_chars_after']:<7.1f}")
    res["elapsed_s"] = round(time.time() - t0, 1)
    with open(OUT, "w") as f:
        json.dump(res, f, ensure_ascii=False, indent=1)
    print(f"\nwrote {OUT} ({res['elapsed_s']}s)")


if __name__ == "__main__":
    main()
