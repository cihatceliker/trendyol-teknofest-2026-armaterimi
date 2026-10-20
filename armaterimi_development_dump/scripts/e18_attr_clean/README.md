# E18 — optional "v2" attribute cleaning (opt-in; default pipeline byte-identical)

**Status: ADOPTED 2026-08-23** as `ATTR_CLEAN=v2` — applied identically at train-build time
(e15 builders), at DEV/holdout eval, and at inference (inlined in the shipped
`e12_lean_infer/lean_infer.py`; train/serve parity asserted byte-identical by
`verify_train_serve_parity.py`). The paragraph below describes the original opt-in design.

## What v2 does

Input: the `attributes` string exactly as `normalize.py` serialises it (`key: value, key: value`).

1. drop pairs whose key is in `drop_keys_v2.csv` (60 keys = the 18 frozen v1 leak keys + 11 boilerplate
   keys: `yıkama talimatı`, `bakım talimatları (genel)`, `bakım talimatları (gıda temas)`,
   `sürdürülebilirlik detayı`, `menşei`, `ek özellik`, `garanti süresi`, `garanti tipi`,
   `ürün güvenliği bilgisi`, `kullanım talimatı`, `kutu durumu`);
2. with `drop_boilerplate=True`, drop pairs whose **value** carries no product information: exact
   `belirtilmemiş` / `mevcut değil` / `ek özellik mevcut değil` / `-` / `unknown` / empty, or any value
   containing `takip ediniz` / `kullanım ömrünü` (the care-instruction sentence). Matching is
   case/diacritic-folded (`lower()` + T1) so private-set spelling variants cannot slip through; the data
   itself has zero such variants. `yok` / `var` / `mevcut` are **never** dropped (`kapüşon: yok` is information).
3. pair order is preserved; if nothing remains the string becomes the sentinel `unknown`.

The parser is not a new parser: `attr_clean.py` imports `SEG` and `t1` from `common/normalize.py` and
mirrors `clean_attrs` line for line. The tests prove it: with an empty drop set and
`drop_boilerplate=False` the function is the identity on all 507,500 real strings except the `""` rows
(see below), and on **raw** text `clean_attributes(raw, v1_keys, False) == normalize.clean_attrs(raw, v1_keys)`.

### Why the sentinel is `unknown` (a small v1 leak found on the way)

`normalize.py` has no explicit empty sentinel: it emits `""` when every key is dropped and passes the raw
sentinels through. Train therefore carries two spellings of "no attributes", and they are an export tell
exactly like `unisex` (T4):

| train_norm `attributes` | rows | label |
|---|---|---|
| `""` | 2,270 | 99.6 % semi-relevant |
| `unknown` | 4,097 | 100 % relevant |

Public has only `""` (85 rows, 1.1 %) — the spelling a v1-trained model has learned to read as "semi".
v2 maps both (and every all-dropped row) to the single token `unknown`, the vocabulary normalize.py
already uses for gender/age_group. This is the **only** input on which v2 with all filters off is not the
identity (`"" -> "unknown"`).

## Files

| file | what |
|---|---|
| `attr_clean.py` | `clean_attributes(attr_str, drop_keys, drop_boilerplate) -> str` (pure, idempotent) and `apply_v2(df, drop_keys_v2_path, drop_boilerplate=True) -> df` (copy; only `attributes` changes). `load_drop_keys(path)`. |
| `drop_keys_v2.csv` | starter list, column `key`, 60 keys (sha256 `63313946b613ed7d…`; v1 list is `3a5bf64df2f2123b…`). |
| `measure.py` / `measure.json` | token-length effect, both tokenizers, CPU only (table below). |
| `test_attr_clean.py` | 13 plain-assert checks, synthetic + real data (~55 s, CPU). |
| `analysis/` | reserved for the separate evidence-based candidate list `drop_keys_v2_candidates.csv` (not merged here). |

## Measured effect (`measure.py`, 2026-08-23)

Deterministic 20,000-row sample of `common/train_norm.parquet` (`random_state=42`) and all 7,500 rows of
`common/public_test_norm.parquet` (attributes only — no labels read). Model input = the e07/e15 template
`query` vs `title | brand | category | gender age_group | attributes`; `pair` is exactly what
`train_student.py`/`eval_student.py` tokenise (`tok(query, doc)`), `single` is `tok(query + " | " + doc)`.
Lengths untruncated; `trunc256` = rows longer than the MAXLEN=256 default, i.e. rows that get cut.

```
split     tokenizer enc      rows mean_before mean_after    Δtok      Δ%     p95 b→a     trunc256 b→a
train20k  berturk   pair    20000        91.6       78.0   -13.6  -14.9%   163→136    0.36%→0.21%
train20k  berturk   single  20000        91.6       78.0   -13.6  -14.9%   163→136    0.36%→0.21%
train20k  qwen3     pair    20000       137.4      114.3   -23.1  -16.8%   251→203    4.32%→1.10%
train20k  qwen3     single  20000       138.3      115.2   -23.1  -16.7%   251→203    4.42%→1.14%
public    berturk   pair     7500        98.6       84.1   -14.5  -14.7%   184→153    0.33%→0.12%
public    berturk   single   7500        98.6       84.1   -14.5  -14.7%   184→153    0.33%→0.12%
public    qwen3     pair     7500       148.2      123.3   -24.9  -16.8%   276→224    7.68%→1.49%
public    qwen3     single   7500       149.1      124.1   -24.9  -16.7%   276→224    7.88%→1.53%

split     empty_before empty_after became_empty  pairs/row b→a   attr chars b→a
train20k         1.23%       5.85%        4.62%   8.69→6.57     198.6→136.2
public           1.13%       5.84%        4.71%  10.48→8.11     231.0→166.1
```

Reading it:
- **-15 % tokens with BERTurk, -17 % with Qwen3**, the same on train and public (no train/test asymmetry
  introduced — the boilerplate keys are common on both sides: `menşei` is on 81 % of train rows and 88 % of
  public rows and is the single largest contributor).
- Truncation at 256 is a BERTurk non-issue either way (0.3 % -> 0.1–0.2 %); for the Qwen3 tokenizer it
  drops from 4.3 %/7.7 % of rows to 1.1 %/1.5 %, which is the one place v2 changes what the model *sees*
  rather than just how much padding it pays for.
- **5.85 % of rows end up with no attributes** (sentinel `unknown`): 1.2 % were already empty, 4.6 % become
  empty because the only pairs they had were boilerplate (typically `menşei: tr` alone, or `menşei` +
  the care sentence). Identical on both sides (5.85 % / 5.84 %).
- Mean pairs per row 8.7 -> 6.6 (train), 10.5 -> 8.1 (public); the surviving pairs are the informative ones
  (colour, size, material, compatibility …).
- `single` vs `pair` differ by at most 1 token; BERTurk is identical because `[SEP]` and `|` both cost one token.
- The only rows that get *longer* are the `"" -> unknown` rows under BERTurk (+1 token; 85 in public).

Where the savings come from (BERTurk, removed pairs tokenised standalone, so approximate; share of
removed tokens, train20k / public): `menşei` 24 % / 24 %, `yıkama talimatı` 18 % / 24 %,
`bakım talimatları (genel)` 13 % / 9 %, `bakım talimatları (gıda temas)` 11 % / 4 %,
`sürdürülebilirlik detayı` 7 % / 11 %, `kutu durumu` 6 % / 9 %, `ek özellik` 6 % / 8 %,
`garanti süresi` 6 % / 4 %, `garanti tipi` 3 % / 2 %, `ürün güvenliği bilgisi` 1 % / 1 %,
`kullanım talimatı` 0 (the key does not occur in either set). The **value** rule is only ~4–5 % of the
savings (almost all `belirtilmemiş`, 0.12–0.15 pairs/row); `-` / `mevcut değil` / care sentences under
non-dropped keys are rare (<0.01 pairs/row). So the key list, not the value rule, is what matters —
which is exactly what the `analysis/` candidate list is meant to refine.

Not measured here: the effect on DEV macro-F1. That needs a student retrain on v2 text and is a DEV-only
decision (holdout stays milestone-only, per `eval_student.py`'s guard). Token savings alone are not a
reason to switch; the argument for v2 is that the model stops spending attention on `menşei: tr` and the
care sentence, and that the `""`/`unknown` tell disappears.

## Tests (`test_attr_clean.py`, all 13 pass, ~55 s CPU)

boilerplate removal (exact + substring, case-robust, no false positives such as `kutu yok`,
`tehlikeli değil`, `parça mevcut değil`) · `yok`/`var`/`mevcut` preserved · key drop with original order ·
sentinel on empty (raw `""`, raw `unknown`, all-dropped, all-boilerplate) · comma/colon-bearing values
split exactly like `normalize.clean_attrs` · idempotence (synthetic, and the full v2 transform on all
507,500 real rows: `f(f(x)) == f(x)`, no residual boilerplate or dropped key) · identity with empty drop set
and `drop_boilerplate=False` on 1,000 real non-empty strings and on the whole corpus except the 2,355 `""`
rows · `apply_v2` returns a copy and touches only `attributes` · mirror of `normalize.clean_attrs` on raw
public + raw train · `v2(normalize(raw)) == v2(raw)` (the re-split round-trips) · `drop_keys_v2.csv` ⊇ v1 list
+ the 11 additions.

```
CUDA_VISIBLE_DEVICES= /home/jupyter/trendyol-teknofest-work/.venv-gcp/bin/python asya/s2_exp/e18_attr_clean/test_attr_clean.py
CUDA_VISIBLE_DEVICES= HF_HUB_OFFLINE=1 /home/jupyter/trendyol-teknofest-work/.venv-gcp/bin/python asya/s2_exp/e18_attr_clean/measure.py
```

## How to enable v2 downstream (when, and only when, a DEV decision says so)

One rule: **every place that calls `NZ.normalize(..., drop_keys)` gets `AC.apply_v2(...)` right after it,
with the same `drop_keys_v2.csv` and the same `drop_boilerplate` flag.** A model trained on v2 text must
be served v2 text; applying v2 at inference to a v1-trained checkpoint (or vice versa) is a train/serve skew.

```python
sys.path.insert(0, f"{W}/asya/s2_exp/e18_attr_clean")
import attr_clean as AC
d = NZ.normalize(raw_df, drop_keys)                       # unchanged v1 step
d = AC.apply_v2(d, AC.DROP_KEYS_V2, drop_boilerplate=True)  # v2 step, same call everywhere
```

Hook points (none of them edited here):
- `e15_student/scripts/build_student_data.py` — `pool()` (`_pool = NZ.normalize(pd.read_parquet(POOL), drop_keys)`):
  wrap with `apply_v2`, and add `drop_keys_v2.csv` to `inputs_used` so the manifest records its sha.
  Rows materialised from `common/train_norm.parquet` / `public_test_norm.parquet` are already normalised
  and need the same `apply_v2` (they are identical inputs to v2 as the parser round-trips).
- `e15_student/scripts/eval_student.py` — `d = NZ.normalize(H.load_public_split(half), drop)`: same wrap.
- `e12_lean_infer/scripts/lean_infer.py` — carries an INLINED frozen copy of normalize.py (no repo imports).
  v2 must be inlined there the same way: `clean_attributes` + the boilerplate constants + the 60 keys
  (the function only needs `re`/`unicodedata` besides `SEG`/`t1`, which that script already has), and
  `verify_parity.py` must be re-run against the v2 checkpoint.
- Tag the run (e.g. `TAG=..._v2`) and keep v1 and v2 checkpoints' decision cuts separate — the
  two-boundary cuts are fitted per checkpoint on DEV and are not transferable.

Things v2 deliberately does **not** do: no value rewriting, no key renaming (the normalize.py docstring
explains why renames re-create leaks), no lowercasing of values, no key discovery from labels (the 11
added keys are a hand list of boilerplate; the evidence-based candidate list lives in `analysis/` and is
not merged). The substring rule catches 2 long free-text `diğer özellikler` values in 500k train rows
that happen to contain `kullanım ömrünü` — accepted as noise.
