# Stage 1 → Stage 2 Understanding (audit as of 2026-08-19)

Sources: `first_stage/` (delivered Stage-1 package), `COMPETITION_RULES.md`,
`sunum_transkript.md` (Stage-2 kickoff webinar), `TeknoFest '26 ... Final Intro.pdf`,
`stage1_data/`, `stage2_data/`, `asya/gcp_jobs/m6a0..m6a3` (prior Stage-2 probes).
Labels: FACT = documented/measured here; DATA OBSERVATION = measured by me on the raw files;
HYPOTHESIS / RISK / OPEN QUESTION as marked.

---

## 1. Stage 1 — what we actually shipped

**Task (FACT).** Binary query–product relevance (1/0) over `submission_pairs.csv`, built by
joining `terms.csv` × `items.csv`. Organizer gave 250,000 positives only
(`training_pairs.csv`); negatives had to be mined.

**Final decision rule (FACT, `core2_path/train/emit2.py`).** Two stages:
1. `v1 < 0.2` (fine-tuned bi-encoder cosine) → hard `0` ("junk band").
2. Everything else: fine-tuned Asure-12B scalar `> CUT` → `1`, else `0`. Selected `CUT = 1.3`
   (best on both public and private LB); a second selected submission used `CUT = 1.1`.

**Scalar definition (FACT).** For every scoring model in the solution:
`scalar = logsumexp(logits["Evet"], logits[" Evet"]) − logsumexp(logits["Hayır"], logits[" Hayır"])`
read at the **last token position**, single forward pass, no generation.

**Pipeline (FACT).**

| # | Stage | Artifact |
|---|---|---|
| 1 | `data/clean_items.py` attribute cleaner (−24% tokens) | `items_clean.csv` |
| 2 | `p0_embed_retrieve.py` TY-ecomm-embed zero-shot embeddings | v0 |
| 3 | `p1_biencoder_train.py` MNRL fine-tune (scale 20, 1 ep, bs128, lr2e-5, seed 42) | `biencoder_v1`, v1 (r@100 = 0.4890, r@200 = 0.6151 on a seeded 500-term holdout) |
| 4 | `p2` (seed 123) v0-band negative mining → 74,585; `p3` BERTurk CE denoiser; `p4` (seed 1234) + CE>0.75 drop → 129,155 | mined negatives v1/v2 |
| 5 | `build_core.py` → 447,322-pair core (seed-42 shuffle fixes row order for all banks) | `v8_train_core.parquet` |
| 6 | `asure_path/bank.py` zero-shot **R2b3** scoring of the core | `core_bank_scalar.npy` |
| 7 | Label cuts: `≥ +3 → 1`, `≤ −3 → 0`, middle 34,839 → LLM-judge majority | `train_v1.parquet` (392,911) |
| 8 | Round 2: `mine2_candidates.py` (seed 777, 3 quadrants) → 317,838; `bank2.py`; judge middle 96,159; `fuse2.py`; `build_mix2.py` (seed 42) | `mix2.parquet` (299,476) |
| 9 | `train2.py` LoRA r16 α32 on NF4 Asure-12B, BCE on the scalar, 1 epoch, bs4×accum16, lr1e-4, 34.7 h | `run2_adapter` (loss 0.483 → 0.320) |
| 10 | `ft2_infer.py` + `emit2.py` | `submission.csv` |

**Judge system (FACT, locked).** `majority(sign(zero-shot Asure scalar), gemini-3.1-flash-lite,
minimax-m3)`, temperature 0, batch 20, system prompt = `rubric.md` + an output-format block.
Soft targets: 3-0 → 0.95/0.05, 2-1 → 0.72/0.28. Judges never touched a submission pair.

**Prompt specs (FACT).** Zero-shot banking = **R2b3** (`asure_path/spec/ASURE_PIPELINE.md`):
persona + category-intent + typo-intent system text, fields in order
title / brand(+descriptor) / gender|age / category / attributes, unknown fields skipped,
unisex rewritten, `Özellikler:` line always present, **query LAST** (documented as a +9pt recency
effect), max_length 512, left padding. Fine-tune prompt is deliberately different: **O3** =
clean system text + query repeated *before and after* the product block ("sandwich"). The 2,120-brand
`brand_desc.json` descriptors are used in banking only, not in the fine-tune prompt.

**Models shipped (FACT).** `run2_adapter` (LoRA), `biencoder_v1`, `asure_gptq4` (batch scoring),
`asure_4bit` (NF4 — training + inference base). All present on this machine, plus a 23 GB bf16
`Trendyol-LLM-Asure-12B` in the HF cache.

**Cost profile (FACT).** Dev machine was a single RTX 4080 SUPER 16 GB: step2 ≈ 2.5–3 days,
step3 ≈ 2.5 days.

**Reproducibility posture (FACT).** Non-training stages exact; v1 mining Jaccard 1.000;
v2 mining Jaccard 0.993 (p3 CE retrain is the only stochastic stage). `VERIFY=1` no-write modes
for `build_core`, `fuse_labels`, `generate_frozen_inputs`.

---

## 2. Stage 2 — the task as specified

**Label set (FACT).** `relevant` / `semi-relevant` / `irrelevant`, metric = fixed three-class
**macro-F1**, joined on `row_id`.

**Data (FACT).** Flat rows — no join tables. `train.csv` 500,000 rows
(250k relevant + 250k semi-relevant, **no irrelevant**); `public_test.csv` 7,500 rows with
`public_solution.csv` labels supplied (2,500 per class); private test 90,000 rows
(30k per class), never shared. Train / public / private are **query-disjoint**, and
public+private queries do not occur anywhere in Stage 1.

**Schema.** `query,title,category,brand,gender,age_group,attributes,real_label`;
test adds `row_id` and drops the label. Output must be exactly `row_id,real_label`.

**Runtime contract (FACT).** Organizer machine = this machine: Debian 12, x86_64,
A100-SXM4 40 GB, driver 580.126.20 / CUDA 13.0, 12 logical CPUs, 70 GB container RAM + 7 GB swap.
`run_solution.sh` runs the image with `--network none --read-only --cap-drop ALL`,
`--user $HOST_UID:$HOST_GID`, `--memory 70g`, `/tmp` tmpfs 8 GB, input bind-mounted read-only at
`/input/input.csv`, output dir at `/output`. **Only `/tmp` and `/output` are writable.**
`run_solution.sh` and `run_explainability.sh` must not be modified.

**Hard limits (FACT).** ZIP ≤ 100 GB · image build ≤ 1 h · 90k inference ≤ 150 min ·
explainability startup ≤ 10 min.

**Scoring (FACT).**
- Final-set score = MaxNorm(macro-F1) × rank multiplier (1st = 1.0 … 10th = 0.8) × **20**.
- Speed = MaxNorm(90000 / runtime_seconds) × **10**, *no* rank multiplier.
- Technical report = **10**, and it is graded on readability as well as content.
- Speed gates: runtime < 150 min **and** macro-F1 above the accuracy floor — otherwise speed = 0.

> **RESOLVED (user, 2026-08-19): the accuracy floor is 0.80.**
> `COMPETITION_RULES.md` §6.3.1.2 says **> 0.80**; the Final-Intro PDF p.11 and the webinar say
> 0.85. The rules file was updated by the organizers *after* the webinar and supersedes both.
> **Plan against 0.80.** Consequence: the gate is much easier than first assumed, so the 10-point
> speed axis is worth materially more than an accuracy-max-at-all-costs reading would suggest.

**Explainability mode (FACT).** Same image, second mode, ngrok tunnel started inside the
container, must fail closed without `NGROK_AUTHTOKEN`. UI must provide: searchable `row_id`
selector over public test, the selected product's visible fields, an **editable** free-text query
box pre-filled from the row, an Explain button, the predicted label, and a rationale.
The rationale must come from **the predictor itself** — SHAP, attention, thinking tokens, priors.
A separate zero-shot LLM explaining input+output is explicitly rejected. Originality here was
called out as a scoring differentiator.

**Deliverable (FACT).** `/home/jupyter/TEAMNAME_submission.zip` containing
`TEAMNAME_docker/`, `TEAMNAME_development_dump/`, `TEAMNAME_FINAL_REPORT.docx`.

**Calendar (FACT).** Development 14–28 Aug 2026 (deadline 28 Aug 23:59) · organizer review
29 Aug – 4 Sep · on-site hackathon + jury presentations 5–6 Sep @ Trendyol Maslak.
Speed and accuracy scores are frozen at the 29 Aug pull; **explainability may keep improving
until 6 Sep**.

**Permissions confirmed in the webinar (FACT).** Stage-1 data *and* Stage-1 trained models may be
reused. Public-test labels may be used for training. No limit on training time or external
compute. AI tooling is unrestricted. Build-time internet is allowed; run-time is not.

---

## 3. Prior Stage-2 probes in this repo

| ID | What | Result |
|---|---|---|
| **m6a0** | Asure-12B zero-shot 3-class on public test, vLLM 0.26.0 | macro-F1 **0.7083**. Per class F1: I 0.802 / S **0.531** / R 0.792. **54.7 rows/s.** |
| **m6a1** | Masked in-batch-negative policy study on `biencoder_v1` | **FAIL** against its preregistered gates |
| **m6a2** | 596M Qwen3 text tower ("Vision Flash") trained as bi-encoder, masked in-batch | **PASS** as a representation: R@10 0.502 / S@10 0.312, R>S acc 0.654. Not a 3-class solution. |
| **m6a3** | Asure-12B zero-shot **binary** non-I vs I | ROC-AUC **0.942** (R-vs-I 0.985, **S-vs-I 0.900**). At 95% I-precision, I-recall 0.344. |

**Read-across.** The Stage-1 style signal is already strong at the I boundary and weak at the
S boundary — semi-relevant is the whole problem. `semi-relevant` recall is 0.437 zero-shot.

---

## 4. What I measured on the raw data

**DATA OBSERVATION 1 — Stage-2 `relevant` IS the Stage-1 positive set.**
All 30,891 Stage-2 train queries are Stage-1 terms. Joining Stage-2 train rows to
`stage1_data/training_pairs.csv` through `terms.csv`/`items.csv` matches **249,984 of the
250,000 `relevant` rows and only 26 `semi-relevant` rows**. So:
`stage2.relevant ≡ stage1.label=1`, and Stage-2 `semi-relevant` is new organizer data drawn from
the space Stage 1 called `0`.

**Consequence.** The Stage-1 `run2_adapter` is already a trained
*relevant-vs-everything-else* discriminator whose positive class is exactly the Stage-2
`relevant` class. Its scalar is a strong prior for the R boundary; it says nothing about S vs I.

**DATA OBSERVATION 2 — Stage-1 mined negatives barely intersect Stage-2 `semi-relevant`.**
Only 140 of 250,000 `semi-relevant` rows appear in `train_v1.parquet` (0.06%), split 86 label-0 /
54 label-1. So the Stage-1 mined-negative pool is *available* as an irrelevant-candidate source,
but it is a **mixture**: Stage 1's rubric labeled accessories and complements `0`, while Stage 2
calls them `semi-relevant`.

**DATA OBSERVATION 3 — large `relevant`-class distribution shift, not explained by query length.**
Mean fraction of query tokens found in title+category+brand:

| Class | Train | Public test |
|---|---:|---:|
| relevant | **0.850** | **0.688** |
| semi-relevant | 0.630 | 0.529 |
| irrelevant | — | 0.363 |

Holding query word-count fixed, train `relevant` stays ≈0.85 at 1/2/3/4 words while public
`relevant` is 0.52 / 0.67 / 0.73 / 0.70. **RISK:** a model fit directly on `train.csv` will learn
"high lexical coverage ⇒ relevant" and under-predict `relevant` on the test distribution.

**DATA OBSERVATION 4 — category and grouping shift.**
`giyim` is 12.5% of train rows but **31.8%** of public test; `ev & mobilya` 26.4% → 12.8%.
Train averages 16.2 rows per query (max 2,138); public test averages **1.28** rows per query.
`gender`/`age_group` are `unknown` for ~67%/65% of train rows but only ~49%/48% of public rows.

**DATA OBSERVATION 5 — organizer `irrelevant` rows are hard, retrieval-adjacent negatives.**
Sampled examples: `siyah beyaz topuklu ayakkabı` → a *babet* (same category root, wrong subtype);
`7. sınıf deneme` → an *8. sınıf* test book; `canon g7x mark iii` → Canon printer ink.
Only 25% have zero query-token overlap. **Random or easy negatives will not match this
distribution.**

**DATA OBSERVATION 6 — the only in-distribution labeled data is `public_solution.csv`.**
7,500 rows, balanced, including **2,500 genuine `irrelevant`** examples in the true test
distribution. It is simultaneously the sole honest validation set and the sole ground-truth
source for the missing class.

---

## 5. Throughput arithmetic

**FACT (measured, m6a0/m6a3).** Asure-12B bf16, vLLM 0.26.0, A100 40 GB, ~1–4 output tokens:
**50–55 rows/s**, ~100 s model load.

90,000 / 50 ≈ **30 minutes** — comfortably inside the 150-minute limit, giving SPS ≈ 47.
A small cross-encoder would reach 1,000+ SPS, so on a max-normalised speed axis a 12B solution
collects roughly 0.5 of the 10 speed points. Accuracy is worth 20 points *and* gates the speed
points, so accuracy dominates — but the gap is not free.

---

## 6. Problem statement, restated

1. **Manufacture the missing `irrelevant` class** at test-set difficulty (hard, retrieval-adjacent,
   frequently same-category) — not easy random negatives.
2. **Learn the S boundary**, which is where zero-shot loses (S F1 0.531, S-vs-I AUC 0.900).
3. **Correct the train→test shift** in the `relevant` class so lexical coverage is not a shortcut.
4. **Clear the accuracy floor (0.80)** inside 150 minutes, then buy back speed points.
5. **Explain the predictor itself**, originally, in a UI that starts in under 10 minutes.
6. **Write a report a tired judge understands on the first read.**

---

## 7. Measured baselines on public test (2026-08-19)

All numbers are **out-of-fold under query-disjoint 5-fold** on the 7,500-row public set
(`asya/s2_exp/common/harness.py`). In-sample optima are within ~0.001 of OOF, so threshold
fitting is stable and not the source of any gain.

| Baseline | macro-F1 | F1 irrelevant | F1 semi | F1 relevant |
|---|---:|---:|---:|---:|
| Surface features only (HGB on 20 lexical/structural features, fit on 6k rows) | 0.6392 | 0.702 | 0.563 | 0.654 |
| Asure-12B zero-shot, **3-class generation** (m6a0) | 0.7083 | 0.802 | 0.531 | 0.792 |
| Asure-12B zero-shot binary prompt, **one scalar + 2 thresholds** (m6a3 scores) | **0.7259** | 0.813 | 0.609 | 0.756 |

**DATA OBSERVATION 7 — the task is ordinal and one scalar carries it.**
The zero-shot binary logit margin orders the classes cleanly (median: I −1.125, S 2.125, R 4.250).
Two thresholds on that single scalar beat the model's own 3-way generation by +0.018, almost
entirely by lifting semi-relevant (F1 0.531 → 0.609). Confusion is adjacent-only:
I↔R is 52 rows out of 7,500, while I↔S is 837 and S↔R is 1,187.
**This argues for an ordinal single-score architecture** — one forward pass, one token position,
no generation. That is also the cheapest thing to serve and the easiest thing to explain.

**DATA OBSERVATION 8 — the zero-shot model's errors are the lexical shortcut.**
Slice macro-F1 by query-token coverage:

| coverage band | n | macro-F1 | F1_I | F1_S | F1_R |
|---|---:|---:|---:|---:|---:|
| 0–25% | 1447 | 0.648 | 0.835 | 0.556 | **0.554** |
| 25–50% | 3037 | 0.707 | 0.808 | 0.633 | 0.680 |
| 50–75% | 1438 | 0.735 | 0.808 | 0.620 | 0.775 |
| 75–100% | 1578 | 0.651 | **0.532** | 0.589 | 0.832 |

Both tails fail in the predicted direction: with no lexical overlap it will not say `relevant`
(F1_R 0.554); with heavy overlap it will not say `irrelevant` (F1_I 0.532). Single-word queries
are the worst slice overall (macro-F1 0.597). Fixing this is worth more than any threshold work.

**DATA OBSERVATION 9 — `semi-relevant` is substitutes + accessories, mostly in-category.**
Over the 7,949 train queries carrying both classes, a semi-relevant item shares the relevant
item's **root** category 89.7% of the time, depth-2 76.9%, exact **leaf** 50.5%. Accessory or
compatibility tokens appear in 19.7% of semi-relevant titles vs 7.6% of relevant.
Top leaf transitions are two clean families — substitutes (`sandalet→terlik`, `klima→vantilatör`,
`tencere→tava`, `yürüyüş ayakkabısı→sneaker`) and accessories (`android cep telefonu→kapak & kılıf`,
`dik süpürge→süpürge aksesuarı`, `playstation 5→playstation aksesuarı`).

**HYPOTHESIS — the S/I boundary is constraint violation, not category distance.**
`sandalet → terlik` is semi-relevant while `siyah beyaz topuklu ayakkabı → babet` is irrelevant;
both are within-root shoe substitutions. The difference is that the second query states a
constraint (`topuklu`) the item contradicts. Consistent with the Stage-1 rubric's Step-2 table.
Not yet tested — it predicts that constraint-bearing queries produce most of the `irrelevant`
mass and should drive negative mining.

**Checks that came back negative (good — no shortcut to lean on):**
- Category-root prior adds only +0.008 to the surface baseline (0.6312 → 0.6392), so the per-root
  class skew (giyim 44.6% I, elektronik 49.4% S) is not an exploitable leak.
- Item-level priors are weak: only 19.9% of public rows have a title seen in train, and the
  train label mix for those items separates the public classes only slightly.

**Throughput measured here:** NF4 Asure-12B + LoRA through HF transformers = **~21 rows/s**
(4,800 tok/s) on the A100. vLLM bf16 without an adapter was 50–55 rows/s. The delivered serving
path will need vLLM (or equivalent), not bnb-NF4.

---

## 8. CRITICAL — the two-snapshot label leak (verified 2026-08-19)

**FACT (independently verified by me, not just reported).** Stage-2 `train.csv` was assembled from
**two different catalog exports**, and which export a row came from *is* its label. Four formatting
markers, measured on all 500,000 train rows:

| marker | train n | P(relevant) | P(semi) | public-test n |
|---|---:|---:|---:|---:|
| `attributes` contains `color detail:` | 196,763 | **1.000000** | 0.000000 | **0** |
| `gender == 'unisex'` | 22,518 | 0.999911 | 0.000089 | **0** |
| `title` contains dotted-i `i̇` (U+0069 U+0307) | 32,822 | 0.000000 | **1.000000** | 1,110 |
| `attributes` empty | 2,247 | 0.000000 | **1.000000** | 85 |

**234,455 train rows — 46.9% — carry at least one marker.** Root cause (A9): snapshot B lowercases
`İ` the Unicode way while snapshot A folds it to plain `i`, plus a real schema revision
(`renk` collapsed from 347 values to 31; sole/lining keys renamed). Train `relevant` is the Stage-1
`items.csv` dump verbatim (snapshot A); train `semi-relevant` **and the entire test set, all three
classes**, come from the later export (snapshot B).

**Why this is a trap and not a gift — measured on DEV:**

| marker | train label | DEV: I / S / R | verdict |
|---|---|---|---|
| `color detail:` | 100% relevant | **absent from test entirely** | pure train artifact |
| `unisex` | 100% relevant | **absent from test entirely** | pure train artifact |
| dotted-i | 100% semi | 0.253 / 0.366 / 0.380 | **no signal** (baseline is .333) |
| empty attributes | 100% semi | 0.485 / 0.242 / 0.273 | **anti-correlated** |

So a model fit on raw `train.csv` will learn a feature that perfectly separates R from S on 39% of
its training data and **either never fires at test time, or fires as pure noise on ~15% of test
rows**. This is a textbook spurious-correlation shortcut, and it is the most likely single cause of
a large train→test generalisation failure.

**RISK — this probably also explains part of DATA OBSERVATION 3.** The `relevant` query-token
coverage gap (0.850 train → 0.688 test) is partly a schema artifact: snapshot A carries richer
attribute strings, so query tokens land more often. A9 additionally measured that the gap survives
*within* every category root and within non-brand queries, so there is a second, independent
generation-difference shift underneath the schema one.

**MANDATORY ACTION, before any model is trained:** normalise both snapshots to a common form —
NFC + Turkish-correct casefolding of the dotted-i, harmonise the `renk` / `color detail` schema,
drop or unify export-specific keys, and re-verify that all four markers fall to chance on train.
A9 ships a verified normaliser; A11's `A11_text_norm.py` covers the Unicode half. **No training run
should consume `train.csv` until this passes.**

### 8b. Confirmed independently by A7, and the normalisation is harder than it looks

A7 found the same leak from a different direction and put a sharper number on it:
**a two-rule oracle classifies 60.8% of `train.csv` at 100.000% accuracy with zero conflicts.**
Its measurements: `U+0307` occurs in **107,483 of 250,000 semi rows (43.0%) and 0 of 250,000
relevant rows**; `color detail` is on **78.7% of relevant rows, 0% of semi rows, 0% of DEV rows**.
On DEV the dotted-i marker is label-independent (χ² p=0.19) — i.e. worth exactly nothing at test time.

**Two things this adds beyond §8:**

1. **The encoding bug fragmented the taxonomy itself.** 252 of 3,173 distinct category strings
   collapse once `U+0307` is stripped. `elektronik/beyaz eşya & i̇klimlendirme` looked 100% semi
   (6,414 rows) while the plain-`i` spelling of the same node looked 100% relevant (1,924 rows).
   Any per-category statistic computed before normalisation is wrong.
2. **It is not only dropped keys — the same concept was RENAMED between snapshots.** Normalisation
   must map synonyms, not just strip characters:
   | snapshot A (relevant rows) | snapshot B (semi rows + all test rows) |
   |---|---|
   | `color detail` | `web color` |
   | `taban materyali` | `alt taban materyali` |
   | `iç taban materyali` | `iç astar & iç taban materyali` |

**Every train-fitted R/S number produced by this project — ours included — is contaminated until
both snapshots are normalised.** That includes the per-vertical and per-constraint rates in the
A1–A13 reports, which were computed on raw `train.csv`.

### 8c. The per-vertical miscalibration is a SCALE effect, not a shift (A7)

Why the category-root feature was worth only +0.008 while category clearly matters: the
`semi-relevant` band is **3.1× wider in compatibility verticals (auto, electronics) than in fashion**.
- per-regime threshold **scaling**: **+0.056** macro-F1 on DEV (0.414 → 0.470, disjoint CIs, vs a
  0.016 minimum detectable difference)
- per-regime **offset**: only +0.015
- training-sample importance reweighting: +0.009 (below detectability)

A linear feature can shift a decision band but cannot widen it — which is exactly why adding
`cat_root` to a classifier did nothing. **RECOMMENDATION: per-vertical threshold scaling, not a
category feature and not reweighting.**

Also measured: lexical AUC(R vs S) spans **0.543 (otomobil & motosiklet) to 0.780 (giyim)**, and
head-noun-in-title is *inverted* in auto and kitap. The gender/age_group availability shift is
**100% a category-mixture artefact** (predicted 0.5062 vs actual 0.4952 unknown-rate), not a class
signal — do not model it.

**Test-weighted priority** (headroom per row is nearly flat at 0.0122–0.0189, so priority ≈ test
frequency, the inverse of train frequency for the top two):
giyim **+0.192** → ev & mobilya +0.073 → elektronik +0.055 → ayakkabı +0.049.

---

## 9. PHASE 0 — snapshot normalisation (done 2026-08-19, verified)

`asya/s2_exp/common/normalize.py` → `train_norm.parquet`, `public_test_norm.parquet`, `drop_keys.csv`.
Runs in ~85 s. **The identical transform must run on the private CSV at inference** (a fixed key
list + a Unicode fix; deterministic, no labels needed).

**Transforms.** T1 NFC + strip `U+0307` on every text field (also collapses 252 duplicate category
strings). T2 drop attribute keys whose *presence* predicts the train label ≥0.93 — the leak's actual
signature, since every relevant row came from snapshot A and every semi row from snapshot B.
T4 `gender: unisex → unknown` (4.5% of train, **0.0%** of test, 22,516/22,518 relevant).

**Three filters I got wrong first, recorded so they are not repeated:**
1. **Renaming instead of dropping.** Mapping `color detail`→`web color` merged a 196,763-row
   snapshot-A marker into a genuine 4,550-row key: 201,313 train rows at P(relevant)=0.977 vs 169
   test rows. The leak survived under a legitimate name.
2. **A one-sided filter.** Dropping only keys *absent from test* misses the other half: snapshot-B
   keys (`iç astar & iç taban materyali`, `alt taban materyali`, `web color`) are P(relevant)=**0.000**
   in train while being **more common in test than in train** (7.1% vs 2.7%).
3. **Prevalence ratio as the proxy.** Confounded by the category-mix shift (giyim 12.5% of train vs
   31.8% of test). It dropped `uyumlu marka`/`uyumlu cihaz` — load-bearing for accessory reasoning —
   and cost 55% of test rows an attribute. Purity-on-label is the correct signal.

**Verified after (all pass):** row/column/label counts preserved · idempotent · runs on an
unlabelled test-shaped frame · holdout still quarantined (`sha256 1e246ab8`, 0 row/query overlap) ·
`harness.macro_f1` bit-identical to the organizer's `evaluate.py` over random trials.

**Effect, measured end to end.** Two-rule oracle: **46.89% of train at 0.999991 → 0.45% at 0.9956.**
Metadata-only adversarial probe (R vs S; no query-item features, so 0.50 is the honest baseline):

| features | RAW train→DEV | NORMALISED train→DEV | verdict |
|---|---|---|---|
| all metadata | 0.9411 → 0.4972 (**gap +0.444**) | 0.7662 → 0.5904 (gap +0.176) | leak cut ~60% |
| attribute richness | — | 0.633 → 0.611 (**+0.022**) | **genuine** — relevant items really are richer in both sets (9.52 vs 6.25 train; 10.73 vs 7.30 DEV) |
| row shape | — | 0.604 → 0.605 (**−0.001**) | **genuine**, transfers perfectly |
| key presence | — | 0.714 → 0.586 (**+0.128**) | **the residual leak** |

**RISK — a residual remains and cannot be removed by attribute surgery.** Dropping 10 keys or 66
gives the same ~0.115 gap, and values-only serialisation only moves it 0.202→0.187 (though it saves
48% of tokens). The two exports differ across the whole attribute vocabulary — `renk` alone collapsed
from 347 values to 31 — so any attribute-derived feature carries some snapshot signal. Accepted, with
two mitigations: the probe is a standing gate before any training run, and see below.

**RISK — mining can create a NEW three-way leak.** Mined irrelevant rows will be built from
`train.csv` items, which are a *mix* of snapshot-A-origin (from relevant rows) and snapshot-B-origin
(from semi rows). Sampling them uniformly would give `irrelevant` a snapshot signature that is
neither A nor B — a fresh artefact correlated with the manufactured class. **Mined item selection
must be stratified so snapshot origin does not correlate with the manufactured label**, and the
adversarial probe must be re-run on the final three-class mixture, not just on train.
