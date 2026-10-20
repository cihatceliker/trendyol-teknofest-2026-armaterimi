> Unless stated otherwise, every macro-F1 in this report is measured on the DEV half of the
> public set with a query-disjoint, out-of-fold (OOF) protocol. The public set was split once
> into DEV (3,750 rows) and HOLDOUT (3,750 rows); all fitting and all model selection used DEV
> only, and HOLDOUT was read only to confirm a decision that DEV had already made.

## Summary of the pipeline flow

The task is to label a (query, product) pair as `relevant`, `semi-relevant` or `irrelevant`.
Scoring is three-class macro-F1 on a private set of 90,000 rows, plus a throughput score
measured from container start to the written output file. Inputs: `train.csv` (500,000 rows,
250k relevant + 250k semi-relevant, no irrelevant) and `public_test.csv` (7,500 labeled rows,
used for measurement only, never for training). Our solution has four parts.

1. **Text normalization, two layers.** The organizer's `train.csv` was assembled from two
   catalogue exports, and which export a row came from predicts its label almost perfectly
   while carrying no signal at test time (see EDA). A deterministic normalization layer (NFC,
   removal of the stray combining dot U+0307, dropping 18 leaked attribute keys, mapping the
   export-specific `gender` values `unisex` and `çocuk` to `unknown`) closes that leak. A second
   attribute-cleaning layer (60 keys plus boilerplate values) shortens the input by 15–17% of
   tokens and adds +0.0155 DEV macro-F1. The same transforms run byte-identically at inference.
2. **Generating the missing `irrelevant` class.** Train has no irrelevant rows; the test set
   is one third each. We generated candidates with ten rule recipes designed around the
   organizer's real irrelevant profile, plus a tf-idf retrieval "near-miss" recipe, and let no
   row be labeled before the candidate pool passed distribution gates. 100% of the candidates
   were then labeled by a validated LLM judge (0.924 macro-F1 against a hand-written rubric); a
   blind three-reader audit of the frozen set measured 0.987 precision. Result: 145,848 frozen
   irrelevant rows plus the semi/relevant "extras" the judge returned.
3. **Model.** `dbmdz/bert-base-turkish-128k-cased` (BERTurk-128k, 12 layers, 184,347,651
   parameters) fine-tuned for 6 epochs as a cross-encoder with a 3-class head. The decision rule
   is ordinal: the three probabilities collapse to one scalar, `x = p_semi + 2·p_rel`, which two
   thresholds split into three classes. The thresholds are fitted on DEV only. DEV OOF macro-F1
   is **0.8546**; the submitted image scores **0.857943456149** on the full public set with the
   organizer's `evaluate.py` (DEV half 0.8557, HOLDOUT half **0.8602**). HOLDOUT scoring above
   DEV is direct evidence that the thresholds are not over-fitted to the half they were fitted on.
4. **Serving.** One Docker image with two modes (prediction and explainability). The
   prediction path avoids importing `transformers`, reads and tokenizes in parallel worker
   processes, and batches by length. Measured end to end inside the container with the jury's
   own flags: **~20.0 s for 90,000 rows ≈ 4,500 rows/s** on a single run (repeated warm runs
   reach 17.8 s ≈ 5,030 rows/s).

**Why BERTurk and not the strongest model.** All candidates below were trained on the same
data; throughput is measured inside the container.

| candidate | DEV OOF | public 7,500 | rows/s in container |
|---|---:|---:|---:|
| TY-ecomm 12L (custom head) | 0.8643 | 0.8653 | 2,680 |
| TY-ecomm 8L (custom head) | 0.8581 | 0.8626 | 3,322 |
| **BERTurk-128k 12L — submitted** | **0.8546** | **0.8579** | **~4,500** |
| turkish-medium 8L/512 | 0.8337 | 0.8409 | 5,427 |

The speed axis is `10 × SPS / best_SPS` with no rank multiplier, and the fastest team sets the
denominator for everyone. BERTurk gives up 0.0074 public macro-F1 for 68% more throughput and
has the best worst-case total across the competitor scenarios we modelled. We chose the Pareto
point, not the quality end.

The explainability UI runs on the same checkpoint: Integrated Gradients on the decision scalar,
word-level attention alignment, counterfactual re-scoring, a faithfulness curve and Turkish
robustness checks. No LLM is used anywhere.

## EDA and key findings regarding the data (if any)

**Finding 1 — a two-export label leak (critical).** `train.csv` was assembled from two
catalogue exports: the relevant rows come from one, the semi-relevant rows from the other, and
the test set's three classes all come from the newer one. Anything that differs between the two
exports is therefore a perfect predictor of the train label and pure noise at test time. Five
format markers prove it (measured on all 500,000 rows):

| marker | train n | P(relevant) | public-test n |
|---|---:|---:|---:|
| `color detail:` present in `attributes` | 196,763 | 1.000000 | **0** |
| `gender == unisex` | 22,518 | 0.999911 | **0** |
| `gender == çocuk` | 16,634 | 0.000000 (100% semi) | 664 (43% irrelevant / 30% relevant at test) |
| combining dot U+0307 in the title | 32,822 | 0.000000 (100% semi) | 1,110 |
| empty `attributes` | 2,247 | 0.000000 (100% semi) | 85 |

46.9% of train rows carry at least one marker; a two-rule oracle classifies 60.8% of raw train at
100% accuracy. The exports also differ in how they lower-case `İ` and in schema (`renk` shrank from 347
values to 31; `color detail` was renamed `web color`). On the test side the markers are absent or
class-independent, so a model fitted on raw train learns a perfect shortcut that at test time
never fires or fires as noise, and every train statistic computed without normalization is
contaminated. The fix is Layer 1 of Data Cleaning.

**Finding 2 — the Stage-2 `relevant` class is the Stage-1 positive set.** 249,984 of the 250,000
relevant rows match Stage-1 training pairs. This was an early sign that the relevant boundary is
comparatively easy.

**Finding 3 — train→test shift.** For relevant rows, the share of query tokens found in the
product text is 0.850 in train and 0.688 in public test; `giyim` is 12.5% of train queries and
31.8% of public ones. Part of the shift is the export schema, part is genuine.

**Finding 4 — the problem is ordinal and the hard boundary is semi vs irrelevant.** With the
Stage-1 12B model used zero-shot, one scalar plus two thresholds beat its own three-way
generation by +0.018 (0.7259 vs 0.7083). An oracle decomposition fixed our priorities: a perfect
relevant gate would add +0.050, a perfect semi/irrelevant gate +0.110, and the exchange rate is
Δmacro-F1 ≈ 0.87 × ΔAUC(semi | irrelevant). Every data and architecture decision was aligned to
those two numbers.

**Finding 5 — the organizer's irrelevant rows are not easy negatives.** 63% share no title word
with the query, 62% have the gender field filled, and the typical pattern is a missing head
noun with the modifiers present. A model that trusts lexical overlap collapses exactly there
(irrelevant F1 0.406 in the slice where 75–100% of query tokens are covered, 21% of rows). This
profile drove the negative-mining design.

**Finding 6 — the semi band is 3.1× wider in compatibility verticals** (automotive, electronics)
than in clothing. We left this to the learned probabilities rather than per-vertical thresholds.

**Finding 7 — even the spelling of "no attributes" was an export marker.** Train writes it two
ways: `""` (2,247 rows, 100% semi) and `unknown` (4,097 rows, 100% relevant); public has only `""`.
Attribute cleaning v2 folds both into one sentinel.

## Negative sampling process (if any)

Train has no irrelevant rows, so the class had to be generated in the organizer's own profile.
Three attempts failed before the final pipeline, each killed by measurement:

- **Cosine-band mining:** candidates from bi-encoder similarity neighbourhoods landed in the
  relevant/semi neighbourhood; the judge found only 6–21% of them irrelevant.
- **"Hard-band" selection:** picking the band where the 12B model was most confused produced
  the opposite of the organizer's profile (their irrelevant rows are *easy* for that model):
  adversarial AUC 0.9101 against the organizer's irrelevant rows, student true-irrelevant
  recall 0.374. Importance matching repaired it (0.6264 / 0.605) but was a patch; selection has
  to happen in the right distribution from the start.
- **Rule labels without a judge:** 61.8k rows stamped `irrelevant` on rule confidence alone
  (≈11% expected contamination). Rejected on discovery, with a rule adopted since: nothing
  unjudged enters training.

**The final pipeline (frozen):**

1. **Generation — ten recipes.** Rule recipes built around the organizer's profile (brand swap,
   type swap within a department, distant colour, wrong device model, wrong size or count, wrong
   accessory, wrong entity, a "lexical trap" where the type word appears only in the attributes,
   distant random, modifier stripping) plus a **tf-idf retrieval near-miss** recipe imitating the
   organizer's own retrieve-then-judge mechanism (top-80 over title + category + brand, minus the
   query's own relevant/semi rows and forbidden leaf categories). Train pairs and public queries
   are dropped; recipes are de-duplicated against each other.
2. **Selection — with a profile and quotas.** A per-query cap, a global per-item cap of 3, a
   department profile taken from the roots of DEV irrelevant rows, and a fashion share matched
   to the real profile (33%). These constants were read from DEV once and never refitted;
   HOLDOUT was never read.
3. **Gates — mandatory, before the judge.** (a) *Adversarial gate:* a probe that tries to
   separate our irrelevant rows from the organizer's may exceed the organizer's own train→test
   shift (AUC of train-relevant vs DEV-relevant) by at most +0.02 — nothing generated from train
   can go below that floor, so an absolute bar is meaningless. History: rule recipes only 0.798 →
   plus the retrieval recipe 0.756 → plus fashion quotas 0.728 (below that day's floor of 0.738). (b) *Difficulty
   gate:* the 12B zero-shot margin profile had to sit in the real irrelevant
   band (fraction with positive margin 0.186, band 0.18–0.28; median −1.5 vs DEV irrelevant
   −1.0). (c) Hygiene and coverage checks. Not one judge call was made before the gates passed.
4. **Labeling — 100% judged.** A stratified 20k canary first (kill rule: retention ≥ 0.85), then
   the rest. 182,000 negative candidates were judged: 152,425 irrelevant, 17,590 semi, 12,778
   relevant. Retention was 0.89–1.00 for the rule recipes and 0.62–0.66 for the two deliberately
   wide recipes (modifier stripping and retrieval near-miss), whose job was to bring borderline
   candidates to the judge.
5. **Audit and freeze.** Blind manual audit of the final set: 300 rows, three independent
   readers, rubric only — **precision 0.987**, zero rows flipped to relevant, 98.3% unanimity,
   every recipe ≥ 0.955. Frozen after a constrained post-judge re-selection: 145,848 irrelevant
   rows (+6,577 reserve), 29,551 judge "extras" that came back semi/relevant (quarantined first,
   admitted after an ablation proved them useful) and 902 device-model contrast rows. Post-judge
   adversarial gate 0.7185, below the organizer's floor of 0.748 + 0.02. Everything is frozen
   with a manifest and sha256.

Control experiment: the same selection trained with rule labels instead of judge labels stays at
DEV 0.752 and true-irrelevant recall 0.72 — the judge's contribution is measured, not assumed.

## All processes that can be considered as data generation

| generated data | process | scale |
|---|---|---:|
| Normalized train / public test | deterministic export normalization (uses no labels) | 500k + 7.5k rows |
| Cleaned attribute text (v2) | 60-key drop + boilerplate-value filter, sentinel merge | all rows |
| Irrelevant candidate pools | ten rule recipes + tf-idf retrieval, seeded | hundreds of thousands |
| Candidate difficulty scores | Stage-1 12B model, zero-shot binary margin (gates and profile only; never a label) | whole pool |
| LLM-judge labels (final production) | Gemini 3.7 Flash, rubric system prompt, canary + kill rules | 182,000 rows |
| Judge label store (earlier rounds included) | same judge, cached; every pair labeled once | ~323,000 pairs |
| Contrast rows | device-model matches for templated queries | 902 rows |
| Blind audit records | 300 rows × 3 independent readers, rubric only | 900 decisions |
| DEV/HOLDOUT split file | query-disjoint, seeded, fixed by sha256 | 3,750 / 3,750 |

The judge never produced a submission prediction; it only labeled training data, at build time,
while internet access was permitted. All generation is seeded and manifested, and the data ships
with the code in the development dump.

## Model Selection

Selection was by measurement: the same frozen data, the same training and evaluation harness,
and a pre-registered rule that a difference below the noise floor (0.012 on 3,750 rows) does not
declare a winner.

**Round 1 — Turkish encoder bake-off (early mining data).** BERTurk-128k 0.7358 > tybert 0.7265
≈ BERTurk-cased-32k 0.7249 (within noise) ≫ XLM-R-base 0.6885. BERTurk-128k became the workhorse.

**Round 2 — wide comparison on the frozen data and clean text** (3 epochs, before the gender
fix; used only to rank backbones):

| backbone (cross-encoder, 3-class head) | DEV OOF | note |
|---|---:|---|
| TY-ecomm-embed-multilingual-base (306M) + our head | 0.8529 | highest quality |
| BERTurk-128k cased (184M) | 0.8205 | gap 0.032 > 0.012 floor |
| mdeberta-v3-base | ~0.812–0.819 | measured under GPU contention; clearly below |
| multilingual-e5-large | ~0.806–0.810 | measured under GPU contention |
| Qwen3-0.6B text tower | 0.7660 | 4× slower, weakest semi/irrelevant boundary |

`TY-ecomm-embed-multilingual-base` is Trendyol's own e-commerce **embedding** model (GTE
architecture), not a classifier. We used it as a cross-encoder by attaching our own 3-class
classification head on the pooled representation and fine-tuning the whole encoder end to end;
about 63% of its 306M parameters sit in the embedding table, so its compute layers are smaller
than the parameter count suggests. Data contributions were backbone-independent (the judge
extras' +0.018 also appeared on BERTurk), so the backbone comparison is clean.

**Round 3 — the final frontier** is the table in the Summary: all four candidates trained on the
same leak-free data, throughput measured inside the container with the jury's flags. On the same
6-epoch recipe BERTurk-128k *uncased* scored 0.8511 (−2.7 sd) — the cased backbone tokenizes
lower-cased Turkish better (2% fewer tokens on the same 90k rows). The recipe's seed spread is
sd 0.0013 (n = 3); the gaps in the frontier table are many multiples of that.

**The submitted model is BERTurk-128k, and the reason is the score formula, not quality.**
TY-ecomm 12L produces 2,680 rows/s in the container, BERTurk ~4,500; the public macro-F1 gap is
0.0074 (0.8653 vs 0.8579). The speed axis is a max-normalized ratio with no rank multiplier, and
our throughput sets the denominator for every team — a slow quality leader loses speed points
against us. Scanned across competitor archetypes, BERTurk has the best worst-case total.

**Ruled out by measurement.** ModernBERT (3.327 tokens per Turkish word vs 2.053 for BERTurk,
plus a slower forward → ~3.6× slower); shipping the 12B LLM itself (~60 rows/s — loses in every
scenario of the score formula); CORAL/CORN ordinal loss heads (our decision is already ordinal
with thresholds fitted to the target metric directly; expected gain ≈ 0); the two-model cascade
and deeper pruning (What didn't work).

## Model Architecture

**Backbone.** `dbmdz/bert-base-turkish-128k-cased` (HF revision `fea3225…`, MIT): 12 layers,
hidden 768, 12 heads, a **128,000-piece WordPiece vocabulary**; 184,347,651 parameters, of which
~98M are the zero-FLOP embedding table. A Turkish-only BERT pre-trained on general corpora. The
large vocabulary matters here: it splits the same 90k rows into 9–18% fewer tokens than the
32k-vocabulary Turkish BERTs, i.e. less compute per row. Used as a cross-encoder: the input is
the `(query, product text)` pair, the product text built from a compact field-tagged template
(`title | brand | category | gender age_group | attributes`; unknown fields skipped), truncated
at 256 tokens (with clean text only ~0.1–0.2% of rows are cut).

**Head.** A 3-class softmax head trained from scratch → `(p_irr, p_semi, p_rel)` per row.

**Decision rule (ordinal, per row).** One scalar, `x = p_semi + 2·p_rel`, split by two thresholds.
The submitted thresholds are **[0.9518, 1.0619]**, fitted on DEV only and frozen in
`decision.json`. Refitting thresholds on a fresh query-disjoint half costs only 0.0031 ± 0.0024,
so threshold fitting is not a source of over-fitting. A per-category-regime variant was also
measured; on this backbone the out-of-fold estimate preferred global thresholds (0.8533 vs
0.8506), and the regime table stays in the checkpoint as inactive documentation. One design
constraint throughout: **every label is produced from its own row only.** The private set is not
processed as one batch, so no post-processing that uses batch statistics (quantile or rank
labeling) exists in the submission.

## Data Cleaning & Input Preprocessing

Two deterministic, label-free layers, applied identically to train and — at inference — to the
private CSV. The serving code carries an embedded copy of the transforms, verified
byte-identical to the repository transform on all 7,500 public rows.

**Layer 1 — export normalization.**

- **T1:** NFC and removal of the combining dot U+0307 in every text field (alone merges 252
  duplicate category strings).
- **T2:** drop the 18 attribute keys whose *presence* predicts the train label with ≥0.93 purity
  (frozen list). Presence, not value, is the leak signature, so purity is the filter.
- **T4:** `gender: unisex → unknown` and `gender: çocuk → unknown` (see below).

Measured effect: the two-rule leak oracle's coverage fell from 46.89% of train (accuracy
0.999991) to **0.45%**; the train→DEV generalization gap of a metadata-only adversarial probe
fell from 0.444 to 0.176. A ~0.13 residual from key presence cannot be removed by any attribute
surgery (the two exports differ across the whole attribute vocabulary); it was accepted, and the
probe is re-run as a standard gate on every data change. Three wrong versions of this filter are
on record: renaming keys instead of dropping them (a 196,763-row marker survived under a
legitimate name), a one-directional filter, and a prevalence-ratio proxy that confounded with
category shift and wrongly dropped the `compatible brand/device` keys. Label purity is the right
signal.

**Layer 2 — attribute cleaning (v2).**

- **60 attribute keys** dropped: the 18 leak keys plus 42 boilerplate keys in which query
  words never occur (care instructions, origin, warranty, sustainability, packaging, legal
  text, manufacturer identity…).
- Uninformative **values** dropped (`belirtilmemiş` / "not specified", `mevcut değil`, care
  sentences) — except under compatibility keys, where "not specified" is information; `yok` /
  `var` ("none" / "present") are never dropped, because `hood: none` is information.
- "No attributes left" folds into one sentinel (`unknown`), closing EDA Finding 7.

Measured effect: 34% of attribute characters and 15–17% of model tokens removed, at the same
rate in train and public (the largest item, `origin`, is present in 81–88% of rows on both
sides); truncation becomes negligible. Paired training: **0.7994 → 0.8149 (+0.0155)**,
relevant-vs-semi AUC 0.872 → 0.909. A model trained on v2 text is served on v2 text only;
train/serve parity is locked by an automatic test.

### The two-export leak in `gender`

This is the single highest-value technical finding of the work, and it is a *data* defect in the
organizer's file, not a hyperparameter. Two values of `gender` differ between the exports:

| value | train relevant | train semi | P(semi) | share of test |
|---|---:|---:|---:|---:|
| `unisex` | 22,516 | 2 | 0.0001 | **0%** — absent from test |
| **`çocuk`** ("child") | **0** | **16,634** | **1.0000** | **8.9%** (664 of 7,500) |

16,634 rows without one exception cannot be chance (probability 2⁻¹⁶⁶³⁴ in a balanced set), and
a model learns it as a certainty — yet in **test** the same rows are 43% irrelevant / 30%
relevant / 27% semi. Measured: **69.0%** accuracy on those rows vs **88.1%** everywhere else, a
19-point hole in a tenth of the test set. `unisex` had been neutralized from the start; we had
cleaned one value of the field and never audited the rest. Mapping `çocuk` to `unknown` as well:

| | no fix | patch at inference only | **leak-free retrain** |
|---|---:|---:|---:|
| TY-ecomm 12L | 0.8587 | +0.0069 | **0.8643** (+0.0056) |
| TY-ecomm 8L | 0.8474 | — | **0.8581** (+0.0107) |
| BERTurk 12L, 6 ep | 0.8428 | +0.0120 | **0.8546** (+0.0118) |
| turkish-medium 8L | 0.8221 | — | **0.8337** (+0.0116) |

The gain appears on **every architecture** and exceeds the sum of all our serving
optimizations. We then scanned **every** value of `gender` and `age_group`: no other broken value
(worst `age_group` shift 0.10; `gender=çocuk` 0.51). `brand = 'genel markalar'` ("generic
brands") carries the same signature (11,243 rows, 0 relevant) but has no neutral sink value;
the principled fix — a sink created by random brand-dropout in training — was trained at 5% and
10% dropout (0.8505 and 0.8554, both inside the seed spread, because dropout taxes real brand
signal). The artifact is real, the fix does not pay; it was **deliberately left unfixed** and is
documented as such in the README.

## Training Strategy

**Approach: single-stage fine-tuning on hard labels — no distillation.** The early plan was
distillation from the 12B model; two findings changed it: (1) the quality bottleneck was **label
quality**, not capacity — once unjudged rule labels were thrown out and every row was judged, the
student jumped; (2) a plain-label student on judged data was already competitive, and its
~30× speed advantage over any LLM path dominates the score formula.

**Final training table (the frozen run): 813,269 rows** — organizer rows 498,374 (250k semi +
248,374 relevant; the 1,626 rows whose query collides with a public query are dropped), 263,932
rows judged `irrelevant` (152k from the final mining round + 112k from the earlier label store),
and 50,963 `contrast` rows (judge semi/relevant extras + device-model contrast rows). Class
balance 284,136 semi / 265,119 relevant / 264,014 irrelevant — close to the private set's
30k/30k/30k. `çocuk`/`unisex` are folded to `unknown` in this table (T4). **No public-test row
entered any training run**; the builder drops public queries from every source and a check
asserts zero query overlap.

**Hyperparameters (the submitted run).** 6 epochs, batch 64, LR 3e-5, AdamW (weight decay
0.01), 5% linear warm-up then cosine, gradient clip 1.0, bf16, max length 256, seed 42;
**class-weighted cross-entropy** (inverse frequency) with label smoothing 0.05; length-grouped
batching. The epoch count is backbone-specific: TY-ecomm 306M peaks at 3 epochs (the 4th
over-fits), BERTurk 184M gains +0.0056 from 3 → 6 epochs and falls back at 9 (−0.0057); LR 2e-5
was below 3e-5 (−0.0031).

**Seed sensitivity** (submitted recipe, n = 3): 0.8546 / 0.8533 / 0.8559 → mean 0.8546, sd
0.0013. The submitted checkpoint (seed 42) sits exactly on its recipe mean — it is the recipe's
expected value, not a lucky draw. Every single-seed comparison in this report is read against
that ±0.0013.

Every data component was proven by a single-variable ablation on the same backbone and text —
the ladders are in the Ablation section. The organizer permits training on public-test labels;
we never showed the model weights a public row (the organizer's 500k rows plus our judged data
sufficed). Public labels were used only to fit the two thresholds and to validate.

## Validation Strategy

The only labeled data from the true distribution is the 7,500-row public set; the private set is
90,000 rows. Every choice made by looking at the public set is a chance to over-fit 7,500 rows,
so the project ran under one discipline from start to finish:

1. **DEV / HOLDOUT — query-disjoint, once, irreversible.** The public set was split by query
   (seeded; no query straddles the halves; rows and classes balanced 3,750 / 3,750). **DEV**: all
   fitting, all variant comparison, all threshold selection. **HOLDOUT**: quarantined. The split
   file is fixed by sha256 and the quarantine is checked automatically.
2. **HOLDOUT protocol: one logged read per milestone**, at most one per day, only on the
   champion DEV already chose; HOLDOUT never selects a model. Reads: (1) DEV 0.8217 → HOLDOUT
   0.8235; (2) DEV 0.8441 → HOLDOUT 0.8385 (−0.006, inside the 0.012 floor); (3) 25 Aug, that
   day's TY-ecomm container: HOLDOUT 0.8646, DEV 0.8633, full 0.8639; (4) 27–28 Aug, **the
   submitted BERTurk image**: HOLDOUT **0.8602**, DEV 0.8557, full **0.857943456149** — HOLDOUT
   above DEV again.
3. **Out-of-fold everywhere.** Thresholds are fitted inside DEV with query-disjoint folds and
   the reported number is always the OOF number. Threshold transfer cost: 0.0031 ± 0.0024. The
   real risk is not threshold fitting but **choosing among many candidates**.
4. **Noise floor and pre-registered gates.** Bootstrap sd of macro-F1 on 3,750 rows is 0.0062
      (95% CI ±0.012) ⇒ rule: no winner is declared below a **0.012** difference and the simpler
   option stays. Picking the best of N candidates inflates the score by ~0.006·√(2 ln N) of pure
   noise, so the number of comparisons was kept small and each ladder ran under a decision rule
   written in advance, as were the kill rules (judge canary retention ≥ 0.85, audit precision
   ≥ 0.85, prompt-probe gate +0.010 AUC).
5. **Metric verified.** Our macro-F1 was checked bit-identical to the organizer's `evaluate.py`
   on random trials; headline numbers carry query-level bootstrap CIs (rows are not independent).
6. **The judge was selected and audited under the same discipline.** Eight candidate LLM judges
   with the rubric system prompt were compared *paired* on the same stratified 1,200 DEV rows:
   Gemini 3.7 Flash won with macro-F1 **0.9238** (CI 0.9073–0.9377); runner-up 0.8795. Its output
   was then blind-audited (300 rows × 3 readers on the final set, precision 0.987).
7. **Adversarial probes as standard gates** — the export-leak residual on every training-data
   change, the generated-irrelevant distribution match on every mining change.
8. **The decision rule was stress-tested** against the worry "the thresholds over-fit 7.5k".
   Parameter-free alternatives: argmax 0.8322, train-prior correction 0.8354, with temperature
   0.8359; fitted thresholds 0.8529 (+0.021, transfer optimism only 0.003). A batch-quantile rule
   beat the fitted thresholds on HOLDOUT (0.8461 vs 0.8385) but violates row independence and was
   **deliberately not shipped**. On the submitted model: argmax 0.8495 → DEV-fitted thresholds
   0.8579 on public (+0.0084). Had the public set never been touched and the thresholds been
   fitted on a slice of our own training data, the score would be 0.8501 — the public labels'
   entire contribution is this +0.008 of calibration, none of it in the model. Re-splitting 12
   times each under random, query-, product- and leaf-disjoint splits, the fitting half
   over-states the other half by only ~0.003; under the hardest category-root-disjoint split
   transfer is 0.850 ± 0.014. That range is our honest expectation for the private set.

**Public macro-F1 of the submitted image, with the organizer's `evaluate.py`: 0.857943456149**
(DEV half 0.8557 / HOLDOUT half 0.8602); output sha256
`a6890ceed4a557bc05a376e295e2005d4240abb1c656aa1b62f77ce65f5a160c`. Command in README §10.

## Inference Strategy for Solution Routine

The speed score is `(SPS / best_SPS) × 10`, and the runtime counts from container start until
the output file is complete — cold start included. The serving path was designed for that, and
every optimization was accepted only through a **parity gate** (rejected unless the output is
byte-identical).

- **Lazy imports, no `transformers` on the serving path.** Script start-up 5.9 s → 0.16 s; model
  set-up runs in a background thread and the class is loaded directly (−1.2 s). The hand-written
  BERT forward matches Hugging Face at max|Δlogit| 1.05e-5 with argmax agreement 1.0000; model
    readiness improved from 6.37 s to 3.44 s. The serving script is selected from the checkpoint's
  own `config.json`; in the submitted image `lean_infer.py` runs.
- **Process-parallel data path.** At 90k scale ~46% of the work was in the single-threaded
  Python data path (read + normalize + tokenize), not in the model. Forked workers brought the
  90k projection from ~17.9 s to ~11.9 s with byte-identical output.
- **Length-bucketed batching.** Stable sort by token length, batches capped on rows and tokens,
  per-batch padding, logits scattered back to the original order.
- **Standardization.** Columns selected by name; frozen normalization and v2 cleaning embedded
  in the serving code; `row_id,real_label` written atomically; schema validated on exit.

**Measured end-to-end speed.** SPS is **wall-clock** from the start of `run_solution.sh` to the
written file, so the table is measured *inside the container* with the jury's flags on the real
90,000 rows (best of two runs). DEV OOF fits thresholds inside its own folds (pessimistic, for
ranking); the DEV/HOLDOUT/public columns come from the built images through the jury's own
`run_solution.sh`.

| candidate | DEV OOF | rows/s, 90k in container | DEV | HOLDOUT | public 7,500 |
|---|---:|---:|---:|---:|---:|
| TY-ecomm 12L | 0.8643 | 2,680 | 0.8677 | 0.8629 | 0.8653 |
| TY-ecomm 8L | 0.8581 | 3,322 | 0.8608 | 0.8643 | 0.8626 |
| **BERTurk-128k 12L — submitted** | **0.8546** | **~4,500** | **0.8557** | **0.8602** | **0.8579** |
| turkish-medium 8L/512 | 0.8337 | 5,427 | 0.8434 | 0.8383 | 0.8409 |

**Where the 20.0 s go** (submitted image): `3.9 s docker --gpus all` (~3.6 s of it is the NVIDIA
hook injecting the driver — isolated with `--entrypoint /bin/true`: 0.69 s without GPU, 4.33 s
with; not our code) + `0.1 s bootstrap` + `14.8 s server` (data path 4.8 s in parallel with model
readiness 5.2 s; inference 9.5 s; write 0.1 s) + `~1.2 s tail`. The model-independent part is
~10.5 s (53%), 3.9 s of it structural. Hence a **physical ceiling of ~90,000 / 6.3 s ≈ 14,000
rows/s for any GPU container** — the speed denominator cannot run away, which bounds the risk of
the fast choice.

**Speed gains** (all output-preserving; **0 label changes** on 90,000 rows verified for each):
hoisting constants out of the normalization loop (serial normalization 11.15 s → 3.22 s);
int32/int8 pool IPC; **removing thread over-subscription** (each worker opened its own tokenizer
thread pool — ~120 threads on 12 cores, starving the model-loading thread by 2.0 s;
`OMP_NUM_THREADS=1 RAYON_NUM_THREADS=1` gave +5…+17%, and the worker count is pinned to 10
because `nproc` reports 1 under the jury flags); the cuInit probe off the critical path (1.39 s
→ 0.02 s); no redundant re-read of the 90k CSV; `os._exit` once the output is fsync'd.

**Measured and rejected.** `torch.compile` (+6% inference, but 16–69 s of per-shape compilation
*inside the scored time*); INT8 matmuls (the raw kernel is 1.32× at our shapes, but the fused
quantize–matmul–dequantize path measures 0.68×); a TensorRT fp16 engine (0.99×); the two-model
cascade and further depth pruning (What didn't work).

**Docker delivery.** One image (`armaterimi_solution`); the organizer's frozen scripts are
byte-identical (sha256 checked on every build); base image digest-pinned, every pip package
version-asserted, weights and environment inside the image (no internet at runtime). Tested
under the full jury flag set (`--network none --read-only --cap-drop ALL --user <uid>`, tmpfs);
fails fast if CUDA is absent; image layers scanned for secrets (clean). Final rehearsal on
28 Aug, image built from the context inside the zip: 90,000 rows in 20.7 s, output
byte-identical to earlier runs, public macro-F1 **0.857943456149**; image 8.69 GB.

### Reproducibility (same input → same output)

The organizer requires that two runs on the 90,000-row test set agree on more than 99% of
labels. Three independent runs of the submitted image through the jury's own script agree on
**100.0000%** of labels (0 differing rows) and produce the same output sha256
`cbacea4b68a5b99b…` — the pipeline is bit-deterministic, not merely within tolerance. This comes
from no sampling, a stable length sort, fixed grouping, one atomic write, and streaming mode
switched off (streaming changed batch composition and, through padding-dependent bf16 noise,
moved 0.1% of labels; measured and rejected).

## Inference Strategy for Explainability Routine

**What explainability means for this model.** Everything on screen is a function of **the
forward pass of the checkpoint that produces the predictions** — same weights, same frozen
preprocessing, same decision rule. There is no second model and no narrating LLM (nothing in the
"separate explainer" class of the organizer's 25 Aug clarification). The explanation target is
**the quantity that actually decides**, the ordinal scalar `x = p_semi + 2·p_rel`, not a class
logit: the served label differs from argmax on ~3% of rows, and on exactly those rows a
class-logit attribution would explain the wrong quantity. An automatic parity test guards that
the UI's label equals the container's label (300/300 identical).

**How the information is produced.** Each panel is the model's own signal:

1. **Decision number line.** The row's scalar, the two thresholds (`[0.9518, 1.0619]`, DEV-fitted)
   and the three class regions; distance to a threshold shows how close the call was, and rows
   near a threshold are flagged as borderline.
2. **Integrated Gradients (primary).** At the word-embedding layer, target = the decision scalar
   (differentiable through softmax · rank), baseline = the tokenizer's own `[UNK]` embedding.
   On this checkpoint the integrand is plateau → cliff → plateau and equally spaced IG missed the
   cliff, so an **adaptive quadrature** is used (17-point pre-scan, gradient budget allocated by
   |Δg|, budget doubled at most twice). The **completeness check** (Σ attribution = f(input) −
   f(baseline)) is computed per request and printed, never re-normalized. Token attributions are
   summed to **words** by character offsets (exact, since IG is additive), then to fields.
3. **Counterfactual re-scoring (occlusion).** Every product field, every `key: value` attribute
   and every query word is removed one at a time and **the model is re-run**; removals that flip
   the label are listed. If none flips it, the decision is distributed — itself a finding.
4. **Faithfulness curve.** The top-k IG words are deleted in order and the scalar re-measured,
   against deleting k random words: a real attribution ranking collapses the score much faster.
   Completeness is the axiomatic check; this is the empirical one.
5. **Method agreement.** The Spearman agreement between the IG and the occlusion field rankings,
   two independent methods, shown as one number.
6. **Attention alignment (supporting).** Last-layer query→product attention with rollout,
   aggregated to **word level** and row-normalized, with a field colour band. Labelled
   *supporting* because of the "attention is not explanation" debate (Jain & Wallace 2019;
   Wiegreffe & Pinter 2019); IG is the primary explanation.
7. **Turkish robustness strip.** The query is re-scored without diacritics, without spaces, with
   a one-character typo and in upper case, showing which variant changes the decision. The model
   is brittle on unspaced compounds (`çelikbileklik`) and single-character deletions — shown on
   purpose, not hidden.
8. **Preprocessing diff.** For each field: the raw value, the value fed to the model and the
   reason (e.g. `gender: çocuk → unknown`, T4 — visible on 8.9% of public rows).
9. **Turkish summary.** Fixed templates filled from measured numbers (label, probability, the
   scalar's position against the thresholds, direction-aware drivers, the flipping counterfactual).

**GUI design.** One static HTML page (vanilla JS + inline SVG/canvas, zero external requests),
in Turkish. At the top, the organizer's mandatory block: searchable `row_id` selector, the
product's visible fields (editable), a free query box pre-filled with the selected row's query,
and an **Explain** button. Below, in order: decision card (label, probability bars, number line,
summary, latency), word contributions (blue = towards "relevant", red = towards "irrelevant"),
field contributions, counterfactual cards, attention heatmap, faithfulness curve, robustness
strip, preprocessing diff and the completeness line. Every card carries a "how to read this"
text; a footnote states the method and its limits.

**Typical use.** Select a row → Explain (~0.3 s on the GPU, 0.9 s end to end through the tunnel)
→ read the label, the scalar's position against the thresholds and the words/fields that drove
it → see in the counterfactual cards what flips the label → edit the query or any product field
and Explain again; the same model re-scores and re-explains (adding one space to the query
`celikbileklik`, for example, moves the label from "semi-relevant" to "relevant" and the point on
the number line crosses the threshold).

**Serving.** The second mode of the same image (`run_explainability.sh`, the organizer's frozen
script). Standard-library `http.server`; the server binds **before** the model loads (`/healthz`
200 within ~1 s, `/readyz` 200 when the model is ready), then the vendored ngrok v3.39.11
(sha256-pinned) opens the tunnel and prints the public URL. Without `NGROK_AUTHTOKEN` it fails
closed; the token is never printed and is in no layer of the image (byte scan). UI and tunnel
shut down with the container. Confirmed with the submitted image on 28 Aug through the
organizer's script and a real tunnel: URL live in under 60 s (limit 10 min), page and API 200
from an external network, Explain 0.9 s end to end, tunnel closed cleanly when the container was
killed. The true-label overlay used during development is behind a CLI flag the organizer's
script never passes.

## What worked & What didn't work?

**What worked**

- **Export normalization:** leak-oracle coverage 46.9% → 0.45%, adversarial gap 0.444 → 0.176.
  The single most valuable step of the project.
- **"Every row goes through the judge" plus the blind audit:** precision 0.987 on the final
  irrelevant set; no rule-labeled row ever entered training.
- **Distribution gates before labeling spend:** adversarial gate 0.7185 (below the organizer's
  floor) plus the difficulty gate; not one label was paid for before the gates passed.
- **The retrieval near-miss recipe:** passed the adversarial gate that rule-only selection could
  not (0.798 → 0.728).
- **Judge extras and the recycled label store:** +0.022 and +0.012 (scalar), both proven by
  ablation; no judge decision was wasted, even those from failed selection rounds.
- **Attribute cleaning v2:** −15–17% tokens and +0.0155 F1 at once — speed and quality.
- **Ordinal scalar + two thresholds:** +0.018 over three-way generation even zero-shot; cheap,
  fast, one-dimensional and therefore explainable.
- **DEV/HOLDOUT + OOF + pre-registered thresholds:** three of four HOLDOUT reads above DEV
  (+0.002, −0.006, +0.001, +0.005); no decision was ever made on a single public number.
- **Parity-gated speed engineering:** every optimization accepted only with byte-identical
  output; data-path parallelism alone gave ~33% end to end.

**What didn't work (rejected by measurement)**

- **Cosine-band negative mining:** candidates fell into the relevant/semi neighbourhood;
  irrelevant yield 6–21%.
- **"Hard band" candidate selection:** the opposite of the organizer's profile (adversarial AUC
  0.9101); importance-matching repaired it (0.6264) but as a patch the pipeline was redesigned.
- **Rule labels without a judge:** a 61.8k "clean" set turned out never judged (~11% expected
  contamination); found and rejected — the trigger for the "label quality over distillation" turn.
- **The distillation pipeline:** 12B teacher + soft-target student, closed once plain-label
  training on judged data proved sufficient on its own. The simple thing won.
- **Prompt/verbalizer probes on the 12B:** best variant +0.008 AUC(semi | irrelevant) against a
  pre-registered gate of ≥ +0.010 → rejected.
- **CORAL/ordinal loss heads:** the decision is already ordinal and CORAL loses to plain CE in the
  literature → not tried, deliberately.
- **Auxiliary semi/irrelevant loss:** no effect (0.8444, AUC unchanged) → the rest of that ladder
  cancelled.
- **ModernBERT:** Turkish fertility 3.327 vs 2.053 tokens per word plus a slower forward ⇒ ~3.6×
  slower; out.
- **Two-model cascade** (fast model takes the confident rows, big model the rest): 0% routed
  0.827 @0.69×, 50% 0.847 @0.93×, 100% 0.850 @1.00× — running the fast model on every row eats
  the gain; no sweet spot. Model choice is binary.
- **Depth pruning (12 → 8 layers) — first rejection reversed.** The first measurement (0.8393
  @1,938 rows/s) moved two variables at once (old data recipe *and* the untuned serving path).
  Re-trained on the champion recipe, TY-ecomm 8L gives 0.8474 (3 seeds, sd 0.0041) at 3,351
  rows/s vs 12L 0.8587 @2,652 — 0.0113 F1 for +26% speed. It became a live candidate and lost to
  BERTurk on the final frontier. Lesson: reject an option by moving one variable.
- **Pruning further (8 → 6 layers):** TY-ecomm 6L 0.7991 — *below* the 0.80 bar where the speed
  score is zeroed. The knee is at 8 layers and it is sharp.
- **torch.compile, INT8, TensorRT:** +6% / 0.68× / 0.99× (Inference) — all net negative or flat.
- **More epochs:** TY-ecomm over-fits at the 4th epoch (~0.845 → ~0.808); BERTurk peaks at 6 and
  falls back at 9 (−0.0057). Epoch count is backbone-specific.
- **Global recalibration (prior / temperature), the batch-quantile rule and per-vertical
  thresholds:** the model calls semi slightly too often (37% vs 33%), but the row-independence
  constraint, the small measured gains and over-fit risk kept every such rule out of the
  submission; one global threshold pair remains.

## Project development environment specifications (Hardware & Software)

**Development and test machine (same spec as the organizer's runtime):** Debian 12, x86_64,
1× NVIDIA A100-SXM4 40 GB (driver 580.126.20, CUDA 13.0), 12 vCPU, 83 GB RAM. Stage-1 development
had also used a single RTX 4080 SUPER 16 GB.

**Software:** Python 3.11 (development) / 3.12 (image); PyTorch 2.10.0+cu128; transformers 4.57.6
(pinned); tokenizers; safetensors; numpy / pandas; scikit-learn (analysis only). During data
generation the 12B reference model was served with vLLM 0.26.0 (build time only). Container base
`pytorch/pytorch:2.10.0-cuda12.8-cudnn9-runtime`, digest-pinned; all versions asserted at build.

**External services (build time only, while permitted):** LLM-judge labeling through the
OpenRouter API (Gemini 3.7 Flash). Spend on record: early rounds ~44 USD (29.47 + 14.7), final
production ~60 USD (canary 7.36 + extra 3.71 + remaining ~49) — **~105 USD** total, reconciled
with the API usage meter. No internet at runtime; all weights inside the image.

**Image:** final build **8.69 GB** (explainability app and vendored ngrok included); model weights
`model.safetensors` **369 MB** bf16 (184,347,651 parameters, BERTurk-128k base, revision
`fea3225…`, sha256 `2a0d4c78…`); served in bf16. Peak VRAM < 4 GB; container RAM far below the
70 GB limit. All figures match the submission README (§9.3).

## Ablation Study (How effective was each stage of the solution on its own?) (if any)

All numbers are query-disjoint OOF on the DEV half of the public set (3,750 rows), three-class
macro-F1 unless stated. Each row is a single-variable, paired comparison.

**A. Problem structure (12B zero-shot references).**

| stage | macro-F1 | contribution |
|---|---:|---|
| Surface features (20 lexical/structural features, gradient boosting) | 0.6392 | baseline |
| 12B zero-shot, 3-class generation | 0.7083 | +0.069 model strength |
| Same model, one scalar + two thresholds | 0.7259 | +0.018 ordinal decision |

**B. Data and recipe ladder (TY-ecomm backbone, same OOF protocol; each row adds one component).**

| configuration | DEV OOF | Δ |
|---|---:|---:|
| judged irrelevant + v1 text, 2 epochs | 0.8149 | — |
| + attribute cleaning v2 | 0.8217 | +0.0068 |
| + judge semi/relevant extras (29.5k) | 0.8441 | **+0.0224** |
| + earlier label store (136k pairs) | 0.8529 | +0.0088 |
| + 3rd epoch (final TY recipe) | 0.8553 | +0.0024 |
| + two-threshold ordinal decision rule | **0.8587** | +0.0034 |
| *comparison:* argmax decision | 0.8434 | −0.0119 |

Two earlier paired measurements complete the picture: replacing the unjudged rule-labeled
irrelevant set with the fully judged one, 0.7766 → 0.7920 (+0.015, BERTurk); backbone BERTurk →
TY-ecomm on the champion data, 0.8205 → 0.8529 (+0.032). The largest single gain is the judge
"extras" (+0.0224) — the judge correcting the boundary that mining missed.

**C. Decision rule (same probability files).** argmax 0.8322 → train-prior correction 0.8354 →
plus temperature 0.8359 → two fitted thresholds **0.8529** (+0.021; transfer optimism 0.003).

**D. The mining chain in isolation.** As candidate selection improved, the gate metric and real
transfer moved together: adversarial AUC 0.9101 → 0.6264 (repair) → 0.7185 (redesign, below the
organizer's floor); the student's true-irrelevant recall 0.374 → 0.466 → 0.605 → semi/irrelevant
AUC 0.9037 in the final model. The judge label's own contribution: the same selection with rule
labels stays at DEV 0.752 / true-irrelevant recall 0.72.

**E. Contributions verified on the submitted BERTurk (same recipe, one variable at a time).**

| change | DEV OOF | Δ |
|---|---:|---:|
| 3 → 6 epochs (before the gender fix) | 0.8372 → 0.8428 | +0.0056 |
| leak-free retrain with `gender: çocuk → unknown` | 0.8428 → 0.8546 | **+0.0118** |
| two-threshold ordinal rule vs argmax (public 7,500) | 0.8495 → 0.8579 | +0.0084 |
| seed spread (n = 3) | 0.8533–0.8559 | sd 0.0013 |

Tried on the same recipe and rejected: 9 epochs −0.0057, LR 2e-5 −0.0031, uncased backbone
−0.0035, brand-sink 5% −0.0041, query-glue / typo augmentation ±0.002 (noise).

The isolated effect of normalization is deliberately not reported as macro-F1: no comparison
model was ever trained on raw train (to avoid producing a contaminated number); its effect is
measured with the leak metrics (oracle 46.9% → 0.45%; adversarial gap 0.444 → 0.176). Ladder B
was measured on the TY-ecomm backbone; on the submitted BERTurk the components were verified to
move in the same direction (leak-free retrain +0.0118; two-threshold rule vs argmax +0.0073 DEV
OOF / +0.0084 public). The full ladder was not repeated on BERTurk, and what was not done is not
written here.
