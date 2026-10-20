# E17 — irrelevant-class mining + labelling, v5 (the frozen pipeline)

Stage-2 train has 250k relevant + 250k semi and **zero** irrelevant rows; the test is balanced
thirds. This directory builds the irrelevant class ONCE: rule-mined candidates shaped like the
organizer's irrelevants, judged by the validated LLM judge, frozen with a manifest.

## Why v5 (what the earlier attempts got wrong)
1. v3.2 selected candidates by Asure-12B confusability ("hard band"). Real organizer irrelevants
   are plainly wrong near-items (63% zero title overlap, 62% gender-tagged, Asure margin median
   −1.0); the hard band was the opposite. Adversarial AUC 0.91, DEV-I transfer 0.37.
2. e08 propensity-matching repaired the shape post hoc (AUC 0.63) — a patch, rejected.
3. v4 (rule recipes, native shape) stalled at 61.8k: item-reuse cap ≤2 discarded 65k rows after
   generation, a flat 15/query cap discarded 32k, fashion (43% of real irrelevants, 13% of train
   queries) was starved, AUC 0.74 — and the rows were stamped `irrelevant` without any judge call.

## Pipeline (scripts/, run in this order; every stage writes parquet + manifest + sha256)
| stage | script | what |
|---|---|---|
| 1 raw | `mine_v5.py` | 10 A14 recipes over Pool A (train item tuples, both snapshots), each emitting a RANKED list per query (depth 30/20/15/8/…), run to seed exhaustion. Train-pair drop, public-query drop, cross-recipe dedup (rarer recipe wins). No preference weighting. |
| 2 select | `select_v5.py` | N=170k. Per-query cap `clip(0.75·n_RS, 1, 30)`; item reuse ≤3; department profile = DEV-irrelevant item roots; recipe mix (M8 ×1.6 to pre-pay judge losses); 55% single-brand clusters; label-store R/S block-list. Fill = rank-major round-robin. |
| 3 gates | `gates_v5.py` (+ `score_v5.sh` → `e05_score/scripts/score_asure.py`) | (a) adversarial AUC ≤ organizer baseline + 0.02 (see below), (b) fingerprint vs DEV-I, (c) hygiene, (d) label-store coverage, (e) Asure hardness: frac(margin>0) in [0.18, 0.28] (DEV-I = 0.234). **No judge call until all pass.** |
| 4 judge | `judge_v5.py` | split → canary 20k (kill rule: keep ≥0.85, no non-M8 recipe <0.80) → remainder → merge into `e16_mine_v4/outputs/label_store.parquet`. Same gemini-3.7-flash setup validated at 0.924 macro-F1. |
| 5 freeze | `finalize_v5.py` | judged-I → irrelevant, judged-S → semi-relevant, judged-R → relevant (flagged). Re-run gates (a)/(b) on the judged-I subset. `outputs/irrelevant_final.parquet` + manifest. 300-row blind hand audit (precision ≥0.85). Read-only afterwards. |

## Recipes (A14 names; purity = hand-adjudicated)
M1 brand swap (right brand, wrong type; .89) · M2 cross-type (same dept, other family, no type
token; .89–.96) · M3a far colour (~1.0) · M3b wrong device model (~1.0, query-templated) · M3c
wrong number/size (~1.0) · M4 wrong accessory (.89–.96) · M5 wrong entity (~1.0) · M6 lexical
trap (type token only in attributes/category; .96) · M7 random far (.96, kept ≤3.5%) · M8
qualifier strip (.61 — judged 100%, its semi rows are kept as semi).
Deliberately absent: gender-mismatch (organizer shows it in 2/3,750 DEV rows; rubric: not a
signal), Stage-1 pools (stale catalog text), Asure as a labeler, any DEV re-fitting.

## Gate (a) recalibration — done BEFORE any judge spend (2026-08-22)
The inherited bar (AUC ≤ 0.65, from e08) is below the train→test domain-shift floor: on the identical
feature set and protocol the organizer's OWN train-relevant rows score 0.738 against DEV-relevant and
train-semi 0.681 against DEV-semi. Nothing built from train queries/items can go under that. The
enforced rule is therefore **our-I vs DEV-I AUC ≤ (train-R vs DEV-R AUC) + 0.02**, both computed in
`gates_v5.py` and written to the manifest. History of this selection: rule recipes only 0.798 →
+M10 retrieval 0.756 → +fashion-query budgets 0.728 (below the organizer's own 0.738).
The 10th recipe **M10 retrieval near-miss** (`recipes_v5.py`): tf-idf top-80 over title+category+brand
minus the query's own R/S rows minus its forbidden leaves — the organizer's retrieve-then-judge step;
purity is the judge's job (every row is judged).

## Constants come from DEV-irrelevant (1,250 rows / 825 queries) ONCE — documented, never re-fitted.
HOLDOUT is never read by anything here.
