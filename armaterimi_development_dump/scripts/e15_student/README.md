# E15 — student distillation (BERTurk-128k <- Asure-12B teacher)

The primary-submission student: `dbmdz/bert-base-turkish-128k-cased` cross-encoder (winner of
the e08 bake-off, `e08_repair/outputs/bakeoff_*.json`), distilled from the e11 Asure-12B
teacher. Evidence base: `model_survey/B1_speed_options.md` §10 (soft/scalar distillation beats
hard labels; **data volume, not capacity, is the binding constraint** — the AliExpress 100k-pair
collapse row), §12 (ladder economics: training is minutes, teacher labelling is the expensive
part).

## Scripts

| script | role |
|---|---|
| `scripts/build_student_data.py` | assemble `outputs/student_train.parquet` from up to 3 sources (below) |
| `scripts/train_student.py` | CE+KL trainer -> `outputs/ckpt_<TAG>/` (HF **and** lean-e12 format) + `train_meta_<TAG>.json` |
| `scripts/eval_student.py` | DEV readouts (e11-comparable) -> `outputs/eval_<TAG>.json` |

Data sources (each optional; see builder docstring for the teacher-parquet contract):
(a) organizer R/S from `common/train_norm.parquet` (hard labels; the 78 public-overlap
queries / 1,626 rows are dropped, e11 logic); (b) judged-I from
`e08_repair/outputs/matched_labels_v1.parquet` (`judge_label=='irrelevant'`, ~47.9k rows, text
via `common/normalize.py` + `_pool_ext`, e08 path); (c) teacher probs parquet(s) via
`TEACHER=` — `(query, item_key | full text cols, p_irr, p_semi, p_rel)`; rows matching an
(a)/(b) row on the full text identity get probs attached, the rest join as `teacher_only` rows with
empty `real_label` (the tomorrow ~1.2M-row teacher job lands here; unlabeled mined candidates =
`e05_score/outputs/asure_scores_v32.parquet`, 684,625 pairs, text via `_pool_ext`).

Loss: `alpha*CE(hard, label-smoothing 0.05, class weights) + beta*T^2*softCE(teacher@T)` per
row; rows with only one signal use it at full weight. Defaults `ALPHA=0.5 BETA=0.5 TEMP=2.0`.
Length-grouped batching (e11 `train_plan`; B1 §3: bucketing ~2x), maxlen 256, bf16 autocast
without GradScaler, linear-warmup cosine, seed 42.

## The experiment ladder (pre-registered 2026-08-21, BEFORE any run)

Throughput basis: ~650 rows/s train (B1: 3,640 rows/s fwd @256 / ~3 for fwd+bwd, x ~1.9
bucketing at natural lengths, conservative). All rungs: `EPOCHS=2 BS=64 LR=3e-5`, seed 42, one
run each — no seed shopping.

| rung | data | rows (x2 ep) | est. wall | build / train |
|---|---|---|---|---|
| **S1** hard labels only | organizer 498,374 R/S + ~47.9k judged-I, class-weighted (I is ~9% of rows) | ~546k (1.09M) | **~28 min** | `TEACHER= python build_student_data.py` then `TAG=s1 ALPHA=1 BETA=0 python train_student.py` |
| **S2** = S1 + teacher KL on the same rows | same rows, teacher probs attached | ~546k (1.09M) | **~28 min** | `TEACHER=<probs.parquet> OUT=<s2 dir> python build_student_data.py` (attached-only file) then `TAG=s2 python train_student.py` |
| **S3** = S2 + teacher-only rows (volume push) | + ~600k unlabeled mined candidates with teacher probs | ~1.15M (2.3M) | **~59 min** | same builder with the full teacher parquet; `TAG=s3 python train_student.py` |

S2 isolates the soft-target effect at fixed data; S3 isolates the volume effect at fixed loss.
(If the teacher job emits one parquet covering everything, S2's set is it filtered to the
(a)/(b) row identities — the builder's attach-vs-append split gives both counts in the
manifest.)

### Decision rule (pre-registered — no cherry-picking, one look per rung)

Primary readout: **DEV OOF two-boundary macro-F1** (`eval_student.py`, the deployment rule).
Reference 0.8328 (e09 DEV-half). Noise floor **0.012** (~2x the DEV binomial SE; differences
below it are noise, `harness.min_detectable_delta`).

- **Pick S3 unless it fails to beat S2 by > 0.012**; in that case pick S2 unless it fails to
  beat S1 by > 0.012; else S1. Ties/regressions always fall DOWN the ladder (simpler wins).
- Secondary sanity gates on the picked rung (diagnostic, not selection): AUC(S|I) and F1_semi
  must not degrade vs the rung below by more than the floor; slice report must not show a
  coverage-band collapse.
- S2/S3 presuppose the teacher itself cleared its e11 kill gates (AUC(S|I) >= 0.930, DEV
  two-boundary > 0.8448). If the teacher fails, the ladder stops at S1.
- **HOLDOUT is read once, milestone-only**, on the picked rung just before submission freeze
  (`HOLDOUT=1` + the two confirmation phrases in `eval_student.py`; it hard-refuses otherwise).
  90k private vs 7.5k public: every extra look is an overfitting opportunity. Never iterate on
  a holdout number.

Ablation knobs already wired for follow-ups (only if the ladder says the direction pays):
`TEMP`, `ALPHA/BETA`, `CW=none`, `ORG_N` (organizer subsample), `BUDGET` (token-budget
batches), `EPOCHS`.

## Checkpoint format & parity

`outputs/ckpt_<TAG>/` is HF format **and** the lean-e12 serving set (`model.safetensors`,
`config.json`, `tokenizer.json`, `decision.json`) in one directory —
`e12_lean_infer/scripts/verify_parity.py` is a top-level script, not an importable exporter,
so its three export lines are mirrored in the trainer. Verify graph parity on CPU (GPU stays
free) with:

```
MODEL=outputs/ckpt_<TAG> OUT=outputs/ckpt_<TAG>_parity N=200 \
  .venv-gcp/bin/python ../e12_lean_infer/scripts/verify_parity.py
```

## Smoke (GPU, later — the teacher is training now)

```
.venv-gcp/bin/python scripts/train_student.py --smoke        # 512 rows, 1 epoch, TAG=smoke
EVAL_N=300 CKPT=outputs/ckpt_smoke .venv-gcp/bin/python scripts/eval_student.py
```
