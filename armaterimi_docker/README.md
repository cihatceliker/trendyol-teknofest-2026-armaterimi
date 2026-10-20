# ArmaTerimi — Trendyol × TEKNOFEST 2026, Stage 2 submission

Turkish query–product relevance, three classes (`relevant` / `semi-relevant` / `irrelevant`).
One Docker image, two modes: batch prediction and an explainability UI.

This file is the **§9.3 Reproducibility Record** required by `COMPETITION_RULES.md`.

## 1. Image name and exact build command

```bash
cd armaterimi_docker
bash prepare_context.sh          # stages the checkpoint + serving code into the build context
CONTAINER_NAME=armaterimi_solution bash build_docker.sh
```

`build_docker.sh` sha256-verifies the organizer's frozen `run_solution.sh` and
`run_explainability.sh` before building, refuses to build if the staged context is incomplete,
and audits the resulting image's history and ENV. It also tags `example_submission` so a judge
who forgets to export `CONTAINER_NAME` still gets a run instead of exit 125.

## 2. Exact prediction invocation

```bash
CONTAINER_NAME=armaterimi_solution \
  ./run_solution.sh /absolute/path/private_input.csv /absolute/path/submission.csv
```

Output schema is exactly `row_id,real_label`, written atomically (temp file + `os.replace`),
UTF-8 without BOM, one row per input row, `row_id` copied through unchanged.

## 3. Explainability invocation

```bash
NGROK_AUTHTOKEN=<token> CONTAINER_NAME=armaterimi_solution \
  ./run_explainability.sh /absolute/path/public_test.csv
```

Fails closed (exit 2) without `NGROK_AUTHTOKEN`; the token is never printed or written to disk.
The ngrok agent is vendored into the image (v3.39.11, sha256-pinned) — **no runtime downloads**.

## 4. Base image and dependency versions

| | |
|---|---|
| base | `pytorch/pytorch:2.10.0-cuda12.8-cudnn9-runtime`, digest-pinned |
| torch | 2.10.0+cu128 (already in the base; deliberately absent from `requirements.txt` so no pip torch can replace the CUDA build) |
| python | 3.12 |
| other pins | see `requirements.txt` (exact `==` pins) |

## 5. Model

| | |
|---|---|
| backbone | `dbmdz/bert-base-turkish-128k-cased` (BERTurk-128k) |
| architecture | `BertForSequenceClassification`, 12 layers, hidden 768, vocab 128,000 |
| parameters | 184,347,651 |
| weights | bf16 on disk (369 MB), sha256 `2a0d4c78c2fe0c491f7915152337c15296af1839332ba0e0a7b470420082e191`. The server casts fp32→bf16 at load anyway, so storing bf16 is bit-identical and halves the read. |
| size / precision | 369 MB bf16 on disk; **bf16** at inference on CUDA |
| fine-tuning | 6 epochs, batch 64, LR 3e-5, max_len 256, seed 42, class-weighted cross-entropy |
| training rows | 813,269 |

Weights are **baked into the image**. There is no download path at run time and the container
runs with networking disabled.

## 6. Decision rule

Not argmax. The three class probabilities are collapsed to one ordinal scalar

```
x = p_semi + 2 * p_relevant
```

sliced by two thresholds into the three classes. Thresholds live in
`/models/current/decision.json`:

```json
{"head": "ordinal", "cuts": [0.951752, 1.061868]}
```

**The cuts were fitted on the DEV half only** — 3,750 of the 7,500 public rows, split
query-disjointly. The HOLDOUT half was never used for fitting and was read exactly once, at
freeze, to produce the number in §10. Choosing global over per-category cuts was decided by the
out-of-fold estimate on this checkpoint (global 0.8533 vs per-regime 0.8506); the per-regime
table is retained in `decision.json` as inert documentation.

## 7. Expected runtime and resources

| | |
|---|---|
| 90,000 rows, end to end | **~20.0 s** wall (container start → submission written) = **~4,500 rows/s**; repeated warm runs reach 17.8 s / ~5,030 rows/s |
| of which | ~3.9 s container start (3.6 s of it the NVIDIA runtime hook, not ours to remove), ~4.6 s data path (concurrent with model load), ~9.5 s inference, ~0.1 s write, ~1.2 s teardown |
| measurement conditions | idle GPU, jury flags, single-run figure quoted (a first run on a fresh machine is the judged case; docker container start is ~2 s slower cold than after many consecutive runs). |
| VRAM | < 4 GB peak (BERT-base class at batch 256) |
| CPU | ≤ 11 active processes (10 pinned data-path workers + server; OMP/rayon at 1 thread each) on the 12-logical-CPU target |
| RAM | well under the 70 GB the judging harness grants |
| GPU | 1 × A100-40GB |

## 8. Batch size, limits, seeds, decoding

`BATCH_ROWS=256`, `TOKEN_BUDGET=65536`, `MAXLEN=256`, `OMP_NUM_THREADS=1`,
`RAYON_NUM_THREADS=1`, `LEAN_WORKERS=10` — all baked as image ENV defaults.

Baking `LEAN_WORKERS` matters: under the judging flag set (`--cap-drop ALL` + cgroup limits)
`nproc` inside the container reports **1**, so auto-detection would collapse the data path to a
single worker.

Every one of these was swept **in-container**, and all of them are bit-exact (0 of 90,000
predictions change). What was measured and rejected: batch 384/512/768/1024 (all inside noise),
`MAXLEN` 224 and 192 (0 predictions changed, 0 speed gained — length-bucketing already isolates
the long tail), `LEAN_CHUNK` 512/1024/2048 (no gain), fp32→bf16 weights (no wall-clock gain, but
kept because the output is provably identical and the image is 370 MB smaller). The one real
finding was worker count: the host optimum (8) does **not** transfer into the container, where 10
wins by +36 SPS over 5 interleaved reps each (t = 2.93).

The thread pins matter: each data-path worker's Rust tokenizer opens its own rayon pool, so on a
12-core box N workers meant ~12N threads and the model-load thread starved. Pinning is
**bit-exact** (0 of 90,000 labels change) and worth +5–17% depending on model.

**No sampling anywhere.** Seed 42 for training; inference is a deterministic forward pass plus a
threshold. No temperature, no top-k, no beam search.

**Checkpoint/resume behavior: none.** The run is single-shot; the output file is written once,
atomically (temp file + `os.replace`), only after all 90,000 rows are scored. Interrupting and
re-running simply repeats the run and produces a byte-identical file — there is no partial-state
resume to probe (§9.2.5).

## 9. Reproducibility

The organizer requires >99% label agreement between two runs over the 90k set. Measured with the
frozen `run_solution.sh`, three independent runs:

| | agreement | differing rows |
|---|---|---|
| run 1 ↔ 2 | **100.0000%** | 0 |
| run 1 ↔ 3 | **100.0000%** | 0 |
| run 2 ↔ 3 | **100.0000%** | 0 |

All three output files share one sha256. The pipeline is **bit-deterministic**, not merely within
tolerance. What makes that true: no sampling; stable length-sort (`np.argsort(kind="stable")`);
fixed batch composition; a single atomic write. Streaming the data path was measured and
**rejected** — it changes batch composition and moves ~0.1% of labels through padding noise.

## 10. Public macro-F1 from the submitted image

```bash
CONTAINER_NAME=armaterimi_solution ./run_solution.sh \
  /home/jupyter/trendyol-teknofest-work/stage2_data/public_test.csv /tmp/public_pred.csv
python stage2_data/evaluate.py /tmp/public_pred.csv --solution stage2_data/public_solution.csv
```

```
rows: 7,500
macro_f1: 0.857943456149
output sha256: a6890ceed4a557bc05a376e295e2005d4240abb1c656aa1b62f77ce65f5a160c
```

Split for honesty about provenance: **DEV half 0.8557** (the half the cuts were fitted on),
**HOLDOUT half 0.8602** (never used for any fitting decision). HOLDOUT scoring *above* DEV is
direct evidence the thresholds are not overfitted to the fitting half.

## 11. External data and model sources (§3.4 disclosure)

- Organizer data: `train.csv` (500,000 rows, `relevant`/`semi-relevant` only — the `irrelevant`
  class is not provided), `public_test.csv`, `public_solution.csv`.
- Backbone: `dbmdz/bert-base-turkish-128k-cased` — HuggingFace, revision
  `fea322505a69df97c8bd7a01863159eb4b45900f`, MIT license. Purpose: pretrained Turkish BERT
  encoder, fine-tuned as our cross-encoder classifier. The only model inside the image.
- LLM judge (development only, never in the image or at inference): `google/gemini-3.7-flash`
  via the OpenRouter API. Purpose: labeling our mined candidate pairs to create the missing
  `irrelevant` class. Validated at 0.924 macro-F1 on 1,200 stratified DEV rows against the
  organizer's labels; a blind cross-model audit of the frozen output measured 0.987 precision.
- Audit readers (development only, never in the image or at inference): Anthropic Claude, run as
  three independent Claude Code agents on 2026-08-23. Purpose: blind second-opinion labeling of
  300 frozen judged-irrelevant pairs from the rubric alone (majority vote; precision 0.987,
  98.3% unanimous). Vote files ship in `generated_data/` of the development dump.
- Rejected-but-tested backbones (development only): `Trendyol/TY-ecomm-embed-multilingual-base-v1.2.0`
  (Apache-2.0), `ytu-ce-cosmos/turkish-*-bert-uncased` (MIT), `dbmdz/bert-base-turkish-128k-uncased` (MIT).
- The missing `irrelevant` class was generated in-house by rule-based and retrieval-based mining
  over the organizer's own catalogue, then labelled by an LLM judge (validated at 0.924 macro-F1
  on DEV rows; a blind cross-model audit by three Claude reader agents measured 0.987 precision
  on the frozen set).
  **No LLM is involved at prediction time or in the explainability path.**
- No other external corpora.

## 12. Explainability — what it does and does not do

Everything shown is computed from the **same checkpoint that produced the submission**, on the
same forward pass, with the same preprocessing:

- **Integrated Gradients** on the word-embedding layer, attributed to the quantity that actually
  decides the row (the ordinal scalar), baseline = the tokenizer's own `[UNK]` embedding, with
  adaptive quadrature and an honest completeness check reported per request.
- **Attention alignment** between query and product tokens, presented as a supporting view.
- **Leave-one-field-out counterfactuals**: each field, each attribute pair and each query word is
  removed and the pair is **re-scored by the model**, showing which removals flip the label.

**No LLM anywhere.** No meta-explainer, no post-hoc narrative model. The Turkish summary is
assembled from the measured numbers by fixed templates.

`verify_ui_parity.py` asserts the UI's label equals the container's label on real public rows by
pushing one set of logits through both deciders — currently 450/450 identical.

## 13. Known limitations and fallback behaviour

- **CUDA is required.** The entrypoint fails fast if the driver is absent rather than silently
  producing a CPU result. A full CPU run would take roughly 30× longer.
- If `decision.json` is missing, serving falls back to argmax — measurably worse. It is present.
- `serve_gte.py` (the GTE port used by an earlier candidate) remains in the image; the serving
  script is selected from the checkpoint's own `config.json`, so a plain-BERT checkpoint routes
  to `lean_infer.py` regardless of the baked default.
- The training data contains mined negatives judged by an LLM at 0.987 precision — not 1.0. The
  residual label noise is in the model.
- `brand = 'genel markalar'` is a known, deliberately **unfixed** train-export artifact (11,243
  rows, 0 relevant). Neutralising it at inference required substituting a token the model has
  never seen in that field, and the measured effect was inside the noise floor, so it was
  rejected. Recorded here rather than silently ignored.
