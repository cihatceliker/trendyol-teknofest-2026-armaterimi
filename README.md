# ArmaTerimi — Trendyol × TEKNOFEST 2026

Source code and reports from ArmaTerimi's delivered Stage 2 solution for **Turkish query–product relevance classification**: `relevant`, `semi-relevant`, or `irrelevant`.

This repository is a reference for the completed competition solution. The submitted code, configuration, and reports are preserved as delivered; this README provides a guide to them.

## Start here

- **[Final solution report](armaterimi_development_dump/docs/FINAL_REPORT_EN.md)** — the approach, experiments, ablations, and measured results.
- **[Original submitted report (Word)](armaterimi_FINAL_REPORT.docx)**.
- **[Complete submission ZIP](https://github.com/cihatceliker/trendyol-teknofest-2026-armaterimi/releases/tag/submission)** — the original delivered bundle, including the checkpoint and generated data.
- **[Docker submission README](armaterimi_docker/README.md)** — the submitted model, decision rule, runtime, and reproducibility record.
- **[Experiment log](armaterimi_development_dump/docs/EXPERIMENT_LOG.md)** — the development history, including rejected approaches.

For a blog reference, link to this repository and the final solution report above. The experiment folders also contain earlier candidates and historical notes; the final report and `armaterimi_docker/` describe the submitted solution.

## Solution overview

1. **Normalize catalogue text.** Remove export-specific formatting and attribute shortcuts, then clean and shorten the product attributes.
2. **Build the missing class.** Mine candidate negatives from the catalogue, apply distribution checks, and label them with an LLM judge during development. The frozen irrelevant set contains 145,848 rows.
3. **Train a Turkish cross-encoder.** Fine-tune `dbmdz/bert-base-turkish-128k-cased` for six epochs on 813,269 rows.
4. **Use an ordinal decision rule.** Convert the three probabilities to `p_semi + 2 × p_relevant`, then apply two thresholds fitted on the query-disjoint DEV split.
5. **Serve offline and explain predictions.** The Docker image uses parallel preprocessing and length-based batching. The explainability UI uses the same checkpoint with Integrated Gradients, attention views, and counterfactual re-scoring. No LLM is used at inference time.

The submitted report records **0.857943 macro-F1 on the 7,500-row public set** and **about 20 seconds for 90,000 rows on one A100-40GB** under the judging configuration. These are the original submission measurements, not benchmarks rerun for this publication; see the report for the split, hardware, and measurement details.

## Code map

| Component | Entry point |
|---|---|
| Text normalization | [`common/normalize.py`](armaterimi_development_dump/scripts/common/normalize.py) |
| Attribute cleaning | [`e18_attr_clean/attr_clean.py`](armaterimi_development_dump/scripts/e18_attr_clean/attr_clean.py) |
| Negative mining and selection | [`e17_mine_v5/`](armaterimi_development_dump/scripts/e17_mine_v5/) |
| LLM labeling and rubric | [`e04_label/scripts/label.py`](armaterimi_development_dump/scripts/e04_label/scripts/label.py) |
| Training | [`e15_student/scripts/train_student.py`](armaterimi_development_dump/scripts/e15_student/scripts/train_student.py) |
| Threshold fitting | [`e15_student/scripts/fit_ship_cuts.py`](armaterimi_development_dump/scripts/e15_student/scripts/fit_ship_cuts.py) |
| Submitted inference | [`armaterimi_docker/src/lean_infer.py`](armaterimi_docker/src/lean_infer.py) |
| Frozen decision configuration | [`models/current/decision.json`](armaterimi_docker/models/current/decision.json) |
| Explainability application | [`armaterimi_docker/explain/`](armaterimi_docker/explain/) |
| Build and run entry points | [`armaterimi_docker/`](armaterimi_docker/) |

## Files and reproduction

Git contains the source, documentation, final report, tokenizer, model configuration, and staging checksums. The following assets from the delivered ZIP are **not included in Git**:

- `armaterimi_docker/models/current/model.safetensors` — the approximately 369 MB trained checkpoint.
- `armaterimi_development_dump/generated_data/` contents other than [`DATA_README.md`](armaterimi_development_dump/generated_data/DATA_README.md) — the generated training tables and audit artifacts.
- `armaterimi_docker/vendor/ngrok` — the vendored executable; its checksum remains in the repository.

A clone alone is therefore a source reference, not a complete runnable submission. To run the delivered solution, download and extract the ZIP attached to the [submission release](https://github.com/cihatceliker/trendyol-teknofest-2026-armaterimi/releases/tag/submission), which contains the checkpoint and vendored executable at their original paths. Prediction requires Docker with NVIDIA GPU support and a competition-format input CSV; training also requires the original data and development environment.

The bundle already contains the staged Docker context. Once those assets are present:

```bash
cd armaterimi_docker
CONTAINER_NAME=armaterimi_solution bash build_docker.sh
CONTAINER_NAME=armaterimi_solution bash run_solution.sh \
  /absolute/path/input.csv /absolute/path/submission.csv
```

The output schema is `row_id,real_label`. See the [Docker README](armaterimi_docker/README.md) for explainability and runtime details. `prepare_context.sh` and development launch scripts retain paths from the original training machine; they require adaptation if used outside that environment. `prepare_context.sh` is not needed to build an already staged bundle.

## Provenance

Published from `armaterimi_submission(1).zip`. The original archive is available as a [release attachment](https://github.com/cihatceliker/trendyol-teknofest-2026-armaterimi/releases/tag/submission); it is not committed to Git.

```text
Archive SHA-256:
b9f980a30411843d8ea2be9a9e291c7fcafcd5f26babe202ea32fee80cc047ea

Submitted model SHA-256:
2a0d4c78c2fe0c491f7915152337c15296af1839332ba0e0a7b470420082e191
```

The original per-file hashes are recorded in [`STAGING_MANIFEST.txt`](armaterimi_docker/models/STAGING_MANIFEST.txt).
