# DEVELOPMENT_DUMP_PLAN — TEAMNAME_development_dump/ layout

Internal packaging plan. Source of truth for requirements: `COMPETITION_RULES.md` §8-9 and
`ORGANIZER_REQUIREMENTS.md` §1.8 (DD1-DD6) + §1.9 (K1-K8, README = K6 / rules §9.3).
Replace `TEAMNAME` with the final registered team name (ArmaTerimi) at packaging.

Organizer requirements being satisfied:

- **DD1** — all generated data (mined candidates, pseudo-labels, negative samples) — *the data
  itself, not just code* ("verilerin hem üretim scriptleri hem de kendileri").
- **DD2** — data preparation code.
- **DD3** — generation processes for every non-competition datum (negative samples, synthetic
  samples, pseudo-labels).
- **DD4** — training + validation code ("kesinlikle bekliyoruz" — non-optional).
- **DD5** — inference code that runs offline. Ours ships inside `TEAMNAME_docker/`; the webinar
  explicitly allows skipping a separate copy in the dump, but we include a pointer README.
- **DD6** — scoring runs only from the Docker; the dump is for audit/reproduction.

## Directory layout

```
TEAMNAME_development_dump/
├── README.md                     # map of this dump + how each DD item is satisfied
├── data_generated/               # DD1 — the data itself
│   ├── normalization/
│   │   ├── drop_keys.csv                   <- asya/s2_exp/common/drop_keys.csv
│   │   ├── train_norm.parquet              <- asya/s2_exp/common/ (T1-T4 output)
│   │   └── public_test_norm.parquet        <- asya/s2_exp/common/
│   ├── mined_irrelevant/
│   │   ├── mined_candidates_v32.parquet    <- asya/s2_exp/e03_mine/outputs/ (684k pool)
│   │   ├── *.manifest.json                 <- same dir (seeds, quotas, env)
│   │   ├── asure_scores_v32.parquet        <- asya/s2_exp/e05_score/outputs/
│   │   ├── pool_matched_selection.parquet  <- asya/s2_exp/e08_repair/outputs/ (frozen 50k)
│   │   └── matched_labels_v1.parquet       <- asya/s2_exp/e08_repair/outputs/
│   ├── judge_labels/
│   │   ├── hard_band_labels.parquet        <- asya/s2_exp/e04_label/outputs/
│   │   ├── cache/*.jsonl                   <- asya/s2_exp/e04_label/cache/ (raw responses)
│   │   └── judge_bench.csv                 <- asya/s2_exp/e02_judge_bench/outputs/
│   ├── training_mixes/
│   │   └── train_mix.parquet + manifest    <- asya/s2_exp/e11_si_lora/outputs/
│   ├── teacher_scalars/                    # [PENDING — produced by E3/distillation]
│   └── stage1_reused/                      # provenance of reused Stage-1 assets
│       └── README.md (run2_adapter lineage; Stage-1 dump already delivered in Stage 1)
├── data_prep_code/               # DD2
│   ├── common/                             <- asya/s2_exp/common/ (normalize.py,
│   │                                          make_split.py, harness.py, public_split.csv)
│   └── rubric_stage2.md                    <- rubric used as judge system prompt
├── generation_code/              # DD3 — process for every non-competition datum
│   ├── e03_mine/scripts/                   <- mining v1/v2/v3 (incl. failed variants)
│   ├── e05_score/scripts/                  <- Asure-margin scoring of the pool
│   ├── e08_repair/scripts/                 <- profile matching + gate + pilots + bake-off
│   ├── e04_label/scripts/                  <- LLM-judge labeling (cached, resumable)
│   └── e02_judge_bench/scripts/            <- judge selection benchmark
├── training_validation_code/     # DD4
│   ├── e11_si_lora/scripts/                <- S/I-LoRA teacher (build_data, train, eval)
│   ├── distillation/                       # [PENDING — student training scripts]
│   ├── e09_fallback/                       <- two-boundary cut fitting + verbalizer probe
│   ├── e01_stage1_predictor_on_stage2/     <- scalar extraction on stage-2 data
│   ├── e06_gate/ + e07_pilot/              <- adversarial gates + pilot harness
│   ├── model_survey/                       <- B1/B2 measurement docs + bench code
│   └── logs_and_results/
│       ├── LOG.md                          <- asya/s2_exp/LOG.md
│       ├── STAGE_UNDERSTANDING.md          <- docs/
│       └── outputs_json/                   <- e08/e09/e11 result JSONs, bakeoff_*.json
└── inference_code/               # DD5 — pointer + copies for convenience
    ├── README.md ("scored inference lives in TEAMNAME_docker/; copies below are identical")
    ├── e12_lean_infer/scripts/             <- lean_infer.py, verify_parity.py, bench_e2e.py
    └── e13_explain/                        <- app.py, model_adapter.py, static/index.html
```

Exclusions (size hygiene, all reproducible from included code + seeds): `item_emb_v0.npy` /
`query_emb_v0.npy` (rebuilt by `encode.py`), `_pool_ext.parquet` and other `_`-prefixed
intermediates, `__pycache__`, model weights (weights live in `TEAMNAME_docker/`; DD6 says
they are not tested from the dump). Keep total well under the 100 GB zip cap (K4).

## The 12 mandatory README items (rules §9.3 — goes in the submission README)

| # | Item | Source of the number | Status |
|---|---|---|---|
| 1 | Image name + exact build command | `submission_e0/build_docker.sh` | [PENDING final image] |
| 2 | Exact prediction invocation | `run_solution.sh` (frozen, sha-checked) | ready |
| 3 | Explainability invocation | `run_explainability.sh` (frozen) | ready |
| 4 | Base image + dependency versions | Dockerfile (`nvidia/cuda:13.0.2-...`), `requirements.txt` | ready |
| 5 | Model name, revision/checksum, precision, size | final ckpt + `models/` manifest | [PENDING final model] |
| 6 | Expected runtime + peak CPU/RAM/VRAM | `bench_e2e.py` 90k run + docker stats | [PENDING — not yet measured on GPU] |
| 7 | Batch size + I/O limits | lean_infer buckets (128 rows / 32,768 tokens) or vLLM config | ready per path |
| 8 | Random seeds + decoding settings | manifests (mine v3 salt, split seed 20260819, e08 seed 20260821); no sampling — deterministic scalar readout | ready |
| 9 | Checkpoint/resume behavior | none in solution mode (state it); labeling cache resumability is dev-only | ready |
| 10 | All external data + model sources | HF model list + OpenRouter judge usage + Stage-1 assets | ready to draft |
| 11 | Known limitations + fallback behavior | fallback = two-boundary 12B image; document switch criteria | ready to draft |
| 12 | Public macro-F1 from the submitted image + exact command + output checksum | §9.1 validation run at packaging | [PENDING — must also be copied into the report, T5] |

## Open gaps this plan surfaces

1. **Teacher scalars + distillation code do not exist yet** (E3 in flight) — dump dirs are
   placeholders until then.
2. **The 300-row hand audit (88.3% precision) is not recorded as a file** in the repo —
   write `data_generated/judge_labels/hand_audit_300.csv` + a short method note, or the
   report cites a number the dump cannot back.
3. **`bench_e2e.py` has never run on GPU** — README item 6 and the report's speed claims
   need that measurement inside Docker (cold page cache).
4. **Judge spend (~$45) reconstruction** — per-run costs exist in `labeling.log` and
   `judge_bench.csv`; add a one-line cost summary to the dump README.
5. Verify at packaging: executables +x (K5), zip extracts to exactly the two dirs + report
   (K1), name `TEAMNAME_submission.zip` at `/home/jupyter/` (K2/K3).
