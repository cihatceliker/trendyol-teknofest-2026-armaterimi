# e20 — explainability web app for the CHAMPION (competition mode 2, "serve-ngrok")

The jury-facing UI that will ship inside the final submission image, adapted from e13 (which
was built against the 12B/BERT plan) to the actual champion: **e15 `ckpt_r1_oldstore`** — a
GTE-"New"-architecture ~306M cross-encoder (`AutoModelForSequenceClassification`,
`trust_remote_code=True`, 3-logit softmax head; DEV two-boundary OOF 0.8553). Development and
verification are CPU-only (`CUDA_VISIBLE_DEVICES=""`); the GPU stays free for training.

**e13 is unchanged**; this directory is a standalone sibling. Reused from e13: the stdlib
`http.server` app skeleton (fastapi/gradio are NOT in `.venv-gcp` — checked again), the
bind-before-model-load /healthz pattern, the single-file no-CDN Turkish UI (layout, number
line, token heat, field bars, attention canvas), and the IG/attention/counterfactual method
framing. Rebuilt for the champion: the model adapter (HF remote-code GTE instead of
hand-rolled BERT), the preprocessing (imported from the v2 sources of truth instead of an
inlined v1 copy), adaptive-quadrature IG (see below — uniform low-step IG provably fails on
this checkpoint), product-side leave-one-segment-out counterfactuals, manual product entry,
random-example loading, and the template-based Turkish summary line.

## What it shows (organizer rules E4/E5/E14-E17: same model + same preprocessing, live UI)

1. **Prediction panel** — the three class probabilities, the served label, and the ordinal
   scalar `x = p_semi + 2·p_rel` (= E[relevance rank]) drawn on a **number line against the
   two decision cuts**. The LABEL follows the container's serving rule byte-for-byte
   (`e12 lean_infer.decide`: softmax3 → argmax; ordinal head → `logits[0]` vs cuts).
   Cuts come from the checkpoint's `decision.json` when present; the shipped champion's
   `decision.json` is `{"head":"softmax3"}` (no cuts), so the number line draws the rank
   midpoints 0.5/1.5 and the UI labels the cut source explicitly (`cuts_source`).
2. **Class-specific Integrated Gradients** on the word-embedding layer (manual, no captum)
   w.r.t. the predicted class logit — per-token signed heat, per-field totals, and an HONEST
   completeness check printed on every explanation.
3. **Counterfactual panels (leave-one-segment-out)** — every "|" template field and every
   individual `key: value` attribute pair removed one at a time and re-scored by the same
   model (field → ""; gender/age_group → `unknown`, the pipeline's missing token;
   attributes → `unknown`, the v2 empty sentinel), plus every single-word query deletion.
   Flips are surfaced first with Δscalar. This is N+1 cheap forwards, batched.
4. **Attention alignment** (query × product tokens, last-2-layer head-mean + full rollout) —
   explicitly labelled a corroborating view (Jain & Wallace '19 ↔ Wiegreffe & Pinter '19).
5. **Turkish summary line** — template-assembled from (1)-(3). No LLM anywhere.
6. **Inputs**: searchable public-test row list (row_id / query / title substring), a
   **"Rastgele örnek"** button (`/api/random`), and **fully editable product fields + query**
   — arbitrary manual input goes through the exact serving pipeline.

## Preprocessing = serving, by construction

`model_adapter.normalize_frame` **imports** `common/normalize.py` (+ `drop_keys.csv`) and
`e18_attr_clean.apply_v2` (+ `drop_keys_v2.csv`, boilerplate filter, `unknown` sentinel) and
then builds the e07 pair template `query` vs `"title | brand | category | gender age_group |
attributes"` — the same call chain as `e15/eval_student.py` (ATTR_CLEAN=v2) that produced the
champion's DEV numbers, and the same transform `e12 lean_infer.py` inlines for the container
(byte-identity of the template asserted in the adapter test). Tokenizer: the checkpoint's own
`tokenizer.json` (Rust), truncation `longest_first` @ 256 — identical to serving. Manual
input and CSV rows share one code path.

## Adaptive IG (why not e13's uniform 32 steps)

On this checkpoint the straight-line path from the baseline to the input is a **plateau →
cliff → plateau**: the class logit barely moves for most of α, then jumps (measured |g′| up
to ~83) inside a narrow window. Uniform midpoint grids alias the cliff — measured
completeness gaps at 8/16/32 uniform steps were 0.40/3.27/0.24 (up to ~90% relative), with
the *sign* of token attributions unstable across step counts; 128 uniform steps converge
(gap 0.0016) but cost ~29 s CPU. Gradient correctness was verified directly (autograd
directional derivative == numeric d/dα along the path), so this is pure quadrature error.

Fix implemented: **two-pass adaptive quadrature**. Pass 1 samples g(α) forward-only on a
17-point grid (endpoints double as the f(input)/f(baseline) of the completeness check);
pass 2 spends the gradient budget (default 16) proportionally to |Δg| per interval with a
15% uniform floor, weighted-midpoint rule. If the honest gap still exceeds
max(0.02, 8%·|Δf|), one retry doubles the budget. Baseline token is **`<unk>`** (the model's
own trained unknown-token embedding): this checkpoint zeroes the `<pad>` row via
`padding_idx`, an extreme off-manifold baseline. Specials keep their embeddings → their
attribution is exactly 0 (asserted in the smoke test). Measured gaps after the fix: 0.2-3.2%
relative on typical rows (worst observed 11% relative when the absolute gap was already
under the 0.02 floor). The gap is printed in the UI footnote — no forced renormalisation.

## GTE-specific traps handled (from the checkpoint's own `modeling.py`)

* `NewSdpaAttention` **raises** on `output_attentions` → the model is loaded with
  `attn_implementation="eager"` (same math as serving's SDPA kernel, exact probabilities).
* `NewEmbeddings` **mutates `token_type_ids` in place** (`mul_(0)` when `type_vocab_size<2`)
  → every forward passes a fresh cloned tensor, never an `expand()` view.
* RoPE positions derive only from the sequence axis, so interpolating `inputs_embeds` for IG
  leaves positional information intact (same story as BERT-IG in e13).

## Architecture

```
e20_explain_ty/
├── scripts/
│   ├── model_adapter.py   ChampionAdapter: HF remote-code load (eager, fp32), decide() with
│   │                      lean_infer parity, adaptive IG, attention+rollout, LOSO
│   │                      counterfactuals, Turkish summary. Preprocessing imported from
│   │                      common/ + e18 (E20_WORK_ROOT / E20_COMMON_DIR / E20_E18_DIR env
│   │                      overrides for container staging).
│   ├── app.py             stdlib ThreadingHTTPServer. Binds + answers /healthz BEFORE model
│   │                      load (background thread; /api/explain 503s until ready). Model
│   │                      calls serialised with a lock. --require-token → exit 2 without
│   │                      NGROK_AUTHTOKEN (fail closed; token never printed). Hides the GPU
│   │                      (CUDA_VISIBLE_DEVICES="") unless --device cuda is given.
│   ├── dev_server.sh      CPU run against the champion ckpt + public_test.csv.
│   ├── verify_ui_parity.py  UI-vs-CONTAINER decision parity: scores real public rows
│   │                      stratified over all three cut regimes, pushes ONE set of logits
│   │                      through BOTH deciders (ChampionAdapter.decide and
│   │                      lean_infer.decide) and requires identical labels. This is the
│   │                      guard that the explanation reflects the model that predicts —
│   │                      run it after ANY change to decision.json or either decider.
│   │                      Current: 450/450 identical; the pre-fix argmax UI differed on 7.3%.
│   ├── smoke.sh           headless endpoint smoke (starts server, runs smoke_client, kills).
│   └── smoke_client.py    34 assertions + wall-clock timings over every endpoint (stdlib).
├── static/index.html      single file, Turkish, vanilla JS, ZERO external requests (no CDN,
│                          no fonts, only same-origin fetch — container runs with the tunnel
│                          as its sole allowed egress). node --check clean.
└── outputs/               smoke_server.log, smoke_results.txt.
```

### API

| Route | Returns |
|---|---|
| `GET /` | the UI |
| `GET /healthz` | 200 immediately (`model: loading\|ready\|error`) |
| `GET /readyz` | 200 when the model is loaded (+head/cuts/labels/device), else 503 |
| `GET /api/rows?q=&limit=` | substring search over row_id + query + title, ≤200 rows |
| `GET /api/row/<row_id>` | raw fields, normalised fields, exact pair-template doc |
| `GET /api/random` | one random public-test row (same shape as /api/row) |
| `POST /api/explain` | `{row_id?, query?, fields?, ig_steps?, counterfactuals?}` → prediction (label, probs, scalar, cuts+source, decision rule) + IG tokens/fields + completeness + attention matrices + product/query counterfactuals + `summary_tr` + timings. `fields` (raw manual product fields) wins over `row_id` and goes through the full serving pipeline. |

## Run locally (CPU)

```bash
asya/s2_exp/e20_explain_ty/scripts/dev_server.sh            # 127.0.0.1:7862, champion ckpt
APP_PORT=7999 scripts/dev_server.sh --ig-steps 24           # overrides
scripts/smoke.sh                                            # headless verification
```

## Measured (2026-08-24, CPU only, 12 vCPU, GPU busy training)

* Server bind + `/healthz` + 7,500 rows loaded AND v2-normalised: **t+2.5 s**.
* `/readyz` (checkpoint load fp32 + warm-up explain): **t+7.6 s** (model_load_s 5.0).
* Full `/api/explain` (forward+attention + adaptive IG@16 + 8-28 counterfactual forwards):
  **2.3-4.0 s warm** (breakdown ≈ 0.1 s attention forward, 1.5-5.7 s IG, 0.3-0.4 s
  counterfactuals; first-ever call adds ~1 s lazy init). IG@12 ≈ 1.9 s, IG@24 ≈ 3.7 s;
  retry doubles IG cost when triggered.
* Static row endpoints ≤ 3 ms.
* Smoke suite: **34/34 PASS** end-to-end in 28 s wall (see `outputs/smoke_results.txt`).

## Docker integration (e19 phase 7 — e19 is NOT modified by e20; this is the recipe)

`e19_docker_ty/run_explainability.sh` (FROZEN organizer script) runs the image as
`<IMAGE> serve-ngrok /input/public_test.csv` with `--network bridge --read-only`, tmpfs
`/tmp`, arbitrary uid, and `NGROK_AUTHTOKEN` as the only env; it exits 2 up-front when the
token is unset. The entrypoint's `serve-ngrok` branch is currently an honest fail-closed
stub. To integrate this app, the image (built from e19's context) must additionally get:

1. **This app**: `scripts/app.py`, `scripts/model_adapter.py` → e.g. `/app/src/explain/`,
   `static/index.html` → `/app/src/explain/static/` (app.py resolves `../static` relative to
   itself — keep that layout).
2. **The preprocessing sources** (the adapter imports them; they are NOT in the image today):
   `common/normalize.py`, `common/drop_keys.csv`, `e18_attr_clean/attr_clean.py`,
   `e18_attr_clean/drop_keys_v2.csv` → e.g. `/app/src/explain/common/` and
   `/app/src/explain/e18_attr_clean/`, with `E20_COMMON_DIR` / `E20_E18_DIR` env (or baked
   ENV) pointing at them.
3. **pandas** — required by normalize.py/attr_clean.py/the adapter, and absent from
   e19/requirements.txt. Add `pandas==2.3.3` (the .venv-gcp version) to the image's pip
   layer at integration time.
4. **A vendored, checksum-pinned ngrok binary** (e.g. `/app/bin/ngrok`), because runtime
   internet exists in this mode only for the tunnel and nothing may be downloaded.
5. **Replace the stub block in `entrypoint.sh`** with the e13-documented supervised startup:
   launch `python /app/src/explain/app.py "$csv" --model-dir /models/current --device cuda
   --host 127.0.0.1 --port ${APP_PORT:-7860} --require-token` in the background → poll
   `127.0.0.1:$APP_PORT/healthz` (python urllib; no curl in the image) → start
   `ngrok http $APP_PORT` → print the public URL to logs (E8) → trap EXIT to kill both (E9).
   Fail-closed on a missing token is enforced three times (frozen script, entrypoint,
   `--require-token`), and the token is never echoed (E7/E10).
6. **Budget check (R8, 10-minute link)**: the server binds and `/healthz` answers at ~2.5 s
   on CPU (row load included) and the model is ready at ~8 s; on the judge's GPU box the
   tunnel can be up within seconds of container start, model load hidden behind `/healthz`.
   Even a pure-CPU fallback is two orders of magnitude inside the budget.
7. **Read-only rootfs is already satisfied**: the app writes nothing; the HF remote-code
   module cache lands in `/tmp/hf/modules` via e19's baked `HF_HOME=/tmp/hf`. `/api/random`
   uses Python's RNG only. The UI makes zero external requests, so the organizer's traffic
   inspection sees nothing but the tunnel itself (E11).
8. `--device cuda` is the only change vs local dev; the adapter is device-agnostic (fp32 on
   CPU; the IG/attention math is identical on GPU — untested here because the GPU is
   training, flag it in the integration test).

## Open items

* Not yet containerized (deliberately — e19 is frozen until phase 7; the recipe above is the
  hand-off).
* GPU path untested (GPU is training; CPU verified end-to-end).
* No headless browser in the venv: the UI's JS is `node --check`-clean and the API contract
  is fully exercised, but eyeball the rendered page once before freeze (`dev_server.sh`,
  then open http://127.0.0.1:7862).
* If `decision.json` is later frozen WITH explicit cuts (e.g. an ordinal head or fitted
  two-cut rule), the adapter/UI pick them up automatically (`cuts_source` flips to
  `decision.json`; ordinal head switches the number line to the native scalar) — re-run
  `scripts/smoke.sh` after any checkpoint swap.
* Attribute-pair counterfactuals are capped at 30 segments per row (`n_attr_skipped`
  reported); no public row came close in testing.
