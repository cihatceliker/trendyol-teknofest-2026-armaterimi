# e12 serving-speed optimisation (v3, 2026-08-24) — CPU-verified, GPU bench pending

Score = SPS = 90000/runtime, clocked from process start to output CSV written. At 90k rows the
old single-threaded data path (CSV read + normalise + tokenise) was ~12-17s and both CLIs spent
seconds in serial imports before touching data. v3 restructures `scripts/lean_infer.py` +
`scripts/serve_hf.py` so that **imports, model load, CSV parse, normalisation and tokenisation
all overlap**, and normalisation runs on all cores. Outputs are proven byte-identical.

Pre-change snapshots: `scripts/lean_infer.py.bak`, `scripts/serve_hf.py.bak` (and the older
`lean_infer.py.v1bak` from the drop-keys v2 adoption). CLIs, env knobs, STAGE-line keys, the
normalisation bytes, decide(), the CSV format and MAXLEN=256 are all unchanged; e15/e17/e18/e19
untouched. `bench_e2e.py` and the e19 docker pair (`prepare_context.sh` copies the same two
filenames) keep working as-is.

## The changes

| # | Change | Where | CPU-verified effect | Expected on A100 @90k |
|---|--------|-------|--------------------|-----------------------|
| 1 | **Parallel data path.** csv.reader streams on the pool feeder thread; rows are normalised in forked worker processes (`LEAN_WORKERS`, default cpu-2 cap 12; `LEAN_CHUNK`=1024 rows); the main thread runs Rust `encode_batch` per returned chunk (GIL released) — parse/normalise/encode pipeline. | `lean_infer.py: make_pool, stream_norm, run_datapath` (shared by both CLIs) | read+normalise 7.5k: 0.977s -> 0.343s (2.8x; only 8 chunks feed 10 workers at 7.5k). Row-scaled datapath @90k projection: **~17.9s -> ~11.9s** (+0.65s fixed tokenizer-file load). | ~6s saved outright, and the remaining ~12s runs **during** model load (see #2/#3), so the effective serial cost of the data path at 90k is ~ max(0, datapath - model_ready). |
| 2 | **Light module imports; torch deferred.** `lean_infer` top-level imports only stdlib+numpy+tokenizers; torch/safetensors load lazily (`_ensure_torch`), normally on the model build thread. | `lean_infer.py` top + `_ensure_torch` | time-to-data-path: lean 1.556s -> **0.126s**; serve_hf 5.907s -> **0.157s**. | Same deltas (import cost is CPU work). |
| 3 | **serve_hf builds the model on a background thread** (was: serial imports -> read -> tokenize -> load). Workers are forked BEFORE the thread starts (fork+threads safety) and before any CUDA init. | `serve_hf.py: _build_model` | 64-row smoke: data path done at +1.1s while model becomes ready at +5.7s — the whole data path is free. At 7.5k the full datapath (0.96s) hid completely under the imports in the prototype. | At 90k both paths run concurrently; total startup+data ~= max(model_ready, datapath). |
| 4 | **Direct auto_map class load** for remote-code checkpoints: `transformers.dynamic_module_utils.get_class_from_dynamic_module("modeling.NewForSequenceClassification", mdir)` instead of the AutoModel registry chain. Plain checkpoints (no local auto_map) and any failure fall back to `AutoModelForSequenceClassification` automatically; `SERVE_AUTO=1` forces the fallback. | `serve_hf.py: _build_model` | model_ready cold CPU: 6.93s (Auto) -> **5.71s** (-1.2s). Outputs byte-identical (A/B on 64 rows). Standalone cold-subprocess A/B: 6.96s -> 5.08s wall. | Same ~1.2-1.9s (import cost is CPU work). |
| 5 | **`torch.set_grad_enabled(False)`** on the build thread before loading (thread-local; the inference loop already used `torch.inference_mode()`). LeanBert's decorator became an in-method `with torch.inference_mode():` (identical semantics, needed for lazy torch). | both | no measurable delta expected; verify_parity confirms the graph is still exact (max diff 4.8e-06). | neutral |
| 6 | **HF offline env** (`HF_HUB_OFFLINE=1`, `TRANSFORMERS_OFFLINE=1`, `HF_HUB_DISABLE_TELEMETRY=1`, all setdefault-overridable) — the model dir is always local; the hub must never be consulted (docker runs `--network none`, a DNS stall there would burn scored seconds). | `serve_hf.py` top | measured: no cost, no behaviour change (model loads fine offline). | robustness insurance |
| 7 | **CUDA context warm-up thread**: `torch.zeros(1, device)` starts before/while transformers imports (serve_hf) or while safetensors reads weights (lean build thread). | both build paths | **UNVERIFIED on GPU** — structure CPU-tested only (no CUDA calls were made). Expected: hides most of the ~2-3s context init under work that was serial before. Verify with the bench A/B below. | up to ~2-3s |
| 8 | `low_cpu_mem_usage` / `device_map`: measured `from_pretrained` is already mmap-lazy in transformers 4.57 (0.16s cold, no delta with the flag) — **not added**. `device_map="cuda"` (skip CPU staging of the 1.2GB) is left unimplemented: accelerate hooks + remote code is a robustness risk for ~sub-second gain. | — | measured no-op | — |

STAGE-line accounting changed subtly (bench_e2e math stays correct): `read` + `tokenize` now sum
to the row-scaled wall of the fused data path (`read` = parse+normalise+wait, `tokenize` = pure
encode); the fixed tokenizer-file load (~0.7s champion) is deliberately excluded so bench_e2e
books it in the fixed-startup residual instead of scaling it 12x.

## Parity / correctness evidence (all CPU, `CUDA_VISIBLE_DEVICES=""`)

- `scripts/parity_datapath.py` on the FULL public_test.csv (7,500 rows, embedded-newline cells
  included): old vs new serial vs parallel vs pipelined — row_ids, (query,doc) pairs, token ids
  AND type_ids **byte-identical on all rows**:
  `PARITY_DATAPATH OK rows=7500 old_read=0.977s new_serial=0.977s new_parallel=0.343s
  pipeline_read_norm_encode=1.647s projected@90k row-scaled: old~17.9s new~11.9s (+0.65s fixed
  tokenizer-file load)`
- `scripts/verify_parity.py` (LeanBert graph after the lazy-torch refactor): token ids identical,
  `max|dlogit| 4.768e-06`, argmax agreement 1.0000 -> **PARITY OK**.
- End-to-end output CSVs byte-identical to the pre-change scripts: lean 512 rows (f5 ckpt),
  serve_hf 64 rows (champion ckpt), `SERVE_AUTO=1` vs direct load, `LEAN_SERIAL_READ=1` fallback
  (both CLIs), `--no-overlap`, and serve_hf(HF Bert) vs lean_infer(hand-rolled) on the same ckpt.

## Measured CPU smokes (cold subprocess, warm page cache, DEVICE=cpu)

- serve_hf champion 64 rows — before: imports 5.907 / read starts +5.915 / model_ready 6.792.
  after: **light imports 0.157 / datapath done +1.101 / model_ready 5.709** (infer unchanged).
- lean f5 512 rows — before: imports 1.556 / model_ready 1.921. after: **imports 0.126 /
  datapath done +0.489 / model_ready 2.075** (torch import now inside the overlap window; at 90k
  it is fully hidden under the ~12s data path, at 512 rows there is little to hide it under).

## GPU benchmark — run when the A100 is free (NOT while training)

```bash
cd /home/jupyter/trendyol-teknofest-work/asya/s2_exp/e12_lean_infer/scripts
PY=/home/jupyter/trendyol-teknofest-work/.venv-gcp/bin/python
CH=/home/jupyter/trendyol-teknofest-work/asya/s2_exp/e15_student/outputs_frozen_v2/ckpt_r1_oldstore

# 1) champion via serve_hf (run twice: cold + warm page cache)
BENCH_SCRIPT=serve_hf.py $PY bench_e2e.py --model-dir=$CH --runs=2

# 2) lean path (any lean-format BERT ckpt, e.g. the f5 parity export)
BENCH_SCRIPT=lean_infer.py $PY bench_e2e.py \
    --model-dir=/home/jupyter/trendyol-teknofest-work/asya/s2_exp/e12_lean_infer/outputs/parity_f5 --runs=2

# 3) A/B the old serial data path (isolates change #1-#3)
LEAN_SERIAL_READ=1 BENCH_SCRIPT=serve_hf.py $PY bench_e2e.py --model-dir=$CH --runs=1

# 4) A/B the AutoModel import chain (isolates change #4)
SERVE_AUTO=1 BENCH_SCRIPT=serve_hf.py $PY bench_e2e.py --model-dir=$CH --runs=1

# knobs if the box differs: LEAN_WORKERS=<n> LEAN_CHUNK=<rows>
```

Baseline to beat (2026-08-24 A100 numbers): serve_hf 51.2s@90k = 1,758 SPS; lean 36.6s@90k =
2,459 SPS. Expected after v3 (estimate — verify with the bench): serve_hf ~35-38s (~2,400-2,600
SPS), lean ~30-32s (~2,800-3,000 SPS); the infer stage itself is untouched.

## Notes for the docker image (e19 — not modified here)

- `src/predict.py` imports torch before exec'ing serve_hf: fine — the build thread's torch
  import then costs ~0s and the fork still happens before any CUDA init (same ordering as a
  torch DataLoader; children never touch CUDA and never import anything).
- The 64MB /dev/shm constraint is safe: the pool passes pickled strings over pipes (POSIX
  semaphores only, bytes) — no tensor shared-memory segments like DataLoader workers.
- The Dockerfile comment "the serving path forks no processes" is stale as of v3: serve_hf now
  forks normalisation workers BEFORE any tokenizer use, so the tokenizers fork warning still
  cannot trigger (children never encode). Worth refreshing the comment on the next e19 edit.
- Rollback story: `.bak` files restore the exact old scripts; at runtime `LEAN_SERIAL_READ=1`
  (old data path) + `SERVE_AUTO=1` (old import chain) approximates old behaviour.
