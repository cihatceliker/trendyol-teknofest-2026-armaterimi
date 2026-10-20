# e12 — E0a lean inference container (B1 §8a, §12a)

The serving loop that produces the speed score (90000/runtime, clocked from container launch to
output-file completion — cold start is inside the clock). Never imports `transformers`.

## Files
- `scripts/lean_infer.py` — self-contained inference: hand-rolled BERT forward (torch +
  safetensors + Rust `tokenizers`), inlined frozen normalisation, length-sorted bucketing,
  atomic CSV write. The ONLY file the scoring container runs.
- `scripts/verify_parity.py` — correctness harness (may import transformers): exports any HF
  BertForSequenceClassification dir to the lean format and asserts logit parity.
- `scripts/bench_e2e.py` — GPU end-to-end timing harness (subprocess wall clock + stage
  breakdown + 90k projection). **Not run yet — GPU busy.**
- `outputs/parity_ckpt/` — BERTurk-base + random seeded 3-way head in lean format
  (model.safetensors, config.json, tokenizer.json, decision.json), used by the parity run.

## Design (constraints from B1 §3, §8, §8a, §8b)
- **Imports**: sys/os/time/json/csv/re/unicodedata/threading/queue + numpy, torch,
  safetensors.torch, tokenizers. `csv` over pandas: `import pandas` costs 0.48 s (13% of the
  3.6 s budget) vs 0.005 s, and the read itself is trivial. No argparse (tiny hand parser).
- **Forward**: embeddings(word+pos+type, LN) → 12× [SDPA attention, residual+LN, GELU FFN,
  residual+LN] → tanh pooler over CLS → classifier — HF BertForSequenceClassification semantics
  exactly (checkpoints without pooler weights fall back to bare CLS). Loader canonicalises
  `bert.`-prefixed and bare keys plus legacy gamma/beta. bf16 on GPU, fp32 on CPU. No
  torch.compile (§8b: net negative inside the timed window).
- **Bucketing** (§3d): stable sort by token length, adjacent rows batched under BOTH a row cap
  (128) and a token budget (rows×maxlen ≤ 32768), pad per batch to its own max, logits scattered
  back to original order. Fully-packed batches pass `attn_mask=None` for the SDPA fast path.
- **Startup overlap**: weights→device (plus CUDA context) load on a background thread while the
  main thread reads/normalises/tokenises — `encode_batch` is Rust and releases the GIL, so the
  two genuinely run in parallel. Tokenisation itself is done upfront (single multi-core
  `encode_batch`); batch assembly + pinning runs on a producer thread (queue depth 4) overlapped
  with GPU compute (`--no-overlap` for A/B — outputs verified byte-identical).
- **Decision**: `decision.json` `{"head":"softmax3"}` → argmax; `{"head":"ordinal","cuts":[c1,c2]}`
  → scalar cut at c1/c2. Labels default to the repo convention (irrelevant/semi-relevant/relevant
  = 0/1/2), honour a real `id2label` from a fine-tune config, `"labels"` overrides.
- **Output**: single atomic write — temp file in the destination dir + `os.replace`.
- **Normalisation**: inlined frozen copy of `common/normalize.py` (T1 NFC + U+0307 strip,
  T2/T3 18 drop keys frozen from `common/drop_keys.csv` @ 2026-08-21, T4 unisex→unknown) and the
  e07 pair template. **Verified byte-identical to the repo transform on all 7,500 public_test
  rows** (0 mismatches). If drop_keys.csv ever changes, re-freeze the list here and re-verify.

## Measured (this machine, 2026-08-21, CPU only — GPU busy)
- **Import chain A/B** (.venv-gcp, warm cache, 2 runs each):
  lean (numpy+torch+safetensors+tokenizers+csv) **1.67–1.71 s** vs
  torch+`transformers` Auto classes **5.81–5.89 s** → **~4.1 s saved**, matching §8a's ~4.35 s.
  Isolated: torch 1.64 s, numpy 0.15 s, tokenizers 0.017 s, pandas 0.476 s, transformers 2.72 s.
- **Parity** (verify_parity.py, BERTurk-base + seeded random 3-way head, 200 public_test rows,
  fp32 CPU): token ids identical on all rows; **max|Δlogit| = 2.7e-06**, mean 2.2e-07, argmax
  agreement 1.0000 (tolerance 0.05). The graph is exact, not merely close.
- **CLI smoke test** (300 rows, CPU): read+normalise 0.010 s, tokenize 0.033 s, model ready at
  +1.66 s (weight load fully hidden behind the CPU pipeline), atomic write 0.002 s; overlap and
  no-overlap outputs byte-identical; ordinal + softmax3 decision paths and bucketing edge cases
  unit-checked.

## Remaining (needs the GPU / a real checkpoint)
1. `bench_e2e.py` on the A100 over full public_test: confirm ~3.6 s cold start and the ~7,600
   rows/s bucketed estimate end-to-end; A/B `--no-overlap`; run twice (cold vs warm page cache).
2. Swap in the real fine-tuned checkpoint (E1 winner) via `verify_parity.py` — rerun parity, then
   freeze thresholds into `decision.json` if the head is ordinal.
3. Container packaging: model dir + `lean_infer.py` + venv; re-measure inside Docker (page cache
   is cold there — §8a run-1 numbers are the honest ones).
