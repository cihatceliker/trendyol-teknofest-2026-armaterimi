# GTE "new-impl" lean serving port (2026-08-24)

Port of `NewForSequenceClassification` (Alibaba GTE new-impl, custom remote code) to the
no-transformers serving stack, so GTE checkpoints get the same fast cold start as the lean
BERT loop. New files only — `lean_infer.py` / `serve_hf.py` untouched:

| file | role |
|---|---|
| `lean_gte.py` | torch+safetensors-only forward, faithful to the checkpoint's local `modeling.py`. `load_gte(model_dir, device, dtype)` -> object with `.forward(input_ids, token_type_ids, attention_mask) -> logits`, drop-in for `lean_infer.infer()`'s bucketed loop. bf16 on cuda / fp32 on cpu. |
| `verify_parity_gte.py` | the ONE gte script allowed to import transformers; asserts logit parity on public_test rows (CPU fp32). |
| `serve_gte.py` | `serve_hf.py` orchestration with `lean_gte` instead of transformers. Same CLI (`serve_gte.py <in.csv> <out> --model-dir <dir>`), same env knobs, same STAGE line; data path / batching / decide / write all imported from `lean_infer`. |

Reference checkpoint: `asya/s2_exp/e15_student/outputs_frozen_v2/ckpt_r1_oldstore`
(12L x 768H x 12 heads, FFN 3072 gated, vocab 250048, RoPE theta 20000 + fixed-NTK factor 8).

## What the port replicates (read from the checkpoint's `modeling.py`)

* embeddings = word + token_type LayerNorm'd; **`type_vocab_size==1` makes `modeling.py`
  force-zero `token_type_ids` in place**, so row 0 of `token_type_embeddings` is added to every
  position; no absolute position embeddings.
* RoPE, half-rotation convention (`rotate_half`), cos/sin computed in fp32 then cast.
  **NTK subtlety:** `NTKScalingRotaryEmbedding.__init__` rebuilds the cache at
  `max_position_embeddings*factor (65536) > max_position_embeddings (8192)`, so ALL positions
  (short sequences included) use `inv_freq = (theta*factor)^-(2i/d) / factor^(2/d)`.
* post-LN blocks: fused `qkv_proj` GEMM -> split thirds -> rope(q,k) -> SDPA (1/sqrt(64)) ->
  `o_proj`; `attn_ln(x+attn)`; gated MLP `up_gate_proj` (bias-free, **first half = up, second
  half = gate**), `down_proj(gelu(gate)*up)`; `mlp_ln(x+mlp)`.
* head: `tanh(pooler.dense(h[:,0]))` -> `classifier` (classifier keys are un-prefixed; the rest
  carry the `new.` base-model prefix). Dropouts are eval no-ops; `logn_attention_scale` is a
  dead TODO in `modeling.py` and is ignored identically.

## Verified parity (CPU float32, N=300 public_test rows, 2026-08-24)

```
[parity-gte] max|Δ| 0.000e+00  mean|Δ| 0.000e+00  argmax agreement 1.0000  (tol 0.02)
PARITY OK  model=.../ckpt_r1_oldstore  n=300  max_abs_diff=0.000e+00  mean_abs_diff=0.000e+00  argmax_agreement=1.0000
```
Bit-exact vs `transformers` 4.57.6 (sdpa attention path). Repro:
`CUDA_VISIBLE_DEVICES="" .venv-gcp/bin/python verify_parity_gte.py`
(`DEBUG_LAYERS=1` prints per-layer max|Δhidden| if it ever regresses.)

## End-to-end CPU runs, full public_test.csv (7500 rows, fp32, GPU busy elsewhere)

Prediction CSVs are **byte-identical** (`cmp` clean, sha256
`4f26f4176f5f4f92298fda58acffde668f144a1a930d841d282ed344ce9a5919` for both
`outputs/pred_hf_cpu.csv` and `outputs/pred_gte_cpu.csv`).

```
serve_hf  STAGE imports=0.161 read=0.203 tokenize=1.095 model_ready=6.369 first_batch=9.900 infer=419.691 write=0.009 total=426.096 rows=7500
serve_gte STAGE imports=0.121 read=0.224 tokenize=1.058 model_ready=3.438 first_batch=7.013 infer=537.349 write=0.014 total=540.801 rows=7500
```

* **Startup (the scored fixed cost): model_ready 6.37s -> 3.44s (−2.93s), first batch
  9.90s -> 7.01s (−2.89s)** on CPU. serve_hf's build thread pays torch 3.41s + transformers
  import/checkpoint init 2.77s; serve_gte pays torch 3.28s + 0.01s weight map (safetensors
  mmap). Cold module import alone: `lean_gte`+`lean_infer` 0.107s vs `import transformers`
  2.705s (transformers measured 4.30s in the parity run when its torch-dependent chain loads).
* CPU inference throughput was lower for the lean loop in this run (14 vs 18 rows/s; identical
  math verified above — suspicion: SDPA bool-mask kernel choice on CPU vs HF's additive float
  bias, worth ignoring). **CPU rows/s is diagnostic only; the scoring config is GPU bf16 —
  bench before relying on the throughput side.**

## GPU bench (run LATER, when the GPU is free — never during training)

```
cd /home/jupyter/trendyol-teknofest-work/asya/s2_exp/e12_lean_infer/scripts
BENCH_SCRIPT=serve_gte.py /home/jupyter/trendyol-teknofest-work/.venv-gcp/bin/python bench_e2e.py \
  --model-dir=/home/jupyter/trendyol-teknofest-work/asya/s2_exp/e15_student/outputs_frozen_v2/ckpt_r1_oldstore --runs=2
```
Compare against `BENCH_SCRIPT=serve_hf.py` with the same flags; the score is
90000/total_runtime, so the −2.9s fixed startup is the expected win and GPU bf16 rows/s the
thing to confirm.
