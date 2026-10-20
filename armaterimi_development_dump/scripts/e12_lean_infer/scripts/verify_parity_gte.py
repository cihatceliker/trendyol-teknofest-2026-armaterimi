"""GTE parity harness — the ONE gte script allowed to import transformers.

Loads the SAME NewForSequenceClassification checkpoint through transformers
(trust_remote_code=True, local modeling.py) and through lean_gte, runs both on N rows of
public_test.csv preprocessed EXACTLY like serve_hf/serve_gte (lean_infer.read_pairs + the Rust
tokenizer with the same truncation, lean_infer.make_batches/_assemble batching), CPU float32,
and asserts max|Δlogit| < TOL and 100% argmax agreement.

Env: MODEL_DIR (default e15_student/outputs_frozen_v2/ckpt_r1_oldstore), N (300), BS (32),
     TOL (0.02), MAXLEN (256), DEBUG_LAYERS=1 -> per-layer max|Δ| of hidden states (localises
     RoPE-convention / key-mapping bugs).  CPU float32 on purpose: the GPU stays free and
     float32 makes the comparison tight; the bf16 GPU path shares the exact same graph.
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""          # GPU is busy; parity is a CPU exercise
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
import sys, time
import numpy as np, torch

W = "/home/jupyter/trendyol-teknofest-work"
E = f"{W}/asya/s2_exp/e12_lean_infer"
sys.path.insert(0, f"{E}/scripts")
import lean_infer as LI
import lean_gte as LG

MDIR   = os.environ.get("MODEL_DIR", f"{W}/asya/s2_exp/e15_student/outputs_frozen_v2/ckpt_r1_oldstore")
N      = int(os.environ.get("N", 300)); BS = int(os.environ.get("BS", 32))
TOL    = float(os.environ.get("TOL", 0.02)); MAXLEN = int(os.environ.get("MAXLEN", 256))
DEBUG  = os.environ.get("DEBUG_LAYERS")

def log(m): print(f"[parity-gte] {m}", flush=True)

# ---- shared input: the serving data path (reader + normaliser + pair template + tokenizer)
row_ids, pairs = LI.read_pairs(f"{W}/stage2_data/public_test.csv")
row_ids, pairs = row_ids[:N], pairs[:N]
tok = LI.Tokenizer.from_file(os.path.join(MDIR, "tokenizer.json"))
tok.enable_truncation(max_length=MAXLEN, strategy="longest_first")   # == lean_infer.run_datapath
encs = tok.encode_batch(pairs)
log(f"{len(encs)} rows tokenized  mean len {np.mean([len(e.ids) for e in encs]):.1f}  "
    f"max len {max(len(e.ids) for e in encs)}")

# ---- lean path (float32 CPU) through the SAME bucketed loop serve_gte uses
t = time.perf_counter()
lean = LG.load_gte(MDIR, device="cpu", dtype=torch.float32)
log(f"lean_gte loaded {time.perf_counter()-t:.2f}s  ({lean.NL}L x {lean.H}H, {lean.num_labels} labels, "
    f"pooler={lean.pool[0] is not None})")
t = time.perf_counter()
lean_logits = LI.infer(lean, encs, batch_rows=BS, overlap=False)
log(f"lean forward {N} rows {time.perf_counter()-t:.2f}s")

# ---- HF path (float32 CPU): identical tensors via the same make_batches/_assemble
t0 = time.perf_counter()
from transformers import AutoModelForSequenceClassification
log(f"import transformers chain {time.perf_counter()-t0:.2f}s")
t = time.perf_counter()
hf = AutoModelForSequenceClassification.from_pretrained(
    MDIR, trust_remote_code=True, dtype=torch.float32).eval()
log(f"HF load {time.perf_counter()-t:.2f}s  ({type(hf).__name__}, "
    f"attn={hf.config._attn_implementation})")

lens = np.array([len(e.ids) for e in encs])
batches = LI.make_batches(lens, BS, 1 << 30)         # rows-capped only: mirror BS batching
hf_logits = np.empty((N, lean.num_labels), dtype=np.float32)
t = time.perf_counter()
with torch.no_grad():
    for idx in batches:
        ids, tt, mk = LI._assemble(encs, idx, pin=False)
        # serve_hf passes token_type_ids (forward accepts it); clone: modeling.py mul_(0)s it
        out = hf(input_ids=ids, attention_mask=mk, token_type_ids=tt.clone())
        hf_logits[idx] = out.logits.float().numpy()
log(f"HF forward {N} rows {time.perf_counter()-t:.2f}s")

# ---- compare
d = np.abs(lean_logits - hf_logits)
agree = float((lean_logits.argmax(1) == hf_logits.argmax(1)).mean())
log(f"max|Δ| {d.max():.3e}  mean|Δ| {d.mean():.3e}  argmax agreement {agree:.4f}  (tol {TOL})")

if DEBUG or d.max() >= TOL or agree < 1.0:
    # per-layer hidden-state diff on one batch to localise the divergence
    idx = batches[0]
    ids, tt, mk = LI._assemble(encs, idx, pin=False)
    lean_states = []
    lean.forward(ids, tt, mk, collect=lean_states)
    hf_states = []
    h1 = hf.new.embeddings.register_forward_hook(lambda m, i, o: hf_states.append(o[0]))
    hooks = [h1] + [l.register_forward_hook(lambda m, i, o: hf_states.append(o[0]))
                    for l in hf.new.encoder.layer]
    with torch.no_grad():
        hf(input_ids=ids, attention_mask=mk, token_type_ids=tt.clone())
    for h in hooks: h.remove()
    for j, (a, b) in enumerate(zip(lean_states, hf_states)):
        name = "embeddings" if j == 0 else f"layer {j-1:2d}"
        log(f"  {name}: max|Δhidden| {(a - b).abs().max().item():.3e}")

assert agree == 1.0, f"PARITY FAIL: argmax agreement {agree:.4f} < 1.0"
assert d.max() < TOL, f"PARITY FAIL: max|Δ|={d.max():.4f} >= {TOL}"
print(f"PARITY OK  model={MDIR}  n={N}  max_abs_diff={d.max():.3e}  "
      f"mean_abs_diff={d.mean():.3e}  argmax_agreement={agree:.4f}")
