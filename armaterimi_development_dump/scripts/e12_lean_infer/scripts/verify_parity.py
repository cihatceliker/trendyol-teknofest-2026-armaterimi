"""E0a parity harness — the ONE script here allowed to import transformers.

Loads an HF BertForSequenceClassification checkpoint (random 3-way head is fine: parity is about
the GRAPH, not the task), exports it to the lean format (model.safetensors + config.json +
tokenizer.json + decision.json), then runs the HF path and the lean path on N rows of
public_test.csv and asserts max|Δlogit| < TOL. Also asserts token-id parity first (sharper
diagnostics than a logit diff). CPU float32 on purpose — the GPU stays free and float32 makes
the comparison tight; the bf16 GPU path shares the exact same graph.

Env: MODEL (default dbmdz/bert-base-turkish-cased), N (200), BS (32), TOL (0.05),
     OUT (outputs/parity_ckpt). Deterministic: torch.manual_seed(42) before head init.
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""          # GPU is busy; parity is a CPU exercise
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
import sys, json, time
import numpy as np, torch

W = "/home/jupyter/trendyol-teknofest-work"
E = f"{W}/asya/s2_exp/e12_lean_infer"
sys.path.insert(0, f"{E}/scripts")
import lean_infer as LI

MODEL = os.environ.get("MODEL", "dbmdz/bert-base-turkish-cased")
OUT   = os.environ.get("OUT", f"{E}/outputs/parity_ckpt")
N     = int(os.environ.get("N", 200)); BS = int(os.environ.get("BS", 32))
TOL   = float(os.environ.get("TOL", 0.05)); MAXLEN = 256

def log(m): print(f"[parity] {m}", flush=True)

t0 = time.perf_counter()
from transformers import AutoModelForSequenceClassification, AutoTokenizer
log(f"import transformers chain {time.perf_counter()-t0:.2f}s")

torch.manual_seed(42)
model = AutoModelForSequenceClassification.from_pretrained(MODEL, num_labels=3).eval()
tok = AutoTokenizer.from_pretrained(MODEL)

# ---- export to lean format
os.makedirs(OUT, exist_ok=True)
model.save_pretrained(OUT, safe_serialization=True)          # model.safetensors + config.json
tok.backend_tokenizer.save(f"{OUT}/tokenizer.json")          # Rust tokenizer, no transformers dep
json.dump({"head": "softmax3"}, open(f"{OUT}/decision.json", "w"))
log(f"exported lean checkpoint -> {OUT}")

# ---- shared input: lean reader + normaliser + pair template (both paths see identical text)
row_ids, pairs = LI.read_pairs(f"{W}/stage2_data/public_test.csv")
row_ids, pairs = row_ids[:N], pairs[:N]
a, b = [p[0] for p in pairs], [p[1] for p in pairs]

# ---- lean path (float32 CPU)
lean = LI.LeanBert(OUT, device="cpu", dtype=torch.float32)
ltok = LI.Tokenizer.from_file(f"{OUT}/tokenizer.json")
ltok.enable_truncation(max_length=MAXLEN, strategy="longest_first")
encs = ltok.encode_batch(pairs)
t = time.perf_counter()
lean_logits = LI.infer(lean, encs, batch_rows=BS, overlap=False)
log(f"lean forward {N} rows {time.perf_counter()-t:.2f}s")

# ---- HF path (float32 CPU), plus token-id parity
hf_ids = tok(a, b, truncation=True, max_length=MAXLEN)["input_ids"]
id_mism = sum(1 for e, h in zip(encs, hf_ids) if list(e.ids) != h)
assert id_mism == 0, f"token-id mismatch on {id_mism}/{N} rows"
log("token ids identical on all rows")
hf_logits = np.empty((N, 3), dtype=np.float32)
t = time.perf_counter()
with torch.no_grad():
    for i in range(0, N, BS):
        enc = tok(a[i:i+BS], b[i:i+BS], truncation=True, max_length=MAXLEN,
                  padding=True, return_tensors="pt")
        hf_logits[i:i+BS] = model(**enc).logits.numpy()
log(f"HF forward {N} rows {time.perf_counter()-t:.2f}s")

# ---- compare
d = np.abs(lean_logits - hf_logits)
agree = float((lean_logits.argmax(1) == hf_logits.argmax(1)).mean())
log(f"max|Δ| {d.max():.3e}  mean|Δ| {d.mean():.3e}  argmax agreement {agree:.4f}  (tol {TOL})")
assert d.max() < TOL, f"PARITY FAIL: max|Δ|={d.max():.4f} >= {TOL}"
print(f"PARITY OK  model={MODEL}  n={N}  max_abs_diff={d.max():.3e}  mean_abs_diff={d.mean():.3e}")
