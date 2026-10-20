"""INT8 probe on the HF module (torchao needs an nn.Module; lean_gte holds raw tensors).

Purpose: decide whether INT8 is worth transplanting into lean_gte via torch._int_mm. Measures
throughput AND label drift on real rows for three modes, using the identical batching the serving
path uses. If int8dyn is not clearly faster here, it will not be faster in the lean loop either.
Env: MODE = bf16 | int8dyn | int8wo ; N (rows) ; BATCH_ROWS ; TOKEN_BUDGET
"""
import os, sys, time, importlib.util
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
spec = importlib.util.spec_from_file_location("lean", os.path.join(HERE, "lean_infer.py"))
L = importlib.util.module_from_spec(spec); _a = sys.argv; sys.argv = ["x"]; spec.loader.exec_module(L); sys.argv = _a
import torch
from transformers import AutoModelForSequenceClassification
from tokenizers import Tokenizer

W = "/home/jupyter/trendyol-teknofest-work"
CK = f"{W}/asya/s2_exp/e15_student/outputs_frozen_v2/ckpt_r1_oldstore"
SC = "/var/tmp/claude-1000/-home-jupyter/483abbaf-6d7f-4b27-838c-53d3d34a821c/scratchpad"
MODE = os.environ.get("MODE", "bf16"); N = int(os.environ.get("N", "20000"))
BR = int(os.environ.get("BATCH_ROWS", "256")); TB = int(os.environ.get("TOKEN_BUDGET", "65536"))

ids, pairs = L.read_pairs(f"{W}/stage2_data/public_test.csv")
pairs = (pairs * 3)[:N]
tok = Tokenizer.from_file(f"{CK}/tokenizer.json"); tok.enable_truncation(max_length=256, strategy="longest_first")
encs = tok.encode_batch(pairs)
m = AutoModelForSequenceClassification.from_pretrained(CK, trust_remote_code=True, dtype=torch.bfloat16).to("cuda").eval()
torch.set_grad_enabled(False)
if MODE != "bf16":
    from torchao.quantization import quantize_, int8_dynamic_activation_int8_weight, int8_weight_only
    cfg = int8_dynamic_activation_int8_weight() if MODE == "int8dyn" else int8_weight_only()
    n_q = [0]
    def filt(mod, fqn):
        keep = isinstance(mod, torch.nn.Linear) and any(k in fqn for k in ("qkv_proj", "o_proj", "up_gate_proj", "down_proj"))
        n_q[0] += int(keep); return keep
    quantize_(m, cfg, filter_fn=filt)
    print(f"quantised {n_q[0]} linears with {MODE}")

lens = np.array([len(e.ids) for e in encs]); batches = L.make_batches(lens, BR, TB)
def run():
    out = np.empty((len(encs), 3), np.float32)
    t = time.perf_counter()
    for idx in batches:
        i_, tt, mk = L._assemble(encs, idx, True)
        o = m(input_ids=i_.to("cuda", non_blocking=True), attention_mask=mk.to("cuda", non_blocking=True),
              token_type_ids=tt.to("cuda", non_blocking=True)).logits
        out[idx] = o.float().cpu().numpy()
    torch.cuda.synchronize(); return out, time.perf_counter() - t
run(); lg, dt = run()
print(f"MODE={MODE}  {len(encs):,} rows in {dt:.2f}s = {len(encs)/dt:,.0f} rows/s  -> 90k infer ~{90000/(len(encs)/dt):.1f}s")
np.save(f"{SC}/i8hf_{MODE}.npy", lg)
ref = f"{SC}/i8hf_bf16.npy"
if MODE != "bf16" and os.path.exists(ref):
    r = np.load(ref); LABS = np.array(["irrelevant", "semi-relevant", "relevant"])
    sr = r[:, 1] + 2 * r[:, 2]; sq = lg[:, 1] + 2 * lg[:, 2]
    print(f"  max|dlogit| {np.abs(r-lg).max():.4f} | scalar corr {np.corrcoef(sr,sq)[0,1]:.6f} | "
          f"argmax agree {(r.argmax(1)==lg.argmax(1)).mean():.3%}")
