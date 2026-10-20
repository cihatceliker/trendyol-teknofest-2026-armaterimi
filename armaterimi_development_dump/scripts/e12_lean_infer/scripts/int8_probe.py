"""INT8 probe: quantise the champion's linears and measure (a) speed, (b) accuracy drift.

lean_gte is pure torch, so torchao's int8 dynamic-activation/int8-weight path applies cleanly to the
transformer linears (qkv_proj, o_proj, up_gate_proj, down_proj). Embeddings, LayerNorms, the pooler
and the classifier stay bf16 — they are a rounding error of the FLOPs and the most precision-sensitive.
Reports tok/s and the max logit deviation vs the bf16 reference on real rows.
Env: MODE = bf16 | int8dyn | int8wo
"""
import os, sys, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import importlib.util
spec = importlib.util.spec_from_file_location("lean", os.path.join(os.path.dirname(os.path.abspath(__file__)), "lean_infer.py"))
L = importlib.util.module_from_spec(spec); _a = sys.argv; sys.argv = ["x"]; spec.loader.exec_module(L); sys.argv = _a
import lean_gte as G
G._ensure_torch()

W = "/home/jupyter/trendyol-teknofest-work"
CK = f"{W}/asya/s2_exp/e15_student/outputs_frozen_v2/ckpt_r1_oldstore"
SC = "/var/tmp/claude-1000/-home-jupyter/483abbaf-6d7f-4b27-838c-53d3d34a821c/scratchpad"
MODE = os.environ.get("MODE", "bf16")
N = int(os.environ.get("N", "20000"))

ids, pairs = L.read_pairs(f"{W}/stage2_data/public_test.csv")
pairs = (pairs * 3)[:N]
from tokenizers import Tokenizer
tok = Tokenizer.from_file(f"{CK}/tokenizer.json"); tok.enable_truncation(max_length=256, strategy="longest_first")
encs = tok.encode_batch(pairs)
print(f"{len(encs):,} rows, mean len {np.mean([len(e.ids) for e in encs]):.1f}")

m = G.load_gte(CK, "cuda", G.torch.bfloat16)
if MODE != "bf16":
    from torchao.quantization import quantize_, int8_dynamic_activation_int8_weight, int8_weight_only
    cfg = int8_dynamic_activation_int8_weight() if MODE == "int8dyn" else int8_weight_only()
    tgt = G.torch.nn.Linear
    quantize_(m.backbone if hasattr(m, "backbone") else m, cfg,
              filter_fn=lambda mod, fqn: isinstance(mod, tgt) and ("proj" in fqn))
    print(f"quantised linears with {MODE}")

def run():
    t = time.perf_counter()
    lg = L.infer(m, encs, int(os.environ.get("BATCH_ROWS", 256)), int(os.environ.get("TOKEN_BUDGET", 65536)), True)
    G.torch.cuda.synchronize()
    return lg, time.perf_counter() - t

run()                                   # warm-up
lg, dt = run()
ntok = sum(len(e.ids) for e in encs)
print(f"MODE={MODE}  infer {dt:.2f}s for {len(encs):,} rows = {len(encs)/dt:,.0f} rows/s "
      f"({ntok/dt/1e6:.2f}M tok/s) -> 90k in ~{90000/(len(encs)/dt):.1f}s")
np.save(f"{SC}/int8_logits_{MODE}.npy", lg)
ref = f"{SC}/int8_logits_bf16.npy"
if MODE != "bf16" and os.path.exists(ref):
    r = np.load(ref)
    d = np.abs(r - lg)
    s_r = r[:, 1] + 2 * r[:, 2] if r.shape[1] == 3 else None
    print(f"  max|dlogit| {d.max():.4f}  mean {d.mean():.5f}  argmax agreement {(r.argmax(1)==lg.argmax(1)).mean():.4%}")
