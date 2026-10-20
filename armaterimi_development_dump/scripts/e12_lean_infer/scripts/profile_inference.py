#!/usr/bin/env python
"""Where does the time actually go inside inference? Kernel-level, on the real 90k batches.

SPS is one number over a pipeline with ~8 distinct stages; this opens the inference stage itself.
Uses torch.profiler over the SAME length-bucketed batches the container runs, so the shapes and
the batch count are exactly what serving sees.
"""
import os, sys, time, argparse, collections
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", required=True); ap.add_argument("--csv", required=True)
ap.add_argument("--label", default=""); ap.add_argument("--batch-rows", type=int, default=256)
ap.add_argument("--token-budget", type=int, default=65536)
a = ap.parse_args()

import lean_infer as L
L.preload_tokenizer(a.ckpt, 256); pool = L.make_pool()
t0 = time.perf_counter(); row_ids, encs, dp = L.run_datapath_wtok(a.csv, a.ckpt, 256, pool); pool.terminate()
t_data = time.perf_counter() - t0
L._ensure_torch(); torch = L.torch
lens = np.array([len(e.ids) for e in encs])
batches = L.make_batches(lens, a.batch_rows, a.token_budget)
cfg = __import__("json").load(open(os.path.join(a.ckpt, "config.json")))
t = time.perf_counter(); m = L.LeanBert(a.ckpt, "cuda", torch.bfloat16); t_load = time.perf_counter() - t

print(f"\n=== {a.label or a.ckpt}")
print(f"    {cfg['num_hidden_layers']}L x {cfg['hidden_size']}H  vocab {cfg['vocab_size']:,}  "
      f"ff {cfg['intermediate_size']}  heads {cfg['num_attention_heads']}")
print(f"    tokens: {lens.sum():,}  mean {lens.mean():.1f}  batches {len(batches)}")
print(f"    data path {t_data:.2f}s   weight load {t_load:.2f}s")

for _ in range(2): L.infer(m, encs[:4096], a.batch_rows, a.token_budget, True)
torch.cuda.synchronize(); t = time.perf_counter()
lg = L.infer(m, encs, a.batch_rows, a.token_budget, True)
torch.cuda.synchronize(); t_inf = time.perf_counter() - t
print(f"    INFERENCE {t_inf:.3f}s  ({len(encs)/t_inf:,.0f} rows/s, {lens.sum()/t_inf/1e6:.2f} Mtok/s)")

from torch.profiler import profile, ProfilerActivity
sub = batches[:40]
idx = np.concatenate(sub)
enc_sub = [encs[i] for i in range(len(encs))]
with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=False) as prof:
    L.infer(m, enc_sub, a.batch_rows, a.token_budget, True)
    torch.cuda.synchronize()
ev = [e for e in prof.key_averages() if e.self_device_time_total > 0]
tot = sum(e.self_device_time_total for e in ev)
BUCK = [("GEMM / matmul", ("gemm","matmul","addmm","bmm","cutlass","sgemm","ampere","nn_")),
        ("attention (SDPA)", ("attention","flash","efficient","softmax")),
        ("layernorm", ("layer_norm","layernorm")),
        ("elementwise / activation", ("elementwise","gelu","tanh","add","mul","copy_","vectorized")),
        ("embedding gather", ("embedding","index_select","gather")),
        ("H2D / D2H copy", ("memcpy","copyd", "cudamemcpy")),
        ("reduce / other", ())]
agg = collections.OrderedDict((k, 0.0) for k, _ in BUCK)
for e in ev:
    n = e.key.lower()
    for k, pats in BUCK:
        if pats and any(p in n for p in pats): agg[k] += e.self_device_time_total; break
    else: agg["reduce / other"] += e.self_device_time_total
print(f"\n    GPU kernel time on a {len(sub)}-batch sample (total {tot/1e3:.1f} ms):")
for k, v in sorted(agg.items(), key=lambda x: -x[1]):
    if v > 0: print(f"      {k:<26} {v/1e3:8.1f} ms   {100*v/tot:5.1f}%")
print(f"\n    top kernels:")
for e in sorted(ev, key=lambda e: -e.self_device_time_total)[:6]:
    print(f"      {e.key[:56]:<58}{e.self_device_time_total/1e3:7.1f} ms  {100*e.self_device_time_total/tot:5.1f}%")
