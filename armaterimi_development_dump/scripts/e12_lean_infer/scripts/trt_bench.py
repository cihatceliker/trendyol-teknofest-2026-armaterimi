#!/usr/bin/env python
"""Benchmark a TensorRT engine against our lean bf16 path, on the real 90k, same batching.

Compares wall time AND labels: TRT fp16 is different arithmetic from torch bf16, and our decision
rule slices a scalar with razor-thin per-regime cuts, so a speed win that moves labels is not free
-- it would need the cuts refitted and another DEV read.
"""
import os, sys, time, argparse
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", required=True); ap.add_argument("--plan", required=True)
ap.add_argument("--csv", required=True); ap.add_argument("--batch-rows", type=int, default=256)
ap.add_argument("--token-budget", type=int, default=65536)
a = ap.parse_args()

import lean_infer as L
L.preload_tokenizer(a.ckpt, 256); pool = L.make_pool()
t0 = time.perf_counter()
row_ids, encs, dp = L.run_datapath_wtok(a.csv, a.ckpt, 256, pool); pool.terminate()
print(f"data path: {len(row_ids):,} rows in {time.perf_counter()-t0:.2f}s")

L._ensure_torch(); torch = L.torch
lens = np.array([len(e.ids) for e in encs])
batches = L.make_batches(lens, a.batch_rows, a.token_budget)
print(f"batches: {len(batches)}")

# ---------- baseline: our shipped lean bf16 path ----------
m = L.LeanBert(a.ckpt, "cuda", torch.bfloat16)
for _ in range(2): L.infer(m, encs[:2048], a.batch_rows, a.token_budget, True)
torch.cuda.synchronize(); t = time.perf_counter()
lg_torch = L.infer(m, encs, a.batch_rows, a.token_budget, True)
torch.cuda.synchronize(); t_torch = time.perf_counter() - t
print(f"torch bf16 (shipped) : {t_torch:7.3f}s   {len(row_ids)/t_torch:8.0f} rows/s")
del m; torch.cuda.empty_cache()

# ---------- TensorRT ----------
import tensorrt as trt
LOG = trt.Logger(trt.Logger.ERROR)
rt = trt.Runtime(LOG)
eng = rt.deserialize_cuda_engine(open(a.plan, "rb").read())
ctx = eng.create_execution_context()
names = [eng.get_tensor_name(i) for i in range(eng.num_io_tensors)]
inp = [n for n in names if eng.get_tensor_mode(n) == trt.TensorIOMode.INPUT]
out = [n for n in names if eng.get_tensor_mode(n) == trt.TensorIOMode.OUTPUT]
stream = torch.cuda.Stream()

def run_trt(idx_list):
    res = np.empty((len(encs), 3), dtype=np.float32)
    with torch.cuda.stream(stream):
        for idx in idx_list:
            Lmax = int(max(len(encs[r].ids) for r in idx)); B = len(idx)
            ids = np.zeros((B, Lmax), np.int64); tt = np.zeros_like(ids); mk = np.zeros_like(ids)
            for b, r in enumerate(idx):
                e = encs[r]; n = len(e.ids)
                ids[b,:n] = e.ids; tt[b,:n] = e.type_ids; mk[b,:n] = 1
            d = {"input_ids": torch.from_numpy(ids).cuda(),
                 "attention_mask": torch.from_numpy(mk).cuda(),
                 "token_type_ids": torch.from_numpy(tt).cuda()}
            for n_ in inp:
                ctx.set_input_shape(n_, tuple(d[n_].shape)); ctx.set_tensor_address(n_, d[n_].data_ptr())
            o = torch.empty((B, 3), dtype=torch.float16, device="cuda")
            ctx.set_tensor_address(out[0], o.data_ptr())
            ctx.execute_async_v3(stream.cuda_stream)
            stream.synchronize()
            res[idx] = o.float().cpu().numpy()
    return res

run_trt(batches[:8])
torch.cuda.synchronize(); t = time.perf_counter()
lg_trt = run_trt(batches)
torch.cuda.synchronize(); t_trt = time.perf_counter() - t
print(f"TensorRT fp16        : {t_trt:7.3f}s   {len(row_ids)/t_trt:8.0f} rows/s   speedup {t_torch/t_trt:.2f}x")

# ---------- do the LABELS move? ----------
def decide(lg):
    x = lg - lg.max(1, keepdims=True); p = np.exp(x); p /= p.sum(1, keepdims=True)
    return p[:,1] + 2*p[:,2]
s1, s2 = decide(lg_torch.astype(np.float64)), decide(lg_trt.astype(np.float64))
c1, c2 = 0.9784192406370285, 1.0837608043183675
l1 = (s1>=c1).astype(int)+(s1>=c2); l2 = (s2>=c1).astype(int)+(s2>=c2)
print(f"\nmax |Δlogit| {np.abs(lg_torch-lg_trt).max():.4f}   max |Δscalar| {np.abs(s1-s2).max():.4f}")
print(f"labels differing (global cuts): {int((l1!=l2).sum())} / {len(l1):,} = {100*(l1!=l2).mean():.3f}%")
