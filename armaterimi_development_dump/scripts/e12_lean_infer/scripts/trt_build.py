#!/usr/bin/env python
"""Export a plain-BERT checkpoint to ONNX and build a TensorRT engine.

Why this exists: our hand-rolled int8 died at 0.68x because the quantise/rescale could not be
fused — 43 MB of activation reads and a 259 MB int32 accumulator around a 0.4 ms GEMM, even under
torch.compile. TensorRT fuses exactly that into the GEMM epilogue, and additionally ships fused
multi-head attention and embedding+LayerNorm for BERT. So TRT can plausibly realise the 1.32x raw
GEMM win that inductor could not. This tests it on OUR checkpoint at OUR shapes.

Usage: trt_build.py <ckpt_dir> <out_dir> [--fp16|--fp32] [--maxlen 256] [--maxbatch 256]
"""
import os, sys, argparse, time
ap = argparse.ArgumentParser()
ap.add_argument("ckpt"); ap.add_argument("out")
ap.add_argument("--fp16", action="store_true", default=True)
ap.add_argument("--maxlen", type=int, default=256)
ap.add_argument("--maxbatch", type=int, default=256)
ap.add_argument("--optlen", type=int, default=96)      # our median token length is 75, p90 129
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)
onnx_path = os.path.join(a.out, "model.onnx")
plan_path = os.path.join(a.out, "model.plan")

import torch
from transformers import AutoModelForSequenceClassification

if not os.path.exists(onnx_path):
    print(f"[1/2] exporting {a.ckpt} -> {onnx_path}")
    # TensorRT 11 is STRONGLY TYPED: BuilderFlag.FP16/INT8 no longer exist and precision comes
    # from the ONNX graph's own dtypes. So the half-precision decision is made HERE, at export.
    dt = torch.float16 if a.fp16 else torch.float32
    dev = "cuda" if (a.fp16 and torch.cuda.is_available()) else "cpu"
    m = AutoModelForSequenceClassification.from_pretrained(a.ckpt, dtype=dt).to(dev).eval()
    B, L = 2, 16
    dummy = tuple(torch.ones(B, L, dtype=torch.long, device=dev) for _ in range(2)) + \
            (torch.zeros(B, L, dtype=torch.long, device=dev),)
    t = time.time()
    torch.onnx.export(
        m, dummy, onnx_path,
        input_names=["input_ids", "attention_mask", "token_type_ids"],
        output_names=["logits"],
        dynamic_axes={k: {0: "batch", 1: "seq"} for k in ("input_ids", "attention_mask", "token_type_ids")}
                     | {"logits": {0: "batch"}},
        opset_version=17, do_constant_folding=True,
    )
    print(f"      exported in {time.time()-t:.1f}s, {os.path.getsize(onnx_path)/1e6:.0f} MB")
else:
    print(f"[1/2] reusing {onnx_path}")

import tensorrt as trt
LOG = trt.Logger(trt.Logger.WARNING)
print(f"[2/2] building TRT engine (fp16={a.fp16}, maxbatch={a.maxbatch}, maxlen={a.maxlen})")
builder = trt.Builder(LOG)
network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED))
parser = trt.OnnxParser(network, LOG)
# parse_from_file, NOT parse(bytes): torch.onnx writes weights as EXTERNAL DATA next to the
# .onnx file (model.onnx itself came out 0 MB), and only the file-based parser resolves those
# relative paths. parse(bytes) fails on the first initializer it cannot find.
if not parser.parse_from_file(onnx_path):
    for i in range(parser.num_errors): print("   ONNX parse error:", parser.get_error(i))
    sys.exit(2)
cfg = builder.create_builder_config()
try: cfg.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 8 << 30)
except Exception: pass
# no FP16 flag in TRT 11 (strongly typed); the ONNX graph already carries fp16 weights.
cfg.set_flag(trt.BuilderFlag.TF32)
prof = builder.create_optimization_profile()
for nm in ("input_ids", "attention_mask", "token_type_ids"):
    prof.set_shape(nm, (1, 8), (a.maxbatch, a.optlen), (a.maxbatch, a.maxlen))
cfg.add_optimization_profile(prof)
t = time.time()
plan = builder.build_serialized_network(network, cfg)
if plan is None: sys.exit("engine build FAILED")
open(plan_path, "wb").write(plan)
print(f"      built in {time.time()-t:.0f}s -> {plan_path} ({os.path.getsize(plan_path)/1e6:.0f} MB)")
