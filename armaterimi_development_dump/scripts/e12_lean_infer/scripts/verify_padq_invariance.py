#!/usr/bin/env python
"""Is the forward pass GROUPING-INVARIANT under quantised padding?

STREAM=1 was reverted because splitting the input into groups changed batch composition, which
changed each batch's padded width, which changed bf16 arithmetic, which flipped ~0.1% of labels on
rows sitting inside the razor-thin cut bands. That cost us the overlap between the data path and
the GPU -- ~5.4s of every 90k run.

LEAN_PAD_Q rounds each batch's width up to a fixed multiple, and forbids a batch from spanning two
buckets, so the padded width is a function of the ROW's own length. If that reasoning is right,
the logits from one big pass and from an arbitrarily grouped pass are BIT-identical -- not close,
identical -- and streaming becomes free. This asserts exactly that, and shows what pad_q=0 does
for contrast.

Usage: verify_padq_invariance.py [--model-dir DIR] [--rows N] [--groups 7]
"""
import sys, os, argparse
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

ap = argparse.ArgumentParser()
ap.add_argument("--model-dir", default="/home/jupyter/trendyol-teknofest-work/asya/s2_exp/"
                                       "e15_student/outputs_frozen_v2/ckpt_v1_berturk_full")
ap.add_argument("--csv", default="/home/jupyter/trendyol-teknofest-work/stage2_data/public_test.csv")
ap.add_argument("--groups", type=int, default=7)
a = ap.parse_args()

import lean_infer as L
L.preload_tokenizer(a.model_dir, 256)
pool = L.make_pool()
row_ids, encs, dp = L.run_datapath_wtok(a.csv, a.model_dir, 256, pool)
pool.terminate()
print(f"{len(row_ids):,} rows tokenised in {dp['total']:.2f}s")

L._ensure_torch(); torch = L.torch
dev = "cuda" if torch.cuda.is_available() else "cpu"
model = L.LeanBert(a.model_dir, dev, torch.bfloat16 if dev == "cuda" else torch.float32)
print(f"model on {dev}")

def run_grouped(pad_q, n_groups):
    """Same rows, but fed as n_groups separate infer() calls -- exactly what STREAM=1 does."""
    out = np.empty((len(encs), model.num_labels), dtype=np.float32)
    step = -(-len(encs) // n_groups)
    for g0 in range(0, len(encs), step):
        sl = slice(g0, min(g0 + step, len(encs)))
        out[sl] = L.infer(model, encs[sl], 256, 65536, True, pad_q=pad_q)
    return out

for pad_q in (0, 8):
    whole = L.infer(model, encs, 256, 65536, True, pad_q=pad_q)
    part  = run_grouped(pad_q, a.groups)
    dmax  = float(np.abs(whole - part).max())
    lw, lp = whole.argmax(1), part.argmax(1)
    # the decision is the ordinal scalar against cuts, so also count scalar movement
    def scal(x):
        e = np.exp(x - x.max(1, keepdims=True)); p = e / e.sum(1, keepdims=True)
        return p[:, 1] + 2 * p[:, 2]
    ds = float(np.abs(scal(whole) - scal(part)).max())
    tag = "LEGACY pad-to-batch-max" if pad_q == 0 else f"QUANTISED pad_q={pad_q}"
    print(f"\n{tag}: 1 pass vs {a.groups} groups")
    print(f"   max |Δlogit|  {dmax:.3e}")
    print(f"   max |Δscalar| {ds:.3e}")
    print(f"   argmax disagreements {int((lw != lp).sum())}/{len(lw)}")
    print(f"   -> {'BIT-IDENTICAL — streaming is safe' if dmax == 0.0 else 'DIFFERS — streaming would change predictions'}")
