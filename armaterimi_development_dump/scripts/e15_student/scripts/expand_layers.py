#!/usr/bin/env python
"""Grow a pretrained BERT encoder by N transformer layers, then hand it to the normal recipe.

tiny2 (turkish-medium, 8L/512) was the only tiny model to clear the 0.80 floor (0.8218) and it
sits 0.0155 under BERTurk. The question this answers: does 10L/512 close that gap while keeping
most of the speed? There is no pretrained 10L/512 Turkish BERT, so we build one.

Init modes (BERT is POST-LayerNorm, so NONE of these is exactly function-preserving — the new
block's LayerNorm re-normalises even when its projections are zero; --probe measures the actual
drift instead of assuming):
  dup   : new layers are copies of the last N pretrained layers (progressive stacking, Gong et
          al. 2019 — the established BERT-growing method; the copied layers are meaningful
          transforms, not noise)
  zero  : dup, then zero attention.output.dense and output.dense so each new block starts as
          LayerNorm(x) — closest to identity that post-LN allows; gradients still flow (the
          weight gradient is input (x) upstream-grad, LoRA-style)
  rand  : new layers freshly initialised from the config (the naive option; measured for
          reference so the choice is evidence-based rather than a preference)

Usage:
  expand_layers.py --src ytu-ce-cosmos/turkish-medium-bert-uncased --add 2 --mode dup \
                   --out /path/to/turkish-medium-10L [--probe]
  --probe scores all three modes on real query-product text and prints representation drift
  vs the unmodified encoder; it writes nothing.
"""
import argparse, copy, os, sys, json

ap = argparse.ArgumentParser()
ap.add_argument("--src", default="ytu-ce-cosmos/turkish-medium-bert-uncased")
ap.add_argument("--add", type=int, default=2)
ap.add_argument("--mode", default="dup", choices=["dup", "zero", "rand"])
ap.add_argument("--out", default="")
ap.add_argument("--probe", action="store_true")
a = ap.parse_args()

import torch
from transformers import AutoTokenizer, AutoModel, AutoConfig

def grow(src, add, mode):
    """Return a model with `add` extra layers appended, built from `src`."""
    base = AutoModel.from_pretrained(src)
    cfg = AutoConfig.from_pretrained(src)
    n0 = cfg.num_hidden_layers
    cfg.num_hidden_layers = n0 + add
    new = AutoModel.from_config(cfg)                 # fresh weights at the new depth
    sd_new = new.state_dict()
    sd_old = base.state_dict()
    keep = {}
    for k, v in sd_old.items():
        if k in sd_new and sd_new[k].shape == v.shape:
            keep[k] = v.clone()
    if mode in ("dup", "zero"):
        # new layer n0+i  <-  pretrained layer (n0-add+i): the top `add` layers, in order
        for i in range(add):
            src_i, dst_i = n0 - add + i, n0 + i
            for k, v in sd_old.items():
                tag = f"encoder.layer.{src_i}."
                if k.startswith(tag):
                    keep[f"encoder.layer.{dst_i}." + k[len(tag):]] = v.clone()
            if mode == "zero":
                for p in ("attention.output.dense.weight", "attention.output.dense.bias",
                          "output.dense.weight", "output.dense.bias"):
                    keep[f"encoder.layer.{dst_i}.{p}"] = torch.zeros_like(
                        keep[f"encoder.layer.{dst_i}.{p}"])
    missing, unexpected = new.load_state_dict(keep, strict=False)
    return base, new, cfg, [m for m in missing if "position_ids" not in m]

TEXTS = [("pejo 307", "peugeot 307 kapı eşiği krom 4 parça | otomobil & motosiklet"),
         ("nar ağacı", "linaline dekoratif yapay nar ağacı ev ofis masa raf | ev & mobilya"),
         ("penti plaj çantası", "koyu yeşil paige hipster bikini altı | giyim"),
         ("kablosuz kulaklık", "bluetooth 5.3 kulak içi kulaklık şarj kutulu | elektronik"),
         ("bebek arabası", "travel sistem bebek arabası puset | anne & bebek & çocuk")]

tok = AutoTokenizer.from_pretrained(a.src)
enc = tok([t[0] for t in TEXTS], [t[1] for t in TEXTS],
          padding=True, truncation=True, max_length=256, return_tensors="pt")

if a.probe:
    print(f"src={a.src}  add={a.add}\n")
    base = AutoModel.from_pretrained(a.src).eval()
    with torch.no_grad():
        ref = base(**enc).last_hidden_state
    refm = (ref * enc["attention_mask"].unsqueeze(-1)).sum(1) / enc["attention_mask"].sum(1, keepdim=True)
    print(f"{'mode':6s} {'cos(mean-pool) vs 8L':>22s} {'rel L2 drift':>14s}   new-layer params")
    for mode in ("dup", "zero", "rand"):
        _b, new, cfg, missing = grow(a.src, a.add, mode)
        new.eval()
        with torch.no_grad():
            out = new(**enc).last_hidden_state
        m = (out * enc["attention_mask"].unsqueeze(-1)).sum(1) / enc["attention_mask"].sum(1, keepdim=True)
        cos = torch.nn.functional.cosine_similarity(m, refm, dim=-1).mean().item()
        rel = ((m - refm).norm(dim=-1) / refm.norm(dim=-1)).mean().item()
        n_new = sum(p.numel() for n, p in new.named_parameters()
                    if any(f"encoder.layer.{cfg.num_hidden_layers - 1 - j}." in n for j in range(a.add)))
        print(f"{mode:6s} {cos:22.4f} {rel:14.4f}   {n_new/1e6:.1f}M"
              + ("   <-- untrained-from-config layers" if mode == "rand" else ""))
    print("\ncos 1.0 / drift 0.0 would be exact preservation. BERT is post-LN, so the new block's\n"
          "LayerNorm re-normalises even with zeroed projections — no mode reaches it. Higher cos\n"
          "means the fine-tune starts nearer the pretrained function.")
    sys.exit(0)

if not a.out:
    sys.exit("--out is required unless --probe")
base, new, cfg, missing = grow(a.src, a.add, a.mode)
os.makedirs(a.out, exist_ok=True)
new.save_pretrained(a.out)
tok.save_pretrained(a.out)
json.dump({"src": a.src, "added_layers": a.add, "mode": a.mode,
           "layers": cfg.num_hidden_layers, "hidden": cfg.hidden_size,
           "params_M": round(sum(p.numel() for p in new.parameters()) / 1e6, 1),
           "randomly_initialised_tensors": missing},
          open(os.path.join(a.out, "EXPANSION.json"), "w"), indent=1)
print(f"wrote {a.out}  {cfg.num_hidden_layers}L x {cfg.hidden_size}H  "
      f"{sum(p.numel() for p in new.parameters())/1e6:.1f}M params  mode={a.mode}")
if missing:
    print(f"  NOTE {len(missing)} tensors left at fresh init: {missing[:4]}{' ...' if len(missing) > 4 else ''}")
