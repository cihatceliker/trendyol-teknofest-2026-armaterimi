"""E0a-GTE — lean forward for the Alibaba GTE "new-impl" architecture (NewForSequenceClassification).

Never imports `transformers`: hand-rolled forward on torch + safetensors, faithful to the
checkpoint's LOCAL modeling.py (e15_student/outputs_frozen_v2/ckpt_r1_oldstore/modeling.py,
GTE Team / Alibaba, transformers_version 4.57.6). Parity: scripts/verify_parity_gte.py.

Graph replicated EXACTLY (padded dense path: unpad_inputs=False, no xformers, sdpa attention):
  embeddings : word_embeddings(ids) [+ token_type row — NOTE: type_vocab_size==1 means
               modeling.py force-zeroes token_type_ids (`token_type_ids.mul_(0)`), so every
               position gets tte row 0 regardless of the tokenizer's type_ids] -> LayerNorm(eps).
               No absolute position embeddings (position_embedding_type == "rope").
  rope       : half-rotation convention (rotate_half = cat(-x2, x1) over the last dim), cos/sin
               shared by all layers, computed in float32 then cast to model dtype (same as HF,
               whose cache is built at torch.get_default_dtype()). NTK scaling (kexue.fm/9706
               eq. 6): NTKScalingRotaryEmbedding.__init__ rebuilds the cache at
               max_position_embeddings*factor > max_position_embeddings, so ALL positions —
               short sequences included — use  inv_freq = (theta*factor)^-(2i/d) / factor^(2/d).
  layer (post-LN):
               qkv = one fused Linear (pack_qkv) -> split thirds -> (B,L,NH,hd) -> rope(q,k)
               -> SDPA (scale 1/sqrt(hd)) -> o_proj;  x = attn_ln(x + attn)
               ug = up_gate_proj(x) [bias=False] -> split: FIRST half = up, SECOND half = gate;
               h = down_proj(act(gate) * up);          x = mlp_ln(x + h)
  head       : pooled = tanh(pooler.dense(x[:, 0])) -> classifier.  All dropouts are eval no-ops
               (and config has attn/classifier dropout 0.0 anyway). logn_attention_scale is a
               dead TODO in modeling.py (attention_scale stays None) — ignored here identically.

Weight map (safetensors key -> here): strip the `new.` base-model prefix; then
  embeddings.word_embeddings.weight -> wte, embeddings.token_type_embeddings.weight -> tte,
  embeddings.LayerNorm.{weight,bias} -> eln,
  encoder.layer.{i}.attention.qkv_proj.{weight,bias} -> layers[i].qkv   (2304x768 fused)
  encoder.layer.{i}.attention.o_proj.{weight,bias}   -> layers[i].o
  encoder.layer.{i}.attn_ln.{weight,bias}            -> layers[i].ln1
  encoder.layer.{i}.mlp.up_gate_proj.weight          -> layers[i].ug    (6144x768, NO bias)
  encoder.layer.{i}.mlp.down_proj.{weight,bias}      -> layers[i].down
  encoder.layer.{i}.mlp_ln.{weight,bias}             -> layers[i].ln2
  pooler.dense.{weight,bias} -> pool, classifier.{weight,bias} -> cls (classifier is unprefixed).
Unpacked q_proj/k_proj/v_proj checkpoints (pack_qkv=False) are fused at load time.

Module import is LIGHT (stdlib only): torch/safetensors load lazily via _ensure_torch(), so
serve_gte.py can pull them on the model-build thread, overlapped with the CSV data path.
Public interface mirrors lean_infer.LeanBert: load_gte(model_dir, device, dtype) -> LeanGTE with
.forward(input_ids, token_type_ids, attention_mask) -> logits, .num_labels, .cfg, .device —
drop-in for lean_infer.infer()'s bucketed loop. bf16 default on cuda, fp32 on cpu.
"""
import os, json, threading

torch = None
F = None
load_file = None
ACT = None
_TORCH_LOCK = threading.Lock()


def _ensure_torch():
    """Idempotent, thread-safe lazy import of torch/safetensors + the activation table."""
    global torch, F, load_file, ACT
    if torch is not None: return
    with _TORCH_LOCK:
        if torch is not None: return
        import torch as _torch
        import torch.nn.functional as _F
        from safetensors.torch import load_file as _lf
        ACT = {"gelu": _F.gelu, "relu": _F.relu, "silu": _F.silu,
               "gelu_new": lambda x: _F.gelu(x, approximate="tanh"),
               "gelu_pytorch_tanh": lambda x: _F.gelu(x, approximate="tanh")}
        F, load_file = _F, _lf
        torch = _torch                      # assigned LAST: `torch is not None` == fully ready


def _canon(k):
    """Canonical key: strip the `new.` base-model prefix, legacy gamma/beta -> weight/bias."""
    if k.startswith("new."): k = k[4:]
    return k.replace(".gamma", ".weight").replace(".beta", ".bias")


class LeanGTE:
    """GTE new-impl NewForSequenceClassification forward from a raw safetensors state dict.
    Eval-only (dropout omitted), plain tensors, no nn.Module overhead."""

    def __init__(self, model_dir, device="cuda", dtype=None):
        _ensure_torch()
        cfg = json.load(open(os.path.join(model_dir, "config.json")))
        self.cfg = cfg
        if dtype is None:
            dtype = torch.bfloat16 if str(device).startswith("cuda") else torch.float32
        self.H  = cfg["hidden_size"];       self.NH  = cfg["num_attention_heads"]
        self.NL = cfg["num_hidden_layers"]; self.eps = cfg.get("layer_norm_eps", 1e-12)
        self.I  = cfg["intermediate_size"]; self.hd  = self.H // self.NH
        self.act = ACT[cfg.get("hidden_act", "gelu")]
        if cfg.get("layer_norm_type", "layer_norm") != "layer_norm":
            raise ValueError("lean_gte supports layer_norm_type='layer_norm' only")
        if cfg.get("position_embedding_type", "rope") != "rope":
            raise ValueError("lean_gte supports position_embedding_type='rope' only")
        self.device, self.dtype = torch.device(device), dtype

        # ---- RoPE inverse frequencies (float32, replicating modeling.py op-for-op)
        theta = float(cfg.get("rope_theta", 10000.0))
        rs = cfg.get("rope_scaling") or None
        ar = torch.arange(0, self.hd, 2, dtype=torch.float32) / self.hd
        if rs is None:
            inv = 1.0 / (theta ** ar)
        else:
            if rs.get("type") != "ntk" or rs.get("mixed_b") is not None:
                raise ValueError(f"unsupported rope_scaling {rs!r} (fixed 'ntk' only)")
            factor = float(rs["factor"])
            if factor <= 1.0:
                raise ValueError("ntk factor <= 1 never triggers the scaled cache in modeling.py")
            # NTKScalingRotaryEmbedding._set_cos_sin_cache, seq_len > max_position_embeddings:
            inv = 1.0 / ((theta * factor) ** ar)
            inv = inv / factor ** (2.0 / self.hd)
        self.inv_freq = inv                                     # cpu float32
        self._cos = self._sin = None; self._nrope = 0
        self._set_rope(int(os.environ.get("GTE_ROPE_CACHE", 512)))

        # ---- weights
        sd = {_canon(k): v for k, v in load_file(os.path.join(model_dir, "model.safetensors")).items()}
        def T(k):   return sd[k].to(self.device, self.dtype)
        def opt(k): return T(k) if k in sd else None
        E = "embeddings."
        self.wte = T(E + "word_embeddings.weight")
        self.tte = opt(E + "token_type_embeddings.weight")      # None if type_vocab_size == 0
        self.eln = (T(E + "LayerNorm.weight"), T(E + "LayerNorm.bias"))
        self.layers = []
        for i in range(self.NL):
            p = f"encoder.layer.{i}."
            if p + "attention.qkv_proj.weight" in sd:           # pack_qkv (this checkpoint)
                qkv = (T(p + "attention.qkv_proj.weight"), T(p + "attention.qkv_proj.bias"))
            else:                                               # fuse unpacked q/k/v at load
                qkv = (torch.cat([T(p + f"attention.{n}_proj.weight") for n in "qkv"], 0),
                       torch.cat([T(p + f"attention.{n}_proj.bias")   for n in "qkv"], 0))
            self.layers.append({
                "qkv":  qkv,
                "o":    (T(p + "attention.o_proj.weight"), T(p + "attention.o_proj.bias")),
                "ln1":  (T(p + "attn_ln.weight"), T(p + "attn_ln.bias")),
                "ug":   T(p + "mlp.up_gate_proj.weight"),       # bias=False in modeling.py
                "down": (T(p + "mlp.down_proj.weight"), T(p + "mlp.down_proj.bias")),
                "ln2":  (T(p + "mlp_ln.weight"), T(p + "mlp_ln.bias"))})
        self.pool = (opt("pooler.dense.weight"), opt("pooler.dense.bias"))  # None -> bare CLS
        self.cls  = (T("classifier.weight"), T("classifier.bias"))
        self.num_labels = self.cls[0].shape[0]

    def _set_rope(self, n):
        """cos/sin cache for positions [0, n): float32 outer product exactly as modeling.py
        (freqs = einsum(t, inv_freq); emb = cat(freqs, freqs)), then cast to model dtype."""
        t = torch.arange(n, dtype=torch.float32)
        emb = torch.einsum("i,j->ij", t, self.inv_freq)
        emb = torch.cat((emb, emb), dim=-1)                     # [n, hd]
        self._cos = emb.cos().to(self.device, self.dtype)
        self._sin = emb.sin().to(self.device, self.dtype)
        self._nrope = n

    @staticmethod
    def _rot(x):
        """modeling.py rotate_half: split the last dim in half, cat(-x2, x1)."""
        h = x.shape[-1] // 2
        return torch.cat((-x[..., h:], x[..., :h]), dim=-1)

    def forward(self, ids, tt, mask, collect=None):
        """ids/tt long (B,L) on device, mask bool (B,L) True=real -> logits (B, num_labels).
        tt is accepted for interface parity with LeanBert but ignored when type_vocab_size < 2
        (modeling.py zeroes it in place). collect: optional list, appended with the hidden state
        after embeddings and after every layer (parity debugging only)."""
        with torch.inference_mode():
            B, L = ids.shape
            if L > self._nrope: self._set_rope(L)
            x = F.embedding(ids, self.wte)
            if self.tte is not None:
                x = x + (self.tte[0] if self.tte.shape[0] < 2 else F.embedding(tt, self.tte))
            x = F.layer_norm(x, (self.H,), *self.eln, self.eps)
            if collect is not None: collect.append(x)
            am = None if bool(mask.all()) else mask[:, None, None, :]   # None -> SDPA fast path
            cos, sin = self._cos[:L], self._sin[:L]                     # [L, hd] -> broadcast
            for l in self.layers:
                q, k, v = F.linear(x, *l["qkv"]).split(self.H, dim=-1)  # one fused GEMM
                q = q.view(B, L, self.NH, self.hd).transpose(1, 2)      # (B,NH,L,hd)
                k = k.view(B, L, self.NH, self.hd).transpose(1, 2)
                v = v.view(B, L, self.NH, self.hd).transpose(1, 2)
                q = q * cos + self._rot(q) * sin
                k = k * cos + self._rot(k) * sin
                a = F.scaled_dot_product_attention(q, k, v, attn_mask=am)
                a = a.transpose(1, 2).reshape(B, L, self.H)
                x = F.layer_norm(x + F.linear(a, *l["o"]), (self.H,), *l["ln1"], self.eps)
                up, gate = F.linear(x, l["ug"]).split(self.I, dim=-1)   # up FIRST, gate SECOND
                h = F.linear(self.act(gate) * up, *l["down"])
                x = F.layer_norm(x + h, (self.H,), *l["ln2"], self.eps)
                if collect is not None: collect.append(x)
            c = x[:, 0]
            if self.pool[0] is not None: c = torch.tanh(F.linear(c, *self.pool))
            return F.linear(c, *self.cls)


def load_gte(model_dir, device="cuda", dtype=None):
    """Loader mirroring lean_infer's LeanBert(...) construction. dtype=None -> bf16 on cuda,
    fp32 elsewhere."""
    return LeanGTE(model_dir, device, dtype)
