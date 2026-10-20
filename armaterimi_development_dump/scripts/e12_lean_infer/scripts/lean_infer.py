"""E0a — lean inference container (B1 §8a, §12a). The serving loop that produces the speed score.

Never imports `transformers`: hand-rolled BERT-family forward on torch + safetensors + the Rust
`tokenizers` lib. v3 (2026-08-24): module import is LIGHT (stdlib + numpy + tokenizers, ~0.25s);
torch/safetensors load lazily via _ensure_torch() — in main() that happens on the model build
thread, overlapped with the data path. The data path is parallel: the CSV is parsed with
csv.reader on the pool feeder thread, rows are normalised in FORKED worker processes (created
before any threads/CUDA — they never import torch), and the main thread tokenises each returned
chunk with Rust encode_batch (releases the GIL) — parse, normalise, tokenise and model load all
overlap. Length-sorted bucketed inference (B1 §3d: ~2.1x over pad-to-256), per-batch padding
only, original row order restored at the end.

Semantics replicate HF BertForSequenceClassification EXACTLY (parity: scripts/verify_parity.py):
embeddings(word+pos+type, LN) -> N x [SDPA self-attn, LN, GELU FFN, LN] -> tanh pooler over CLS
(or bare CLS if the checkpoint has no pooler) -> classifier. Loader accepts `bert.`-prefixed
(BertForSequenceClassification) and bare checkpoints, plus legacy gamma/beta LayerNorm names.

Normalisation is an INLINED FROZEN copy of common/normalize.py (T1 NFC+U+0307 strip, T2/T3 drop
keys = e18 drop_keys_v2 @ 2026-08-23 + boilerplate-value filter, T4 unisex/çocuk->unknown) — no repo imports, the
container is self-contained. Text pair: query vs "title | brand | category | gender age_group |
attributes" (same template as e07 training). The parallel read path is asserted byte-identical
to the pre-parallel serial one by scripts/parity_datapath.py on the full public_test.csv.

CLI: lean_infer.py <input_csv> <output_path> --model-dir <dir>
     model dir holds model.safetensors + config.json + tokenizer.json [+ decision.json]
     decision.json: {"head":"softmax3"} (argmax) or {"head":"ordinal","cuts":[c1,c2]} (scalar)
Env overrides: DEVICE, BATCH_ROWS(128), TOKEN_BUDGET(32768), MAXLEN(256), LEAN_NO_OVERLAP,
     LEAN_WORKERS(default cpu-2, cap 12), LEAN_CHUNK(1024 rows/chunk),
     LEAN_SERIAL_READ=1 -> single-process data path (A/B + emergency fallback).
Deterministic: no sampling, argmax ties -> lowest class id, stable length sort.
"""
import sys, os, time
T0 = time.perf_counter()
os.environ.setdefault("TOKENIZERS_PARALLELISM", "true")
import json, csv, re, unicodedata, threading, queue, multiprocessing

_t = time.perf_counter(); import numpy as np;                 T_NP  = time.perf_counter() - _t
_t = time.perf_counter(); from tokenizers import Tokenizer;   T_TOK = time.perf_counter() - _t
T_IMPORTS = time.perf_counter() - T0

# torch + safetensors are deliberately NOT imported at module level (1.4-2.0s): _ensure_torch()
# pulls them on first need, normally on the model build thread while this thread runs the data
# path. verify_parity.py-style callers that touch LeanBert/infer directly still work: every
# torch-needing entry point calls _ensure_torch() itself.
torch = None
F = None
load_file = None
ACT = None
T_TORCH_IMPORT = 0.0
_TORCH_LOCK = threading.Lock()

def _ensure_torch():
    """Idempotent, thread-safe lazy import of torch/safetensors + the activation table."""
    global torch, F, load_file, ACT, T_TORCH_IMPORT
    if torch is not None: return
    with _TORCH_LOCK:
        if torch is not None: return
        t = time.perf_counter()
        import torch as _torch
        import torch.nn.functional as _F
        from safetensors.torch import load_file as _lf
        T_TORCH_IMPORT = time.perf_counter() - t
        ACT = {"gelu": _F.gelu, "relu": _F.relu,
               "gelu_new": lambda x: _F.gelu(x, approximate="tanh"),
               "gelu_pytorch_tanh": lambda x: _F.gelu(x, approximate="tanh")}
        F, load_file = _F, _lf
        torch = _torch                      # assigned LAST: `torch is not None` == fully ready

def log(m): print(f"[lean +{time.perf_counter()-T0:7.3f}s] {m}", flush=True)

# ---------------------------------------------------------------- frozen normalisation
# Inlined from asya/s2_exp/common/normalize.py + drop_keys.csv (2026-08-21). Keep byte-identical
# to training-time preprocessing; see that module's docstring for the leak analysis behind it.
COMBINING_DOT = "̇"
TEXT_COLS = ["query", "title", "category", "brand", "gender", "age_group", "attributes"]
SEG = re.compile(r", (?=[^:,]{1,40}: )")
# v2 (e18_attr_clean/drop_keys_v2.csv, adopted 2026-08-23): the 18 snapshot-leak keys PLUS 42 keys
# measured at <=0.3% query-landing (care/washing, warranty, origin, sustainability, packaging,
# legal, manufacturer identity...). MUST stay byte-identical to the training-time transform.
DROP_KEYS = frozenset([
    "alt taban materyali", "astar materyali", "cam materyali",
    "cam renk", "color detail", "dolgu tekniği",
    "ekartman", "iç astar & iç taban materyali", "iç taban materyali",
    "kasa çapı", "kordon renk", "priz tipi",
    "saya materyali", "taban materyali", "web color",
    "çerçeve formu", "çerçeve materyali", "çerçeve renk",
    "yıkama talimatı", "bakım talimatları (genel)", "bakım talimatları (gıda temas)",
    "sürdürülebilirlik detayı", "menşei", "ek özellik",
    "garanti süresi", "garanti tipi", "ürün güvenliği bilgisi",
    "kullanım talimatı", "kutu durumu", "paket görseli (arka)",
    "paket görseli (ön)", "ürün kutusu", "piller kutuya dahil mi?",
    "ücretsiz kurulum/montaj", "kurulum gerekli mi?", "tip-19. ruhsat tarihi",
    "ürün bilgi formu", "enerji sınıfı görseli", "enerji sınıf aralığı",
    "gıda temas sembolü", "tamir edilebilirlik", "ruhsat tarih",
    "ikincil ithalatçı adı", "üçüncül ithalatçı mail adresi", "ikincil ithalatçı adres bilgisi",
    "üçüncül ithalatçı adres bilgisi", "üçüncül ithalatçı adı", "ikincil ithalatçı mail adresi",
    "üretici adres bilgisi", "birincil ithalatçı mail adresi", "birincil ithalatçı adres bilgisi",
    "birincil ithalatçı adı", "üretici mail adresi", "üretici adı",
    "persona", "enerji ve besin değerleri", "paket derinlik",
    "paket genişlik", "paket yükseklik", "refill"])
# boilerplate values carry no information; compatibility keys keep "belirtilmemiş" (device
# silence is a signal, rubric 4C). "yok"/"var"/"mevcut" are information and are NEVER dropped.
BOILERPLATE_EXACT = frozenset({"belirtilmemiş", "mevcut değil", "ek özellik mevcut değil", "-", "unknown", ""})
BOILERPLATE_SUBSTR = ("takip ediniz", "kullanım ömrünü")
KEEP_VALUES = frozenset({"yok", "var", "mevcut"})
EMPTY_SENTINEL = "unknown"
_TR_FOLD = str.maketrans({"ç":"c","ğ":"g","ı":"i","ö":"o","ş":"s","ü":"u","İ":"i","I":"ı"})
def _fold(v): return " ".join(str(v).translate(_TR_FOLD).lower().split())
# The folded forms of the CONSTANTS are constants. Folding them inside _is_boiler re-derived all
# ten of them on every call: 8.6M _fold() calls over a 90k file where only ~1.4M are real work,
# and _fold was 9.4s of the 11.3s serial normalise. Precomputed here; output is byte-identical
# (same strings, same folds) and parity_datapath.py asserts it.
_KEEP_FOLDED = frozenset(_fold(x) for x in KEEP_VALUES)
_BOIL_FOLDED = frozenset(_fold(x) for x in BOILERPLATE_EXACT)
_BOIL_SUBSTR_FOLDED = tuple(_fold(x) for x in BOILERPLATE_SUBSTR)
def _is_boiler(v):
    f = _fold(v)
    if f in _KEEP_FOLDED: return False
    if f in _BOIL_FOLDED: return True
    return any(s in f for s in _BOIL_SUBSTR_FOLDED)

def t1(s: str) -> str:
    """T0 Turkish-aware lowercase + T1 NFC + strip stray combining dot. Idempotent.

    Lowercasing is a PROVABLE NO-OP on the competition data -- train.csv and public_test.csv
    contain zero uppercase characters in any of the seven fields -- so it cannot move a scored
    prediction. It is here because the model was trained entirely on lowercase text and collapses
    on uppercase input (measured: "ÇELIK BILEKLIK" scores 0.05 where "çelik bileklik" scores
    1.96), which is reachable through the explainability UI's free-text query box.
    Plain str.lower(), deliberately NOT Turkish-locale lowering: locale-correct rules map dotless
    I to 'ı', so "ÇELIK" would become "çelık" -- a word the model has never seen -- whereas someone
    typing ASCII capitals means "çelik". Python lowers 'İ' to 'i'+U+0307 and the combining-dot strip
    below removes that, so both capital forms land on the same lowercase text.
    """
    return unicodedata.normalize("NFC", s.lower()).replace(COMBINING_DOT, "")

def clean_attrs(attr: str) -> str:
    """Drop frozen leak keys from the attributes string (T2/T3), lowercase keys."""
    out = []
    for seg in SEG.split(t1(attr)):
        if ":" not in seg:
            if seg.strip(): out.append(seg.strip())
            continue
        k, v = seg.split(":", 1)
        kl = k.strip().lower(); vs = v.strip()
        if kl in DROP_KEYS: continue
        if _is_boiler(vs) and "uyum" not in _fold(kl): continue      # v2 value filter
        out.append(f"{kl}: {vs}")
    return ", ".join(out) if out else EMPTY_SENTINEL

# ---------------------------------------------------------------- data path (serial + parallel)
def _header_ix(hdr):
    return {c: hdr.index(c) for c in ["row_id"] + TEXT_COLS if c in hdr}

def _rows_to_pairs(rows, ix):
    """The frozen per-row transform: parsed csv rows -> (row_ids, [(query, doc), ...]).
    Columns by NAME, NA/short rows -> '', all text normalised; doc template matches e07
    training: title | brand | category | gender age_group | attributes. Shared by the serial
    reader and the pool workers (parity_datapath.py asserts byte-identical output)."""
    def get(row, c):
        i = ix.get(c, -1)
        return row[i] if 0 <= i < len(row) else ""
    ids, pairs = [], []
    for row in rows:
        g = t1(get(row, "gender"))
        if g == "unisex" or g == "çocuk": g = "unknown"      # T4 (both are train-export artifacts)
        b = t1(get(row, "brand"))
        if _BRAND_SINK and b == "genel markalar": b = "unknown"     # T5, checkpoint-declared
        doc = (f"{t1(get(row,'title'))} | {b} | {t1(get(row,'category'))}"
               f" | {g} {t1(get(row,'age_group'))} | {clean_attrs(get(row,'attributes'))}")
        ids.append(get(row, "row_id"))
        pairs.append((t1(get(row, "query")), doc))
    return ids, pairs

def read_pairs(path):
    """CSV -> (row_ids, [(query, doc), ...]) — serial reference implementation."""
    with open(path, newline="", encoding="utf-8") as f:
        rd = csv.reader(f)
        ix = _header_ix(next(rd))
        return _rows_to_pairs(rd, ix)

class _Enc:
    """Minimal stand-in for a tokenizers Encoding: _assemble/make_batches only ever touch .ids
    and .type_ids, so workers can ship plain numpy arrays instead of pickling Encoding objects."""
    __slots__ = ("ids", "type_ids")
    def __init__(self, ids, type_ids): self.ids = ids; self.type_ids = type_ids

_TOK = None

_BRAND_SINK = False   # set by configure_text() from the CHECKPOINT's own decision.json

def configure_text(mdir):
    """Read preprocessing switches the CHECKPOINT declares about itself, BEFORE the pool forks.

    T5 (brand 'genel markalar' -> 'unknown') is only correct for a model TRAINED with the brand
    sink; applying it to a model trained without one feeds a token that appears in none of its
    56,085 training brands. Tying the switch to the checkpoint instead of to an image ENV makes
    the mismatch impossible: swap the checkpoint and the preprocessing follows it.
    """
    global _BRAND_SINK
    try:
        with open(os.path.join(mdir, "decision.json"), encoding="utf-8") as f:
            _BRAND_SINK = bool(json.load(f).get("brand_sink", False))
    except Exception:
        _BRAND_SINK = False
    return _BRAND_SINK

def preload_tokenizer(mdir, maxlen):
    """Load the Rust tokenizer into a module global BEFORE the pool is forked, so every worker
    inherits it copy-on-write (no per-worker load cost, no extra memory)."""
    global _TOK
    configure_text(mdir)
    if _TOK is None:
        _TOK = Tokenizer.from_file(os.path.join(mdir, "tokenizer.json"))
        _TOK.enable_truncation(max_length=maxlen, strategy="longest_first")
    return _TOK

def _norm_tok_chunk(args):
    """Pool worker: normalise AND tokenise one chunk.

    MEASURED 2026-08-25: the `tokenizers` Rust parallelism is disabled in the PARENT once any fork
    has happened (encode drops 6.2s -> 9.9s at 90k, unchanged by worker count 2..10, so it is the
    fork and not CPU contention). Tokenising inside the workers restores parallelism — across
    processes instead of rayon threads. Returns flat arrays + lengths: cheap to pickle and small
    enough for the container's 64MB /dev/shm at the default chunk size.
    """
    ids, pairs = _rows_to_pairs(*args)
    encs = _TOK.encode_batch(pairs)
    lens = np.fromiter((len(e.ids) for e in encs), dtype=np.int32, count=len(encs))
    # int32 ids / int8 type_ids instead of int64+int64: the flat arrays are the ONLY thing that
    # crosses the pool pipe, ~151 MB per 90k file at int64. Vocabularies here are 32k-250k so
    # int32 is exact, and type_ids are 0/1. _assemble copies into its own int64 buffer, so numpy
    # widens on the way in and the tensors handed to torch are unchanged.
    flat_i = np.concatenate([np.asarray(e.ids, dtype=np.int32) for e in encs]) if encs else np.zeros(0, np.int32)
    flat_t = np.concatenate([np.asarray(e.type_ids, dtype=np.int8) for e in encs]) if encs else np.zeros(0, np.int8)
    return ids, flat_i, flat_t, lens

def _unpack_chunk(flat_i, flat_t, lens):
    cuts = np.cumsum(lens)[:-1]
    return [_Enc(i, t) for i, t in zip(np.split(flat_i, cuts), np.split(flat_t, cuts))]

def _norm_chunk(args):
    """Pool worker: normalise one chunk of parsed rows. Runs in a forked child that has never
    imported torch; pure stdlib work."""
    return _rows_to_pairs(*args)

def _iter_row_chunks(path, chunk):
    """csv.reader stream -> (rows, ix) chunks. Consumed lazily by the pool's feeder thread, so
    parsing overlaps worker normalisation. csv module handles quoted embedded newlines — never
    split this file by raw lines."""
    with open(path, newline="", encoding="utf-8") as f:
        rd = csv.reader(f)
        ix = _header_ix(next(rd))
        buf = []
        for row in rd:
            buf.append(row)
            if len(buf) >= chunk:
                yield buf, ix
                buf = []
        if buf: yield buf, ix

def default_workers():
    return int(os.environ.get("LEAN_WORKERS", max(1, min((os.cpu_count() or 4) - 2, 12))))

def make_pool(workers=None):
    """Fork-based normalisation pool. MUST be created before any other threads exist (fork and
    live threads don't mix) and before CUDA init. Children inherit only the light module state
    (no torch). Returns None -> caller falls back to the serial path."""
    try:
        n = workers if workers is not None else default_workers()
        if n <= 1: return None
        return multiprocessing.get_context("fork").Pool(n)
    except Exception:
        return None

def stream_norm(path, pool, chunk=None):
    """Yield (row_ids, pairs) chunks IN ORDER. pool=None -> one serial chunk (read_pairs)."""
    if pool is None:
        yield read_pairs(path)
        return
    chunk = chunk or int(os.environ.get("LEAN_CHUNK", 1024))
    yield from pool.imap(_norm_chunk, _iter_row_chunks(path, chunk))

def read_pairs_parallel(path, pool=None, chunk=None):
    """Parallel read_pairs: byte-identical output, workers do the normalisation."""
    own = pool is None
    if own: pool = make_pool()
    try:
        ids, pairs = [], []
        for i, p in stream_norm(path, pool, chunk):
            ids.extend(i); pairs.extend(p)
        return ids, pairs
    finally:
        if own and pool is not None: pool.terminate()

def run_datapath(inp, mdir, maxlen, pool):
    """Full pipelined data path used by BOTH serving CLIs: parse (feeder thread) -> normalise
    (pool workers) -> encode_batch per chunk (this thread; Rust releases the GIL). Everything
    overlaps a concurrent model-load thread. Returns (row_ids, encs, detail)."""
    t_start = time.perf_counter()
    it = stream_norm(inp, pool)                       # parse+normalise start immediately
    t = time.perf_counter()
    tok = Tokenizer.from_file(os.path.join(mdir, "tokenizer.json"))
    tok.enable_truncation(max_length=maxlen, strategy="longest_first")
    t_ldtok = time.perf_counter() - t
    row_ids, encs, t_enc = [], [], 0.0
    for ids, pairs in it:
        row_ids.extend(ids)
        t = time.perf_counter(); encs.extend(tok.encode_batch(pairs)); t_enc += time.perf_counter() - t
    return row_ids, encs, {"total": time.perf_counter() - t_start, "tok_load": t_ldtok,
                           "encode": t_enc, "mode": "pipelined" if pool is not None else "serial"}

def run_datapath_wtok(inp, mdir, maxlen, pool, chunk=None):
    """Data path with tokenisation done INSIDE the pool workers (see _norm_tok_chunk). Output is
    byte-identical to run_datapath; only where the work happens changes."""
    t_start = time.perf_counter()
    preload_tokenizer(mdir, maxlen)                    # must already be loaded pre-fork; idempotent
    chunk = chunk or int(os.environ.get("LEAN_CHUNK", 1024))
    row_ids, encs = [], []
    for ids, fi, ft, lens in pool.imap(_norm_tok_chunk, _iter_row_chunks(inp, chunk)):
        row_ids.extend(ids); encs.extend(_unpack_chunk(fi, ft, lens))
    return row_ids, encs, {"total": time.perf_counter() - t_start, "tok_load": 0.0,
                           "encode": 0.0, "mode": "worker-tokenised"}

def run_datapath_streaming(inp, mdir, maxlen, pool, model_ready, on_group=None):
    """Same parse -> normalise -> encode pipeline as run_datapath, but yields GROUPS of encodings
    as soon as each is ready so the GPU can start while later rows are still being tokenised.

    Why this exists: run_datapath materialises every encoding before infer() is called, so at 90k
    rows the device idles for the whole data path (~11s measured). Grouping costs a little padding
    efficiency (length sorting becomes per-group instead of global) — with GROUP_ROWS >> batch size
    the length distribution inside a group matches the whole file, so the loss is negligible while
    the overlap saves seconds.

    Yields (row_ids_chunk, encs_chunk). `model_ready` is an Event: groups are buffered until the
    model exists, then released, so we never block the loader.
    """
    group_rows = int(os.environ.get("LEAN_GROUP_ROWS", 12288))
    it = stream_norm(inp, pool)
    tok = Tokenizer.from_file(os.path.join(mdir, "tokenizer.json"))
    tok.enable_truncation(max_length=maxlen, strategy="longest_first")
    buf_ids, buf_encs = [], []
    for ids, pairs in it:
        buf_ids.extend(ids); buf_encs.extend(tok.encode_batch(pairs))
        if len(buf_ids) >= group_rows:
            model_ready.wait()
            if on_group: on_group()
            yield buf_ids, buf_encs
            buf_ids, buf_encs = [], []
    if buf_ids:
        model_ready.wait()
        if on_group: on_group()
        yield buf_ids, buf_encs

# ---------------------------------------------------------------- model
def _canon(k):
    """Canonical HF BertModel key: strip `bert.` prefix, legacy gamma/beta -> weight/bias."""
    if k.startswith("bert."): k = k[5:]
    return k.replace(".gamma", ".weight").replace(".beta", ".bias")

class LeanBert:
    """BertForSequenceClassification forward from a raw safetensors state dict. Eval-only
    (dropout omitted), plain tensors, no nn.Module overhead. dtype=None -> torch.bfloat16
    (the pre-lazy-import default)."""

    def __init__(self, model_dir, device="cuda", dtype=None):
        _ensure_torch()
        if dtype is None: dtype = torch.bfloat16
        cfg = json.load(open(os.path.join(model_dir, "config.json")))
        self.cfg = cfg
        self.H    = cfg["hidden_size"];      self.NH  = cfg["num_attention_heads"]
        self.NL   = cfg["num_hidden_layers"]; self.eps = cfg.get("layer_norm_eps", 1e-12)
        self.act  = ACT[cfg.get("hidden_act", "gelu")]
        self.device, self.dtype = torch.device(device), dtype
        sd = {_canon(k): v for k, v in load_file(os.path.join(model_dir, "model.safetensors")).items()}
        def T(k):     return sd[k].to(self.device, self.dtype)
        def opt(k):   return T(k) if k in sd else None
        E = "embeddings."
        self.wte, self.wpe, self.tte = T(E+"word_embeddings.weight"), T(E+"position_embeddings.weight"), T(E+"token_type_embeddings.weight")
        self.eln = (T(E+"LayerNorm.weight"), T(E+"LayerNorm.bias"))
        self.layers = []
        for i in range(self.NL):
            p = f"encoder.layer.{i}."
            self.layers.append({
                "q":   (T(p+"attention.self.query.weight"),  T(p+"attention.self.query.bias")),
                "k":   (T(p+"attention.self.key.weight"),    T(p+"attention.self.key.bias")),
                "v":   (T(p+"attention.self.value.weight"),  T(p+"attention.self.value.bias")),
                "ao":  (T(p+"attention.output.dense.weight"), T(p+"attention.output.dense.bias")),
                "ln1": (T(p+"attention.output.LayerNorm.weight"), T(p+"attention.output.LayerNorm.bias")),
                "ff1": (T(p+"intermediate.dense.weight"),    T(p+"intermediate.dense.bias")),
                "ff2": (T(p+"output.dense.weight"),          T(p+"output.dense.bias")),
                "ln2": (T(p+"output.LayerNorm.weight"),      T(p+"output.LayerNorm.bias"))})
        self.pool = (opt("pooler.dense.weight"), opt("pooler.dense.bias"))   # None -> bare CLS
        self.cls  = (T("classifier.weight"), T("classifier.bias"))
        self.num_labels = self.cls[0].shape[0]

    def forward(self, ids, tt, mask):
        """ids/tt long (B,L) on device, mask bool (B,L) True=real. Returns logits (B, num_labels).
        Wrapped in torch.inference_mode() (same semantics as the former decorator)."""
        with torch.inference_mode():
            B, L = ids.shape
            x = F.embedding(ids, self.wte) + self.wpe[:L] + F.embedding(tt, self.tte)
            x = F.layer_norm(x, (self.H,), *self.eln, self.eps)
            am = None if bool(mask.all()) else mask[:, None, None, :]   # None -> SDPA fast path
            hd = self.H // self.NH
            for l in self.layers:
                q = F.linear(x, *l["q"]).view(B, L, self.NH, hd).transpose(1, 2)
                k = F.linear(x, *l["k"]).view(B, L, self.NH, hd).transpose(1, 2)
                v = F.linear(x, *l["v"]).view(B, L, self.NH, hd).transpose(1, 2)
                a = F.scaled_dot_product_attention(q, k, v, attn_mask=am)
                a = a.transpose(1, 2).reshape(B, L, self.H)
                x = F.layer_norm(x + F.linear(a, *l["ao"]), (self.H,), *l["ln1"], self.eps)
                h = self.act(F.linear(x, *l["ff1"]))
                x = F.layer_norm(x + F.linear(h, *l["ff2"]), (self.H,), *l["ln2"], self.eps)
            c = x[:, 0]
            if self.pool[0] is not None: c = torch.tanh(F.linear(c, *self.pool))
            return F.linear(c, *self.cls)

# ---------------------------------------------------------------- bucketed inference
PAD_Q = int(os.environ.get("LEAN_PAD_Q", "0"))   # 0 = legacy pad-to-batch-max

def make_batches(lens, batch_rows=128, token_budget=32768, pad_q=None):
    """Stable length-sort, then adjacent rows into batches capped by rows AND rows*maxlen.

    pad_q > 0 switches on QUANTISED buckets: a row is assigned to bucket ceil(len/pad_q) and a
    batch never spans two buckets, so the padded width is pad_q*bucket -- a function of the ROW,
    not of which other rows happen to sit beside it. That makes the forward pass bit-identical
    under ANY grouping of the input, which is what lets the data path stream (STREAM=1) without
    changing a single logit. Legacy behaviour (pad to the batch max) is exactly reproduced by
    pad_q=0. Cost is a little extra padding: mean length ~105, so pad_q=8 wastes ~3.5% of tokens.
    """
    pad_q = PAD_Q if pad_q is None else pad_q
    if pad_q > 0:
        buck = -(-np.asarray(lens) // pad_q)            # ceil-div bucket id
        order = np.lexsort((lens, buck))                # primary bucket, secondary length
        out, i, n = [], 0, len(order)
        while i < n:
            b = buck[order[i]]; w = int(b) * pad_q      # this batch's fixed padded width
            j = i + 1
            while (j < n and buck[order[j]] == b and j - i < batch_rows
                   and (j - i + 1) * w <= token_budget):
                j += 1
            out.append(order[i:j]); i = j
        return out
    order = np.argsort(lens, kind="stable")
    out, i, n = [], 0, len(order)
    while i < n:
        j = i + 1
        while j < n and j - i < batch_rows and (j - i + 1) * lens[order[j]] <= token_budget:
            j += 1
        out.append(order[i:j]); i = j
    return out

def _assemble(encs, idx, pin, pad_q=None):
    """Pad one batch to its own max length -> (ids, type_ids, mask) CPU tensors.
    With pad_q > 0 the width is rounded up to the bucket ceiling instead, which is identical for
    every row in the batch by construction (see make_batches)."""
    _ensure_torch()
    pad_q = PAD_Q if pad_q is None else pad_q
    L = max(len(encs[r].ids) for r in idx)
    if pad_q > 0:
        L = -(-L // pad_q) * pad_q
    ids = np.zeros((len(idx), L), dtype=np.int64); tt = np.zeros_like(ids)
    mk  = np.zeros((len(idx), L), dtype=bool)
    for b, r in enumerate(idx):
        e = encs[r]; n = len(e.ids)
        ids[b, :n] = e.ids; tt[b, :n] = e.type_ids; mk[b, :n] = True
    ts = tuple(torch.from_numpy(a) for a in (ids, tt, mk))
    return tuple(t.pin_memory() for t in ts) if pin else ts

def infer(model, encs, batch_rows=128, token_budget=32768, overlap=True, on_first_batch=None,
          pad_q=None):
    """Length-sorted bucketed forward over all encodings; logits returned in ORIGINAL row order.
    overlap: batch assembly on a producer thread (queue depth 4) while the device computes."""
    _ensure_torch()
    pad_q = PAD_Q if pad_q is None else pad_q
    lens = np.array([len(e.ids) for e in encs])
    batches = make_batches(lens, batch_rows, token_budget, pad_q)
    logits = np.empty((len(encs), model.num_labels), dtype=np.float32)
    pin = model.device.type == "cuda"
    if overlap:
        q = queue.Queue(maxsize=4)
        def produce():
            for idx in batches: q.put((idx, _assemble(encs, idx, pin, pad_q)))
            q.put(None)
        threading.Thread(target=produce, daemon=True).start()
        feed = iter(q.get, None)
    else:
        feed = ((idx, _assemble(encs, idx, pin, pad_q)) for idx in batches)
    first = True
    for idx, (ids, tt, mk) in feed:
        nb = model.device.type == "cuda"
        out = model.forward(ids.to(model.device, non_blocking=nb),
                            tt.to(model.device, non_blocking=nb),
                            mk.to(model.device, non_blocking=nb))
        logits[idx] = out.float().cpu().numpy()
        if first and on_first_batch: on_first_batch(); first = False
    return logits

# ---------------------------------------------------------------- decision + output
LABELS = ["irrelevant", "semi-relevant", "relevant"]     # repo convention (harness.py, e07)

def _probs3(logits):
    """logits -> (p_irr, p_semi, p_rel). 2 columns = the e21 TWO-HEAD parameterisation
    (A = relevant-vs-rest, B = semi-vs-irrelevant given not-relevant); 3 columns = softmax."""
    if logits.shape[1] == 2:
        pa = 1.0 / (1.0 + np.exp(-logits[:, 0].astype(np.float64)))
        pb = 1.0 / (1.0 + np.exp(-logits[:, 1].astype(np.float64)))
        return np.stack([(1 - pa) * (1 - pb), (1 - pa) * pb, pa], 1)
    e = np.exp(logits.astype(np.float64) - logits.max(1, keepdims=True))
    return e / e.sum(1, keepdims=True)

def _ordinal_scalar(logits):
    """The frozen serving scalar x = p_semi + 2*p_rel (1-column checkpoints emit it directly)."""
    if logits.shape[1] == 1:
        return logits[:, 0].astype(np.float64)
    P = _probs3(logits)
    return P[:, 1] + 2.0 * P[:, 2]

def _roots_from_csv(path):
    """Category ROOT per row, in file order. Only read when the regime rule is active; one
    single-column pass, no effect on the frozen (query, doc) data path."""
    with open(path, newline="", encoding="utf-8") as f:
        rd = csv.reader(f); ix = _header_ix(next(rd)); i = ix.get("category", -1)
        # .lower() mirrors e23 fit_regime_cuts.py, which fitted the table on lowercased roots.
        # No-op on public (all roots already lowercase) so no measured number moves; it keeps the
        # regime rule alive if the private export ever capitalises a category.
        return [t1(r[i]).split("/")[0].strip().lower() if 0 <= i < len(r) else "" for r in rd]

def decide(logits, model_dir, cfg, input_csv=None):
    """logits -> label strings, per decision.json head.
       softmax3        : argmax (2-column checkpoints go through the two-head probabilities)
       ordinal         : scalar + two global cuts
       ordinal_regime  : scalar + cuts chosen by the row's OWN category root (still per-row;
                         no batch statistics are used). Falls back to the global cuts for any
                         root outside the pre-registered regimes, or if input_csv is unavailable."""
    dp = os.path.join(model_dir, "decision.json")
    d = json.load(open(dp)) if os.path.exists(dp) else {}
    head = d.get("head", "softmax3" if logits.shape[1] >= 2 else "ordinal")
    labels = d.get("labels")
    if labels is None:
        i2l = cfg.get("id2label")
        real = i2l and len(i2l) == 3 and not any(str(v).startswith("LABEL_") for v in i2l.values())
        labels = [i2l[str(i)] for i in range(len(i2l))] if real else LABELS
    if head == "ordinal_regime":
        s = _ordinal_scalar(logits)
        gl = d.get("global", d.get("cuts", [0.5, 1.5]))
        table = d.get("regimes", {})
        roots = _roots_from_csv(input_csv) if input_csv else None
        if roots is None or len(roots) != len(s) or not table:
            c1, c2 = gl
            cls = (s >= c1).astype(int) + (s >= c2)
        else:
            fash = set(d.get("fashion_roots", [])); tech = set(d.get("technical_roots", []))
            cls = np.zeros(len(s), int)
            reg = np.array(["fashion" if r in fash else ("technical" if r in tech else "other")
                            for r in roots])
            for g in np.unique(reg):
                c1, c2 = table.get(g, gl)
                m = reg == g
                cls[m] = (s[m] >= c1).astype(int) + (s[m] >= c2)
    elif head == "ordinal":
        c1, c2 = d.get("cuts", [0.5, 1.5])
        s = _ordinal_scalar(logits)
        cls = (s >= c1).astype(int) + (s >= c2)
    else:
        cls = _probs3(logits).argmax(1) if logits.shape[1] == 2 else logits.argmax(1)
    return [labels[i] for i in cls]

LAST_WRITE = None      # (path, row_ids, labels) from the most recent write_csv

def write_csv(path, row_ids, labels):
    """Single atomic write: temp file in the destination dir + os.replace.
    Also publishes what was written, so predict.py can run its contract check against these
    objects instead of re-parsing the 90k input CSV a second time purely to recover row_ids."""
    global LAST_WRITE
    tmp = f"{path}.tmp{os.getpid()}"
    with open(tmp, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(["row_id", "real_label"]); w.writerows(zip(row_ids, labels))
    os.replace(tmp, path)
    LAST_WRITE = (path, list(row_ids), list(labels))

# ---------------------------------------------------------------- main
def main(argv):
    pos = [a for a in argv if not a.startswith("--")]
    kw  = dict(a.lstrip("-").split("=", 1) if "=" in a else (a.lstrip("-"), "") for a in argv if a.startswith("--"))
    if len(pos) != 2 or "model-dir" not in kw:
        sys.exit("usage: lean_infer.py <input_csv> <output_path> --model-dir <dir> "
                 "[--device=cuda] [--batch-rows=128] [--token-budget=32768] [--no-overlap]")
    inp, outp, mdir = pos[0], pos[1], kw["model-dir"]
    dev_req = kw.get("device") or os.environ.get("DEVICE") or ""   # "" -> resolved on build thread
    br   = int(kw.get("batch-rows")   or os.environ.get("BATCH_ROWS", 128))
    tb   = int(kw.get("token-budget") or os.environ.get("TOKEN_BUDGET", 32768))
    ml   = int(kw.get("max-len")      or os.environ.get("MAXLEN", 256))
    ovl  = "no-overlap" not in kw and not os.environ.get("LEAN_NO_OVERLAP")
    log(f"light imports {T_IMPORTS:.3f}s (numpy {T_NP:.3f} tokenizers {T_TOK:.3f}); torch deferred to build thread")

    # normalisation workers are forked BEFORE any thread/CUDA exists; then torch import + CUDA
    # context + weights->device run on a background thread while this thread runs the data path.
    # WTOK=1 (default): tokenise inside the pool workers. tokenizers' Rust parallelism is disabled
    # in the parent after any fork (measured: encode 6.2s -> 9.9s at 90k, flat across worker counts),
    # so the tokenizer must be preloaded pre-fork and the workers must do the encoding.
    WTOK = os.environ.get("WTOK", "1") == "1" and not os.environ.get("LEAN_SERIAL_READ")
    configure_text(mdir)
    if WTOK: preload_tokenizer(mdir, ml)
    pool = None if os.environ.get("LEAN_SERIAL_READ") else make_pool()
    holder = {}
    def build():
        try:
            _ensure_torch()
            dev = dev_req or ("cuda" if torch.cuda.is_available() else "cpu")
            dt  = torch.bfloat16 if dev.startswith("cuda") else torch.float32
            warm = None
            if dev.startswith("cuda"):                    # context init overlaps the weight read
                warm = threading.Thread(target=lambda: torch.zeros(1, device=dev), daemon=True)
                warm.start()
            holder["m"] = LeanBert(mdir, dev, dt)
            if warm: warm.join()
            holder["dev"] = dev
            holder["ready"] = time.perf_counter() - T0
        except BaseException as e:                        # surface load failures in main
            holder["err"] = e
    th = threading.Thread(target=build, daemon=True); th.start()

    try:
        row_ids, encs, dp = (run_datapath_wtok(inp, mdir, ml, pool) if (WTOK and pool is not None)
                             else run_datapath(inp, mdir, ml, pool))
    finally:
        if pool is not None: pool.terminate()
    # STAGE accounting: read+tokenize must sum to the row-scaled datapath wall; the fixed
    # tokenizer-file load is excluded (it lands in bench_e2e's fixed-startup residual).
    t_read = max(dp["total"] - dp["tok_load"] - dp["encode"], 0.0); t_tok = dp["encode"]
    lens = [len(e.ids) for e in encs]
    log(f"data path [{dp['mode']}] {len(row_ids):,} rows {dp['total']:.3f}s "
        f"(tok load {dp['tok_load']:.3f} + encode {t_tok:.3f} + parse/normalise/wait {t_read:.3f})  "
        f"mean len {np.mean(lens):.1f}  p95 {np.percentile(lens,95):.0f}")

    th.join()
    if "err" in holder: raise holder["err"]
    m = holder["m"]; t_model = time.perf_counter() - T0
    log(f"model ready on {holder['dev']} at +{holder['ready']:.3f}s (torch import {T_TORCH_IMPORT:.3f}s; "
        f"{m.NL}L x {m.H}H, {m.num_labels} labels, pooler={m.pool[0] is not None})")

    fb = {}
    t = time.perf_counter()
    logits = infer(m, encs, br, tb, ovl, on_first_batch=lambda: fb.setdefault("t", time.perf_counter() - T0))
    t_inf = time.perf_counter() - t
    log(f"inference {t_inf:.3f}s  ({len(row_ids)/max(t_inf,1e-9):,.0f} rows/s)  first batch at +{fb.get('t',0):.3f}s")

    t = time.perf_counter(); write_csv(outp, row_ids, decide(logits, mdir, m.cfg, input_csv=inp)); t_wr = time.perf_counter() - t
    tot = time.perf_counter() - T0
    log(f"wrote {outp}  {t_wr:.3f}s")
    print(f"STAGE imports={T_IMPORTS:.3f} read={t_read:.3f} tokenize={t_tok:.3f} model_ready={t_model:.3f} "
          f"first_batch={fb.get('t',0):.3f} infer={t_inf:.3f} write={t_wr:.3f} total={tot:.3f} "
          f"rows={len(row_ids)} sps={len(row_ids)/tot:.1f}", flush=True)

if __name__ == "__main__":
    main(sys.argv[1:])
