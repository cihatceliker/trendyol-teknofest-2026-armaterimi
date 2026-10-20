"""Serving path for checkpoints the lean BERT loop cannot run (e.g. TY-ecomm-embed = GTE "New"
architecture, custom remote code). Apples-to-apples with lean_infer.py: the SAME v2 text
normalisation, the SAME Rust tokenizer, the SAME length-bucketed batching, the SAME decision
rule and CSV writer — everything is imported from lean_infer so the two paths cannot drift.
Only the model forward differs (transformers instead of the hand-rolled BERT).

v3 (2026-08-24) startup + data-path optimisation (CPU-verified; GPU deltas in README_SPEED.md):
  * lean_infer is a light import now (torch deferred) — this process reaches the data path in
    ~0.3s instead of ~5.9s of serial torch+transformers imports.
  * normalisation workers are forked FIRST (before any thread/CUDA; they never import torch),
    then torch+transformers import AND the checkpoint load run on a background thread while the
    main thread streams the CSV: parse (feeder thread) -> normalise (workers) -> encode_batch
    (Rust, GIL released). At 7.5k rows the whole data path hides under the imports.
  * the model class is resolved directly from the checkpoint's LOCAL auto_map entry via
    transformers.dynamic_module_utils (skips the AutoModel registry chain, ~1.9s cold);
    SERVE_AUTO=1 forces the plain AutoModelForSequenceClassification path, which also remains
    the automatic fallback (and the path taken by non-remote-code checkpoints).
  * on CUDA the context is warmed by a tiny thread concurrent with the transformers import
    (UNVERIFIED on GPU: structure CPU-tested, the overlap saving is not).
  * HF offline/telemetry env vars are set (setdefault, overridable): the model dir is always
    local; the hub must never be consulted under the scoring clock.

CLI: serve_hf.py <input_csv> <output_path> --model-dir <dir>
Env: DEVICE, BATCH_ROWS(128), TOKEN_BUDGET(32768), MAXLEN(256), SERVE_AUTO,
     LEAN_WORKERS / LEAN_CHUNK / LEAN_SERIAL_READ (data path, see lean_infer.py)
"""
import sys, os, time
T0 = time.perf_counter()
os.environ.setdefault("TOKENIZERS_PARALLELISM", "true")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
import json, threading
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path: sys.path.insert(0, _HERE)
import lean_infer as L      # light since v3; a REAL import so pool workers can unpickle L._norm_chunk
T_IMPORTS = time.perf_counter() - T0

def log(m): print(f"[serve_hf +{time.perf_counter()-T0:7.3f}s] {m}", flush=True)

def _build_model(mdir, dev_req, out):
    """Background thread: torch + transformers imports, CUDA context warm-up, checkpoint load,
    move to device + eval. Fills `out`; exceptions are re-raised by main after join."""
    try:
        t = time.perf_counter(); import torch; out["t_torch"] = time.perf_counter() - t
        torch.set_grad_enabled(False)              # thread-local; covers the load below
        dev = dev_req or ("cuda" if torch.cuda.is_available() else "cpu")
        dt  = torch.bfloat16 if dev.startswith("cuda") else torch.float32
        warm = None
        if dev.startswith("cuda"):                 # context init overlaps the transformers import
            warm = threading.Thread(target=lambda: torch.zeros(1, device=dev), daemon=True)
            warm.start()
        t = time.perf_counter()
        m = None
        if not os.environ.get("SERVE_AUTO"):
            try:
                cfg = json.load(open(os.path.join(mdir, "config.json")))
                ref = (cfg.get("auto_map") or {}).get("AutoModelForSequenceClassification")
                if ref and "--" not in ref:        # local remote-code ref, e.g. "modeling.NewForSequenceClassification"
                    from transformers.dynamic_module_utils import get_class_from_dynamic_module
                    cls = get_class_from_dynamic_module(ref, mdir)
                    m = cls.from_pretrained(mdir, dtype=dt)
            except Exception as e:
                log(f"direct auto_map load failed ({type(e).__name__}: {e}) -> AutoModel fallback")
                m = None
        if m is None:
            from transformers import AutoModelForSequenceClassification
            m = AutoModelForSequenceClassification.from_pretrained(mdir, trust_remote_code=True, dtype=dt)
        out["t_tf_load"] = time.perf_counter() - t
        if warm: warm.join()
        t = time.perf_counter(); out["m"] = m.to(dev).eval(); out["t_to_dev"] = time.perf_counter() - t
        out["dev"] = dev
        out["ready"] = time.perf_counter() - T0
    except BaseException as e:
        out["err"] = e

def main(argv):
    pos = [a for a in argv if not a.startswith("--")]
    kw = dict(a.lstrip("-").split("=", 1) if "=" in a else (a.lstrip("-"), "") for a in argv if a.startswith("--"))
    if len(pos) != 2 or "model-dir" not in kw:
        sys.exit("usage: serve_hf.py <input_csv> <output_path> --model-dir <dir>")
    inp, outp, mdir = pos[0], pos[1], kw["model-dir"]
    dev_req = kw.get("device") or os.environ.get("DEVICE") or ""   # "" -> resolved on build thread
    br = int(kw.get("batch-rows") or os.environ.get("BATCH_ROWS", 128))
    tb = int(kw.get("token-budget") or os.environ.get("TOKEN_BUDGET", 32768))
    ml = int(kw.get("max-len") or os.environ.get("MAXLEN", 256))
    log(f"light imports {T_IMPORTS:.3f}s; torch/transformers deferred to build thread")

    pool = None if os.environ.get("LEAN_SERIAL_READ") else L.make_pool()   # fork BEFORE threads
    out = {}
    th = threading.Thread(target=_build_model, args=(mdir, dev_req, out), daemon=True); th.start()

    try:
        row_ids, encs, dp = L.run_datapath(inp, mdir, ml, pool)
    finally:
        if pool is not None: pool.terminate()
    # read+tokenize sum to the row-scaled datapath wall; the fixed tokenizer-file load is
    # excluded so bench_e2e counts it in the fixed-startup residual (see lean_infer.main).
    t_read = max(dp["total"] - dp["tok_load"] - dp["encode"], 0.0); t_tok = dp["encode"]
    log(f"data path [{dp['mode']}] {len(row_ids):,} rows {dp['total']:.3f}s "
        f"(tok load {dp['tok_load']:.3f} + encode {t_tok:.3f} + parse/normalise/wait {t_read:.3f})  "
        f"mean len {np.mean([len(e.ids) for e in encs]):.1f}")

    th.join()
    if "err" in out: raise out["err"]
    m, dev = out["m"], out["dev"]
    t_model = time.perf_counter() - T0
    import torch                                   # already loaded by the build thread
    log(f"model ready on {dev} at +{out['ready']:.3f}s ({type(m).__name__}, {m.config.num_labels} labels; "
        f"torch {out['t_torch']:.2f}s, tf+load {out['t_tf_load']:.2f}s, to-device {out['t_to_dev']:.2f}s)")

    n_lab = m.config.num_labels
    lens = np.array([len(e.ids) for e in encs])
    batches = L.make_batches(lens, br, tb)
    logits = np.empty((len(encs), n_lab), dtype=np.float32)
    accepts_tt = "token_type_ids" in getattr(m, "forward").__code__.co_varnames
    first_batch = {}
    t = time.perf_counter()
    with torch.inference_mode():
        for idx in batches:
            ids, tt, mk = L._assemble(encs, idx, dev.startswith("cuda"))
            kwargs = dict(input_ids=ids.to(dev, non_blocking=True), attention_mask=mk.to(dev, non_blocking=True))
            if accepts_tt: kwargs["token_type_ids"] = tt.to(dev, non_blocking=True)
            logits[idx] = m(**kwargs).logits.float().cpu().numpy()
            first_batch.setdefault("t", time.perf_counter() - T0)
    if dev.startswith("cuda"): torch.cuda.synchronize()
    t_inf = time.perf_counter() - t
    log(f"inference {t_inf:.3f}s ({len(row_ids)/max(t_inf,1e-9):,.0f} rows/s)")

    t = time.perf_counter(); L.write_csv(outp, row_ids, L.decide(logits, mdir, json.load(open(os.path.join(mdir, "config.json"))), input_csv=inp)); t_wr = time.perf_counter() - t
    tot = time.perf_counter() - T0
    print(f"STAGE imports={T_IMPORTS:.3f} read={t_read:.3f} tokenize={t_tok:.3f} model_ready={t_model:.3f} "
          f"first_batch={first_batch.get('t',0):.3f} infer={t_inf:.3f} write={t_wr:.3f} total={tot:.3f} "
          f"rows={len(row_ids)} sps={len(row_ids)/tot:.1f}", flush=True)

if __name__ == "__main__":
    main(sys.argv[1:])
