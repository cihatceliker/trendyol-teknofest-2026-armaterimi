"""Serving loop for GTE "new-impl" checkpoints (NewForSequenceClassification) WITHOUT
transformers: same orchestration as serve_hf.py, but the forward is lean_gte.LeanGTE
(torch + safetensors only), so process start pays neither the ~5.8s transformers import nor
its ~4s dynamic-module model init.

Apples-to-apples with lean_infer.py/serve_hf.py: the SAME v2 text normalisation, the SAME Rust
tokenizer, the SAME length-bucketed batching (lean_infer.infer with producer-thread batch
assembly), the SAME decision rule and CSV writer — everything is imported from lean_infer so
the paths cannot drift. Parity vs transformers: scripts/verify_parity_gte.py (CPU fp32,
max|Δlogit| and 100% argmax agreement on public_test rows).

Startup mirrors lean_infer.main: normalisation workers are forked FIRST (before any
thread/CUDA; they never import torch), then torch import + CUDA context warm-up + weight load
run on a background thread while the main thread streams the CSV (parse -> normalise ->
encode_batch, Rust releases the GIL). bf16 on cuda, fp32 on cpu.

CLI: serve_gte.py <input_csv> <output_path> --model-dir <dir>
     model dir holds model.safetensors + config.json + tokenizer.json [+ decision.json]
Env: DEVICE, BATCH_ROWS(128), TOKEN_BUDGET(32768), MAXLEN(256), LEAN_NO_OVERLAP,
     LEAN_WORKERS / LEAN_CHUNK / LEAN_SERIAL_READ (data path, see lean_infer.py)
Deterministic: no sampling, argmax ties -> lowest class id, stable length sort.
"""
import sys, os, time
T0 = time.perf_counter()
os.environ.setdefault("TOKENIZERS_PARALLELISM", "true")
import threading
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path: sys.path.insert(0, _HERE)
import lean_infer as L      # light import; a REAL import so pool workers can unpickle L._norm_chunk
import lean_gte as G        # light import; torch deferred to the build thread
T_IMPORTS = time.perf_counter() - T0

def log(m): print(f"[serve_gte +{time.perf_counter()-T0:7.3f}s] {m}", flush=True)

def _build_model(mdir, dev_req, out):
    """Background thread: torch/safetensors import, CUDA context warm-up, weights -> device.
    Fills `out`; exceptions are re-raised by main after join."""
    try:
        t = time.perf_counter()
        G._ensure_torch()
        out["t_torch"] = time.perf_counter() - t
        torch = G.torch
        dev = dev_req or ("cuda" if torch.cuda.is_available() else "cpu")
        dt  = torch.bfloat16 if dev.startswith("cuda") else torch.float32
        warm = None
        if dev.startswith("cuda"):                 # context init overlaps the weight read
            warm = threading.Thread(target=lambda: torch.zeros(1, device=dev), daemon=True)
            warm.start()
        t = time.perf_counter()
        out["m"] = G.load_gte(mdir, dev, dt)
        out["t_load"] = time.perf_counter() - t
        if warm: warm.join()
        out["dev"] = dev
        out["ready"] = time.perf_counter() - T0
    except BaseException as e:
        out["err"] = e

def main(argv):
    pos = [a for a in argv if not a.startswith("--")]
    kw = dict(a.lstrip("-").split("=", 1) if "=" in a else (a.lstrip("-"), "") for a in argv if a.startswith("--"))
    if len(pos) != 2 or "model-dir" not in kw:
        sys.exit("usage: serve_gte.py <input_csv> <output_path> --model-dir <dir> "
                 "[--device=cuda] [--batch-rows=128] [--token-budget=32768] [--no-overlap]")
    inp, outp, mdir = pos[0], pos[1], kw["model-dir"]
    dev_req = kw.get("device") or os.environ.get("DEVICE") or ""   # "" -> resolved on build thread
    br  = int(kw.get("batch-rows")   or os.environ.get("BATCH_ROWS", 128))
    tb  = int(kw.get("token-budget") or os.environ.get("TOKEN_BUDGET", 32768))
    ml  = int(kw.get("max-len")      or os.environ.get("MAXLEN", 256))
    ovl = "no-overlap" not in kw and not os.environ.get("LEAN_NO_OVERLAP")
    log(f"light imports {T_IMPORTS:.3f}s; torch deferred to build thread")

    # WTOK=1 (default): tokenise inside the pool workers. The tokenizer must be loaded BEFORE the
    # fork so children inherit it copy-on-write, and the fork must still precede any thread.
    WTOK = os.environ.get("WTOK", "1") == "1" and not os.environ.get("LEAN_SERIAL_READ")
    if WTOK: L.preload_tokenizer(mdir, ml)
    pool = None if os.environ.get("LEAN_SERIAL_READ") else L.make_pool()   # fork BEFORE threads
    out = {}
    th = threading.Thread(target=_build_model, args=(mdir, dev_req, out), daemon=True); th.start()

    # STREAM=1 (default): tokenise in groups and run each group on the device as soon as it is
    # ready, so the data path hides behind inference instead of running before it. At 90k the
    # serial order left the GPU idle for the whole ~11s data path. STREAM=0 restores the old
    # materialise-everything path (kept for A/B and as a fallback).
    STREAM = os.environ.get("STREAM", "0") == "1"   # default OFF: +1.5% only, and it changes batch
    # composition, which flips ~0.1% of labels through padding float-noise on our narrow cut bands.
    if STREAM:
        model_ready = threading.Event()
        def _await_model():
            th.join(); model_ready.set()
        threading.Thread(target=_await_model, daemon=True).start()
        row_ids, parts, t_dp0, fb = [], [], time.perf_counter(), {}
        n_groups = 0
        try:
            for gid, genc in L.run_datapath_streaming(inp, mdir, ml, pool, model_ready):
                if "err" in out: raise out["err"]
                mm = out["m"]; n_groups += 1
                lg = L.infer(mm, genc, br, tb, ovl,
                             on_first_batch=(lambda: fb.setdefault("t", time.perf_counter() - T0)))
                row_ids.extend(gid); parts.append(lg)
        finally:
            if pool is not None: pool.terminate()
        th.join()
        if "err" in out: raise out["err"]
        m = out["m"]; t_model = time.perf_counter() - T0
        logits = np.concatenate(parts, 0) if len(parts) > 1 else parts[0]
        t_stream = time.perf_counter() - t_dp0
        log(f"streamed data path + inference: {len(row_ids):,} rows in {n_groups} group(s), "
            f"{t_stream:.3f}s combined (model ready at +{out['ready']:.3f}s)")
        t_read = 0.0; t_tok = 0.0; t_inf = t_stream
        log(f"inference (overlapped) {t_inf:.3f}s ({len(row_ids)/max(t_inf,1e-9):,.0f} rows/s) "
            f"first batch at +{fb.get('t',0):.3f}s")
        t = time.perf_counter(); L.write_csv(outp, row_ids, L.decide(logits, mdir, m.cfg, input_csv=inp)); t_wr = time.perf_counter() - t
        tot = time.perf_counter() - T0
        log(f"wrote {outp}  {t_wr:.3f}s")
        print(f"STAGE imports={T_IMPORTS:.3f} read={t_read:.3f} tokenize={t_tok:.3f} model_ready={t_model:.3f} "
              f"first_batch={fb.get('t',0):.3f} infer={t_inf:.3f} write={t_wr:.3f} total={tot:.3f} "
              f"rows={len(row_ids)} sps={len(row_ids)/tot:.1f}", flush=True)
        return

    try:
        row_ids, encs, dp = (L.run_datapath_wtok(inp, mdir, ml, pool) if (WTOK and pool is not None)
                             else L.run_datapath(inp, mdir, ml, pool))
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
    m = out["m"]; t_model = time.perf_counter() - T0
    log(f"model ready on {out['dev']} at +{out['ready']:.3f}s (torch import {out['t_torch']:.2f}s, "
        f"load {out['t_load']:.2f}s; GTE {m.NL}L x {m.H}H, {m.num_labels} labels, "
        f"pooler={m.pool[0] is not None})")

    fb = {}
    t = time.perf_counter()
    logits = L.infer(m, encs, br, tb, ovl, on_first_batch=lambda: fb.setdefault("t", time.perf_counter() - T0))
    t_inf = time.perf_counter() - t
    log(f"inference {t_inf:.3f}s  ({len(row_ids)/max(t_inf,1e-9):,.0f} rows/s)  first batch at +{fb.get('t',0):.3f}s")

    t = time.perf_counter(); L.write_csv(outp, row_ids, L.decide(logits, mdir, m.cfg, input_csv=inp)); t_wr = time.perf_counter() - t
    tot = time.perf_counter() - T0
    log(f"wrote {outp}  {t_wr:.3f}s")
    print(f"STAGE imports={T_IMPORTS:.3f} read={t_read:.3f} tokenize={t_tok:.3f} model_ready={t_model:.3f} "
          f"first_batch={fb.get('t',0):.3f} infer={t_inf:.3f} write={t_wr:.3f} total={tot:.3f} "
          f"rows={len(row_ids)} sps={len(row_ids)/tot:.1f}", flush=True)

if __name__ == "__main__":
    main(sys.argv[1:])
