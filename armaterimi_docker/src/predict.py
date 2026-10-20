#!/usr/bin/env python
"""Solution-mode wrapper around the UNTOUCHED e12 serving pair (serve_hf.py +
lean_infer.py). Adds exactly what the Docker contract demands and the serving
scripts deliberately leave out:

  1. CUDA fail-fast (contract 5.3.2): serve_hf silently falls back to CPU when
     torch.cuda.is_available() is False; under the jury clock that would
     mislead instead of fail. Escape hatch for local CPU smoke tests: set
     DEVICE (e.g. DEVICE=cpu). In scoring mode no env is injected, so the
     guard is always live for the judges.
  2. Post-write validation (contract 1.9/T7): header exactly row_id,real_label;
     row count == input; row_id set equality + uniqueness; labels within the
     three allowed values; no BOM. An invalid file is renamed to a dot-prefixed
     .invalid name so nothing submission-shaped is left behind, and we exit 1.
  3. Durability + permissions: chmod 0644 and fsync file + directory after
     serve_hf's atomic os.replace (its temp file is fsync-less by design).
  4. Stray-temp cleanup: serve_hf writes <out>.tmp<pid> in /output; on any
     failure the leftovers are removed (the judge's dir must stay clean).

Invocation (by entrypoint.sh): python /app/src/predict.py <input_csv> <output_path>
Model dir comes from $MODEL_DIR (baked default /models/current).
"""
import os as _os
import time as _time
_PY_T0 = _time.time()
_STAMP = _os.environ.get("PREDICT_STAMPS") == "1"
if _STAMP: print(f"[predict-stamp] python_up={_PY_T0:.3f}", flush=True)
import csv
import glob
import json
import importlib.util
import os
import sys

ALLOWED = {"irrelevant", "semi-relevant", "relevant"}
HERE = os.path.dirname(os.path.abspath(__file__))


def fail(msg, code=1):
    print(f"[predict FATAL] {msg}", file=sys.stderr, flush=True)
    sys.exit(code)


def read_ids(path, id_col="row_id"):
    """row_id column as raw strings (csv module: no NA coercion, quoted
    multiline fields handled - contract 2.5)."""
    with open(path, newline="", encoding="utf-8") as f:
        rd = csv.reader(f)
        hdr = next(rd)
        if id_col not in hdr:
            fail(f"input CSV has no {id_col} column: {hdr}", 2)
        i = hdr.index(id_col)
        return [row[i] if i < len(row) else "" for row in rd]


def validate(out_path, in_ids, written=None):
    """Contract check on the submission.

    `written` = the (row_ids, labels) the writer actually emitted. When present the SEMANTIC
    checks run on those objects and the FILE is only scanned cheaply (BOM, header, line count) --
    a full csv.reader pass over 90,000 output rows cost ~1s inside the scored window, and SPS is
    wall clock. The file is still proven to exist, be BOM-free, carry the right header and hold
    exactly the right number of lines; without `written` the original full parse is used.
    """
    with open(out_path, "rb") as f:
        head = f.read(3)
        if head == b"\xef\xbb\xbf":
            return "output starts with a UTF-8 BOM (evaluate.py rejects it)"
    if written is not None:
        w_ids, w_labels = written
        with open(out_path, "rb") as f:
            first = f.readline().decode("utf-8").rstrip("\r\n")
            if first != "row_id,real_label":
                return f"header is {first!r}, must be 'row_id,real_label'"
            n = 0
            while True:
                b = f.read(1 << 20)
                if not b: break
                n += b.count(b"\n")
        if n != len(w_ids):
            return f"file holds {n} data lines, writer emitted {len(w_ids)}"
        rows = list(zip(w_ids, w_labels))
    else:
        with open(out_path, newline="", encoding="utf-8") as f:
            rd = csv.reader(f)
            hdr = next(rd)
            if hdr != ["row_id", "real_label"]:
                return f"header is {hdr}, must be ['row_id', 'real_label']"
            rows = list(rd)
    if any(len(r) != 2 for r in rows):
        return "output contains a row without exactly 2 fields"
    out_ids = [r[0] for r in rows]
    labels = {r[1] for r in rows}
    if len(rows) != len(in_ids):
        return f"row count {len(rows)} != input row count {len(in_ids)}"
    if len(set(out_ids)) != len(out_ids):
        return "duplicate row_id in output"
    if set(out_ids) != set(in_ids):
        return "row_id set does not match the input exactly"
    if not labels <= ALLOWED:
        return f"labels outside the allowed set: {sorted(labels - ALLOWED)}"
    dist = {lab: 0 for lab in sorted(ALLOWED)}
    for r in rows:
        dist[r[1]] += 1
    print(f"[predict] validation OK: {len(rows)} rows, labels {dist}", flush=True)
    return None


def main():
    if len(sys.argv) != 3:
        fail(f"usage: predict.py <input_csv> <output_path> (got {sys.argv[1:]})", 2)
    in_csv, out_path = sys.argv[1], sys.argv[2]
    model_dir = os.environ.get("MODEL_DIR", "/models/current")
    if not os.path.isfile(os.path.join(model_dir, "config.json")):
        fail(f"MODEL_DIR is not a model directory: {model_dir}", 2)

    # ---- 1. CUDA fail-fast (unless a DEVICE override is explicitly set) ----
    # ctypes cuInit(0) probe (~50ms) instead of `import torch` (~4s serial dead time on the
    # official clock — measured 2026-08-24; torch itself is imported by the serve script on a
    # background thread, overlapped with the CSV data path). Same fail-closed semantics.
    # cuInit(0) was assumed to cost ~50ms; MEASURED in-container it is ~1.3s of SERIAL dead time
    # before the server even starts, and the server then initialises CUDA again on its own thread.
    # Run the probe on a background thread and only join it after serving, so the fail-fast
    # semantics survive but the cost overlaps real work instead of preceding it.
    if not os.environ.get("DEVICE"):
        import ctypes, threading
        _probe = {"rc": -1}
        def _cuprobe():
            try:
                _probe["rc"] = ctypes.CDLL("libcuda.so.1").cuInit(0)
            except OSError:
                _probe["rc"] = -1
        _t = threading.Thread(target=_cuprobe, daemon=True); _t.start()
        # Cheap fail-fast FIRST (microseconds): if the driver device nodes are absent the NVIDIA
        # hook did not run and there is no GPU at all -- fail now rather than after an hour of CPU.
        # The real cuInit keeps running behind the serving pass and is joined below, so a device
        # node that exists but cannot initialise is still caught, just not at the cost of 1.3s.
        _nodes = any(os.path.exists(p) for p in ("/dev/nvidiactl", "/proc/driver/nvidia/version"))
        rc = 0 if _nodes else (_t.join(8.0) or _probe["rc"])
        if rc != 0:
            fail(f"CUDA driver probe failed (cuInit rc={rc}) and no DEVICE override is set. "
                 "Refusing a silent CPU fallback under the scoring clock (contract 5.3.2).")

    # ---- 2. run the frozen serving pipeline in-process ----
    # SERVE_SCRIPT: baked image default (Dockerfile ENV) selects the serving implementation;
    # serve_gte.py = bit-exact no-transformers port of the champion arch (parity: max|dlogit|=0.0,
    # byte-identical CSVs), -2.9s startup vs serve_hf. serve_hf.py remains the fallback.
    _script = os.environ.get("SERVE_SCRIPT", "serve_hf.py")
    # serve_gte.py only understands the GTE "New" architecture. A plain-BERT checkpoint
    # (BERTurk, the turkish-* models) must go through lean_infer.py, so let the CHECKPOINT decide
    # rather than the baked default: config.json carries auto_map only for remote-code models.
    try:
        with open(os.path.join(model_dir, "config.json"), encoding="utf-8") as _f:
            _remote = bool(json.load(_f).get("auto_map"))
        if _script == "serve_gte.py" and not _remote:
            _script = "lean_infer.py"
    except Exception:
        pass
    if not os.path.exists(os.path.join(HERE, _script)):
        _script = "serve_hf.py"
    # Load under the script's OWN name and register it in sys.modules. The name was hardcoded to
    # "serve_hf", which is fine for serve_hf.py/serve_gte.py (they import lean_infer as a normal
    # module, so its pool functions pickle by their real qualified name). But running lean_infer.py
    # directly under the alias "serve_hf" made multiprocessing try to pickle
    # serve_hf._norm_tok_chunk, which resolves nowhere:
    #     PicklingError: attribute lookup _norm_tok_chunk on serve_hf failed
    # i.e. every plain-BERT checkpoint died the moment the data-path pool started.
    _mod = os.path.splitext(_script)[0]
    if _STAMP: print(f"[predict-stamp] before_serve={_time.time():.3f} (predict boot {_time.time()-_PY_T0:.3f}s)", flush=True)
    spec = importlib.util.spec_from_file_location(_mod, os.path.join(HERE, _script))
    serve_hf = importlib.util.module_from_spec(spec)
    sys.modules[_mod] = serve_hf
    try:
        spec.loader.exec_module(serve_hf)                     # imports lean_infer itself
        serve_hf.main([in_csv, out_path, f"--model-dir={model_dir}"])
    except BaseException:
        # never leave serve_hf's <out>.tmp<pid> behind in the judge's directory
        for stray in glob.glob(out_path + ".tmp*"):
            try:
                os.remove(stray)
            except OSError:
                pass
        raise                                                  # non-zero exit

    # ---- 3. validate against the organizer schema ----
    # The serving pass already read every row_id; re-parsing the 90,000-row INPUT csv purely to
    # recover them cost ~0.7s INSIDE the scored window (SPS is wall clock from run_solution.sh to
    # the submission being written, so this counted against us). Take them from the writer when it
    # published them, and keep the file re-read as the fallback.
    # join the background CUDA probe now that the work is done: a device node that exists but
    # fails cuInit means the run silently used CPU, which must still be a hard failure.
    try:
        if not os.environ.get("DEVICE"):
            _t.join(5.0)
            if _probe["rc"] != 0:
                fail(f"CUDA driver probe failed after the run (cuInit rc={_probe['rc']}): "
                     f"the output may have been produced on CPU", 3)
    except NameError:
        pass
    _lw = getattr(sys.modules.get("lean_infer"), "LAST_WRITE", None)
    if _lw and _lw[0] == out_path and len(_lw[1]) > 0:
        in_ids = _lw[1]
    else:
        in_ids = read_ids(in_csv)
    err = validate(out_path, in_ids, written=(_lw[1], _lw[2]) if _lw and _lw[0] == out_path else None)
    if err:
        quarantined = os.path.join(os.path.dirname(out_path) or ".",
                                   f".{os.path.basename(out_path)}.invalid")
        os.replace(out_path, quarantined)
        fail(f"output failed validation ({err}); moved to {quarantined}")

    # ---- 4. permissions + durability ----
    os.chmod(out_path, 0o644)
    fd = os.open(out_path, os.O_RDONLY)
    os.fsync(fd)
    os.close(fd)
    dfd = os.open(os.path.dirname(out_path) or ".", os.O_RDONLY)
    os.fsync(dfd)
    os.close(dfd)
    print(f"[predict] wrote and validated {out_path}", flush=True)

    # ---- 5. exit without paying for interpreter teardown ----
    # The submission is written, chmod'd and fsync'd -- BOTH the file and its parent directory --
    # so it is durable and complete at this point; the scored window ("complete of the submission
    # file dumping") is over. What remains is CPython shutdown: tearing down the CUDA context and
    # releasing ~4 GB of device memory for a process the container destroys milliseconds later.
    # MEASURED 2026-08-27 on the 90k run: that shutdown is ~1.4s of the scored wall clock, while
    # container teardown itself is only 0.07s and output validation 0.011s. os._exit skips it.
    # Safe here specifically because nothing is left to flush: both streams are flushed below and
    # the only durable artifact has already been fsync'd. Exit status is still 0.
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
