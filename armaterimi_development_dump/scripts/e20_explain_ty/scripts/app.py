#!/usr/bin/env python3
"""e20 — CHAMPION explainability HTTP server (stdlib http.server; fastapi is NOT in .venv-gcp).

Serves the jury-facing single-page UI (static/index.html, fully inlined, no CDN) and a JSON
API over the public test set + THE shipped predictor (e15 ckpt_gf2_berturk_6ep via
model_adapter.ChampionAdapter):

  GET  /                      the UI
  GET  /healthz               200 immediately (before any model work; tunnel comes up first)
  GET  /readyz                200 when the model is loaded, else 503 (the UI polls this)
  GET  /api/rows?q=&limit=    searchable row_id + query listing (substring, case-insensitive)
  GET  /api/row/<row_id>      the product's visible fields, raw AND normalised-for-the-model
  GET  /api/random            one random public-test row (same shape as /api/row + row_id)
  POST /api/explain           {"row_id"?, "query"?, "fields"?, "ig_steps"?} -> prediction +
                              rationale payload. `fields` (raw manual product fields) wins
                              over row_id — this is the arbitrary-manual-input path; the raw
                              fields go through the SAME normalize+v2 pipeline as the CSV.

Design constraints (ORGANIZER_REQUIREMENTS E1-E13, R8): the ngrok URL must be live within 10
minutes of container start, so the server binds and answers /healthz BEFORE loading the model
(model loads on a background thread; /api/explain returns 503 with state until ready).
--require-token makes the process fail closed (exit 2) without NGROK_AUTHTOKEN — the container
entrypoint passes it in serve-ngrok mode; local dev omits it. The token is never printed.
No outbound network calls anywhere in this app (HF offline env is set by the adapter).

Usage:
  app.py <public_test_csv> --model-dir <ckpt_dir> [--host 127.0.0.1] [--port 7862]
         [--device cpu] [--ig-steps 16] [--max-len 256] [--require-token]
"""
from __future__ import annotations

import os
import sys

# CPU-only guard: hide the GPU (it is training) unless cuda was explicitly requested.
_dev = "cpu"
for _i, _a in enumerate(sys.argv):
    if _a == "--device" and _i + 1 < len(sys.argv):
        _dev = sys.argv[_i + 1]
    elif _a.startswith("--device="):
        _dev = _a.split("=", 1)[1]
if not _dev.startswith("cuda"):
    os.environ["CUDA_VISIBLE_DEVICES"] = ""

import argparse            # noqa: E402
import csv                 # noqa: E402
import json                # noqa: E402
import random              # noqa: E402
import re                  # noqa: E402
import threading           # noqa: E402
import time                # noqa: E402
import traceback           # noqa: E402

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer   # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
STATIC_DIR = os.path.join(os.path.dirname(HERE), "static")
sys.path.insert(0, HERE)

import model_adapter as MA                                   # noqa: E402
from model_adapter import ChampionAdapter, RAW_COLS, t1      # noqa: E402

MAX_BODY = 256 * 1024
MAX_FIELD_CHARS = 6000
T0 = time.time()


def log(msg: str) -> None:
    print(f"[e20 +{time.time()-T0:7.2f}s] {msg}", flush=True)


# ---------------------------------------------------------------- shared state
class State:
    def __init__(self) -> None:
        self.rows: dict[str, dict] = {}     # row_id -> {"raw": {...}, "norm": {...}, "doc": str}
        self.order: list[str] = []
        self.search: list[tuple[str, str, str]] = []   # (haystack, row_id, raw query)
        self.adapter: ChampionAdapter | None = None
        self.model_error: str | None = None
        self.model_load_s: float | None = None
        self.lock = threading.Lock()        # serialises model calls
        self.args: argparse.Namespace | None = None


S = State()


def load_rows(path: str) -> None:
    import pandas as pd
    t = time.time()
    raws, rids = [], []
    with open(path, newline="", encoding="utf-8") as f:
        rd = csv.DictReader(f)
        if rd.fieldnames is None or "row_id" not in rd.fieldnames:
            raise SystemExit(f"public test CSV lacks a row_id column: {path}")
        for r in rd:
            rids.append(r.get("row_id", ""))
            raws.append({c: (r.get(c) or "") for c in RAW_COLS})
    norm = MA.normalize_frame(pd.DataFrame(raws, dtype=str)).to_dict("records")
    for rid, raw, nf in zip(rids, raws, norm):
        nf = {c: str(nf[c]) for c in RAW_COLS}
        doc, _spans = MA.build_doc(nf)
        S.rows[rid] = {"raw": raw, "norm": nf, "doc": doc}
        S.order.append(rid)
        S.search.append(((rid + " " + raw["query"] + " " + raw["title"]).casefold(), rid, raw["query"]))
    log(f"loaded + normalised {len(S.order):,} rows from {path} in {time.time()-t:.1f}s "
        f"(normalize.py T1-T4 + e18 apply_v2 — the serving pipeline)")


def load_model() -> None:
    a = S.args
    try:
        t = time.time()
        adapter = ChampionAdapter(a.model_dir, device=a.device, max_len=a.max_len)
        # warm-up: one tiny explain so the first juror click is not the slow one
        adapter.explain("deneme", {"title": "deneme", "brand": "", "category": "",
                                   "gender": "unknown", "age_group": "unknown",
                                   "attributes": "unknown"},
                        ig_steps=2, counterfactuals=False)
        S.adapter = adapter
        S.model_load_s = time.time() - t
        log(f"model ready on {a.device} in {S.model_load_s:.1f}s "
            f"(head={adapter.head}, cuts={adapter.cuts}, labels={adapter.labels}, dir={a.model_dir})")
    except Exception:
        S.model_error = traceback.format_exc(limit=6)
        log(f"MODEL LOAD FAILED:\n{S.model_error}")


def row_payload(rid: str) -> dict:
    row = S.rows[rid]
    out = {"row_id": rid, "raw": row["raw"], "normalized": row["norm"], "doc": row["doc"]}
    if row.get("truth"): out["truth"] = row["truth"]
    return out


# ---------------------------------------------------------------- HTTP handler
ROW_RE = re.compile(r"^/api/row/([^/?#]+)$")


class Handler(BaseHTTPRequestHandler):
    server_version = "e20-explain/1.0"
    protocol_version = "HTTP/1.1"

    # -- plumbing ------------------------------------------------------------
    def _send(self, code: int, body: bytes, ctype: str) -> None:
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _json(self, code: int, obj) -> None:
        self._send(code, json.dumps(obj, ensure_ascii=False).encode("utf-8"),
                   "application/json; charset=utf-8")

    def log_message(self, fmt: str, *args) -> None:
        log(f"{self.address_string()} {fmt % args}")

    def _query_params(self) -> dict:
        if "?" not in self.path:
            return {}
        from urllib.parse import parse_qs, urlparse
        return {k: v[0] for k, v in parse_qs(urlparse(self.path).query).items()}

    # -- GET -----------------------------------------------------------------
    def do_GET(self) -> None:   # noqa: N802
        path = self.path.split("?", 1)[0]
        if path == "/healthz":
            self._json(200, {"ok": True, "rows": len(S.order),
                             "model": "ready" if S.adapter else ("error" if S.model_error else "loading"),
                             "uptime_s": round(time.time() - T0, 1)})
        elif path == "/readyz":
            if S.adapter:
                self._json(200, {"ready": True, "model_load_s": round(S.model_load_s or 0, 1),
                                 "head": S.adapter.head, "cuts": S.adapter.cuts,
                                 "labels": S.adapter.labels, "device": str(S.adapter.device),
                                 "model_dir": S.adapter.model_dir})
            else:
                self._json(503, {"ready": False,
                                 "state": "error" if S.model_error else "loading",
                                 "error": S.model_error})
        elif path == "/":
            try:
                with open(os.path.join(STATIC_DIR, "index.html"), "rb") as f:
                    self._send(200, f.read(), "text/html; charset=utf-8")
            except OSError:
                self._json(500, {"error": "static/index.html missing"})
        elif path == "/api/rows":
            self._api_rows()
        elif path == "/api/random":
            if not S.order:
                self._json(500, {"error": "no rows loaded"})
            else:
                self._json(200, row_payload(random.choice(S.order)))
        elif (m := ROW_RE.match(path)):
            rid = m.group(1)
            if rid in S.rows:
                self._json(200, row_payload(rid))
            else:
                self._json(404, {"error": f"unknown row_id: {rid}"})
        else:
            self._json(404, {"error": "not found"})

    def _api_rows(self) -> None:
        p = self._query_params()
        q = p.get("q", "").casefold().strip()
        try:
            limit = max(1, min(int(p.get("limit", "50")), 200))
        except ValueError:
            limit = 50
        out = []
        for hay, rid, query in S.search:
            if q and q not in hay:
                continue
            out.append({"row_id": rid, "query": query})
            if len(out) >= limit:
                break
        self._json(200, {"total": len(S.order), "matched": len(out), "limit": limit, "rows": out})

    # -- POST ----------------------------------------------------------------
    def do_POST(self) -> None:   # noqa: N802
        path = self.path.split("?", 1)[0]
        if path != "/api/explain":
            self._json(404, {"error": "not found"})
            return
        try:
            n = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            n = 0
        if n <= 0 or n > MAX_BODY:
            self._json(400, {"error": f"body required, at most {MAX_BODY} bytes"})
            return
        try:
            body = json.loads(self.rfile.read(n).decode("utf-8"))
            assert isinstance(body, dict)
        except Exception:
            self._json(400, {"error": "invalid JSON body"})
            return
        if S.adapter is None:
            self._json(503, {"error": "model not ready",
                             "state": "error" if S.model_error else "loading"})
            return

        # -- resolve the (raw fields, query) input: manual fields > row_id ---
        rid = body.get("row_id")
        fields_in = body.get("fields")
        if isinstance(fields_in, dict):
            raw = {c: str(fields_in.get(c, "") or "")[:MAX_FIELD_CHARS] for c in RAW_COLS}
            try:
                norm = MA.normalize_fields(raw)
            except Exception:
                log(f"NORMALIZE FAILED:\n{traceback.format_exc(limit=4)}")
                self._json(400, {"error": "field normalization failed; see server log"})
                return
            source = "manual"
            rid_out = rid if rid in S.rows else None
        elif rid in S.rows:
            raw = S.rows[rid]["raw"]
            norm = S.rows[rid]["norm"]
            source = "row"
            rid_out = rid
        else:
            self._json(404 if rid else 400,
                       {"error": f"unknown row_id: {rid}" if rid else "need row_id or fields"})
            return

        q_raw = body.get("query")
        if not isinstance(q_raw, str) or not q_raw.strip():
            q_raw = raw.get("query", "")
        query = t1(str(q_raw).strip())[:MAX_FIELD_CHARS]

        a = S.args
        try:
            ig_steps = max(4, min(int(body.get("ig_steps", a.ig_steps)), 64))
        except (TypeError, ValueError):
            ig_steps = a.ig_steps
        want_cf = bool(body.get("counterfactuals", True))

        t = time.time()
        try:
            with S.lock:
                payload = S.adapter.explain(query, norm, ig_steps=ig_steps,
                                            counterfactuals=want_cf)
        except Exception:
            log(f"EXPLAIN FAILED for {rid_out or 'manual'}:\n{traceback.format_exc(limit=6)}")
            self._json(500, {"error": "explain failed; see server log"})
            return
        payload["row_id"] = rid_out
        payload["source"] = source
        # The model never sees the raw row -- it sees the normalised one. Ship BOTH so the UI can
        # show exactly what the preprocessing changed. This matters here: T4 rewrites
        # gender 'çocuk'->'unknown' on 8.9% of public rows and measurably moves predictions,
        # so a juror asking "what did the model actually receive" deserves to see it.
        payload["raw"] = dict(raw)
        if rid_out and S.rows.get(rid_out, {}).get("truth"):
            payload["truth"] = S.rows[rid_out]["truth"]
        payload["query_used"] = query
        payload["query_original"] = raw.get("query", "")
        payload["elapsed_ms"] = round((time.time() - t) * 1000)
        self._json(200, payload)


# ---------------------------------------------------------------- main
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("csv", help="public test CSV path")
    ap.add_argument("--model-dir", required=True, help="serve_hf-compatible checkpoint directory")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=int(os.environ.get("APP_PORT", "7862")))
    ap.add_argument("--device", default="cpu", help="cpu (default) or cuda")
    ap.add_argument("--ig-steps", type=int, default=16)
    ap.add_argument("--max-len", type=int, default=256)
    ap.add_argument("--truth", default="",
                    help="DEV ONLY: csv with row_id,real_label to overlay ground truth in the UI. "
                         "Never pass this in the delivered image — run_explainability.sh does not, "
                         "and the answer key is not staged into the build context.")
    ap.add_argument("--require-token", action="store_true",
                    help="fail closed (exit 2) unless NGROK_AUTHTOKEN is set (serve-ngrok mode)")
    S.args = ap.parse_args()

    if S.args.require_token and not os.environ.get("NGROK_AUTHTOKEN"):
        print("NGROK_AUTHTOKEN is required; explainability mode fails closed.", file=sys.stderr)
        return 2

    load_rows(S.args.csv)
    if S.args.truth:
        import csv as _csv
        with open(S.args.truth, newline="", encoding="utf-8") as f:
            for r in _csv.DictReader(f):
                rid = r.get("row_id")
                if rid in S.rows:
                    S.rows[rid]["truth"] = r.get("real_label") or r.get("label") or ""
        n = sum(1 for v in S.rows.values() if v.get("truth"))
        log(f"DEV MODE: ground truth loaded for {n:,} rows from {S.args.truth} "
            f"(experimentation only; not part of the delivered image)")

    # Bind FIRST so /healthz answers inside seconds; the model loads behind it.
    httpd = ThreadingHTTPServer((S.args.host, S.args.port), Handler)
    threading.Thread(target=load_model, daemon=True).start()
    log(f"listening on http://{S.args.host}:{S.args.port} (model loading in background)")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        httpd.server_close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
