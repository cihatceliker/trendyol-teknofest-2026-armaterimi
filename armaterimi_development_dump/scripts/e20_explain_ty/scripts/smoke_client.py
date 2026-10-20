#!/usr/bin/env python3
"""e20 smoke client — hits every endpoint of a running app.py, asserts payload sanity, prints
wall-clock timings. Exits non-zero on the first failed assertion. stdlib only (urllib)."""
from __future__ import annotations

import json
import sys
import time
import urllib.error
import urllib.request

BASE = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:7893"
T0 = time.time()
FAIL = 0


def req(method: str, path: str, body: dict | bytes | None = None, timeout: float = 300.0):
    url = BASE + path
    data = None
    headers = {}
    if body is not None:
        data = body if isinstance(body, bytes) else json.dumps(body).encode("utf-8")
        headers["Content-Type"] = "application/json"
    r = urllib.request.Request(url, data=data, headers=headers, method=method)
    t = time.time()
    try:
        with urllib.request.urlopen(r, timeout=timeout) as resp:
            code, raw = resp.status, resp.read()
    except urllib.error.HTTPError as e:
        code, raw = e.code, e.read()
    except (urllib.error.URLError, ConnectionError) as e:
        return None, None, time.time() - t, e
    try:
        payload = json.loads(raw.decode("utf-8"))
    except Exception:
        payload = raw
    return code, payload, time.time() - t, None


def check(name: str, cond: bool, detail: str = "") -> None:
    global FAIL
    mark = "PASS" if cond else "FAIL"
    if not cond:
        FAIL += 1
    print(f"  [{mark}] {name}" + (f" — {detail}" if detail else ""))


def section(s: str) -> None:
    print(f"\n== {s} (t+{time.time()-T0:.1f}s)")


# ---------------------------------------------------------------- 1. server bind + health
section("server bind / /healthz")
code = None
for _ in range(120):
    code, h, dt, err = req("GET", "/healthz", timeout=5)
    if code is not None:
        break
    time.sleep(0.5)
check("/healthz answers", code == 200, f"HTTP {code} in {dt*1000:.0f} ms at t+{time.time()-T0:.1f}s")
check("healthz shape", isinstance(h, dict) and h.get("ok") is True and h.get("rows", 0) > 7000,
      f"rows={h.get('rows')} model={h.get('model')}")

# ---------------------------------------------------------------- 2. model ready
section("model load / /readyz")
ready = None
for _ in range(1200):
    code, ready, dt, err = req("GET", "/readyz", timeout=10)
    if code == 200:
        break
    time.sleep(1.0)
check("/readyz turned 200", code == 200, f"at t+{time.time()-T0:.1f}s")
check("readyz shape", isinstance(ready, dict) and ready.get("head") in ("softmax3", "ordinal", "ordinal_regime")
      and len(ready.get("cuts", [])) == 2,
      f"head={ready.get('head')} cuts={ready.get('cuts')} model_load_s={ready.get('model_load_s')} "
      f"device={ready.get('device')}")

# ---------------------------------------------------------------- 3. UI + row listing
section("GET / and /api/rows")
code, html, dt, _ = req("GET", "/")
check("/ serves UI", code == 200 and isinstance(html, (bytes, bytearray)) and b"Uygunluk" in html,
      f"{len(html) if isinstance(html,(bytes,bytearray)) else '?'} bytes in {dt*1000:.0f} ms")
code, rows, dt, _ = req("GET", "/api/rows?q=pejo&limit=5")
check("/api/rows search", code == 200 and rows.get("matched", 0) >= 1,
      f"matched={rows.get('matched')} in {dt*1000:.0f} ms")
RID = rows["rows"][0]["row_id"]
code, rows_all, dt, _ = req("GET", "/api/rows?limit=3")
check("/api/rows unfiltered", code == 200 and rows_all.get("total", 0) > 7000,
      f"total={rows_all.get('total')} in {dt*1000:.0f} ms")

# ---------------------------------------------------------------- 4. row / random
section("GET /api/row/<id> and /api/random")
code, row, dt, _ = req("GET", f"/api/row/{RID}")
check("row payload", code == 200 and set(row) >= {"row_id", "raw", "normalized", "doc"}
      and " | " in row["doc"], f"{RID} in {dt*1000:.0f} ms")
code, rnd, dt, _ = req("GET", "/api/random")
check("random payload", code == 200 and rnd.get("row_id") in (rnd.get("row_id"),)
      and set(rnd) >= {"row_id", "raw", "normalized", "doc"},
      f"row_id={rnd.get('row_id')} in {dt*1000:.0f} ms")
code, _r2, dt, _ = req("GET", "/api/row/ROW_does_not_exist")
check("unknown row -> 404", code == 404)

# ---------------------------------------------------------------- 5. full explanation (row)
section("POST /api/explain — public row")
code, ex, dt, _ = req("POST", "/api/explain", {"row_id": RID})
check("explain 200", code == 200, f"wall {dt:.2f}s server {ex.get('elapsed_ms','?')} ms")
p = ex.get("prediction", {})
check("prediction shape", p.get("label") in ("irrelevant", "semi-relevant", "relevant")
      and len(p.get("cuts", [])) == 2 and isinstance(p.get("scalar"), float),
      f"label={p.get('label')} scalar={p.get('scalar'):.3f} cuts={p.get('cuts')}")
probs = p.get("probs") or {}
check("probs sum to 1", abs(sum(probs.values()) - 1.0) < 1e-4,
      " ".join(f"{k}={v:.3f}" for k, v in probs.items()))
check("scalar = p_semi + 2*p_rel", abs(p["scalar"] - (probs.get("semi-relevant", 0)
      + 2 * probs.get("relevant", 0))) < 1e-6)
check("argmax parity", p["label"] == max(probs, key=probs.get))
toks = ex.get("tokens", [])
check("tokens present", len(toks) > 10 and all("ig" in t for t in toks), f"n={len(toks)}")
ig = ex.get("ig", {})
check("IG completeness", ig.get("completeness_gap", 9) < 0.15,
      f"gap={ig.get('completeness_gap')} steps={ig.get('steps')} target={ig.get('target')}")
check("IG specials are 0", all(abs(t["ig"]) < 1e-9 for t in toks if t["special"]))
att = ex.get("attention", {})
check("attention dims", len(att.get("matrix_last", [])) == len(att.get("query_tokens", []))
      and len(att.get("matrix_last", [[]])[0]) == len(att.get("product_tokens", [])),
      f"{len(att.get('query_tokens', []))}x{len(att.get('product_tokens', []))}")
cf = ex.get("counterfactuals", {})
check("counterfactuals present", cf.get("product", {}).get("n_variants", 0) >= 1
      and "query_words" in cf,
      f"product n={cf.get('product', {}).get('n_variants')} "
      f"query n={cf.get('query_words', {}).get('n_variants')} "
      f"product flips={len(cf.get('product', {}).get('flips', []))}")
check("summary_tr present", len(ex.get("summary_tr", "")) > 30,
      ex.get("summary_tr", "")[:160])
print(f"    timings_ms={ex.get('timings_ms')}")

# ---------------------------------------------------------------- 6. edited query (counterfactual live path)
section("POST /api/explain — edited query")
code, ex2, dt, _ = req("POST", "/api/explain", {"row_id": RID, "query": "pejo 307 far camı"})
check("edited-query explain 200", code == 200,
      f"wall {dt:.2f}s label={ex2.get('prediction', {}).get('label')}")
check("query_used echoes edit", ex2.get("query_used") == "pejo 307 far camı")

# ---------------------------------------------------------------- 7. manual fields (arbitrary input)
section("POST /api/explain — manual product fields")
manual = {
    "fields": {
        "query": "kablosuz kulaklık",
        "title": "logitech m171 kablosuz mouse",
        "brand": "logitech",
        "category": "elektronik/bilgisayar/çevre birimleri/mouse",
        "gender": "unisex",
        "age_group": "",
        "attributes": "renk: siyah, bağlantı: kablosuz, menşei: tr",
    }
}
code, ex3, dt, _ = req("POST", "/api/explain", manual)
check("manual explain 200", code == 200, f"wall {dt:.2f}s server {ex3.get('elapsed_ms','?')} ms")
check("manual source flag", ex3.get("source") == "manual" and ex3.get("row_id") is None)
nrm = ex3.get("normalized", {})
check("pipeline applied to manual input",
      nrm.get("gender") == "unknown" and "menşei" not in nrm.get("attributes", "menşei"),
      f"gender={nrm.get('gender')!r} attrs={nrm.get('attributes')!r}")
check("manual prediction sane",
      ex3.get("prediction", {}).get("label") in ("irrelevant", "semi-relevant", "relevant"),
      f"label={ex3.get('prediction', {}).get('label')} "
      f"scalar={ex3.get('prediction', {}).get('scalar'):.3f}")
check("manual summary", len(ex3.get("summary_tr", "")) > 30, ex3.get("summary_tr", "")[:160])

# ---------------------------------------------------------------- 8. latency repeat (warm)
section("repeat explain — warm latency sample")
walls = []
for _ in range(3):
    code, exr, dt, _ = req("POST", "/api/explain", {"row_id": rnd["row_id"]})
    check("warm explain 200", code == 200,
          f"wall {dt:.2f}s server {exr.get('elapsed_ms','?')} ms "
          f"(n_tokens={exr.get('n_tokens')})")
    walls.append(dt)
print(f"    warm walls: {', '.join(f'{w:.2f}s' for w in walls)}")

# ---------------------------------------------------------------- 9. error paths
section("error paths")
code, e1, dt, _ = req("POST", "/api/explain", b"this is not json")
check("bad JSON -> 400", code == 400)
code, e2, dt, _ = req("POST", "/api/explain", {"row_id": "ROW_nope"})
check("unknown row_id -> 404", code == 404)
code, e3, dt, _ = req("POST", "/api/explain", {})
check("no input -> 400", code == 400)
code, e4, dt, _ = req("GET", "/api/nope")
check("unknown route -> 404", code == 404)

print(f"\n{'ALL PASS' if FAIL == 0 else f'{FAIL} FAILURES'} — total {time.time()-T0:.1f}s")
sys.exit(1 if FAIL else 0)
