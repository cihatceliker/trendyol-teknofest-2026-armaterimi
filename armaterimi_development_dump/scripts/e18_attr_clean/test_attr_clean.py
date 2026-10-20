"""E18 unit tests for attr_clean.clean_attributes / apply_v2 — plain asserts, no pytest needed.

    /home/jupyter/trendyol-teknofest-work/.venv-gcp/bin/python asya/s2_exp/e18_attr_clean/test_attr_clean.py

Synthetic cases first (boilerplate removal, yok/var preserved, order, sentinel, idempotence,
comma/colon-bearing values), then real-data properties on common/train_norm.parquet,
common/public_test_norm.parquet and the raw CSVs:
  * identity with drop_boilerplate=False and an empty drop set on 1,000 real non-empty strings,
    and on ALL 507,500 strings the only non-identity rows are exactly the "" rows (-> sentinel);
  * idempotence of the full v2 transform on real rows;
  * clean_attributes(raw, v1_keys, False) == normalize.clean_attrs(raw, v1_keys) on raw public and
    raw train text, i.e. the parser really is a mirror, not a reimplementation;
  * drop_keys_v2.csv is a superset of drop_keys.csv and contains the 11 requested additions.
CPU only; reads parquet/CSV, writes nothing.
"""
from __future__ import annotations
import os, sys, time
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import attr_clean as AC
import normalize as NZ                      # attr_clean put common/ on sys.path

W = "/home/jupyter/trendyol-teknofest-work"
S2 = f"{W}/asya/s2_exp"
V1_KEYS = f"{S2}/common/drop_keys.csv"
TRAIN_NORM = f"{S2}/common/train_norm.parquet"
PUBLIC_NORM = f"{S2}/common/public_test_norm.parquet"
RAW_TRAIN = f"{W}/stage2_data/train.csv"
RAW_PUBLIC = f"{W}/stage2_data/public_test.csv"
NEW_KEYS = ["yıkama talimatı", "bakım talimatları (genel)", "bakım talimatları (gıda temas)",
            "sürdürülebilirlik detayı", "menşei", "ek özellik", "garanti süresi", "garanti tipi",
            "ürün güvenliği bilgisi", "kullanım talimatı", "kutu durumu"]

CARE = "ürünün kullanım ömrünü korumak için ürüne özel bakım talimatlarını takip ediniz."
passed = []
def ok(name):
    passed.append(name); print(f"  PASS {name}")

# ----------------------------------------------------------------------------- synthetic
def test_boilerplate_removal():
    f = lambda s: AC.clean_attributes(s, set(), True)
    assert f("renk: siyah, desen: belirtilmemiş") == "renk: siyah"
    assert f("renk: siyah, ek özellik: ek özellik mevcut değil") == "renk: siyah"
    assert f("renk: siyah, parça: mevcut değil") == "renk: siyah"
    assert f("renk: siyah, boy: -") == "renk: siyah"
    assert f("renk: siyah, boy: unknown") == "renk: siyah"
    assert f("renk: siyah, boy: ") == "renk: siyah"
    assert f(f"renk: siyah, bakım talimatları (genel): {CARE}") == "renk: siyah"
    assert f("renk: siyah, yıkama: düşük ısıda yıkayabilir, kullanım ömrünü uzatabilirsiniz") == "renk: siyah"
    # case / stray-dot robust: private-set spelling variants must not slip through
    assert f("renk: siyah, desen: Belirtilmemiş") == "renk: siyah"
    assert f("renk: siyah, parça: MEVCUT DEĞİL") == "renk: siyah"
    # NOT boilerplate: informative values that merely contain similar words stay
    assert f("kutu: kutu yok, durum: tehlikeli değil, parça: parça mevcut değil") == \
           "kutu: kutu yok, durum: tehlikeli değil, parça: parça mevcut değil"
    # drop_boilerplate=False leaves everything in place
    s = f"renk: siyah, desen: belirtilmemiş, bakım: {CARE}"
    assert AC.clean_attributes(s, set(), False) == s
    ok("boilerplate removal (exact + substring, case-robust, no false positives)")

def test_yok_var_preserved():
    s = "kapüşon: yok, cep: var, taş cinsi: yok, stok: mevcut, bluetooth: yok"
    assert AC.clean_attributes(s, set(), True) == s
    assert AC.clean_attributes("kapüşon: YOK, cep: Var", set(), True) == "kapüşon: YOK, cep: Var"
    for v in ("yok", "var", "mevcut"):
        assert not AC.is_boilerplate(v), v
    ok("yok / var / mevcut preserved")

def test_key_drop_and_order():
    s = "a: 1, menşei: tr, b: 2, garanti süresi: 2 yıl, c: 3, d: belirtilmemiş, e: 5"
    assert AC.clean_attributes(s, {"menşei", "garanti süresi"}, True) == "a: 1, b: 2, c: 3, e: 5"
    assert AC.clean_attributes(s, {"menşei", "garanti süresi"}, False) == "a: 1, b: 2, c: 3, d: belirtilmemiş, e: 5"
    assert AC.clean_attributes("a: 1, Menşei : tr, b: 2", {"menşei"}, True) == "a: 1, b: 2"   # key folded like normalize
    ok("key drop + original order preserved")

def test_sentinel_on_empty():
    for s in ("", "unknown", "menşei: tr", "a: belirtilmemiş, b: -", f"x: {CARE}", "   "):
        assert AC.clean_attributes(s, {"menşei"}, True) == AC.EMPTY_SENTINEL, repr(s)
    assert AC.EMPTY_SENTINEL == "unknown"
    assert AC.clean_attributes("", set(), False) == "unknown"           # the ONLY non-identity input
    assert AC.clean_attributes("unknown", set(), False) == "unknown"
    ok("sentinel 'unknown' when nothing remains (and both raw empties unify)")

def test_comma_colon_values_mirror_normalize():
    # values that contain commas / colons: SEG only splits at ", " followed by a <=40-char key and ": "
    raw = ("diğer özellikler: sayfa sayısı :  96, ölçü: 10, 20 ve 30 cm, not: a,b:c, "
           "uzun açıklama: bu değer, virgül içerir ama anahtar gibi görünmez çünkü çok uzundur: evet, renk: kırmızı")
    v1 = NZ.clean_attrs(raw, set())
    assert AC.clean_attributes(raw, set(), False) == v1
    assert AC.clean_attributes(v1, set(), False) == v1
    assert AC.clean_attributes(raw, {"renk"}, False) == NZ.clean_attrs(raw, {"renk"})
    ok("comma/colon-bearing values split exactly like normalize.clean_attrs")

def test_idempotence_synthetic():
    drop = AC.load_drop_keys()
    for s in ("renk: siyah, menşei: tr, desen: belirtilmemiş", "", "unknown", f"a: {CARE}",
              "kapüşon: yok, cep: var", "diğer özellikler: sayfa sayısı :  96, menşei: tr"):
        once = AC.clean_attributes(s, drop, True)
        assert AC.clean_attributes(once, drop, True) == once, repr(s)
    ok("idempotence (synthetic)")

def test_drop_keys_v2_file():
    v1 = set(pd.read_csv(V1_KEYS).key)
    v2 = AC.load_drop_keys()
    assert v1 <= v2, v1 - v2
    assert set(NEW_KEYS) <= v2, set(NEW_KEYS) - v2
    assert set(v1) <= set(v2) and set(NEW_KEYS) <= set(v2) and len(v2) == len(set(v2)), len(v2)
    ok(f"drop_keys_v2.csv contains v1 + additions, no duplicates ({len(v2)} keys)")

# ----------------------------------------------------------------------------- real data
def test_real_data():
    t0 = time.time()
    tr = pd.read_parquet(TRAIN_NORM, columns=["attributes"]).attributes
    pu = pd.read_parquet(PUBLIC_NORM, columns=["attributes"]).attributes
    allv = pd.concat([tr, pu], ignore_index=True)
    assert len(allv) == 507_500

    # identity: empty drop set, drop_boilerplate=False, 1,000 real non-empty strings
    samp = tr[tr != ""].sample(n=1000, random_state=42)
    assert all(AC.clean_attributes(s, set(), False) == s for s in samp)
    ok("identity on 1,000 real non-empty attribute strings (empty drop set, no boilerplate drop)")

    # ... and on the whole corpus the only non-identity rows are the "" rows -> sentinel
    ident = allv.map(lambda s: AC.clean_attributes(s, set(), False))
    diff = allv != ident
    assert diff.sum() == (allv == "").sum() == 2355, (int(diff.sum()), int((allv == "").sum()))
    assert (ident[diff] == AC.EMPTY_SENTINEL).all()
    ok("identity on all 507,500 real strings except the 2,355 '' rows (-> 'unknown')")

    # idempotence of the full v2 transform on real rows
    drop = AC.load_drop_keys()
    once = allv.map(lambda s: AC.clean_attributes(s, drop, True))
    twice = once.map(lambda s: AC.clean_attributes(s, drop, True))
    assert (once == twice).all()
    # v2 never emits a boilerplate value, never emits "" and never emits a dropped key
    bad = once[once.map(lambda s: s != AC.EMPTY_SENTINEL and any(
        (k in drop) or (AC.is_boilerplate(v) and not AC._is_compat_key(k))   # compat keys keep 'belirtilmemiş' by design
        for k, v in ((seg.split(":", 1)[0].strip().lower(), seg.split(":", 1)[1].strip())
                     for seg in NZ.SEG.split(s) if ":" in seg)))]
    assert len(bad) == 0, bad.head()
    assert not (once == "").any()
    ok("idempotence + no residual boilerplate (except compatibility keys) / dropped keys on all 507,500 real rows")

    # apply_v2 frame API: other columns untouched, input not mutated
    d = pd.read_parquet(PUBLIC_NORM)
    before = d.copy()
    out = AC.apply_v2(d, AC.DROP_KEYS_V2, True)
    assert d.equals(before) and out is not d
    assert out.drop(columns="attributes").equals(d.drop(columns="attributes"))
    assert (out.attributes == pu.map(lambda s: AC.clean_attributes(s, drop, True))).all()
    ok("apply_v2 returns a copy, touches only `attributes`")

    # mirror check on RAW text: clean_attributes with v1 keys == normalize.clean_attrs with v1 keys
    v1 = set(pd.read_csv(V1_KEYS).key)
    rp = pd.read_csv(RAW_PUBLIC, dtype=str, keep_default_na=False).attributes
    rt = pd.read_csv(RAW_TRAIN, dtype=str, keep_default_na=False, usecols=["attributes"]).attributes
    for name, raw in (("public", rp), ("train", rt)):
        mine = raw.map(lambda s: AC.clean_attributes(s, v1, False))
        ref = raw.map(lambda s: NZ.clean_attrs(s, v1))
        # the only allowed difference is "" -> sentinel
        d_ = mine != ref
        assert (ref[d_] == "").all() and (mine[d_] == AC.EMPTY_SENTINEL).all(), name
    ok("clean_attributes(raw, v1_keys) == normalize.clean_attrs(raw, v1_keys) on raw public + train (mod sentinel)")

    # composability: v2 on normalize(raw) == v2 straight on raw (so the parser round-trips)
    for name, raw in (("public", rp), ("train", rt.sample(n=50_000, random_state=42))):
        via_norm = raw.map(lambda s: NZ.clean_attrs(s, v1)).map(lambda s: AC.clean_attributes(s, drop, True))
        direct = raw.map(lambda s: AC.clean_attributes(s, drop, True))
        assert (via_norm == direct).all(), name
    ok("v2(normalize(raw)) == v2(raw): the re-split round-trips")
    print(f"  (real-data checks took {time.time() - t0:.1f}s)")

if __name__ == "__main__":
    for fn in (test_boilerplate_removal, test_yok_var_preserved, test_key_drop_and_order,
               test_sentinel_on_empty, test_comma_colon_values_mirror_normalize,
               test_idempotence_synthetic, test_drop_keys_v2_file, test_real_data):
        fn()
    print(f"ALL {len(passed)} CHECKS PASSED")
