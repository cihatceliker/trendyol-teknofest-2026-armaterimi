"""E18 attr_clean — OPTIONAL "v2" attribute cleaning, layered on top of common/normalize.py.

Default pipeline behaviour is untouched: nothing here is imported by normalize.py or by any
e15/e16/e17 script. v2 is opt-in — call `apply_v2()` on a frame that has ALREADY been through
`normalize.normalize(df, drop_keys)`, and do so identically at train-build time and at inference.

What v2 does to the `attributes` string ("key: value, key: value", as normalize.py serialises it):
  (a) drops pairs whose key is in `drop_keys` (drop_keys_v2.csv = the frozen v1 list + care /
      warranty / origin / packaging boilerplate keys);
  (b) with drop_boilerplate=True, drops pairs whose VALUE is pure boilerplate — an exact member of
      BOILERPLATE_EXACT or any value containing a BOILERPLATE_SUBSTR care-instruction phrase.
      Values "yok" / "var" / "mevcut" are NEVER dropped: "kapüşon: yok" is information.
  Pair order is preserved. If nothing remains the string becomes EMPTY_SENTINEL ("unknown").

Why "unknown" and not "": normalize.py emits "" when every key is dropped and passes the raw
sentinels through verbatim, so train carries BOTH "" (2,270 rows, 99.6% semi-relevant) and
"unknown" (4,097 rows, 100% relevant) — the pair is an export tell exactly like `unisex` (T4).
Public carries only "" (85 rows). One sentinel, applied identically on both sides, removes the
tell; "unknown" is the vocabulary normalize.py already uses for gender/age_group.

Parsing: the segment split is normalize.SEG / normalize.t1 IMPORTED from common/normalize.py,
and the per-segment logic mirrors normalize.clean_attrs line for line (split on the first ":",
strip+lower the key, strip the value, keep a key-less segment if non-empty). Nothing is
re-parsed differently: test_attr_clean.py asserts that with an empty drop set and
drop_boilerplate=False this function is the identity on every real attribute string except
"" (which maps to the sentinel), and that it reproduces normalize.clean_attrs on raw text.
"""
from __future__ import annotations
import os, sys
import pandas as pd

_COMMON = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "common"))
if _COMMON not in sys.path:
    sys.path.insert(0, _COMMON)
from normalize import SEG, t1          # the frozen splitter — never redefined here

HERE = os.path.dirname(os.path.abspath(__file__))
DROP_KEYS_V2 = os.path.join(HERE, "drop_keys_v2.csv")

EMPTY_SENTINEL = "unknown"
BOILERPLATE_EXACT = frozenset({"belirtilmemiş", "mevcut değil", "ek özellik mevcut değil", "-", "unknown", ""})
BOILERPLATE_SUBSTR = ("takip ediniz", "kullanım ömrünü")      # care-instruction sentences
KEEP_VALUES = frozenset({"yok", "var", "mevcut"})             # informative; never boilerplate


def _fold(v: str) -> str:
    """Case/diacritic-robust form of a value for boilerplate matching only (never emitted).
    lower() first, then t1(): Python lowercases 'İ' to 'i'+U+0307 and t1 strips that dot."""
    return t1(v.lower()).strip()


def is_boilerplate(value: str) -> bool:
    """True when `value` carries no product information (see module docstring)."""
    v = _fold(value)
    if v in KEEP_VALUES:
        return False
    if v in BOILERPLATE_EXACT:
        return True
    return any(s in v for s in BOILERPLATE_SUBSTR)


COMPAT_KEY_MARK = ("uyum",)   # compatibility keys keep "belirtilmemiş": device silence is a signal (rubric §4C)


def _is_compat_key(key: str) -> bool:
    k = _fold(key)
    return any(m in k for m in COMPAT_KEY_MARK)


def clean_attributes(attr_str: str, drop_keys: set, drop_boilerplate: bool) -> str:
    """Pure function on one serialised attribute string. See module docstring.

    Mirrors normalize.clean_attrs exactly (same SEG split, same key/value handling) and adds the
    two v2 filters. Idempotent. Returns EMPTY_SENTINEL when nothing remains."""
    if not isinstance(drop_keys, (set, frozenset)):
        drop_keys = set(drop_keys)
    out = []
    for seg in SEG.split(t1(attr_str)):
        if ":" not in seg:                         # key-less segment: normalize keeps it if non-empty
            s = seg.strip()
            if not s:
                continue
            if drop_boilerplate and is_boilerplate(s):
                continue
            out.append(s)
            continue
        k, v = seg.split(":", 1)
        kl, vs = k.strip().lower(), v.strip()
        if kl in drop_keys:
            continue
        if drop_boilerplate and is_boilerplate(vs) and not _is_compat_key(kl):
            continue
        out.append(f"{kl}: {vs}")
    return ", ".join(out) if out else EMPTY_SENTINEL


def load_drop_keys(path: str = DROP_KEYS_V2) -> set:
    """Read a `key` column; fold each key the way normalize._keys folds data keys (t1, strip, lower)."""
    keys = pd.read_csv(path, dtype=str, keep_default_na=False).key.tolist()
    return {t1(k).strip().lower() for k in keys if k.strip()}


def apply_v2(df: pd.DataFrame, drop_keys_v2_path: str = DROP_KEYS_V2, drop_boilerplate: bool = True) -> pd.DataFrame:
    """Apply clean_attributes to the `attributes` column of a NORMALIZED frame. Returns a copy.

    Must be applied identically wherever normalize.normalize() is applied (training builders,
    DEV/holdout eval, the inlined copy in e12 lean_infer). Other columns are untouched."""
    drop = load_drop_keys(drop_keys_v2_path)
    d = df.copy()
    d["attributes"] = d["attributes"].map(lambda a: clean_attributes(a, drop, drop_boilerplate))
    return d


if __name__ == "__main__":                         # tiny smoke
    ex = ("renk: siyah, kapüşon: yok, menşei: tr, garanti süresi: 2 yıl, "
          "bakım talimatları (genel): ürünün kullanım ömrünü korumak için ürüne özel bakım "
          "talimatlarını takip ediniz., diğer özellikler: sayfa sayısı :  96, desen: belirtilmemiş")
    print(repr(clean_attributes(ex, load_drop_keys(), True)))
