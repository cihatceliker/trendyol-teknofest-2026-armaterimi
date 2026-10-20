"""Data-path parity harness: OLD serial reader (lean_infer.py.bak) vs the NEW refactored
serial + parallel + pipelined readers. Byte-identical is the bar: row_ids, query, doc,
token ids AND token type_ids — all asserted equal on the FULL input CSV. CPU only, no CUDA.

Usage: parity_datapath.py [csv_path] [model_dir]
  defaults: stage2_data/public_test.csv + champion ckpt_r1_oldstore (tokenizer source only).
Env: MAXLEN(256), LEAN_WORKERS / LEAN_CHUNK as in lean_infer.
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""              # never touch the busy GPU
os.environ.setdefault("TOKENIZERS_PARALLELISM", "true")
import importlib.machinery, importlib.util, sys, time

W = "/home/jupyter/trendyol-teknofest-work"
HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path: sys.path.insert(0, HERE)
CSV = sys.argv[1] if len(sys.argv) > 1 else f"{W}/stage2_data/public_test.csv"
MD  = sys.argv[2] if len(sys.argv) > 2 else f"{W}/asya/s2_exp/e15_student/outputs_frozen_v2/ckpt_r1_oldstore"
MAXLEN = int(os.environ.get("MAXLEN", 256))

def log(m): print(f"[parity_dp] {m}", flush=True)

# ---- OLD module (pre-parallel snapshot)
_bak = os.path.join(HERE, "lean_infer.py.bak")
spec = importlib.util.spec_from_file_location("lean_old", _bak,
                                              loader=importlib.machinery.SourceFileLoader("lean_old", _bak))
OLD = importlib.util.module_from_spec(spec)
sys.modules["lean_old"] = OLD
t = time.perf_counter(); spec.loader.exec_module(OLD)
log(f"old module import (torch at top): {time.perf_counter()-t:.2f}s")
t = time.perf_counter()
import lean_infer as NEW
log(f"new module import (light): {time.perf_counter()-t:.2f}s  (torch loaded now: {NEW.torch is not None})")

# ---- 1. serial reference vs refactored serial
t = time.perf_counter(); o_ids, o_pairs = OLD.read_pairs(CSV); t_old = time.perf_counter() - t
log(f"OLD serial read_pairs          : {t_old:.3f}s  ({len(o_ids):,} rows)")
t = time.perf_counter(); s_ids, s_pairs = NEW.read_pairs(CSV); t_ser = time.perf_counter() - t
assert s_ids == o_ids, "NEW serial read_pairs: row_ids differ"
assert s_pairs == o_pairs, "NEW serial read_pairs: (query,doc) pairs differ"
log(f"NEW serial read_pairs          : {t_ser:.3f}s  byte-identical: OK")

# ---- 2. parallel read (fork pool)
pool = NEW.make_pool()
nw = NEW.default_workers()
t = time.perf_counter(); p_ids, p_pairs = NEW.read_pairs_parallel(CSV, pool=pool); t_par = time.perf_counter() - t
assert p_ids == o_ids, "parallel read: row_ids differ"
assert p_pairs == o_pairs, "parallel read: (query,doc) pairs differ"
log(f"NEW parallel read ({nw:2d} workers): {t_par:.3f}s  byte-identical: OK  ({t_old/max(t_par,1e-9):.1f}x vs old)")

# ---- 3. tokenisation parity: single-shot encode of the OLD pairs vs the pipelined chunked path
from tokenizers import Tokenizer
tok = Tokenizer.from_file(os.path.join(MD, "tokenizer.json"))
tok.enable_truncation(max_length=MAXLEN, strategy="longest_first")
t = time.perf_counter(); ref_encs = tok.encode_batch(o_pairs); t_enc1 = time.perf_counter() - t
log(f"reference single-shot encode   : {t_enc1:.3f}s")

t = time.perf_counter(); r_ids, r_encs, dp = NEW.run_datapath(CSV, MD, MAXLEN, pool); t_pipe = time.perf_counter() - t
pool.terminate()
assert r_ids == o_ids, "pipelined datapath: row_ids differ"
assert len(r_encs) == len(ref_encs), "pipelined datapath: row count differs"
bad = sum(1 for a, b in zip(ref_encs, r_encs) if list(a.ids) != list(b.ids) or list(a.type_ids) != list(b.type_ids))
assert bad == 0, f"token id/type_id mismatch on {bad} rows"
t_row = max(t_pipe - dp['tok_load'], 1e-9)      # row-scaled part; the tokenizer-file load is fixed
log(f"NEW pipelined datapath         : {t_pipe:.3f}s [{dp['mode']}] "
    f"(fixed tok load {dp['tok_load']:.3f} + row-scaled {t_row:.3f}, encode share {dp['encode']:.3f}) "
    f"vs old serial read+encode {t_old + t_enc1:.3f}s -> {(t_old + t_enc1)/t_row:.1f}x on the row-scaled part")
log(f"token ids + type_ids identical on all {len(r_encs):,} rows")

print(f"PARITY_DATAPATH OK rows={len(o_ids)} old_read={t_old:.3f}s new_serial={t_ser:.3f}s "
      f"new_parallel={t_par:.3f}s pipeline_read_norm_encode={t_pipe:.3f}s "
      f"projected@90k row-scaled: old~{(t_old+t_enc1)*90000/len(o_ids):.1f}s new~{t_row*90000/len(o_ids):.1f}s "
      f"(+{dp['tok_load']:.2f}s fixed tokenizer-file load)", flush=True)
