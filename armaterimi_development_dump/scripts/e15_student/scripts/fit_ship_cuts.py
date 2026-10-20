"""FREEZE tooling: fit the shipped ordinal cuts and write decision.json.

Runs the champion over public_test.csv via the exact serving stack (lean_gte + lean_infer data
path, GPU), joins labels, fits two cuts on the scalar (p_semi + 2*p_rel) maximizing macro-F1,
writes {"head":"ordinal","cuts":[c1,c2]} into --model-dir (backing up the old decision.json)
and a fit manifest next to it.

DISCIPLINE GUARD: --half dev fits on the DEV half only (code validation, no holdout labels
touched). --half full requires FREEZE_CONFIRM=cuts-on-full-public-at-freeze in the env — run it
ONLY after the day's HOLDOUT read, per the pinned freeze sequence.
"""
import argparse, json, os, shutil, sys, time
import numpy as np
W = '/home/jupyter/trendyol-teknofest-work'
sys.path.insert(0, f'{W}/asya/s2_exp/common')
sys.path.insert(0, f'{W}/asya/s2_exp/e12_lean_infer/scripts')
ap = argparse.ArgumentParser()
ap.add_argument('--model-dir', default=f'{W}/asya/s2_exp/e15_student/outputs_frozen_v2/ckpt_r1_oldstore')
ap.add_argument('--half', choices=['dev', 'full'], default='dev')
ap.add_argument('--write', action='store_true', help='actually write decision.json (default: report only)')
a = ap.parse_args()
if a.half == 'full' and os.environ.get('FREEZE_CONFIRM') != 'cuts-on-full-public-at-freeze':
    sys.exit('REFUSE: --half full needs FREEZE_CONFIRM=cuts-on-full-public-at-freeze (run after the HOLDOUT read)')
import harness as H
from sklearn.metrics import f1_score
import lean_infer as L, lean_gte as G, torch
LAB = ['irrelevant', 'semi-relevant', 'relevant']

# --- forward over the needed rows via the exact serving stack
ids, pairs = L.read_pairs(f'{W}/stage2_data/public_test.csv')
from tokenizers import Tokenizer
tok = Tokenizer.from_file(os.path.join(a.model_dir, 'tokenizer.json'))
tok.enable_truncation(max_length=256, strategy='longest_first')
encs = tok.encode_batch(pairs)
dev = 'cuda' if torch.cuda.is_available() else 'cpu'
m = G.load_gte(a.model_dir, dev, torch.bfloat16 if dev == 'cuda' else torch.float32)
t0 = time.time()
logits = L.infer(m, encs, 128, 32768, True)
print(f'forward {len(ids):,} rows in {time.time()-t0:.1f}s on {dev}')
p = torch.softmax(torch.tensor(np.asarray(logits), dtype=torch.float32), -1).numpy()
scal = p[:, 1] + 2 * p[:, 2]
id2s = dict(zip(ids, scal))

# --- labels for the chosen half
half = H.load_public_split('dev') if a.half == 'dev' else None
if half is None:
    import pandas as pd
    sol = pd.read_csv(f'{W}/stage2_data/public_solution.csv', dtype=str)
    rows = list(zip(sol.row_id, sol.real_label))
else:
    rows = list(zip(half.row_id, half.real_label))
y = np.array([r[1] for r in rows]); s = np.array([id2s[r[0]] for r in rows])
print(f'fitting on {len(y):,} rows ({a.half})')

best = (-1, None)
q1s = np.quantile(s, np.linspace(0.05, 0.60, 120)); q2s = np.quantile(s, np.linspace(0.40, 0.95, 120))
for c1 in q1s:
    for c2 in q2s:
        if c2 <= c1: continue
        pred = np.where(s < c1, 'irrelevant', np.where(s < c2, 'semi-relevant', 'relevant'))
        f = f1_score(y, pred, average='macro', labels=LAB)
        if f > best[0]: best = (f, (float(c1), float(c2)))
f, cuts = best
print(f'BEST macro-F1 {f:.4f} at cuts [{cuts[0]:.4f}, {cuts[1]:.4f}]')
pred = np.where(s < cuts[0], 'irrelevant', np.where(s < cuts[1], 'semi-relevant', 'relevant'))
import collections
print('pred mix:', dict(collections.Counter(pred)))
if a.write:
    dj = os.path.join(a.model_dir, 'decision.json')
    shutil.copy(dj, dj + f'.bak_{a.half}')
    json.dump({'head': 'ordinal', 'cuts': list(cuts)}, open(dj, 'w'))
    json.dump({'fit_half': a.half, 'n': int(len(y)), 'macro_f1_fit': round(f, 4), 'cuts': list(cuts),
               'created_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
               'model_dir': a.model_dir, 'scalar': 'p_semi + 2*p_rel', 'grid': '120x120 quantile'},
              open(os.path.join(a.model_dir, 'cuts_fit_manifest.json'), 'w'), indent=1)
    print(f'WROTE {dj} (old kept as .bak_{a.half})')
