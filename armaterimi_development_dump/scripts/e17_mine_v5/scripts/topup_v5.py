"""E17 top-up: extra negatives from the HIGH-PURITY recipes (canary keep >= 0.95: M1, M2, M4, M5, M6,
M7) so the judged-irrelevant total reaches the 150k target. Same hygiene as select_v5 (normalized
identity, label-store R/S block-list, public-query drop), caps computed AGAINST the current
selected_v5 (item reuse <= 3 incl. existing rows, per-query budget minus rows already selected),
fashion roots first, rank-major. Output: outputs/topup_v5.parquet (+manifest), schema = selected_v5.
Usage: topup_v5.py --n 12000
"""
import argparse, hashlib, json, sys
from collections import defaultdict
import numpy as np, pandas as pd
W = '/home/jupyter/trendyol-teknofest-work'; E17 = f'{W}/asya/s2_exp/e17_mine_v5'; E16 = f'{W}/asya/s2_exp/e16_mine_v4'
sys.path.insert(0, f'{W}/asya/s2_exp/common'); import normalize as NZ
ap = argparse.ArgumentParser(); ap.add_argument('--n', type=int, default=12000); args = ap.parse_args()
SEED = 20260823; rng = np.random.default_rng(SEED)
PURE = {'M1', 'M2', 'M4', 'M5', 'M6', 'M7'}
FASHION = {'giyim', 'ayakkabı', 'aksesuar'}
K_Q, Q_FLOOR, Q_CAP, K_FASHION, FLOOR_FASHION, ITEM_CAP = 1.25, 8, 30, 2.0, 12, 3

raw = pd.read_parquet(f'{E17}/outputs/mined_v5_raw.parquet')
sel = pd.read_parquet(f'{E17}/outputs/selected_v5.parquet')
nid = pd.read_parquet(f'{E17}/outputs/item_norm_identity.parquet').set_index('item_key').norm_key
pool = pd.read_parquet(f'{W}/asya/s2_exp/e05_score/outputs/_pool_ext.parquet', columns=['item_key', 'category']).set_index('item_key')
root_of = pool.category.map(NZ.t1).str.split('/').str[0].str.strip().str.lower()
raw['norm_key'] = nid.reindex(raw.item_key.values).values; sel['norm_key'] = nid.reindex(sel.item_key.values).values
raw['code'] = raw.recipe.str.extract(r'^(M\d+[abc]?)', expand=False); raw['root'] = root_of.reindex(raw.item_key.values).values
raw = raw[raw.code.isin(PURE)]
taken = set(zip(sel['query'], sel.norm_key))
raw = raw[[(q, k) not in taken for q, k in zip(raw['query'], raw.norm_key)]]
raw = raw.drop_duplicates(['query', 'norm_key'])
store = pd.read_parquet(f'{E16}/outputs/label_store.parquet', columns=['query', 'item_key', 'judge_label'])
store['norm_key'] = nid.reindex(store.item_key.values).values
bad = set(zip(store[store.judge_label != 'irrelevant']['query'], store[store.judge_label != 'irrelevant'].norm_key))
raw = raw[[(q, k) not in bad for q, k in zip(raw['query'], raw.norm_key)]]
pubq = set(pd.read_parquet(f'{W}/asya/s2_exp/common/public_test_norm.parquet', columns=['query'])['query'].map(NZ.t1))
raw = raw[~raw['query'].map(NZ.t1).isin(pubq)].reset_index(drop=True)
# budgets vs existing selection
trc = pd.read_parquet(f'{W}/asya/s2_exp/common/train_norm.parquet', columns=['query', 'category'])
n_rs = trc['query'].value_counts()
qroot = trc.assign(r=trc.category.str.split('/').str[0].str.strip().str.lower()).groupby('query').r.agg(lambda x: x.mode().iloc[0])
used_q = sel[~sel.contrast].groupby('query').size()
item_cnt = defaultdict(int, sel.norm_key.value_counts().to_dict())
def budget(q):
    if q not in n_rs.index: return 12
    f = qroot.get(q) in FASHION
    return int(np.clip(round((K_FASHION if f else K_Q) * n_rs[q]), FLOOR_FASHION if f else Q_FLOOR, Q_CAP))
raw['remaining'] = [max(budget(q) - int(used_q.get(q, 0)), 0) for q in raw['query']]
raw = raw[raw.remaining > 0]
# recipe priority first (near-miss recipes before random-far), fashion as tie-breaker, rank-major
PRIO = {'M1': 0, 'M6': 1, 'M2': 2, 'M4': 3, 'M5': 3, 'M7': 9}
M7_CAP = int(0.10 * args.n)
raw['fash'] = raw.root.isin(FASHION).astype(int)
raw['prio'] = raw.code.map(PRIO)
raw = raw.iloc[np.lexsort((rng.random(len(raw)), raw['rank'].to_numpy(), -raw.fash.to_numpy(), raw.prio.to_numpy()))].reset_index(drop=True)
bq = dict(zip(raw['query'].unique(), [0] * raw['query'].nunique()))
keep = np.zeros(len(raw), bool); got = 0
rem = dict(zip(raw['query'], raw.remaining))
m7 = 0
for i, (q, k, c) in enumerate(zip(raw['query'].to_numpy(), raw.norm_key.to_numpy(), raw.code.to_numpy())):
    if got >= args.n: break
    if rem[q] <= 0 or item_cnt[k] >= ITEM_CAP: continue
    if c == 'M7':
        if m7 >= M7_CAP: continue
        m7 += 1
    keep[i] = True; rem[q] -= 1; item_cnt[k] += 1; got += 1
top = raw[keep].copy(); top['phase'] = 'topup'; top['contrast'] = False
cols = [c for c in sel.columns if c in top.columns]
top = top[cols].sort_values(['query', 'item_key']).reset_index(drop=True)
out = f'{E17}/outputs/topup_v5.parquet'; top.to_parquet(out, index=False)
man = {'n': int(len(top)), 'recipes': top.recipe.value_counts().to_dict(), 'roots': top.root.value_counts(normalize=True).round(3).head(8).to_dict(),
       'fashion_share': round(float(top.root.isin(FASHION).mean()), 3), 'n_queries': int(top['query'].nunique()),
       'item_reuse_max_incl_selection': int(max(item_cnt.values())), 'seed': SEED, 'base_selection_sha12': hashlib.sha256(open(f'{E17}/outputs/selected_v5.parquet', 'rb').read()).hexdigest()[:12],
       'sha256': hashlib.sha256(open(out, 'rb').read()).hexdigest()}
json.dump(man, open(out.replace('.parquet', '.manifest.json'), 'w'), indent=1, ensure_ascii=False)
print(json.dumps(man, indent=1, ensure_ascii=False))
