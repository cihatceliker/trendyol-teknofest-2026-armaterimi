"""E17 selection (v5): mined_v5_raw (ranked lists) -> selected_v5.parquet (N_TARGET rows).

Constraints, all DOCUMENTED CONSTANTS (never re-fitted here):
  * per-query cap      b_q = clip(round(K_Q * n_RS_q), Q_FLOOR, 30)   n_RS_q = the query's relevant+semi
                       rows in train. Templated (synthetic) queries: b_q = TEMPLATED_BUDGET.
  * item reuse cap     ITEM_CAP appearances of one item across the whole selection.
  * department profile ROOT_TARGET = DEV-irrelevant item-root shares (A14 §4.3-1); the
                       remainder spread over other roots by raw availability.
  * recipe mix         RECIPE_TARGET within each root; M8 carries an x1.6 over-allocation
                       because the judge will relabel ~40% of it semi (kept as semi data).
  * brand clusters     BRAND_CLUSTER_FRAC of queries planned for >=3 rows take ALL their rows
                       from M1 with a single brand (A14 §4.3-2).
Fill order inside a cell: RANK-MAJOR ROUND-ROBIN — every query's rank-0 row before any
query's rank-1 row — so budget spreads across queries and the hardest candidates go first.
Hygiene drops: dupes, items absent from the pool, label-store R/S pairs, public-query collisions.
Usage: select_v5.py [--raw PATH] [--tag T] [--n N]
"""
import argparse, hashlib, json, os, sys
from collections import defaultdict
import numpy as np, pandas as pd

W = '/home/jupyter/trendyol-teknofest-work'
E17 = f'{W}/asya/s2_exp/e17_mine_v5'
E16 = f'{W}/asya/s2_exp/e16_mine_v4'
sys.path.insert(0, f'{W}/asya/s2_exp/common')
import normalize as NZ

SEED = 20260823
K_Q = 1.25
Q_CAP = 30
Q_FLOOR = 8
# fashion-root QUERIES carry more negatives: in DEV, rows under fashion queries are 45.6% irrelevant
# (0.84 I per R/S row) vs 33% overall (0.5) — and fashion is 43% of DEV-I items but only 13% of
# train queries, so the supply has to come from those queries. Per-root multiplier on the cap/floor.
FASHION_ROOTS = {'giyim', 'ayakkabı', 'aksesuar'}
K_FASHION = 2.0
FLOOR_FASHION = 12                 # every query may carry at least this many negatives (train median
                            # n_RS is 6; the test shows ~1.2 rows/query so per-query count is not
                            # an organizer property — the cap only prevents concentration)
TEMPLATED_BUDGET = 12
ITEM_CAP = 3
BRAND_CLUSTER_FRAC = 0.55
ROOT_TARGET = {'giyim': .434, 'elektronik': .133, 'ayakkabı': .087, 'ev & mobilya': .086,
               'kozmetik & kişisel bakım': .058, 'aksesuar': .051}
ROOT_REMAINDER = 1.0 - sum(ROOT_TARGET.values())
RECIPE_TARGET = {'M1': .33, 'M10': .22, 'M8': .16, 'M6': .12, 'M2': .08, 'M7': .03, 'M3b': .02, 'M4': .02}
RECIPE_REMAINDER = 1.0 - sum(RECIPE_TARGET.values())
SHORTFALL_SINK = ['M10', 'M6', 'M2']

ap = argparse.ArgumentParser()
ap.add_argument('--raw', default=f'{E17}/outputs/mined_v5_raw.parquet')
ap.add_argument('--tag', default='')
ap.add_argument('--n', type=int, default=170_000)
args = ap.parse_args()
N_TARGET = args.n
tag = f'_{args.tag}' if args.tag else ''
OUT = f'{E17}/outputs/selected_v5{tag}.parquet'
rng = np.random.default_rng(SEED)
log = []

# ---------------------------------------------------------------- 0. load + hygiene
raw = pd.read_parquet(args.raw)
n0 = len(raw)
# NORMALIZED ITEM IDENTITY (review 2026-08-22 pt 5): 5,556 pool rows are the same product after
# normalize.py (mostly the same item exported in both snapshots). Dedup, reuse cap and the
# label-store block-list all operate on norm_key so identical products cannot evade caps or
# be paid for twice. Map built by outputs/item_norm_identity.parquet (hash of the normalized
# 6-field tuple).
nid = pd.read_parquet(f'{E17}/outputs/item_norm_identity.parquet').set_index('item_key').norm_key
raw['norm_key'] = nid.reindex(raw.item_key.values).values
assert raw.norm_key.notna().all(), 'items missing from the normalized identity map'
raw = raw.drop_duplicates(['query', 'norm_key']).reset_index(drop=True)
n_dupes = n0 - len(raw)
pool = pd.read_parquet(f'{W}/asya/s2_exp/e05_score/outputs/_pool_ext.parquet',
                       columns=['item_key', 'category', 'brand']).set_index('item_key')
root_of = pool.category.map(NZ.t1).str.split('/').str[0].str.strip().str.lower()
bkey_of = pool.brand.map(NZ.t1).str.strip().str.lower()
raw['root'] = root_of.reindex(raw.item_key.values).values
n_nopool = int(raw.root.isna().sum())
raw = raw[raw.root.notna()].reset_index(drop=True)
store = pd.read_parquet(f'{E16}/outputs/label_store.parquet', columns=['query', 'item_key', 'judge_label'])
store['norm_key'] = nid.reindex(store.item_key.values).values
bad = (store[store.judge_label != 'irrelevant'].dropna(subset=['norm_key'])[['query', 'norm_key']]
       .drop_duplicates().assign(_bad=1))
n1 = len(raw)
raw = raw.merge(bad, on=['query', 'norm_key'], how='left')
raw = raw[raw._bad.isna()].drop(columns='_bad').reset_index(drop=True)
n_known_rs = n1 - len(raw)
pub = pd.read_parquet(f'{W}/asya/s2_exp/common/public_test_norm.parquet', columns=['query'])
pubq = set(pub['query'].map(NZ.t1))
n2 = len(raw)
raw = raw[~raw['query'].map(NZ.t1).isin(pubq)].reset_index(drop=True)
n_pubq = n2 - len(raw)
raw['code'] = raw.recipe.astype(str).str.extract(r'^(M\d+[abc]?)', expand=False).fillna('other')
raw['bkey'] = bkey_of.reindex(raw.item_key.values).fillna('').values
print(f"raw {n0:,} -> {len(raw):,} (dupes {n_dupes:,}, no-pool {n_nopool:,}, "
      f"store R/S {n_known_rs:,}, public-q {n_pubq:,})")

# ---------------------------------------------------------------- 1. per-query budgets
tr = pd.read_parquet(f'{W}/asya/s2_exp/common/train_norm.parquet', columns=['query'])
n_rs = tr['query'].value_counts()
qs = raw.groupby('query').size()
w = n_rs.reindex(qs.index)
is_templ = raw.groupby('query').templated.any().reindex(qs.index).fillna(False)
n_noweight = int((w.isna() & ~is_templ).sum())
trc = pd.read_parquet(f'{W}/asya/s2_exp/common/train_norm.parquet', columns=['query', 'category'])
qroot = (trc.assign(r=trc.category.str.split('/').str[0].str.strip().str.lower())
            .groupby('query').r.agg(lambda x: x.mode().iloc[0]))
is_fash = qroot.reindex(qs.index).isin(FASHION_ROOTS).to_numpy()
kq = np.where(is_fash, K_FASHION, K_Q); fl = np.where(is_fash, FLOOR_FASHION, Q_FLOOR)
b = np.clip(np.round(kq * w.fillna(0).to_numpy(float)), fl, Q_CAP).astype(int)
b = np.where(is_templ.to_numpy(), TEMPLATED_BUDGET, b)
b = np.where(w.isna().to_numpy() & ~is_templ.to_numpy(), int(np.clip(round(K_Q * n_rs.median()), Q_FLOOR, Q_CAP)), b)
capq = np.minimum(b, qs.to_numpy())                     # can't take more than raw supplies
budget = dict(zip(qs.index, capq))
print(f"budgets: {len(budget):,} queries, mean cap {np.mean(b):.1f}, mean supply-limited "
      f"{np.mean(capq):.1f}, total capacity {int(capq.sum()):,} (target {N_TARGET:,}), "
      f"{n_noweight} non-templated queries without train weight")
if capq.sum() < N_TARGET:
    log.append({'stage': 'capacity', 'note': 'capacity below target', 'capacity': int(capq.sum())})
N_base = min(N_TARGET, int(capq.sum()))

# ---------------------------------------------------------------- 2. root x recipe targets
root_avail = raw.root.value_counts()
other_roots = [r for r in root_avail.index if r not in ROOT_TARGET]
oa = root_avail.reindex(other_roots).fillna(0)
root_tgt = {r: s * N_base for r, s in ROOT_TARGET.items()}
root_tgt.update({r: ROOT_REMAINDER * N_base * oa[r] / max(oa.sum(), 1) for r in other_roots})
for r in list(root_tgt):
    av = int(root_avail.get(r, 0))
    if root_tgt[r] > av:
        log.append({'stage': 'root_clip', 'root': r, 'target': round(root_tgt[r]), 'avail': av})
        root_tgt[r] = av
cell_tgt = {}
for r, nt in root_tgt.items():
    sub = raw[raw.root == r]
    natural = sub[~sub.code.isin(RECIPE_TARGET)].code.value_counts()
    tgt = {c: s * nt for c, s in RECIPE_TARGET.items()}
    tgt.update({c: RECIPE_REMAINDER * nt * natural[c] / max(natural.sum(), 1) for c in natural.index})
    avail = sub.code.value_counts()
    moved = 0.0
    for c in list(tgt):
        av = int(avail.get(c, 0))
        if tgt[c] > av:
            moved += tgt[c] - av
            log.append({'stage': 'cell_clip', 'root': r, 'code': c, 'target': round(tgt[c]), 'avail': av})
            tgt[c] = av
    for c in SHORTFALL_SINK:
        if moved <= 0: break
        room = int(avail.get(c, 0)) - tgt.get(c, 0.0)
        take = min(moved, max(room, 0.0))
        if take > 0:
            tgt[c] = tgt.get(c, 0.0) + take; moved -= take
            log.append({'stage': 'cell_redistribute', 'root': r, 'to': c, 'rows': round(take)})
    if moved > 0.5:
        log.append({'stage': 'cell_unplaced', 'root': r, 'rows': round(moved)})
    cell_tgt.update({(r, c): int(round(v)) for c, v in tgt.items() if v >= 1})

# ---------------------------------------------------------------- greedy state
Q = raw['query'].to_numpy(); IK = raw.norm_key.to_numpy(); RK = raw['rank'].to_numpy()   # reuse on norm identity
taken = np.zeros(len(raw), bool)
item_cnt = defaultdict(int)
bq = dict(budget)
phase = np.full(len(raw), '', dtype=object)

def rank_major(idx):
    """rank-major round-robin: all rank-0 rows (random query order), then rank-1, ..."""
    idx = np.asarray(idx, int)
    jitter = rng.random(len(idx))
    return idx[np.lexsort((jitter, RK[idx]))]

def fill(idx, need, tag_):
    got = 0
    for i in idx:
        if got >= need: break
        q, k = Q[i], IK[i]
        if taken[i] or bq.get(q, 0) <= 0 or item_cnt[k] >= ITEM_CAP: continue
        taken[i] = True; phase[i] = tag_; bq[q] -= 1; item_cnt[k] += 1; got += 1
    return got

cell_rows = {k: v.to_numpy() for k, v in raw.groupby(['root', 'code']).groups.items()}

# ---------------------------------------------------------------- 3. brand clusters (M1)
m1 = raw[(raw.code == 'M1') & (raw.bkey != '')]
best = {}
for (q, bk), idx in sorted(m1.groupby(['query', 'bkey']).groups.items()):
    ix = np.asarray(sorted(idx.to_numpy(), key=lambda i: RK[i]))
    if len(ix) >= 3 and (q not in best or len(ix) > len(best[q])):
        best[q] = ix
eligible = sorted(q for q, ix in best.items() if budget.get(q, 0) >= 3)
G_plan = int(sum(1 for v in budget.values() if v >= 3))
K = int(round(BRAND_CLUSTER_FRAC * len(eligible)))   # of queries that CAN cluster (M1 seeds)
committed = 0
for j in rng.permutation(len(eligible)):
    if committed >= K: break
    q = eligible[j]
    ix = [i for i in best[q] if item_cnt[IK[i]] < ITEM_CAP][:budget[q]]
    if len(ix) < 3: continue
    for i in ix:
        taken[i] = True; phase[i] = 'brand_cluster'; item_cnt[IK[i]] += 1
    bq[q] = 0; committed += 1
if committed < K:
    log.append({'stage': 'brand_cluster_short', 'wanted': K, 'got': committed})
print(f"brand clusters: {committed:,}/{K:,} queries (eligible {len(eligible):,}, G_plan {G_plan:,}), "
      f"rows {int(taken.sum()):,}")

# ---------------------------------------------------------------- 4. cell fill (tightest first)
consumed = defaultdict(int)
for r, c in zip(raw.root.to_numpy()[taken], raw.code.to_numpy()[taken]):
    consumed[(r, c)] += 1
# fill order: scarcest ROOT first (supply/target), then tightest cell within the root — a
# query's budget is shared across cells, so the scarce department must claim it first
root_scarcity = {r: root_avail.get(r, 0) / max(root_tgt.get(r, 1), 1) for r in root_tgt}
tightness = sorted(cell_tgt, key=lambda k: (root_scarcity.get(k[0], 1e9),
                                            len(cell_rows.get(k, ())) / max(cell_tgt[k], 1), k))
for key in tightness:
    need = cell_tgt[key] - consumed.get(key, 0)
    if need <= 0: continue
    idx = cell_rows.get(key, np.array([], int))
    consumed[key] += fill(rank_major(idx), need, 'cell_fill')

# ---------------------------------------------------------------- 5. shortfall redistribution
deficit = {k: cell_tgt[k] - consumed.get(k, 0) for k in cell_tgt if cell_tgt[k] > consumed.get(k, 0)}
for (r, _), d in sorted(deficit.items()):
    for c in SHORTFALL_SINK:
        if d <= 0: break
        idx = cell_rows.get((r, c), np.array([], int))
        got = fill(rank_major(idx), d, 'refill')
        consumed[(r, c)] += got
        if got: log.append({'stage': 'refill', 'root': r, 'to': c, 'rows': int(got)})
        d -= got
rem = N_base - int(taken.sum())
if rem > 0:
    for c in SHORTFALL_SINK:
        if rem <= 0: break
        idx = np.flatnonzero((raw.code == c).to_numpy() & ~taken)
        got = fill(rank_major(idx), rem, 'topup')
        if got: log.append({'stage': 'global_topup', 'to': c, 'rows': int(got)})
        rem -= got
if rem > 0:
    log.append({'stage': 'unfilled', 'rows': int(rem)})

# ---------------------------------------------------------------- output + manifest
sel = raw[taken].copy()
sel['phase'] = phase[taken]
sel['contrast'] = False
# device-model CONTRAST rows (review pt 6): every selected templated query also carries up to
# CONTRAST_PER_QUERY right-model candidates so synthetic queries are not an irrelevant-only
# shortcut. They are judged like everything else and enter the set with the judge's label.
CONTRAST_PER_QUERY = 3
_cf = f'{E17}/outputs/mined_v5_contrast.parquet'
if os.path.exists(_cf):
    con = pd.read_parquet(_cf)
    con['norm_key'] = nid.reindex(con.item_key.values).values
    tq = set(sel[sel.templated]['query'])
    con = con[con['query'].isin(tq)].sort_values(['query', 'rank'])
    con = con.groupby('query').head(CONTRAST_PER_QUERY)
    con['root'] = root_of.reindex(con.item_key.values).values
    con['code'] = 'M3b_match'; con['phase'] = 'contrast'; con['contrast'] = True
    sel = pd.concat([sel, con[sel.columns.intersection(con.columns)]], ignore_index=True)
    print(f"contrast rows attached: {len(con):,} for {con['query'].nunique():,} templated queries")
sel = sel.drop(columns=['bkey'], errors='ignore').sort_values(['query', 'item_key']).reset_index(drop=True)
sel.to_parquet(OUT, index=False)

def bucket(s): return s.where(s.isin(ROOT_TARGET), 'other')
tgt_prof = dict(ROOT_TARGET, other=ROOT_REMAINDER)
tvd = lambda p: 0.5 * sum(abs(p.get(k, 0.0) - v) for k, v in tgt_prof.items())
tvd_before = tvd(bucket(raw.root).value_counts(normalize=True).to_dict())
tvd_after = tvd(bucket(sel.root).value_counts(normalize=True).to_dict())
per_q = sel.groupby('query').size()
ge3 = per_q[per_q >= 3].index
sub = sel[sel['query'].isin(ge3)]
obm = (sub.assign(bk=bkey_of.reindex(sub.item_key.values).fillna('').values)
          .groupby('query').agg(nc=('code', 'nunique'), c0=('code', 'first'), nb=('bk', 'nunique')))
achieved_cluster = float(((obm.nc == 1) & (obm.c0 == 'M1') & (obm.nb == 1)).mean()) if len(obm) else 0.0
manifest = {
    'seed': SEED, 'raw': args.raw, 'n_raw': int(n0), 'n_selected': int(len(sel)), 'n_negatives': int((~sel.contrast).sum()), 'n_contrast': int(sel.contrast.sum()), 'n_target': N_TARGET,
    'drops': {'dupes': int(n_dupes), 'no_pool': int(n_nopool), 'known_RS_in_store': int(n_known_rs),
              'public_query_collisions': int(n_pubq)},
    'constants': {'K_Q': K_Q, 'Q_CAP': Q_CAP, 'K_FASHION': K_FASHION, 'FLOOR_FASHION': FLOOR_FASHION, 'FASHION_ROOTS': sorted(FASHION_ROOTS), 'TEMPLATED_BUDGET': TEMPLATED_BUDGET, 'ITEM_CAP': ITEM_CAP,
                  'BRAND_CLUSTER_FRAC': BRAND_CLUSTER_FRAC, 'ROOT_TARGET': ROOT_TARGET,
                  'RECIPE_TARGET': RECIPE_TARGET, 'SHORTFALL_SINK': SHORTFALL_SINK,
                  'fill_order': 'scarcest root first, tightest cell, rank-major round-robin', 'Q_FLOOR': Q_FLOOR},
    'budgets': {'n_queries': int(len(budget)), 'mean_cap': round(float(np.mean(b)), 2),
                'mean_supply_limited': round(float(np.mean(capq)), 2), 'capacity': int(capq.sum())},
    'root_shares': {'target': tgt_prof,
                    'before': bucket(raw.root).value_counts(normalize=True).round(4).to_dict(),
                    'achieved': bucket(sel.root).value_counts(normalize=True).round(4).to_dict(),
                    'achieved_full': sel.root.value_counts(normalize=True).round(4).to_dict(),
                    'tvd_before': round(tvd_before, 4), 'tvd_after': round(tvd_after, 4)},
    'recipe_shares': {'target': RECIPE_TARGET,
                      'natural_raw': raw.code.value_counts(normalize=True).round(4).to_dict(),
                      'achieved': sel.code.value_counts(normalize=True).round(4).to_dict()},
    'per_query': {'mean': round(float(per_q.mean()), 3), 'max': int(per_q.max()),
                  'n_queries': int(sel['query'].nunique()), 'frac_ge3': round(float((per_q >= 3).mean()), 4)},
    'rank': {'mean': round(float(sel['rank'].mean()), 2), 'p90': int(sel['rank'].quantile(.9))},
    'brand_cluster': {'target_frac': BRAND_CLUSTER_FRAC, 'planned_K': K, 'committed': committed,
                      'achieved_frac_of_ge3': round(achieved_cluster, 4)},
    'item_reuse': {'max': int(sel.norm_key.value_counts().max()),
                   'n_items': int(sel.norm_key.nunique()),
                   'n_reused': int((sel.norm_key.value_counts() > 1).sum()),
                   'identity': 'normalized 6-field tuple'},
    'origin_by_root': (lambda o: sel.assign(origin=o.reindex(sel.item_key.values).fillna('missing').values)
                       .groupby('root').origin.value_counts(normalize=True).round(3).unstack().fillna(0)
                       .to_dict(orient='index'))(pd.read_parquet(f'{W}/asya/s2_exp/e03_mine/outputs/item_pool.parquet',
                                                                  columns=['item_key', 'origin']).set_index('item_key').origin),
    'templated_rows': int(sel.templated.sum()),
    'phases': sel.phase.value_counts().to_dict(),
    'redistribution_log': log,
}
manifest['sha256'] = hashlib.sha256(open(OUT, 'rb').read()).hexdigest()
json.dump(manifest, open(OUT.replace('.parquet', '.manifest.json'), 'w'), indent=1, default=str, ensure_ascii=False)
print(f"\nselected {len(sel):,} -> {OUT}")
print(f"root TVD vs DEV-I: {tvd_before:.3f} -> {tvd_after:.3f} | achieved {manifest['root_shares']['achieved']}")
print("recipe shares:", manifest['recipe_shares']['achieved'])
print(f"per-query mean {per_q.mean():.2f} max {per_q.max()} | items {sel.norm_key.nunique():,} reuse max "
      f"{sel.norm_key.value_counts().max()} | brand-cluster frac {achieved_cluster:.3f} | "
      f"rank mean {sel['rank'].mean():.1f} | log {len(log)} entries | unfilled {rem}")
