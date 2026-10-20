"""E17 pilot-transfer data for the (unjudged) v5 negatives — the e07 protocol, unchanged:
balanced organizer R/S + our negatives labelled irrelevant by rule, query-disjoint 1/8 of the
negatives held out in-distribution, BERTurk-128k 2 ep (e07_pilot/scripts/train.py), DEV read.
Reference on this protocol: v4 clean 0.7708 DEV / DEV-I acc 0.695; matched 0.7358 / 0.605.
Rule labels carry the recipe contamination (M10/M8 especially) — this is a SHAPE test, the
judge fixes the labels. Output: outputs/pilotV5/{pilot_train,pilot_indist_test}.parquet
"""
import os, sys, numpy as np, pandas as pd
W = '/home/jupyter/trendyol-teknofest-work'
sys.path.insert(0, f'{W}/asya/s2_exp/common'); import normalize as NZ
E17 = f'{W}/asya/s2_exp/e17_mine_v5'
drop = pd.read_csv(f'{W}/asya/s2_exp/common/drop_keys.csv').key.tolist()
pool = NZ.normalize(pd.read_parquet(f'{W}/asya/s2_exp/e05_score/outputs/_pool_ext.parquet'), drop).set_index('item_key')
sel = pd.read_parquet(f'{E17}/outputs/selected_v5.parquet'); sel = sel[~sel.contrast]
it = pool.loc[sel.item_key]
A = pd.DataFrame({'query': sel['query'].map(NZ.t1).values, 'title': it.title.values, 'category': it.category.values,
                  'brand': it.brand.values, 'gender': it.gender.values, 'age_group': it.age_group.values,
                  'attributes': it.attributes.values, 'real_label': 'irrelevant'})
COLS = list(A.columns)
tr = pd.read_parquet(f'{W}/asya/s2_exp/common/train_norm.parquet')
O = f'{E17}/outputs/pilotV5'; os.makedirs(O, exist_ok=True)
MI = A[COLS].drop_duplicates(['query', 'title'])
uq = np.array(sorted(MI['query'].unique())); r2 = np.random.default_rng(7); r2.shuffle(uq)
hold = set(uq[:max(1, len(uq) // 8)])
MI_tr, MI_te = MI[~MI['query'].isin(hold)], MI[MI['query'].isin(hold)]
n = len(MI_tr)
RS = tr.groupby('real_label', group_keys=False).apply(lambda g: g.sample(min(n, len(g)), random_state=42))
train = pd.concat([RS, MI_tr], ignore_index=True).sample(frac=1, random_state=42).reset_index(drop=True)
train.to_parquet(f'{O}/pilot_train.parquet', index=False); MI_te.to_parquet(f'{O}/pilot_indist_test.parquet', index=False)
print(f"pilotV5: train {len(train):,} {train.real_label.value_counts().to_dict()} | indist {len(MI_te):,}")
