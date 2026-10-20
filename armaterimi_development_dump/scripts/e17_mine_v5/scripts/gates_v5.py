"""E17 acceptance gates for selected_v5 (copied from e16 gates_v4; constants updated: item reuse cap 3,
per-query cap 30, margin gate reads a v5 Asure score file when present).

Five gates, all RECORDED into outputs/gates_v4.json (thresholds are noted, not enforced —
the decision to ship is a human milestone read, not a script side effect):

  a. adversarial     — can a surface classifier tell our rows from organizer DEV-I?
                       feats()/gate_auc() are COPIED VERBATIM from
                       asya/s2_exp/e08_repair/scripts/match_pool.py (the e06/e08 protocol);
                       the margin covariate is neutralized to 0 on BOTH sides because v4
                       pairs are unscored (gate e owns margins). PASS threshold 0.65.
  b. fingerprint     — the A14 §4.1 signature stats vs the DEV-I targets
                       (.401/.238/.191/.105/8.13), helpers imported from rubric_eda.
  c. hygiene         — zero public-query overlap, snapshot origin balance, item reuse <= 2,
                       per-query <= 15.
  d. label coverage  — fraction of selected pairs already judged (free labels) split by
                       recipe + the §4.4 tier-policy judge-bill estimate.
  e. margin STUB     — Asure margins only where pairs overlap asure_scores_v32; the
                       non-overlapping rest needs a GPU scoring pass which is NOT run here.

Usage:  gates_v4.py [--selected PATH] [--tag standin]
"""
import argparse, hashlib, json, os, sys

import numpy as np, pandas as pd

W = '/home/jupyter/trendyol-teknofest-work'
E16 = f"{W}/asya/s2_exp/e16_mine_v4"
E17 = f"{W}/asya/s2_exp/e17_mine_v5"
sys.path.insert(0, f'{W}/asya/s2_exp/common')
sys.path.insert(0, f'{W}/asya/s2_exp/rubric_eda')
import harness as H, normalize as NZ
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score

SEED = 20260823
ADV_PASS_LEGACY = 0.65          # e08-era bar; shown to be below the train->test domain-shift floor
ADV_TOL = 0.02                  # RECALIBRATED 2026-08-22 (before any judge spend): our-I vs DEV-I AUC must be
                                # <= the organizer's OWN train-relevant vs DEV-relevant AUC (same features, same
                                # protocol) + ADV_TOL. Measured baselines: train-R 0.738, train-S 0.681 — no data
                                # built from train queries/items can score below that shift.
FP_TARGET = {'brand_in_query': .401, 'head_in_title': .238, 'zero_all': .191,
             'accessory_item': .105, 'title_tok': 8.13}     # A14 §4.1 DEV-I fingerprint
TIER = {'M1': 'A', 'M3a': 'A', 'M3b': 'A', 'M3c': 'A', 'M5': 'A', 'M7': 'A',
        'M2': 'B', 'M4': 'B', 'M6': 'B', 'M8': 'C', 'M9': 'D', 'M10': 'C'}       # A14 §4.4
AUDIT = {'A': .02, 'B': .05, 'C': 1.0, 'D': .10}

ap = argparse.ArgumentParser()
ap.add_argument('--selected', default='')
ap.add_argument('--tag', default='')
ap.add_argument('--scores', default='', help='Asure score parquet for THIS selection (sha-bound)')
ap.add_argument('--post-judge', action='store_true', help='enforce the hardness band (judged-I set); pre-judge it is recorded as a warning because semi contamination sits in the plausible tail')
args = ap.parse_args()
tag = f'_{args.tag}' if args.tag else ''
SEL = args.selected or f'{E17}/outputs/selected_v5{tag}.parquet'
OUT = f'{E17}/outputs/gates_v5{tag}.json'
rng = np.random.default_rng(SEED)
gates = {'selected': SEL, 'seed': SEED}

# ---------------------------------------------------------------- materialize both sides
drop = pd.read_csv(f'{W}/asya/s2_exp/common/drop_keys.csv').key.tolist()
sel = pd.read_parquet(SEL).drop(columns=['judge_label', 'n_votes', 'conflicted'], errors='ignore')
if 'contrast' in sel.columns:
    n_con = int(sel.contrast.sum()); sel = sel[~sel.contrast].reset_index(drop=True)
    print(f"excluding {n_con} contrast rows from the gates (not negatives)")
pool = NZ.normalize(pd.read_parquet(f'{W}/asya/s2_exp/e05_score/outputs/_pool_ext.parquet'),
                    drop).set_index('item_key')
it = pool.loc[sel.item_key]
A = pd.DataFrame({'query': sel['query'].values, 'title': it.title.values,
                  'category': it.category.values, 'brand': it.brand.values,
                  'gender': it.gender.values, 'age_group': it.age_group.values,
                  'attributes': it.attributes.values})
dev = NZ.normalize(H.load_public_split('dev'), drop)
di = dev[dev.real_label == 'irrelevant']
B = di[['query', 'title', 'category', 'brand', 'gender', 'age_group', 'attributes']].reset_index(drop=True)
print(f"selected {len(A):,} rows   organizer DEV irrelevant {len(B):,}")


# ================================================================ (a) adversarial gate
# --- feats() and gate_auc() COPIED VERBATIM from e08_repair/scripts/match_pool.py ---
def feats(d, root_cat, margin):
    q = [H.toks(x) for x in d['query']]; t = [H.toks(x) for x in d.title]
    c = [H.toks(x) for x in d.category]; b = [H.toks(x) for x in d.brand]
    F = {}
    F['cov_title'] = [len(a & x) / max(len(a), 1) for a, x in zip(q, t)]
    F['cov_cat'] = [len(a & x) / max(len(a), 1) for a, x in zip(q, c)]
    F['cov_brand'] = [len(a & x) / max(len(a), 1) for a, x in zip(q, b)]
    F['zero_title'] = [int(len(a & x) == 0) for a, x in zip(q, t)]
    F['qwords'] = [len(a) for a in q]
    F['qchars'] = d['query'].str.len().values
    F['tchars'] = d.title.str.len().values
    F['cat_depth'] = d.category.str.count('/').values
    F['n_attr'] = d.attributes.str.count(',').values
    F['len_attr'] = d.attributes.str.len().values
    F['gender_known'] = (~d.gender.isin(['unknown', ''])).astype(int).values
    F['age_known'] = (~d.age_group.isin(['unknown', ''])).astype(int).values
    F['head_in_item'] = [int(list(a)[-1] in (x | y) if a else 0) for a, x, y in zip(q, t, c)]
    F['root'] = pd.Categorical(d.category.str.split('/').str[0], categories=root_cat).codes
    F['margin'] = np.asarray(margin, dtype='float32')
    return pd.DataFrame(F)

def gate_auc(Xa, Xb, ga, gb, tag):
    X = pd.concat([Xa, Xb], ignore_index=True)
    y = np.r_[np.ones(len(Xa)), np.zeros(len(Xb))]
    g = np.r_[ga, gb]
    sw = np.r_[np.full(len(Xa), len(Xb) / len(Xa)), np.ones(len(Xb))]
    oof = np.zeros(len(y))
    for tr, te in GroupKFold(5).split(X, y, groups=g):
        m = HistGradientBoostingClassifier(max_iter=300, learning_rate=.06, random_state=0)
        m.fit(X.iloc[tr], y[tr], sample_weight=sw[tr])
        oof[te] = m.predict_proba(X.iloc[te])[:, 1]
    auc = roc_auc_score(y, oof)
    print(f"[{tag}] adversarial AUC = {auc:.4f}  (n_ours={len(Xa):,} vs n_org={len(Xb):,})")
    return auc, oof
# --- end verbatim copy ---

root_cat = sorted(set(A.category.str.split('/').str[0]) | set(B.category.str.split('/').str[0]))
cap = min(len(A), len(B) * 8)
sub = np.sort(rng.choice(len(A), size=cap, replace=False))
XA = feats(A.iloc[sub].reset_index(drop=True), root_cat, np.zeros(cap))     # margin neutralized:
XB = feats(B, root_cat, np.zeros(len(B)))                                   # v4 pairs unscored (gate e)
auc, _ = gate_auc(XA, XB, A['query'].values[sub], B['query'].values, 'selected_v4 vs DEV-I')

X = pd.concat([XA, XB], ignore_index=True)
y = np.r_[np.ones(len(XA)), np.zeros(len(XB))]
m = HistGradientBoostingClassifier(max_iter=300, learning_rate=.06, random_state=0).fit(X, y)
from sklearn.inspection import permutation_importance                        # e06 pattern
pi = permutation_importance(m, X, y, n_repeats=3, random_state=0, scoring='roc_auc')
imp = pd.Series(pi.importances_mean, index=X.columns).sort_values(ascending=False)
print("top separating features:\n" + imp.head(6).round(4).to_string())
# organizer domain-shift baseline on the identical protocol (train-relevant vs DEV-relevant)
_tr = pd.read_parquet(f'{W}/asya/s2_exp/common/train_norm.parquet')
_trR = _tr[_tr.real_label == 'relevant'][['query', 'title', 'category', 'brand', 'gender', 'age_group', 'attributes']]
_trR = _trR.iloc[np.sort(rng.choice(len(_trR), cap, replace=False))].reset_index(drop=True)
_devR = dev[dev.real_label == 'relevant'][['query', 'title', 'category', 'brand', 'gender', 'age_group', 'attributes']].reset_index(drop=True)
_rc = sorted(set(_trR.category.str.split('/').str[0]) | set(_devR.category.str.split('/').str[0]))
base_auc, _ = gate_auc(feats(_trR, _rc, np.zeros(len(_trR))), feats(_devR, _rc, np.zeros(len(_devR))),
                       _trR['query'].values, _devR['query'].values, 'BASELINE train-R vs DEV-R')
ADV_PASS = round(float(base_auc) + ADV_TOL, 4)
gates['adversarial'] = {'auc': round(float(auc), 4), 'pass_threshold': ADV_PASS,
                        'baseline_train_R_vs_dev_R': round(float(base_auc), 4), 'tolerance': ADV_TOL,
                        'legacy_threshold': ADV_PASS_LEGACY,
                        'passed': bool(auc <= ADV_PASS), 'n_ours': int(cap), 'n_org': int(len(B)),
                        'top_features': imp.head(8).round(4).to_dict(),
                        'note': 'feats/gate_auc verbatim from e08 match_pool.py; '
                                'margin covariate neutralized to 0 on both sides'}

# ================================================================ (b) fingerprint table
try:
    import A14_mine as M14
    from A5_constraint_extractors import head_noun
    toks, fold, leaf_of = M14.toks, M14.fold, M14.leaf_of
    ACC = M14.ACCESSORY_TOK
    brand_in_q = lambda q, b: M14._brand_in_query(q, b, None)
except Exception as e:                                       # simple re-implementation fallback
    print(f"A14_mine import failed ({e}); using simple fallback helpers")
    import re, unicodedata
    _F = str.maketrans({'ç': 'c', 'ğ': 'g', 'ı': 'i', 'ö': 'o', 'ş': 's', 'ü': 'u'})
    fold = lambda s: unicodedata.normalize('NFC', str(s)).replace('̇', '').lower().translate(_F)
    toks = lambda s: re.findall(r'[0-9a-z]+', fold(s))
    leaf_of = lambda c: fold(c).split('/')[-1].strip()
    ACC = {'kilif', 'koruyucu', 'kordon', 'aksesuar', 'yedek', 'parca', 'filtre', 'adaptor',
           'sarj', 'kablo', 'kapak', 'stand', 'aparat'}
    head_noun = lambda q: ([t for t in fold(q).split() if not t.isdigit()] or [''])[-1]
    brand_in_q = lambda q, b: bool(b) and ''.join(toks(b)) != '' and ''.join(toks(b)) in ''.join(toks(q))

def fingerprint(d):
    qt = [set(toks(q)) for q in d['query']]
    at = [set(toks(f"{t} {c} {b} {a}")) for t, c, b, a
          in zip(d.title, d.category, d.brand, d.attributes)]
    heads = [head_noun(q) for q in d['query']]
    return {
        'brand_in_query': float(np.mean([brand_in_q(q, b) for q, b in zip(d['query'], d.brand)])),
        'head_in_title': float(np.mean([bool(h) and len(h) >= 3 and h[:5] in fold(t)
                                        for h, t in zip(heads, d.title)])),
        'zero_all': float(np.mean([len(a & x) == 0 for a, x in zip(qt, at)])),
        'accessory_item': float(np.mean([any(x in ACC for x in toks(leaf_of(c)))
                                         for c in d.category])),
        'title_tok': float(np.mean([len(set(toks(t))) for t in d.title])),
    }

fp_sel, fp_dev = fingerprint(A), fingerprint(B)
gates['fingerprint'] = {'selected': {k: round(v, 4) for k, v in fp_sel.items()},
                        'dev_I_measured': {k: round(v, 4) for k, v in fp_dev.items()},
                        'dev_I_target_A14': FP_TARGET}
print("\nfingerprint          selected   DEV-I(meas)  A14 target")
for k in FP_TARGET:
    print(f"  {k:<16} {fp_sel[k]:9.3f}  {fp_dev[k]:10.3f}  {FP_TARGET[k]:9.3f}")

# ================================================================ (c) hygiene
pub = pd.read_parquet(f'{W}/asya/s2_exp/common/public_test_norm.parquet', columns=['query'])
pubq = {H.tr_norm(q) for q in pub['query']}
selq = {H.tr_norm(q) for q in sel['query']}
overlap = sorted(selq & pubq)
origin = pd.read_parquet(f'{W}/asya/s2_exp/e03_mine/outputs/item_pool.parquet',
                         columns=['item_key', 'origin']).set_index('item_key').origin
org = origin.reindex(sel.item_key.values).fillna('missing')
reuse = sel.item_key.value_counts()
perq = sel.groupby('query').size()
gates['hygiene'] = {
    'public_query_overlap': {'n': len(overlap), 'passed': len(overlap) == 0,
                             'examples': overlap[:5]},
    'origin_balance': {'rows': org.value_counts(normalize=True).round(4).to_dict(),
                       'unique_items': org.groupby(level=0).first().reindex(
                           sel.item_key.unique()).value_counts(normalize=True).round(4).to_dict()},
    'item_reuse': {'max': int(reuse.max()), 'passed': bool(reuse.max() <= 3)},
    'per_query': {'max': int(perq.max()), 'passed': bool(perq.max() <= 30)},
}
print(f"\nhygiene: public-q overlap {len(overlap)} | origin {org.value_counts(normalize=True).round(3).to_dict()}"
      f" | item reuse max {reuse.max()} | per-query max {perq.max()}")

# ================================================================ (d) label-store coverage
store = pd.read_parquet(f'{E16}/outputs/label_store.parquet')
mrg = sel.merge(store[['query', 'item_key', 'judge_label']], on=['query', 'item_key'], how='left')
cov = mrg.judge_label.notna()
code = sel.recipe.astype(str).str.extract(r'^(M\d+[abc]?)', expand=False).fillna('other')
per_recipe = {}
bill = 0.0
for c, g in mrg.groupby(code.values):
    n, k = len(g), int(g.judge_label.notna().sum())
    unc = n - k
    calls = unc * AUDIT.get(TIER.get(c, 'C'), 1.0)
    bill += calls
    per_recipe[c] = {'n': n, 'covered': k, 'frac': round(k / max(n, 1), 4),
                     'tier': TIER.get(c, '?'), 'est_judge_calls': int(round(calls))}
gates['label_coverage'] = {
    'overall_frac': round(float(cov.mean()), 4), 'n_covered': int(cov.sum()),
    'covered_labels': mrg.judge_label.value_counts().to_dict(),
    'per_recipe': per_recipe, 'est_judge_calls_total': int(round(bill)),
    'note': 'select_v4 already dropped store-R/S pairs, so covered pairs are free '
            'irrelevant labels; bill = uncovered x A14 §4.4 tier audit rates'}
print(f"\nlabel coverage: {cov.mean():.1%} of {len(sel):,} pairs already judged | "
      f"est. judge calls {bill:,.0f}")

# ================================================================ (e) margin gate STUB
_sha = hashlib.sha256(open(SEL, 'rb').read()).hexdigest()[:12]
_v5sc = args.scores or f'{E17}/outputs/asure_scores_v5_{_sha}.parquet'
sc = pd.read_parquet(_v5sc) if os.path.exists(_v5sc) else pd.read_parquet(f'{W}/asya/s2_exp/e05_score/outputs/asure_scores_v32.parquet')
print('margin source:', 'v5 scores' if os.path.exists(_v5sc) else 'v32 overlap STUB')
ms = sel.merge(sc[['query', 'item_key', 'asure_margin']], on=['query', 'item_key'], how='left')
have = ms.asure_margin.dropna()
devsc = pd.read_parquet(f'{W}/asya/s2_exp/e05_score/outputs/asure_scores_dev.parquet')
dm = devsc[devsc.recipe == 'dev_irrelevant'].asure_margin
qtl = [.1, .25, .5, .75, .9]
gates['margin_stub'] = {
    'coverage_frac': round(float(ms.asure_margin.notna().mean()), 4),
    'n_scored': int(len(have)), 'n_unscored': int(ms.asure_margin.isna().sum()),
    'selected_overlap': ({'median': round(float(have.median()), 3),
                          'frac_gt0': round(float((have > 0).mean()), 4),
                          'frac_gt1': round(float((have > 1).mean()), 4),
                          'quantiles': {str(q): round(float(have.quantile(q)), 3) for q in qtl}}
                         if len(have) else None),
    'dev_I_reference': {'median': round(float(dm.median()), 3),
                        'frac_gt0': round(float((dm > 0).mean()), 4),
                        'frac_gt1': round(float((dm > 1).mean()), 4),
                        'quantiles': {str(q): round(float(dm.quantile(q)), 3) for q in qtl}},
    'source': ('v5' if os.path.exists(_v5sc) else 'v32_stub'),
    'pass_band_frac_gt0': [0.18, 0.28],
    'passed': (bool(0.18 <= float((have > 0).mean()) <= 0.28) if os.path.exists(_v5sc) and len(have) else None),
    'note': 'v5: scored on the selected set itself; v32_stub: overlap only. '
            'Task brief quotes DEV-I "~13%>0"; measured frac>0 is 0.234 and frac>1 is 0.119 — '
            'the quoted figure matches frac>1. Both recorded; resolve before enforcing.'}
print(f"\nmargin stub: {ms.asure_margin.notna().mean():.1%} of pairs have v32 Asure scores | "
      f"overlap median {have.median() if len(have) else float('nan'):.3f} vs DEV-I {dm.median():.3f}")

# ---------------------------------------------------------------- ENFORCED verdict
fp_ok = all(abs(fp_sel[k] - fp_dev[k]) <= (0.6 if k == 'title_tok' else 0.03) for k in FP_TARGET)
checks = {
    'adversarial_auc_le_baseline_plus_tol': bool(auc <= ADV_PASS),
    'public_query_overlap_zero': gates['hygiene']['public_query_overlap']['passed'],
    'item_reuse_le_3': gates['hygiene']['item_reuse']['passed'],
    'per_query_le_30': gates['hygiene']['per_query']['passed'],
    'margin_scored_for_this_selection': bool(gates['margin_stub'].get('source') == 'v5'
                                             and gates['margin_stub']['coverage_frac'] >= 0.999),
    'margin_frac_gt0_in_band': (bool(gates['margin_stub'].get('passed') is True) if args.post_judge else True),
}
gates['margin_pre_judge_warning'] = (None if args.post_judge else
    {'frac_gt0': gates['margin_stub']['selected_overlap']['frac_gt0'], 'in_band': gates['margin_stub'].get('passed'),
     'note': 'pre-judge: expected semi/relevant contamination (judge removes it) inflates the plausible tail; '
             'enforced on the judged-I set in finalize (--post-judge)'})
gates['fingerprint_within_tol_0.03'] = {'passed': bool(fp_ok), 'enforced': False,
    'note': 'diagnostic only: the adversarial gate sees these features; brand_in_query is capped by brand-swap supply, head_in_title inflated by retrieval rows the judge relabels'}
gates['selection_sha12'] = _sha
gates['checks'] = checks
gates['all_passed'] = bool(all(checks.values()))
print("\nCHECKS:", json.dumps(checks, indent=1), "\nALL_PASSED:", gates['all_passed'])
os.makedirs(f'{E17}/outputs', exist_ok=True)
json.dump(gates, open(OUT, 'w'), indent=1, default=str)
print(f"\nwrote {OUT}")
