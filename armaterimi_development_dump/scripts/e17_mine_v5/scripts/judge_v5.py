"""E17 judge runner: selected_v5 -> label store, fail-closed, in two disjoint slices.

  --stage split    REFUSES unless outputs/gates_v5.json has all_passed=True for the CURRENT
                   selected_v5 sha. Drops pairs already in the label store (free labels, by
                   normalized item identity), then splits the rest into a canary stratified by
                   recipe x root x templated (>= MIN_PER_RECIPE rows per recipe where available,
                   ~CANARY_N total) and a remainder. Contrast rows ride in the canary.
  --stage canary   label.py on the canary slice (CAP in $), re-run once for missing batches,
                   REQUIRE coverage >= 0.995 of sent rows, then the pre-registered kill rule:
                   overall irrelevant keep-rate >= 0.85 AND no non-M8 negative recipe with
                   n >= 200 below 0.80. Exports outputs/audit_300.parquet (300 judged-I rows,
                   recipe-stratified, full text, NO judge label) for the blind hand audit.
  --stage rest     REFUSES unless canary verdict passed AND outputs/audit_300_verdict.json says
                   precision >= 0.85. Labels the remainder.
  --stage merge    rebuild the label store from ALL caches (labels_v4.py) and write
                   outputs/judged_v5.parquet = selected_v5 + judge_label (+ provenance).
Cache keys in label.py include the exact row set, so disjoint slices never re-pay.
Provenance recorded: sha256 of rubric_stage2.md, common/normalize.py, e04 label.py, judge model.
"""
import argparse, hashlib, json, os, subprocess, sys, time
import numpy as np, pandas as pd

W = '/home/jupyter/trendyol-teknofest-work'
E17 = f'{W}/asya/s2_exp/e17_mine_v5'
E16 = f'{W}/asya/s2_exp/e16_mine_v4'
E04 = f'{W}/asya/s2_exp/e04_label'
PY = f'{W}/.venv-gcp/bin/python'
sys.path.insert(0, f'{W}/asya/s2_exp/common')
SEED = 20260823
CANARY_N = 20_000
MIN_PER_RECIPE = 600
COST_PER_ROW = 0.00037
MODEL = 'google/gemini-3.7-flash'

ap = argparse.ArgumentParser()
ap.add_argument('--stage', required=True, choices=['split', 'canary', 'rest', 'topup', 'merge'])
ap.add_argument('--cap', type=float, default=None)
ap.add_argument('--conc', type=int, default=6)
args = ap.parse_args()

def log(*a): print(f"[judge_v5 {time.strftime('%H:%M:%S')}]", *a, flush=True)
def sha(p): return hashlib.sha256(open(p, 'rb').read()).hexdigest()
SEL = f'{E17}/outputs/selected_v5.parquet'
PROV = {'rubric_sha256': sha(f'{W}/rubric_stage2.md'), 'normalize_py_sha256': sha(f'{W}/asya/s2_exp/common/normalize.py'),
        'label_py_sha256': sha(f'{E04}/scripts/label.py'), 'model': MODEL, 'batch': 25, 'temperature': 0}

def require_gates():
    g = json.load(open(f'{E17}/outputs/gates_v5.json'))
    cur = sha(SEL)[:12]
    if g.get('selection_sha12') != cur:
        sys.exit(f"REFUSE: gates_v5.json is for selection {g.get('selection_sha12')}, current is {cur}")
    if not g.get('all_passed'):
        sys.exit(f"REFUSE: gates not all passed: {g.get('checks')}")
    return g

def run_label(cand_path, out_path, cap):
    env = dict(os.environ, CAND=cand_path, OUT=out_path, N_PER_RECIPE='0', BATCH='25',
               CONC=str(args.conc), CAP=str(cap), MODEL=MODEL, SEED=str(SEED))
    for attempt in (1, 2):                       # second pass only re-sends batches missing from cache
        log(f"label.py CAND={os.path.basename(cand_path)} CAP=${cap:.2f} (pass {attempt})")
        r = subprocess.run([PY, f'{E04}/scripts/label.py'], env=env, cwd=E04)
        if r.returncode != 0:
            sys.exit(f"label.py failed rc={r.returncode}")
        res = pd.read_parquet(out_path)
        n = len(pd.read_parquet(cand_path))
        cov = len(res) / n
        log(f"coverage {len(res):,}/{n:,} = {cov:.4f}")
        if cov >= 0.995:
            return res
    sys.exit(f"REFUSE: coverage {cov:.4f} < 0.995 after 2 passes")

if args.stage == 'split':
    g = require_gates()
    sel = pd.read_parquet(SEL)
    nid = pd.read_parquet(f'{E17}/outputs/item_norm_identity.parquet').set_index('item_key').norm_key
    store = pd.read_parquet(f'{E16}/outputs/label_store.parquet', columns=['query', 'item_key', 'judge_label'])
    store['norm_key'] = nid.reindex(store.item_key.values).values
    sel['norm_key'] = nid.reindex(sel.item_key.values).values
    m = sel.merge(store.dropna(subset=['norm_key']).drop_duplicates(['query', 'norm_key'])[['query', 'norm_key', 'judge_label']],
                  on=['query', 'norm_key'], how='left')
    covered = m.judge_label.notna().values
    unc = sel[~covered].copy()
    unc['code'] = unc.recipe.str.extract(r'^(M\d+[abc]?|M3b_device_match)', expand=False)
    unc['stratum'] = unc.code + '|' + unc.root.astype(str) + '|' + unc.templated.astype(str)
    rng = np.random.default_rng(SEED)
    # stratified canary: proportional to stratum size, with a per-recipe floor; contrast rows all in
    frac = CANARY_N / max(len(unc), 1)
    take = np.zeros(len(unc), bool)
    idx_by_code = unc.groupby('code').indices
    for code, ix in idx_by_code.items():
        if code == 'M3b_device_match':
            take[ix] = True; continue
        want = max(int(round(frac * len(ix))), min(MIN_PER_RECIPE, len(ix)))
        sub = unc.iloc[ix]
        # within recipe: proportional across (root, templated) strata
        for st, jx in sub.groupby('stratum').indices.items():
            k = int(round(want * len(jx) / len(ix)))
            if k > 0:
                take[ix[jx][rng.choice(len(jx), min(k, len(jx)), replace=False)]] = True
    can = unc[take][['query', 'item_key', 'recipe']].sort_values(['recipe', 'query', 'item_key']).reset_index(drop=True)
    rest = unc[~take][['query', 'item_key', 'recipe']].sort_values(['recipe', 'query', 'item_key']).reset_index(drop=True)
    can.to_parquet(f'{E17}/outputs/judge_canary.parquet', index=False)
    rest.to_parquet(f'{E17}/outputs/judge_remainder.parquet', index=False)
    info = {'selection_sha12': g['selection_sha12'], 'selected': int(len(sel)),
            'covered_by_store': int(covered.sum()), 'covered_labels': m.judge_label.value_counts().to_dict(),
            'uncovered': int(len(unc)), 'canary': int(len(can)), 'remainder': int(len(rest)),
            'est_cost_canary': round(len(can) * COST_PER_ROW, 2), 'est_cost_remainder': round(len(rest) * COST_PER_ROW, 2),
            'canary_recipes': can.recipe.value_counts().to_dict(), 'provenance': PROV}
    json.dump(info, open(f'{E17}/outputs/judge_split.json', 'w'), indent=1)
    log(json.dumps(info, indent=1))

elif args.stage == 'canary':
    require_gates()
    cand = f'{E17}/outputs/judge_canary.parquet'
    n = len(pd.read_parquet(cand))
    cap = args.cap or round(n * COST_PER_ROW * 1.3, 2)
    res = run_label(cand, f'{E17}/outputs/judge_canary_labels.parquet', cap)
    res['code'] = res.recipe.str.extract(r'^(M\d+[abc]?|M3b_device_match)', expand=False)
    neg = res[res.code != 'M3b_device_match']
    keep = (neg.judge_label == 'irrelevant').mean()
    per = neg.groupby('code').judge_label.apply(lambda s: (s == 'irrelevant').mean()).round(3)
    n_per = neg.code.value_counts()
    floor_fail = [c for c in per.index if c != 'M8' and n_per[c] >= 200 and per[c] < 0.80]
    con = res[res.code == 'M3b_device_match']
    verdict = {'n_labelled': int(len(res)), 'n_sent': int(n), 'keep_rate': round(float(keep), 4),
               'per_recipe_keep': per.to_dict(), 'per_recipe_n': n_per.to_dict(),
               'label_mix': neg.judge_label.value_counts().to_dict(),
               'contrast_label_mix': con.judge_label.value_counts().to_dict(),
               'rule': 'coverage>=0.995 and keep>=0.85 and no non-M8 recipe (n>=200) <0.80',
               'passed': bool(keep >= 0.85 and not floor_fail), 'floor_fail': floor_fail, 'provenance': PROV}
    json.dump(verdict, open(f'{E17}/outputs/judge_canary_verdict.json', 'w'), indent=1)
    log(json.dumps(verdict, indent=1))
    # blind audit export: 300 judged-I negatives, recipe-stratified, full normalized text, no label
    import normalize as NZ
    drop = pd.read_csv(f'{W}/asya/s2_exp/common/drop_keys.csv').key.tolist()
    pool = NZ.normalize(pd.read_parquet(f'{W}/asya/s2_exp/e03_mine/outputs/item_pool.parquet'), drop).set_index('item_key')
    ji = neg[neg.judge_label == 'irrelevant']
    rng = np.random.default_rng(SEED)
    parts = []
    for c, g in ji.groupby('code'):
        k = max(1, int(round(300 * len(g) / len(ji))))
        parts.append(g.iloc[rng.choice(len(g), min(k, len(g)), replace=False)])
    aud = pd.concat(parts).sample(frac=1, random_state=SEED).head(300).reset_index(drop=True)
    it = pool.loc[aud.item_key]
    aud_out = pd.DataFrame({'audit_id': range(len(aud)), 'query': aud['query'].map(NZ.t1).values,
                            'title': it.title.values, 'category': it.category.values, 'brand': it.brand.values,
                            'gender': it.gender.values, 'age_group': it.age_group.values,
                            'attributes': it.attributes.values, 'recipe': aud.recipe.values,
                            'item_key': aud.item_key.values})
    aud_out.to_parquet(f'{E17}/outputs/audit_300.parquet', index=False)
    log(f"audit export: {len(aud_out)} rows -> outputs/audit_300.parquet (blind)")
    log("CANARY PASSED -> run the blind audit, then --stage rest" if verdict['passed'] else "CANARY FAILED -> STOP")

elif args.stage == 'rest':
    require_gates()
    v = json.load(open(f'{E17}/outputs/judge_canary_verdict.json'))
    if not v['passed']:
        sys.exit("REFUSE: canary verdict FAILED")
    a = json.load(open(f'{E17}/outputs/audit_300_verdict.json'))
    if not a.get('passed'):
        sys.exit(f"REFUSE: blind audit not passed: {a}")
    cand = f'{E17}/outputs/judge_remainder.parquet'
    n = len(pd.read_parquet(cand))
    cap = args.cap or round(n * COST_PER_ROW * 1.3, 2)
    run_label(cand, f'{E17}/outputs/judge_remainder_labels.parquet', cap)

elif args.stage == 'topup':
    require_gates()
    v = json.load(open(f'{E17}/outputs/judge_canary_verdict.json'))
    a = json.load(open(f'{E17}/outputs/audit_300_verdict.json'))
    if not a.get('passed'):
        sys.exit(f"REFUSE: blind audit not passed: {a}")
    top = pd.read_parquet(f'{E17}/outputs/topup_v5.parquet')[['query', 'item_key', 'recipe']]
    cand = f'{E17}/outputs/judge_topup.parquet'; top.sort_values(['recipe', 'query', 'item_key']).to_parquet(cand, index=False)
    cap = args.cap or round(len(top) * COST_PER_ROW * 1.3, 2)
    run_label(cand, f'{E17}/outputs/judge_topup_labels.parquet', cap)

elif args.stage == 'merge':
    r = subprocess.run([PY, f'{E16}/scripts/labels_v4.py'], cwd=E16)
    if r.returncode != 0:
        sys.exit("labels_v4.py failed")
    store = pd.read_parquet(f'{E16}/outputs/label_store.parquet')
    nid = pd.read_parquet(f'{E17}/outputs/item_norm_identity.parquet').set_index('item_key').norm_key
    store['norm_key'] = nid.reindex(store.item_key.values).values
    st = store.dropna(subset=['norm_key']).drop_duplicates(['query', 'norm_key'])
    sel = pd.read_parquet(SEL)
    _tp = f'{E17}/outputs/topup_v5.parquet'
    if os.path.exists(_tp) and os.path.exists(f'{E17}/outputs/judge_topup_labels.parquet'):
        sel = pd.concat([sel, pd.read_parquet(_tp)[sel.columns.intersection(pd.read_parquet(_tp).columns)]], ignore_index=True)
        log(f"merged top-up rows: selection now {len(sel):,}")
    sel['norm_key'] = nid.reindex(sel.item_key.values).values
    j = sel.merge(st[['query', 'norm_key', 'judge_label', 'n_votes', 'conflicted']], on=['query', 'norm_key'], how='left')
    j.to_parquet(f'{E17}/outputs/judged_v5.parquet', index=False)
    json.dump({'selection_sha12': sha(SEL)[:12], 'n': int(len(j)), 'labelled': int(j.judge_label.notna().sum()),
               'labels': j.judge_label.value_counts().to_dict(), 'provenance': PROV,
               'label_store_sha256': sha(f'{E16}/outputs/label_store.parquet')},
              open(f'{E17}/outputs/judged_v5.manifest.json', 'w'), indent=1)
    log(f"judged_v5: {len(j):,} rows | labelled {j.judge_label.notna().sum():,} | {j.judge_label.value_counts().to_dict()}")
