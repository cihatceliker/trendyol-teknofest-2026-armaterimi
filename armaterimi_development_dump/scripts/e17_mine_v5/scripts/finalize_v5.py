"""E17 finalize: judged_v5 -> the frozen irrelevant set (post-judge constrained re-selection).

Recipe keep-rates differ (M8 ~0.6), so pre-judge quotas drift after judging. This stage:
  1. takes judged-I negatives (judge_label == irrelevant, contrast == False) as the new "raw"
     (same schema as mined_v5_raw) and runs select_v5.py over them with --tag final --n N_FINAL,
     so the FROZEN set meets the department/recipe/cap constraints on judged rows;
  2. writes outputs/irrelevant_final.parquet (query, item_key, recipe, rank, templated, root,
     judge_label='irrelevant') + outputs/irrelevant_reserve.parquet (judged-I not selected; for
     the dose ablation, never silently merged);
  3. QUARANTINES judge-semi / judge-relevant rows (negatives only) into
     outputs/judged_extras_quarantine.parquet — not organizer-equivalent truth; an ablation arm;
  4. keeps contrast rows with their judge label in outputs/contrast_final.parquet (the
     templated-query balance set; small, documented);
  5. re-runs gates_v5 on the final selection (--tag final, scores from the sha-bound file of the
     pre-judge selection, re-indexed to the final rows);
  6. writes outputs/FREEZE.manifest.json with every input sha. After this nothing here changes.
Usage: finalize_v5.py [--n N_FINAL]
"""
import argparse, hashlib, json, os, subprocess, sys, time
import pandas as pd
W = '/home/jupyter/trendyol-teknofest-work'
E17 = f'{W}/asya/s2_exp/e17_mine_v5'
PY = f'{W}/.venv-gcp/bin/python'
ap = argparse.ArgumentParser(); ap.add_argument('--n', type=int, default=150_000); args = ap.parse_args()
def sha(p): return hashlib.sha256(open(p, 'rb').read()).hexdigest()
def log(*a): print(f"[finalize_v5 {time.strftime('%H:%M:%S')}]", *a, flush=True)

j = pd.read_parquet(f'{E17}/outputs/judged_v5.parquet')
assert j.judge_label.notna().mean() >= 0.995, 'judged_v5 is not fully labelled'
neg = j[~j.contrast]
con = j[j.contrast]
ji = neg[neg.judge_label == 'irrelevant']
extras = neg[neg.judge_label.isin(['semi-relevant', 'relevant'])]
log(f"judged negatives {len(neg):,}: irrelevant {len(ji):,} | extras {len(extras):,} | contrast {len(con):,}")

raw_cols = ['query', 'item_key', 'recipe', 'step', 'rank', 'why', 'templated', 'seed_query_leaf', 'item_leaf']
ji_raw = ji[[c for c in raw_cols if c in ji.columns]].reset_index(drop=True)
ji_path = f'{E17}/outputs/_judged_I_as_raw.parquet'
ji_raw.to_parquet(ji_path, index=False)
n_final = min(args.n, len(ji_raw))
r = subprocess.run([PY, f'{E17}/scripts/select_v5.py', '--raw', ji_path, '--tag', 'final', '--n', str(n_final)])
if r.returncode != 0:
    sys.exit('select_v5 --tag final failed')
fin = pd.read_parquet(f'{E17}/outputs/selected_v5_final.parquet')
fin = fin[~fin.contrast] if 'contrast' in fin.columns else fin
fin = fin.drop(columns=[c for c in ('phase', 'contrast', 'code') if c in fin.columns])
fin['judge_label'] = 'irrelevant'
fin.to_parquet(f'{E17}/outputs/irrelevant_final.parquet', index=False)
key = set(zip(fin['query'], fin.item_key))
res = ji[[not (q, k) in key for q, k in zip(ji['query'], ji.item_key)]]
res.to_parquet(f'{E17}/outputs/irrelevant_reserve.parquet', index=False)
extras.to_parquet(f'{E17}/outputs/judged_extras_quarantine.parquet', index=False)
con.to_parquet(f'{E17}/outputs/contrast_final.parquet', index=False)

# post-judge gates on the final set, Asure scores re-indexed from the pre-judge scoring file
pre_sha = sha(f'{E17}/outputs/selected_v5.parquet')[:12]
scores = f'{E17}/outputs/asure_scores_v5_{pre_sha}.parquet'
_tp = f'{E17}/outputs/topup_v5.parquet'
if os.path.exists(_tp):                       # combine with the sha-bound top-up scores
    tp_sha = sha(_tp)[:12]
    tp_scores = f'{E17}/outputs/asure_scores_v5_{tp_sha}.parquet'
    assert os.path.exists(tp_scores), f'top-up scores missing: {tp_scores}'
    comb = pd.concat([pd.read_parquet(scores)[['query', 'item_key', 'asure_margin']],
                      pd.read_parquet(tp_scores)[['query', 'item_key', 'asure_margin']]], ignore_index=True)
    scores = f'{E17}/outputs/asure_scores_v5_combined_{pre_sha}_{tp_sha}.parquet'
    comb.to_parquet(scores, index=False)
r = subprocess.run([PY, f'{E17}/scripts/gates_v5.py', '--selected', f'{E17}/outputs/irrelevant_final.parquet',
                    '--tag', 'final', '--scores', scores, '--post-judge'])
g = json.load(open(f'{E17}/outputs/gates_v5_final.json'))
man = {
    'created_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
    'irrelevant_final': {'n': int(len(fin)), 'sha256': sha(f'{E17}/outputs/irrelevant_final.parquet'),
                         'recipes': fin.recipe.value_counts().to_dict(),
                         'roots': fin.root.value_counts(normalize=True).round(4).to_dict()},
    'irrelevant_reserve': {'n': int(len(res)), 'sha256': sha(f'{E17}/outputs/irrelevant_reserve.parquet')},
    'judged_extras_quarantine': {'n': int(len(extras)), 'labels': extras.judge_label.value_counts().to_dict(),
                                 'by_recipe': extras.groupby('recipe').judge_label.value_counts().unstack().fillna(0).astype(int).to_dict(orient='index'),
                                 'sha256': sha(f'{E17}/outputs/judged_extras_quarantine.parquet')},
    'contrast_final': {'n': int(len(con)), 'labels': con.judge_label.value_counts().to_dict(),
                       'sha256': sha(f'{E17}/outputs/contrast_final.parquet')},
    'post_judge_gates': {'all_passed': g.get('all_passed'), 'checks': g.get('checks'),
                         'adversarial_auc': g['adversarial']['auc']},
    'inputs': {p: sha(f'{E17}/outputs/{p}') for p in ('mined_v5_raw.parquet', 'selected_v5.parquet', 'judged_v5.parquet',
                                                        'gates_v5.json', 'judge_canary_verdict.json', 'audit_300_verdict.json')
               if os.path.exists(f'{E17}/outputs/{p}')},
    'label_store_sha256': sha(f'{W}/asya/s2_exp/e16_mine_v4/outputs/label_store.parquet'),
    'scripts': {s: sha(f'{E17}/scripts/{s}') for s in os.listdir(f'{E17}/scripts') if s.endswith('.py')},
}
json.dump(man, open(f'{E17}/outputs/FREEZE.manifest.json', 'w'), indent=1, ensure_ascii=False)
log(f"FROZEN: irrelevant_final {len(fin):,} | reserve {len(res):,} | quarantine {len(extras):,} | contrast {len(con):,} | post-judge gates all_passed={g.get('all_passed')}")
