"""E17 blind hand-audit scorer. Input: outputs/audit_300.parquet (300 judged-irrelevant rows,
full normalized text, NO judge label) + outputs/audit_votes_<k>.csv for k independent auditors
(columns: audit_id,label in {relevant,semi-relevant,irrelevant}). Each auditor works from the
rubric only, blind to the judge and to each other. Majority vote per row (ties -> semi).
Precision = fraction of rows the auditors also call irrelevant. Kill gate: precision >= 0.85
AND fraction flipped to RELEVANT <= 0.03. Writes outputs/audit_300_verdict.json (required by
judge_v5.py --stage rest) with per-recipe precision and the disagreeing rows.
"""
import glob, json, os, sys
import pandas as pd
E17 = '/home/jupyter/trendyol-teknofest-work/asya/s2_exp/e17_mine_v5'
aud = pd.read_parquet(f'{E17}/outputs/audit_300.parquet')
files = sorted(glob.glob(f'{E17}/outputs/audit_votes_*.csv'))
if len(files) < 3:
    sys.exit(f"need >=3 auditor files, have {files}")
votes = {}
for f in files:
    v = pd.read_csv(f)
    assert {'audit_id', 'label'} <= set(v.columns), f
    assert v.label.isin(['relevant', 'semi-relevant', 'irrelevant']).all(), f"bad labels in {f}"
    votes[os.path.basename(f)] = v.set_index('audit_id').label.reindex(aud.audit_id)
V = pd.DataFrame(votes)
missing = int(V.isna().any(axis=1).sum())
PRIO = {'semi-relevant': 0, 'irrelevant': 1, 'relevant': 2}
def maj(r):
    vc = r.dropna().value_counts()
    if vc.empty: return None
    top = vc[vc == vc.max()].index.tolist()
    return sorted(top, key=lambda l: PRIO[l])[0]
aud['audit_label'] = V.apply(maj, axis=1).values
ok = aud.audit_label.notna()
prec = float((aud.audit_label[ok] == 'irrelevant').mean())
flip_rel = float((aud.audit_label[ok] == 'relevant').mean())
per = aud[ok].groupby('recipe').audit_label.apply(lambda s: (s == 'irrelevant').mean()).round(3)
agree = float((V.nunique(axis=1) == 1).mean())
verdict = {'n': int(ok.sum()), 'n_missing': missing, 'auditors': list(votes), 'precision': round(prec, 4),
           'flip_to_relevant': round(flip_rel, 4), 'inter_auditor_unanimous': round(agree, 4),
           'per_recipe_precision': per.to_dict(), 'per_recipe_n': aud[ok].recipe.value_counts().to_dict(),
           'rule': 'precision>=0.85 and flip_to_relevant<=0.03',
           'passed': bool(prec >= 0.85 and flip_rel <= 0.03)}
aud.assign(**{k: V[k].values for k in V.columns}).to_parquet(f'{E17}/outputs/audit_300_scored.parquet', index=False)
json.dump(verdict, open(f'{E17}/outputs/audit_300_verdict.json', 'w'), indent=1)
print(json.dumps(verdict, indent=1))
