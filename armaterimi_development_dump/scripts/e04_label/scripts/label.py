"""Label mined candidates with an LLM judge. Keeps ONLY rows returned `irrelevant`
(we already have 250k organizer relevant + 250k semi, and judge labels are ~92% accurate,
so teacher-labelled R/S would inject noise into classes we already have clean).

Conditions are held IDENTICAL to the benchmark that measured this judge at macro-F1 0.9238
on 1,200 disjoint DEV rows: same rubric as system prompt, batch 25, temperature 0, no
reasoning flag. Changing conditions would invalidate that measurement.

Text is materialised HERE from the raw pool + normalize.py, so normalisation stays a late,
re-runnable step and the mined artifacts remain identity-only.
Env: N_PER_RECIPE (0 = all) MODEL BATCH CONC CAP OUT SEED
"""
import os, sys, json, re, time, hashlib, threading
from concurrent.futures import ThreadPoolExecutor
import numpy as np, pandas as pd, urllib.request

W='/home/jupyter/trendyol-teknofest-work'
sys.path.insert(0,f'{W}/asya/s2_exp/common'); import normalize as NZ
HERE=f'{W}/asya/s2_exp/e04_label'; MINE=f'{W}/asya/s2_exp/e03_mine/outputs'
def _load_key(p='/home/jupyter/openrouter.txt'):
    """The key file may carry extra lines (notes). Take the first sk- token only."""
    for line in open(p, encoding='utf-8'):
        s = line.strip()
        if s.startswith('sk-'):
            return s
    raise SystemExit('no sk- key found in ' + p)
KEY=_load_key()
MODEL=os.environ.get("MODEL","google/gemini-3.7-flash")
BATCH=int(os.environ.get("BATCH",25)); CONC=int(os.environ.get("CONC",6))
CAP=float(os.environ.get("CAP",1.0)); NPR=int(os.environ.get("N_PER_RECIPE",100))
SEED=int(os.environ.get("SEED",20260819))
OUT=os.environ.get("OUT",f"{HERE}/outputs/yield_probe.parquet")
def log(m): print(f"[{time.strftime('%H:%M:%S')}] {m}",flush=True)

RUB=open(f'{W}/rubric_stage2.md').read()
SYS=RUB+"""

---

# OUTPUT INSTRUCTION (overrides any example above)

You will receive a numbered batch of query-product pairs. Apply the rubric to each one
independently. Return ONLY a JSON array, one object per pair, in input order:

[{"id": 1, "label": "relevant"}, {"id": 2, "label": "irrelevant"}, ...]

`label` must be exactly one of: "relevant", "semi-relevant", "irrelevant".
No prose, no markdown fences, no explanation. Every input id must appear exactly once.
"""

_c=os.environ.get('CAND','mined_candidates.parquet')
cand=pd.read_parquet(_c if os.path.isabs(_c) else f'{MINE}/{_c}')
pool=pd.read_parquet(os.environ.get('POOL',f'{MINE}/item_pool.parquet'))
drop=pd.read_csv(f'{W}/asya/s2_exp/common/drop_keys.csv').key.tolist()
pool_n=NZ.normalize(pool, drop)                      # normalise LATE, here
pool_n=pool_n.set_index('item_key')
if NPR>0:
    cand=(cand.groupby('recipe', group_keys=False)
              .apply(lambda g: g.sample(min(NPR,len(g)), random_state=SEED))
              .sort_values(['recipe','query','item_key']).reset_index(drop=True))
log(f"{len(cand):,} candidates | recipes {cand.recipe.value_counts().to_dict()}")

def block(i,r):
    it=pool_n.loc[r.item_key]
    q=NZ.t1(r.query)
    return (f'--- {i} ---\nArama sorgusu: {q}\nÜrün adı: {it.title}\nKategori: {it.category}\n'
            f'Marka: {it.brand}\nCinsiyet: {it.gender} | Yaş: {it.age_group}\nÖzellikler: {it.attributes}')

LAB={"relevant","semi-relevant","irrelevant"}
def parse(t, n):
    t=re.sub(r"^```(?:json)?|```$","",t.strip(),flags=re.M).strip()
    m=re.search(r"\[.*\]",t,re.S); out={}
    if m:
        try:
            for o in json.loads(m.group(0)):
                if isinstance(o,dict):
                    l=str(o.get("label","")).strip().lower().replace("_","-")
                    if l=="semi relevant": l="semi-relevant"
                    if l in LAB and o.get("id") is not None: out[int(o["id"])]=l
        except Exception: pass
    if len(out)<n:
        for line in t.splitlines():
            mm=re.search(r'"?id"?\s*[:=]\s*(\d+).*?"?(relevant|semi-relevant|irrelevant)"?',line,re.I)
            if mm: out.setdefault(int(mm.group(1)),mm.group(2).lower())
    return out

def call(msgs):
    body=json.dumps({"model":MODEL,"messages":msgs,"temperature":0,"usage":{"include":True}}).encode()
    req=urllib.request.Request("https://openrouter.ai/api/v1/chat/completions",data=body,
        headers={"Authorization":f"Bearer {KEY}","Content-Type":"application/json",
                 "X-Title":"teknofest-label"})
    with urllib.request.urlopen(req,timeout=300) as r: return json.loads(r.read())

# cache key MUST include the candidate source + the exact row set, or two different
# probes with the same row COUNT silently replay each other's results.
sig=hashlib.sha256((MODEL+os.environ.get('CAND','')+str(SEED)+str(BATCH)+
     hashlib.sha256(pd.util.hash_pandas_object(cand[['query','item_key']],index=False)
       .values.tobytes()).hexdigest()).encode()).hexdigest()[:10]
cache=f"{HERE}/cache/{re.sub(r'[^a-z0-9]','_',MODEL.lower())}.{sig}.jsonl"
done={}
if os.path.exists(cache):
    for l in open(cache,encoding='utf-8'):
        try: d=json.loads(l); done[d["batch"]]=d
        except Exception: pass
batches=[(b,cand.iloc[b*BATCH:(b+1)*BATCH]) for b in range((len(cand)+BATCH-1)//BATCH)]
todo=[x for x in batches if x[0] not in done]
log(f"{len(batches)} batches, {len(todo)} to run, cache {os.path.basename(cache)}")
lock=threading.Lock(); spent=[sum(float(d.get('cost') or 0) for d in done.values())]
def work(item):
    b,df=item
    if spent[0]>=CAP: return
    user="\n\n".join(block(i+1,r) for i,(_,r) in enumerate(df.iterrows()))
    msgs=[{"role":"system","content":SYS},{"role":"user","content":user}]
    for a in range(4):
        try:
            resp=call(msgs); txt=resp["choices"][0]["message"].get("content") or ""
            u=resp.get("usage",{}) or {}
            rec={"batch":b,"text":txt,"keys":df.item_key.tolist(),"queries":df['query'].tolist(),
                 "recipes":df.recipe.tolist(),"cost":u.get("cost"),
                 "pt":u.get("prompt_tokens"),"ct":u.get("completion_tokens")}
            with lock:
                open(cache,'a',encoding='utf-8').write(json.dumps(rec,ensure_ascii=False)+"\n")
                spent[0]+=float(u.get("cost") or 0)
            return
        except Exception as e:
            if a==3: log(f"  ! batch {b} failed: {type(e).__name__} {str(e)[:100]}")
            time.sleep(3*(a+1))
t0=time.time()
with ThreadPoolExecutor(CONC) as ex: list(ex.map(work,todo))
recs={}
for l in open(cache,encoding='utf-8'):
    try: d=json.loads(l); recs[d["batch"]]=d
    except Exception: pass
rows=[]; cost=0.0
for b,d in recs.items():
    got=parse(d["text"],len(d["keys"]))
    for i,(k,q,rc) in enumerate(zip(d["keys"],d["queries"],d["recipes"]),start=1):
        if i in got: rows.append((q,int(k),rc,got[i]))
    cost+=float(d.get("cost") or 0)
res=pd.DataFrame(rows,columns=['query','item_key','recipe','judge_label'])
res.to_parquet(OUT,index=False)
log(f"labelled {len(res):,}/{len(cand):,}  cost ${cost:.4f}  ({time.time()-t0:.0f}s)")
piv=pd.crosstab(res.recipe,res.judge_label,normalize='index').round(3)
cnt=res.recipe.value_counts()
piv['n']=cnt
log("\n=== IRRELEVANT YIELD BY RECIPE ===\n"+piv.sort_values('irrelevant',ascending=False).to_string())
log(f"\noverall irrelevant yield: {(res.judge_label=='irrelevant').mean():.3f}")
log(f"cost per KEPT irrelevant row: ${cost/max((res.judge_label=='irrelevant').sum(),1)*1000:.2f}/1k")
