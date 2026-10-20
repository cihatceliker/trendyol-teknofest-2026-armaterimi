"""E15 train_student — distill the Asure-12B teacher into a BERTurk-128k cross-encoder.

Extends the e07 pilot recipe (same text pair serialization: query vs
"title | brand | category | gender age_group | attributes") with the distillation loss of
model_survey/B1 §10 (soft targets beat hard labels; volume is the binding constraint):

  per row:  L = alpha * CE(hard label, label_smoothing=SMOOTH, class weights CW)   [if hard]
              + beta  * T^2 * softCE(teacher probs tempered at T || student logits/T) [if teacher]
  a row missing one signal uses the other at FULL weight (alpha/beta -> 1.0 for that row).
  softCE(q, s) = -sum q * log_softmax(s/T) == KL(q || softmax(s/T)) + const(q): same gradients,
  cheaper. Teacher probs are tempered as q_i ∝ p_i^(1/T) (the teacher emits probs, not logits).

Batching: length-grouped batches via e11's token-budget planner (si_common.train_plan) —
BUDGET=0 (default) degenerates it to fixed-size BS batches cut from length-sorted pools
(B1 §3: bucketing ~2x over pad-to-256); BUDGET>0 switches to true token-budget batches.
bf16 autocast WITHOUT GradScaler (bf16 needs none — e07's GradScaler+bf16 was a no-op).
AdamW(wd 0.01), linear-warmup(5%)-cosine, grad clip 1.0, seed 42. Not resumable by design
(runs are ~15-60 min); checkpoint at END only.

Checkpoint outputs/ckpt_<TAG>/ is simultaneously HF format (save_pretrained) AND the lean
e12 format: model.safetensors + config.json + tokenizer.json + decision.json — the same file
set e12_lean_infer/scripts/verify_parity.py exports (that script is a top-level parity
harness, not an importable exporter, so the three export lines are mirrored here). Verify
parity on CPU afterwards with:
  MODEL=<ckpt_dir> OUT=<ckpt_dir>_parity N=200 .venv-gcp/bin/python \
      asya/s2_exp/e12_lean_infer/scripts/verify_parity.py

Env: MODEL (dbmdz/bert-base-turkish-128k-cased) DATA TAG OUT EPOCHS(2) BS(64) LR(3e-5)
     MAXLEN(256) TEMP(2.0) ALPHA(0.5) BETA(0.5) SMOOTH(0.05) CW(auto|none|w_i,w_s,w_r)
     BUDGET(0) ROWS(0=all) LOG_EVERY(200) NW(2 dataloader workers) DEVICE(cuda; cpu needs --smoke)
CLI: --smoke -> ROWS=512 EPOCHS=1 LOG_EVERY=2 (plumbing check; CPU allowed)
"""
import os, sys, json, time, math
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
import numpy as np, pandas as pd, torch, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

W = "/home/jupyter/trendyol-teknofest-work"
sys.path.insert(0, f"{W}/asya/s2_exp/common")
sys.path.insert(0, f"{W}/asya/s2_exp/e11_si_lora/scripts")
from si_common import log, sha256, train_plan

E15    = f"{W}/asya/s2_exp/e15_student"
SMOKE  = "--smoke" in sys.argv
MODEL  = os.environ.get("MODEL", "dbmdz/bert-base-turkish-128k-cased")
DATA   = os.environ.get("DATA", f"{E15}/outputs/student_train.parquet")
TAG    = os.environ.get("TAG", "smoke" if SMOKE else "s1")
OUT    = os.environ.get("OUT", f"{E15}/outputs")
EPOCHS = int(os.environ.get("EPOCHS", 1 if SMOKE else 2))
BS     = int(os.environ.get("BS", 64))
LR     = float(os.environ.get("LR", 3e-5))
MAXLEN = int(os.environ.get("MAXLEN", 256))
TEMP   = float(os.environ.get("TEMP", 2.0))
ALPHA  = float(os.environ.get("ALPHA", 0.5))
BETA   = float(os.environ.get("BETA", 0.5))
SMOOTH = float(os.environ.get("SMOOTH", 0.05))
CW     = os.environ.get("CW", "auto")
BUDGET = int(os.environ.get("BUDGET", 0))          # 0 = fixed BS from length-sorted pools
ROWS   = int(os.environ.get("ROWS", 512 if SMOKE else 0))
LOG_E  = int(os.environ.get("LOG_EVERY", 2 if SMOKE else 200))
NW     = int(os.environ.get("NW", 2))
DEVICE = os.environ.get("DEVICE", "cuda")
SEED = int(os.environ.get("SEED", "42"))
AUX_SI = float(os.environ.get("AUX_SI", "0"))    # weight of the auxiliary semi-vs-irrelevant BCE
# AUX_SI_HEAD (e21 salvage): keep the 3-way softmax EXACTLY as shipped (logits 0..2) and add a 4th
# logit trained only on stage-2 non-relevant rows as a dedicated semi-vs-irrelevant discriminator.
# The two-head runs showed a dedicated S/I output reaches AUC(S|I) 0.9126 vs 0.9037, but paid for it
# by losing the 3-way contrast on R|S; this keeps that contrast. The 4th logit is IGNORED at
# inference (eval/serving read logits[:, :3]), so it can only act through the shared encoder.
AUX_SI_HEAD = float(os.environ.get("AUX_SI_HEAD", "0"))
# ---- TWO-HEAD mode (e21): head A = relevant-vs-rest, head B = semi-vs-irrelevant CONDITIONAL on
# not-relevant. Stage-2 rows supervise both heads; Stage-1 rows (label "s1_rel"/"s1_nonrel") are a
# COMPLEMENTARY label — they supervise head A only and head B abstains on them.
#   p_rel = s(A) ; p_semi = (1-s(A))*s(B) ; p_irr = (1-s(A))*(1-s(B))
# so the ordinal scalar p_semi+2*p_rel and the two fitted cuts carry over unchanged.
TWO_HEAD = os.environ.get("TWO_HEAD", "0") == "1"
S1_W     = float(os.environ.get("S1_W", "1.0"))   # weight on Stage-1 head-A rows (they carry ~17% FN)
HEAD_B_W = float(os.environ.get("HEAD_B_W", "1.0"))
WARMUP, WD, CLIP = 0.05, 0.01, 1.0
L2I = {"irrelevant": 0, "semi-relevant": 1, "relevant": 2}
if TWO_HEAD:
    L2I = dict(L2I, s1_nonrel=3, s1_rel=4)      # head-A-only rows

torch.manual_seed(SEED); np.random.seed(SEED)
if DEVICE == "cuda":
    assert torch.cuda.is_available(), "no CUDA — for a CPU plumbing check use --smoke DEVICE=cpu"
else:
    assert SMOKE, "DEVICE=cpu is a smoke-only escape hatch; real runs are GPU"
AMP = DEVICE == "cuda"                              # bf16 autocast, NO GradScaler (bf16 needs none)

# ---------------------------------------------------------------- data + targets
df = pd.read_parquet(DATA); DATA_SHA = sha256(DATA)
if ROWS: df = df.sample(n=min(ROWS, len(df)), random_state=SEED).reset_index(drop=True)
has_hard = df.real_label.isin(L2I).to_numpy()
has_teacher = df.p_irr.notna().to_numpy() if "p_irr" in df else np.zeros(len(df), bool)
assert (has_hard | has_teacher).all(), "row with neither hard label nor teacher probs"
Y = np.where(has_hard, df.real_label.map(L2I).fillna(0).astype(np.int64), -1)
QT = np.zeros((len(df), 3), np.float32)             # tempered teacher targets q ∝ p^(1/T)
if has_teacher.any():
    P = df.loc[has_teacher, ["p_irr", "p_semi", "p_rel"]].to_numpy(np.float32)
    q = np.clip(P, 1e-8, None) ** (1.0 / TEMP)
    QT[has_teacher] = q / q.sum(1, keepdims=True)
if CW == "auto":                                    # inverse-frequency over hard-label rows
    cnt = np.maximum(np.bincount(Y[has_hard], minlength=3), 1)
    cw = cnt.sum() / (3.0 * cnt)
elif CW == "none":
    cw = np.ones(3)
else:
    cw = np.array([float(x) for x in CW.split(",")]); assert len(cw) == 3
cw_t = torch.tensor(cw, dtype=torch.float32, device=DEVICE)
if TWO_HEAD:
    _rel = np.isin(Y, [2, 4]); _nrel = np.isin(Y, [0, 1, 3])
    POSW_A = float(max(_nrel.sum(), 1) / max(_rel.sum(), 1))       # relevant is the minority overall
    _irr, _semi = (Y == 0).sum(), (Y == 1).sum()
    POSW_B = float(max(_irr, 1) / max(_semi, 1))                   # among stage-2 non-relevant rows
    log(f"TWO_HEAD: headA rows {len(Y):,} (rel {_rel.sum():,} pos_w {POSW_A:.3f}) | "
        f"headB rows {_irr + _semi:,} (semi {_semi:,} pos_w {POSW_B:.3f}) | "
        f"stage1 rows {int(np.isin(Y, [3, 4]).sum()):,} w={S1_W}")
log(f"{len(df):,} rows from {DATA} (sha {DATA_SHA[:12]}) | hard {int(has_hard.sum()):,} "
    f"teacher {int(has_teacher.sum()):,} both {int((has_hard & has_teacher).sum()):,} | "
    f"CW {np.round(cw, 3).tolist()} | alpha {ALPHA} beta {BETA} T {TEMP}")

from transformers import AutoTokenizer, AutoModelForSequenceClassification
# TRUST=1 for repos shipping custom code (gte / TY-ecomm-embed); decoder backbones
# (Qwen3, Vision-Flash) usually declare no pad token -> reuse EOS, and their
# sequence-classification head reads the LAST non-pad token, which needs RIGHT padding.
TRUST = os.environ.get("TRUST", "0") == "1"
tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=TRUST)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token
tok.padding_side = "right"
A = df["query"].tolist()                            # e07 serialization, verbatim
B = (df.title + " | " + df.brand + " | " + df.category + " | "
     + df.gender + " " + df.age_group + " | " + df.attributes).tolist()
if os.environ.get("SANDWICH","0")=="1":                    # causal models: re-read the query after the item
    B=[b+" || "+q for b,q in zip(B,A)]
    print("SANDWICH=1 serialization")

def tok_lens(chunk=4096):
    out = []
    for s in range(0, len(A), chunk):
        e = tok(A[s:s + chunk], B[s:s + chunk], truncation=True, max_length=MAXLEN)
        out += [len(x) for x in e["input_ids"]]
    return np.asarray(out)

lens = tok_lens()
budget, cap = (BUDGET, max(BS * 4, 96)) if BUDGET else (1 << 60, BS)   # 0 -> pure fixed-BS cut
plan = []
for ep in range(EPOCHS):
    plan += [(b, ep) for b in train_plan(lens, budget, seed=SEED + ep, cap=cap)]
steps = len(plan)

# ---- query-glue augmentation (GLUE_P), RESAMPLED EVERY EPOCH ----------------------------------
# Turkish shoppers type compounds glued: "çelik bileklik" -> "celikbileklik". The shipped model
# scores 74.4% on that slice against 85.8% elsewhere. Gluing ONCE into the training file would be
# a fixed corrupted subset the model can memorise, and would leave the other ~97% of rows never
# glued in any epoch. Drawing a FRESH mask per epoch means a row is sometimes seen glued and
# sometimes spaced, which is what teaches the invariance rather than the specific strings.
# Deterministic in SEED, so runs stay reproducible.
GLUE_P = float(os.environ.get("GLUE_P", "0"))
# TYPO_P: character-level query noise, resampled per epoch like GLUE_P.
# MEASURED on the frozen model over 1,472 real public rows: deleting ONE character from the query
# flips 33.4% of labels and swapping two adjacent characters flips 41.0% (mean |dScalar| 0.39 and
# 0.48). That is by far the largest invariance deficit we found -- attribute-order shuffling flips
# only 1.2% and dropping an attribute 1.4%, so the product side is already robust and needs no
# augmentation. Real test queries do contain misspellings (plastation4, stadivaryus, prezerfatif).
TYPO_P = float(os.environ.get("TYPO_P", "0"))
GLUE_MASK = None
TYPO_MASK = None
if GLUE_P > 0:
    _multi = np.array([" " in a.strip() for a in A])          # only multi-word queries can glue
    GLUE_MASK = np.zeros((EPOCHS, len(A)), dtype=bool)
    for ep in range(EPOCHS):
        GLUE_MASK[ep] = (np.random.RandomState(SEED * 7919 + ep).rand(len(A)) < GLUE_P) & _multi
    log(f"query-glue augmentation p={GLUE_P:.3f}, resampled per epoch: "
        f"{GLUE_MASK.sum():,} row-visits glued of {EPOCHS*len(A):,} "
        f"({100*GLUE_MASK.mean():.2f}%); a row is glued in >=1 epoch with prob "
        f"{100*(1-(1-GLUE_P)**EPOCHS):.1f}%")
if TYPO_P > 0:
    TYPO_MASK = np.zeros((EPOCHS, len(A)), dtype=bool)
    _long = np.array([len(a.strip()) >= 6 for a in A])
    for ep in range(EPOCHS):
        TYPO_MASK[ep] = (np.random.RandomState(SEED * 104729 + ep).rand(len(A)) < TYPO_P) & _long
    log(f"query-typo augmentation p={TYPO_P:.3f}, resampled per epoch: "
        f"{TYPO_MASK.sum():,} row-visits perturbed ({100*TYPO_MASK.mean():.2f}%)")
log(f"{steps} batches / {sum(len(b) for b, _ in plan):,} row-visits / {EPOCHS} epoch(s) | "
    f"tokens p50 {np.median(lens):.0f} p95 {np.percentile(lens, 95):.0f} max {lens.max()} | "
    f"batching {'token-budget ' + str(BUDGET) if BUDGET else f'fixed BS={BS} length-grouped'}")

class PlanDS(Dataset):
    """One item = one pre-planned batch, tokenized in the worker (overlaps with the GPU)."""
    def __init__(s, plan): s.plan = plan
    def __len__(s): return len(s.plan)
    def __getitem__(s, i):
        b, ep = s.plan[i]
        qs = [A[j] for j in b]
        if GLUE_MASK is not None:
            m = GLUE_MASK[ep]
            qs = ["".join(q.split()) if m[j] else q for q, j in zip(qs, b)]
        if TYPO_MASK is not None:
            m = TYPO_MASK[ep]
            out = []
            for q, j in zip(qs, b):
                if not m[j] or len(q) < 6:
                    out.append(q); continue
                # deterministic per (seed, epoch, row): delete or transpose one interior character
                r = np.random.RandomState((SEED * 104729 + ep * 7919 + int(j)) % (2 ** 31))
                i = int(r.randint(1, len(q) - 1))
                out.append(q[:i] + q[i + 1:] if r.rand() < 0.5 else
                           (q[:i] + q[i + 1] + q[i] + q[i + 2:] if i + 2 <= len(q) else q[:i] + q[i + 1:]))
            qs = out
        enc = tok(qs, [B[j] for j in b], truncation=True, max_length=MAXLEN,
                  padding=True, return_tensors="pt")
        return (enc, torch.from_numpy(Y[b]), torch.from_numpy(QT[b]),
                torch.from_numpy(has_hard[b]), torch.from_numpy(has_teacher[b]))

dl = DataLoader(PlanDS(plan), batch_size=None, shuffle=False, num_workers=NW)

# ---------------------------------------------------------------- model + optimizer
_mk = dict(num_labels=(2 if TWO_HEAD else (4 if AUX_SI_HEAD > 0 else 3)), trust_remote_code=TRUST, ignore_mismatched_sizes=True)  # rerankers ship 1-output heads; ours is retrained
if os.environ.get("DTYPE", "") == "bf16":
    _mk["dtype"] = torch.bfloat16                      # billion-param backbones
model = AutoModelForSequenceClassification.from_pretrained(MODEL, **_mk).to(DEVICE)
if model.config.pad_token_id is None:
    model.config.pad_token_id = tok.pad_token_id
if os.environ.get("GRAD_CKPT", "0") == "1":
    model.gradient_checkpointing_enable()
    model.config.use_cache = False
opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
warm = max(1, int(WARMUP * steps))
sched = torch.optim.lr_scheduler.LambdaLR(
    opt, lambda s: (s + 1) / warm if s < warm
    else 0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, steps - warm))))

def loss_fn(logits, y, qt, hh, ht):
    """Per-row alpha/beta mix; a row missing one signal takes the other at full weight."""
    n, total = logits.shape[0], logits.new_zeros(())
    if TWO_HEAD:
        A, Bv = logits[:, 0], logits[:, 1]
        tgtA = torch.isin(y, y.new_tensor([2, 4])).float()
        smA = tgtA * (1 - SMOOTH) + 0.5 * SMOOTH
        wA = torch.where(y >= 3, A.new_tensor(S1_W), A.new_tensor(1.0)) * \
             torch.where(tgtA > 0.5, A.new_tensor(POSW_A), A.new_tensor(1.0))
        total = (F.binary_cross_entropy_with_logits(A, smA, reduction="none") * wA).sum()
        mB = (y == 0) | (y == 1)                       # stage-2 non-relevant rows only
        if mB.any():
            tgtB = (y[mB] == 1).float()
            smB = tgtB * (1 - SMOOTH) + 0.5 * SMOOTH
            wB = torch.where(tgtB > 0.5, A.new_tensor(POSW_B), A.new_tensor(1.0))
            total = total + HEAD_B_W * (
                F.binary_cross_entropy_with_logits(Bv[mB], smB, reduction="none") * wB).sum()
        return total / n
    if hh.any():
        ce = F.cross_entropy(logits[hh][:, :3], y[hh], weight=cw_t,
                             label_smoothing=SMOOTH, reduction="none")
        total = total + (torch.where(ht[hh], ce.new_tensor(ALPHA), ce.new_tensor(1.0)) * ce).sum()
    if ht.any():
        ls = F.log_softmax(logits[ht][:, :3] / TEMP, -1)
        kd = -(qt[ht] * ls).sum(-1) * TEMP * TEMP    # softCE == KL(q||s_T)+const; Hinton T^2 scale
        total = total + (torch.where(hh[ht], kd.new_tensor(BETA), kd.new_tensor(1.0)) * kd).sum()
    if AUX_SI_HEAD > 0:
        nr = hh & (y != 2)                       # stage-2 non-relevant rows only
        if nr.any():
            z = logits[nr][:, 3]                 # the dedicated S|I logit
            total = total + AUX_SI_HEAD * F.binary_cross_entropy_with_logits(
                z, (y[nr] == 1).float(), reduction="none").sum()
    if AUX_SI > 0:
        # B2 §1.2/1.3: the semi/irrelevant boundary owns 2.2x the headroom of the relevant one and
        # converts at ~0.87 macro-F1 per AUC point. 3-way CE spends most of its gradient on the
        # (already easy) relevant decision, so add an explicit binary margin on the S|I pair,
        # restricted to hard-labelled non-relevant rows. Same shape as the e11 12B loss.
        nr = hh & (y != 2)
        if nr.any():
            z = logits[nr][:, 1] - logits[nr][:, 0]           # semi-vs-irrelevant margin
            bce = F.binary_cross_entropy_with_logits(z, (y[nr] == 1).float(), reduction="none")
            total = total + AUX_SI * bce.sum()
    return total / n

model.train()
t0, ntok, win = time.time(), 0, []
for i, (enc, y, qt, hh, ht) in enumerate(dl, 1):
    enc = {k: v.to(DEVICE) for k, v in enc.items()}
    y, qt, hh, ht = (t.to(DEVICE) for t in (y, qt, hh, ht))
    if AMP:
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            logits = model(**enc).logits
    else:
        logits = model(**enc).logits
    loss = loss_fn(logits.float(), y, qt, hh, ht)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP)
    opt.step(); sched.step(); opt.zero_grad(set_to_none=True)
    win.append(loss.item()); ntok += int(enc["attention_mask"].sum())
    if i % LOG_E == 0:
        el = time.time() - t0
        rows = sum(len(plan[k]) for k in range(i))
        log(f"  step {i}/{steps} loss {np.mean(win):.4f} lr {sched.get_last_lr()[0]:.2e} | "
            f"{rows/el:.0f} rows/s {ntok/el:.0f} tok/s | eta {el/i*(steps-i)/60:.1f} min")
        win = []

# ---------------------------------------------------------------- checkpoint (HF + lean e12)
ckpt = f"{OUT}/ckpt_{TAG}"
os.makedirs(ckpt, exist_ok=True)
if AUX_SI_HEAD > 0:
    # The 4th logit was a TRAINING-TIME device only. Strip it so the shipped artifact is an
    # ordinary 3-class checkpoint: identical serving code, identical cost, nothing downstream
    # (eval, lean/GTE serving, docker, parity tests) has to know this run ever had an aux head.
    with torch.no_grad():
        cl = model.classifier
        new_cl = torch.nn.Linear(cl.in_features, 3).to(cl.weight.device, cl.weight.dtype)
        new_cl.weight.copy_(cl.weight[:3]); new_cl.bias.copy_(cl.bias[:3])
        model.classifier = new_cl
    model.config.num_labels = 3
    model.config.id2label = {i: l for i, l in enumerate(["irrelevant", "semi-relevant", "relevant"])}
    model.config.label2id = {l: i for i, l in enumerate(["irrelevant", "semi-relevant", "relevant"])}
    log("AUX_SI_HEAD: stripped the auxiliary 4th logit — checkpoint saved as a plain 3-class head")
model.save_pretrained(ckpt, safe_serialization=True)     # model.safetensors + config.json
tok.save_pretrained(ckpt)
if not os.path.exists(f"{ckpt}/tokenizer.json"):         # lean path needs the Rust tokenizer
    tok.backend_tokenizer.save(f"{ckpt}/tokenizer.json")
json.dump({"head": "two_head" if TWO_HEAD else "softmax3"}, open(f"{ckpt}/decision.json", "w"))

el = time.time() - t0
meta = dict(tag=TAG, model=MODEL, data=DATA, data_sha256=DATA_SHA, rows=int(len(df)),
            n_hard=int(has_hard.sum()), n_teacher=int(has_teacher.sum()),
            n_both=int((has_hard & has_teacher).sum()),
            epochs=EPOCHS, bs=BS, budget=BUDGET, lr=LR, maxlen=MAXLEN, seed=SEED,
            temp=TEMP, alpha=ALPHA, beta=BETA, smooth=SMOOTH, aux_si=AUX_SI, aux_si_head=AUX_SI_HEAD,
            two_head=TWO_HEAD, s1_w=S1_W, head_b_w=HEAD_B_W, class_weights=np.round(cw, 4).tolist(),
            loss="alpha*CE(hard,ls=%.2f,cw)+beta*T^2*softCE(teacher@T) ; solo signal at weight 1" % SMOOTH,
            steps=steps, wall_seconds=round(el, 1),
            rows_per_s=round(sum(len(b) for b, _ in plan) / max(el, 1e-6), 1),
            tok_per_s=round(ntok / max(el, 1e-6), 1),
            final_loss=float(np.mean(win)) if win else None,
            amp="bf16-autocast-no-gradscaler" if AMP else "fp32-cpu-smoke", smoke=SMOKE,
            device=torch.cuda.get_device_name(0) if DEVICE == "cuda" else "cpu",
            torch=torch.__version__,
            parity_cmd=f"MODEL={ckpt} OUT={ckpt}_parity .venv-gcp/bin/python "
                       f"{W}/asya/s2_exp/e12_lean_infer/scripts/verify_parity.py",
            created_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
json.dump(meta, open(f"{OUT}/train_meta_{TAG}.json", "w"), indent=1)
log(f"DONE {steps} steps in {el/60:.1f} min ({meta['rows_per_s']} rows/s) -> {ckpt} "
    f"(HF + lean e12 format) + train_meta_{TAG}.json")
