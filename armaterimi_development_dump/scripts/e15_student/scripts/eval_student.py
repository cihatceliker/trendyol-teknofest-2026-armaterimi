"""E15 eval_student — DEV readouts for a student checkpoint, matching e11's decision numbers.

Per row (p_irr, p_semi, p_rel) from the 3-way head. Readouts (all through common/harness.py so
numbers stay comparable across experiments):
  argmax          macro-F1 + per-class + confusion, zero fitted parameters
  ordinal scalar  x = p_semi + 2*p_rel -> harness.oof_scalar (query-disjoint OOF two-cut)
  two-boundary    R if p_rel > h_R; else I if (1-p_irr) <= h_I; else S — OOF-fitted via
                  e11 si_common.fit_2b/apply_2b. PRIMARY readout (it is the deployment rule).
  AUCs            harness.ordinal_auc on the scalar — AUC(S|I) and AUC(R|rest) are the gates
  slices          harness.slice_report on the OOF two-boundary preds (includes the
                  query-token-coverage bands, plus cat_root and query-word-count)
  bootstrap CI    grouped by query, on the primary preds
Writes OUT/eval_<tag>.json + OUT/dev_probs_<tag>.parquet. Reference: two-boundary DEV-half
OOF 0.8328 (e09 freeze_cuts); noise floor 0.012 (pre-registered, README).

DEV text goes through common/normalize.py + drop_keys.csv — identical to the training path.

HOLDOUT DISCIPLINE: the quarantined public half is MILESTONE-ONLY (90k private vs 7.5k
public — every look is a chance to overfit; see harness.load_public_split docstring). This
script hard-refuses HOLDOUT=1 unless BOTH confirmation phrases are also set:
  HOLDOUT=1 HOLDOUT_CONFIRM1=this-is-a-pre-registered-milestone \
  HOLDOUT_CONFIRM2=dev-decisions-are-frozen
Never iterate on a holdout number; it is the audit, not the tuning set.

Env: CKPT (default: newest outputs/ckpt_*) TAG (default: ckpt dir suffix) OUT BS(256)
     MAXLEN(256) DEVICE(cuda if available) EVAL_N(0=all DEV rows; smoke-only subsample)
     HOLDOUT(0) HOLDOUT_CONFIRM1 HOLDOUT_CONFIRM2
"""
import os, sys, json, glob, time
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
import numpy as np, pandas as pd, torch

W = "/home/jupyter/trendyol-teknofest-work"
sys.path.insert(0, f"{W}/asya/s2_exp/common")
sys.path.insert(0, f"{W}/asya/s2_exp/e11_si_lora/scripts")
import harness as H, normalize as NZ
from si_common import log, fit_2b, apply_2b

E15    = f"{W}/asya/s2_exp/e15_student"
CKPT   = os.environ.get("CKPT", "")
OUT    = os.environ.get("OUT", f"{E15}/outputs")
BS     = int(os.environ.get("BS", 256))
MAXLEN = int(os.environ.get("MAXLEN", 256))
DEVICE = os.environ.get("DEVICE", "cuda" if torch.cuda.is_available() else "cpu")
EVAL_N = int(os.environ.get("EVAL_N", 0))
REF, NOISE = 0.8328, 0.012      # e09 freeze_cuts DEV-half two-boundary OOF; pre-registered floor

if not CKPT:
    cks = sorted(glob.glob(f"{E15}/outputs/ckpt_*"), key=os.path.getmtime)
    assert cks, "no outputs/ckpt_* — train first or pass CKPT"
    CKPT = cks[-1]
TAG = os.environ.get("TAG", os.path.basename(CKPT.rstrip("/")).removeprefix("ckpt_"))

# ---------------------------------------------------------------- holdout triple guard
half = "dev"
if os.environ.get("HOLDOUT", "0") == "1":
    ok = (os.environ.get("HOLDOUT_CONFIRM1") == "this-is-a-pre-registered-milestone"
          and os.environ.get("HOLDOUT_CONFIRM2") == "dev-decisions-are-frozen")
    if not ok:
        sys.exit("REFUSED: HOLDOUT=1 without both confirmation phrases. Holdout reads are "
                 "milestone-only (see docstring). If this IS a milestone, set\n"
                 "  HOLDOUT_CONFIRM1=this-is-a-pre-registered-milestone\n"
                 "  HOLDOUT_CONFIRM2=dev-decisions-are-frozen")
    half, TAG = "holdout", TAG + "_HOLDOUT"
    log("*** HOLDOUT read — milestone only. Do not iterate on this number. ***")

_dj = os.path.join(CKPT, "decision.json")
TWO_HEAD = (json.load(open(_dj)).get("head") == "two_head") if os.path.exists(_dj) else False
if TWO_HEAD: print("TWO_HEAD checkpoint: probs from sigmoid heads")
drop = pd.read_csv(f"{W}/asya/s2_exp/common/drop_keys.csv").key.tolist()
d = NZ.normalize(H.load_public_split(half), drop)
if os.environ.get("ATTR_CLEAN", "v1") == "v2":                      # e18 opt-in, must match the training build
    sys.path.insert(0, f"{W}/asya/s2_exp/e18_attr_clean"); import attr_clean as AC
    d = AC.apply_v2(d, f"{W}/asya/s2_exp/e18_attr_clean/drop_keys_v2.csv", drop_boilerplate=True)
    print("ATTR_CLEAN=v2 applied to DEV")
assert len(d) == 3750
if EVAL_N:                                       # smoke-only: keep folds query-disjoint anyway
    d = d.sample(n=min(EVAL_N, len(d)), random_state=42).reset_index(drop=True)
    log(f"EVAL_N={EVAL_N}: subsampled {len(d)} rows — plumbing check, numbers meaningless")
y = d.real_label.values

# ---------------------------------------------------------------- score
from transformers import AutoTokenizer, AutoModelForSequenceClassification
TRUST = os.environ.get("TRUST", "0") == "1"           # gte / custom-code backbones
tok = AutoTokenizer.from_pretrained(CKPT, trust_remote_code=TRUST)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token
tok.padding_side = "right"                            # decoder heads read the last non-pad token
_mk = dict(trust_remote_code=TRUST)
if os.environ.get("DTYPE", "") == "bf16":
    _mk["dtype"] = torch.bfloat16
model = AutoModelForSequenceClassification.from_pretrained(CKPT, **_mk).to(DEVICE).eval()
if model.config.pad_token_id is None:
    model.config.pad_token_id = tok.pad_token_id
assert model.config.num_labels == (2 if TWO_HEAD else 3)
A = d["query"].tolist()                          # e07/e15 serialization, verbatim
B = (d.title + " | " + d.brand + " | " + d.category + " | "
     + d.gender + " " + d.age_group + " | " + d.attributes).tolist()
if os.environ.get("SANDWICH","0")=="1":                    # causal models: re-read the query after the item
    B=[b+" || "+q for b,q in zip(B,A)]
    print("SANDWICH=1 serialization")
lens = np.asarray([len(x) for x in tok(A, B, truncation=True, max_length=MAXLEN)["input_ids"]])
order = np.argsort(lens, kind="mergesort")       # length-sorted eval batches
probs = np.full((len(d), 3), np.nan, np.float32)
t0 = time.time()
with torch.no_grad():
    for s in range(0, len(order), BS):
        b = order[s:s + BS]
        enc = tok([A[i] for i in b], [B[i] for i in b], truncation=True, max_length=MAXLEN,
                  padding=True, return_tensors="pt").to(DEVICE)
        if DEVICE == "cuda":
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                lg = model(**enc).logits.float()
        else:
            lg = model(**enc).logits
        if TWO_HEAD:                                   # e21: p_rel=s(A); p_semi=(1-s(A))s(B)
            pa = torch.sigmoid(lg[:, 0]); pb = torch.sigmoid(lg[:, 1])
            probs[b] = torch.stack([(1 - pa) * (1 - pb), (1 - pa) * pb, pa], -1).cpu().numpy()
        else:
            probs[b] = lg.softmax(-1).cpu().numpy()
assert np.isfinite(probs).all()
log(f"scored {len(d):,} rows in {time.time()-t0:.1f}s ({len(d)/max(time.time()-t0,1e-6):.0f} rows/s, {DEVICE})")
p_irr, p_semi, p_rel = probs.T
scalar = p_semi + 2 * p_rel
folds = H.query_folds(d["query"].values)

# ---------------------------------------------------------------- readouts (e11 eval_dev set)
argmax_pred = np.array(H.LABELS, dtype=object)[probs.argmax(1)]
res_scalar = H.oof_scalar(scalar, y, folds)
aucs = H.ordinal_auc(scalar, y)

sR, sI = p_rel, 1.0 - p_irr
gR = np.unique(np.percentile(sR, np.linspace(0.2, 99.8, 300)))
gI = np.unique(np.percentile(sI, np.linspace(0.2, 99.8, 300)))
pred2b = np.empty(len(y), dtype=object); cuts2b = []
for f in np.unique(folds):
    trm, tem = folds != f, folds == f
    _, hR, hI = fit_2b(sR[trm], sI[trm], y[trm], gR, gI)
    pred2b[tem] = apply_2b(sR[tem], sI[tem], hR, hI); cuts2b.append((hR, hI))
ins2b, hR0, hI0 = fit_2b(sR, sI, y, gR, gI)
f1_2b = H.macro_f1(y, pred2b)

pd.DataFrame({"row_id": d.row_id, "p_irr": p_irr, "p_semi": p_semi, "p_rel": p_rel,
              "scalar": scalar}).to_parquet(f"{OUT}/dev_probs_{TAG}.parquet", index=False)

def slices():
    """harness.slice_report records; tolerant of tiny EVAL_N subsamples (empty -> [])."""
    try:
        return H.slice_report(d, pred2b, min_n=150 if not EVAL_N else 10).to_dict(orient="records")
    except KeyError:                      # harness sorts an empty frame when no slice clears min_n
        return []

res = {
    "ckpt": CKPT, "tag": TAG, "half": half, "n": int(len(d)),
    "reference_two_boundary_oof": REF, "noise_floor": NOISE,
    "two_boundary_oof": {"macro_f1": f1_2b,                     # PRIMARY readout
                         "per_class": H.per_class_f1(y, pred2b),
                         "confusion": H.confusion(y, pred2b).values.tolist(),
                         "in_sample": ins2b, "fold_cuts": cuts2b,
                         "in_sample_cuts": [hR0, hI0],
                         "delta_vs_ref": round(f1_2b - REF, 4)},
    "scalar_oof": {"macro_f1": res_scalar["oof_macro_f1"],
                   "per_class": res_scalar["oof_per_class"],
                   "in_sample": res_scalar["in_sample_macro_f1"],
                   "fold_cuts": res_scalar["fold_cuts"],
                   "delta_vs_ref": round(res_scalar["oof_macro_f1"] - REF, 4)},
    "argmax": {"macro_f1": H.macro_f1(y, argmax_pred),
               "per_class": H.per_class_f1(y, argmax_pred),
               "confusion": H.confusion(y, argmax_pred).values.tolist()},
    "auc": aucs,
    "slices_two_boundary": slices(),
    "bootstrap_ci_two_boundary": H.bootstrap_ci(y, pred2b, groups=d["query"].values),
    "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
}
out_p = f"{OUT}/eval_{TAG}.json"
json.dump(res, open(out_p, "w"), indent=1, default=float)
log(f"[{half.upper()}] 2-boundary OOF {f1_2b:.4f} (ref {REF}, Δ {f1_2b-REF:+.4f}) | "
    f"scalar OOF {res_scalar['oof_macro_f1']:.4f} | argmax {res['argmax']['macro_f1']:.4f}")
log(f"AUC(S|I) {aucs['S_vs_I']:.4f}  AUC(R|rest) {aucs['R_vs_rest']:.4f}  -> {out_p}")
print(H.confusion(y, pred2b).to_string())
