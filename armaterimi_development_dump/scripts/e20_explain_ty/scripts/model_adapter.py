"""e20 — model adapter for the CHAMPION explainability server (adapts e13 to the 306M student).

Wraps THE champion predictor (e15 ckpt_r1_oldstore: GTE-"New"-architecture ~306M cross-encoder,
AutoModelForSequenceClassification via the checkpoint's own local modeling.py,
trust_remote_code) and produces, from the predictor's own forward/backward passes only:

  * prediction — logits -> label EXACTLY as serving does (e12 lean_infer.decide parity:
      softmax3 -> argmax; ordinal -> logits[0] vs decision.json cuts). Plus the ordinal scalar
      x = p_semi + 2*p_rel (= E[relevance rank], the scalar e15's readouts are built on) drawn
      against the two cuts (decision.json "cuts" when present, else the 0.5/1.5 rank midpoints).
  * integrated gradients — class-specific IG on the word-embedding layer (manual, no captum),
      with ADAPTIVE step placement: a cheap forward-only scan of g(a) = f(x0 + a*(x-x0)) first
      locates where the logit actually moves (this checkpoint's path is plateau -> cliff ->
      plateau; uniform low-step grids alias the cliff and blow the completeness check), then
      the gradient budget is allocated per interval proportional to |dg| (15% uniform floor).
      Signed per-token attribution toward the predicted class, per-field aggregation, honest
      completeness check on every call (no forced renormalisation).
  * attention — query-token x product-token alignment (last layers, heads averaged) +
      attention rollout (Abnar & Zuidema 2020). Corroborating view only; forced eager
      attention because the GTE SDPA path raises on output_attentions.
  * counterfactuals — leave-one-segment-out over the product side (the "|" template fields and
      each "key: value" attribute pair) + single-word query deletions; every variant re-scored
      by the same model (N+1 cheap forwards, batched).
  * summary — template-assembled Turkish explanation line from IG + counterfactuals. No LLM.

Preprocessing is THE serving pipeline imported from its sources of truth (never re-implemented
here): common/normalize.py (T1 NFC+U+0307 strip, T2/T3 drop_keys.csv, T4 unisex->unknown),
then e18_attr_clean.apply_v2 (drop_keys_v2 + boilerplate-value filter, empty -> "unknown"),
then the e07 pair template   query   vs   "title | brand | category | gender age_group |
attributes" — the byte-identical path eval_student.py used for the champion's DEV numbers and
the same v2 transform e12 lean_infer.py inlines for the container. Tokenizer: the checkpoint's
own tokenizer.json (Rust), truncation longest_first @ 256 — identical to serving.

GTE gotchas handled here (found by reading the checkpoint's modeling.py):
  * NewSdpaAttention raises on output_attentions -> the model is loaded attn_implementation=
    "eager" (same math as SDPA, exact probabilities; logit deltas are float-noise level).
  * NewEmbeddings does token_type_ids.mul_(0) IN PLACE when type_vocab_size < 2 -> a fresh
    (cloned/contiguous) tensor is passed on every forward, never an expand() view.
  * RoPE positions derive from the sequence axis of inputs_embeds -> interpolating word
    embeddings for IG leaves positional information intact, exactly like BERT-IG in e13.
"""
from __future__ import annotations

import json
import os
import sys
import time

import random
import numpy as np
import pandas as pd
import torch

# Path overrides for container staging (e19 phase 7): point these at the in-image copies of
# common/{normalize.py,drop_keys.csv} and e18_attr_clean/{attr_clean.py,drop_keys_v2.csv}.
W = os.environ.get("E20_WORK_ROOT", "/home/jupyter/trendyol-teknofest-work")
COMMON = os.environ.get("E20_COMMON_DIR", f"{W}/asya/s2_exp/common")
E18 = os.environ.get("E20_E18_DIR", f"{W}/asya/s2_exp/e18_attr_clean")
for _p in (COMMON, E18):
    if _p not in sys.path:
        sys.path.insert(0, _p)
import normalize as NZ    # noqa: E402  frozen T1-T4 (the training/serving source of truth)
import attr_clean as AC   # noqa: E402  e18 v2 attribute cleaning (imports normalize itself)

t1 = NZ.t1
DROP_KEYS_V1 = pd.read_csv(f"{COMMON}/drop_keys.csv", dtype=str, keep_default_na=False).key.tolist()
DROP_V2_PATH = AC.DROP_KEYS_V2   # e18/drop_keys_v2.csv

RAW_COLS = ["query", "title", "category", "brand", "gender", "age_group", "attributes"]


def normalize_frame(df: pd.DataFrame) -> pd.DataFrame:
    """The full v2 preprocessing on a raw-text frame: normalize.normalize + attr_clean.apply_v2.
    Identical call chain to e15/eval_student.py (ATTR_CLEAN=v2), which produced the champion's
    DEV numbers, and to the inlined copy in e12 lean_infer (the container serving path)."""
    d = NZ.normalize(df, DROP_KEYS_V1)
    d = AC.apply_v2(d, DROP_V2_PATH, drop_boilerplate=True)
    return d


def normalize_fields(raw: dict) -> dict:
    """One raw field dict -> normalised field dict through the SAME frame pipeline."""
    df = pd.DataFrame([{c: str(raw.get(c, "") or "") for c in RAW_COLS}])
    d = normalize_frame(df)
    rec = d.iloc[0]
    return {c: str(rec[c]) for c in RAW_COLS}


# Product-side field order inside the e07 doc template. gender+age_group share one " | "
# segment but keep separate char spans so attributions aggregate per REAL field.
DOC_FIELDS = ["title", "brand", "category", "gender", "age_group", "attributes"]


def build_doc(f: dict) -> tuple[str, list[tuple[str, int, int]]]:
    """Normalised fields -> (doc string, char spans). Byte-identical to the serving template:
    title | brand | category | gender age_group | attributes  (e12 lean_infer._rows_to_pairs,
    e15 eval_student's B column)."""
    doc, spans = "", []

    def put(name: str, txt: str) -> None:
        nonlocal doc
        s = len(doc)
        doc += txt
        spans.append((name, s, len(doc)))

    put("title", f["title"]);         doc += " | "
    put("brand", f["brand"]);         doc += " | "
    put("category", f["category"]);   doc += " | "
    put("gender", f["gender"]);       doc += " "
    put("age_group", f["age_group"]); doc += " | "
    put("attributes", f["attributes"])
    return doc, spans


def _token_to_word_idx(tok_idx: list, tokens: list, words: list) -> list:
    """For each attention row/col (a TOKEN position), the index of the word it belongs to.

    Words were merged from sub-word pieces by character offset (_words_from_tokens), so a token
    belongs to the word whose [start, end) span contains it, within the same sequence and field.
    Returns -1 for anything unmatched (specials), which the UI drops.
    """
    out = []
    for i in tok_idx:
        t = tokens[i]
        w = -1
        if not t.get("special"):
            for k, wd in enumerate(words):
                if (wd["seq"] == t["seq"] and wd["start"] <= t["start"] and t["end"] <= wd["end"]):
                    w = k
                    break
        out.append(w)
    return out


def _words_from_tokens(tokens: list[dict], src: dict) -> list[dict]:
    """Merge sub-word pieces into whole words using the tokenizer's character offsets.

    Two consecutive pieces belong to the same word when they sit in the same sequence and the
    same doc field AND the source text between them is empty (no whitespace, no punctuation gap).
    IG is additive over input positions, so summing pieces is exact, not an approximation.
    Surface text is sliced from the ORIGINAL string, so the juror reads "pejo", never "pe"+"jo".
    """
    out: list[dict] = []
    cur = None
    for tk in tokens:
        if tk["special"] or tk["end"] <= tk["start"]:
            cur = None
            continue
        sid, a, b = tk["seq"], tk["start"], tk["end"]
        text = src.get(sid, "")[a:b]
        same = (cur is not None and cur["seq"] == sid and cur["field"] == tk["field"]
                and a == cur["end"] and text[:1].strip() != "")
        if same:
            cur["end"] = b
            cur["ig"] += tk["ig"]
            cur["n_pieces"] += 1
        else:
            cur = {"seq": sid, "side": "query" if sid == 0 else "product", "field": tk["field"],
                   "start": a, "end": b, "ig": tk["ig"], "n_pieces": 1}
            out.append(cur)
    for w in out:
        w["text"] = src.get(w["seq"], "")[w["start"]:w["end"]]
        w["ig"] = round(w["ig"], 6)
    return out


def _assign_field(off: tuple[int, int], spans: list[tuple[str, int, int]]) -> str:
    """Token char span -> field name by maximum overlap; separators map to 'sep'."""
    best, best_ov = "sep", 0
    a, b = off
    for name, s, e in spans:
        ov = min(b, e) - max(a, s)
        if ov > best_ov:
            best, best_ov = name, ov
    return best


# ---------------------------------------------------------------- decision head
LABELS = ["irrelevant", "semi-relevant", "relevant"]   # repo convention (e07/e12/e15)
RANK = {"irrelevant": 0, "semi-relevant": 1, "relevant": 2}
TR_LABEL = {"irrelevant": "İlgisiz", "semi-relevant": "Kısmen ilgili", "relevant": "İlgili"}
TR_FIELD = {"query": "sorgu", "title": "başlık", "brand": "marka", "category": "kategori",
            "gender": "cinsiyet", "age_group": "yaş grubu", "attributes": "nitelikler"}


class ChampionAdapter:
    """The champion predictor + its native explanation signals. `model_dir` is any
    serve_hf-compatible checkpoint dir (config/modeling/tokenizer/decision.json)."""

    def __init__(self, model_dir: str, device: str = "cpu", max_len: int = 256):
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
        os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
        from tokenizers import Tokenizer
        from transformers import AutoModelForSequenceClassification

        self.model_dir = model_dir
        self.device = torch.device(device)
        self.max_len = max_len

        # eager attention: exact attention probabilities for the alignment view (the GTE
        # NewSdpaAttention raises on output_attentions). Same math as serving's SDPA kernel.
        try:
            m = AutoModelForSequenceClassification.from_pretrained(
                model_dir, trust_remote_code=True, dtype=torch.float32,
                attn_implementation="eager")
        except TypeError:   # older transformers signature
            from transformers import AutoConfig
            cfg = AutoConfig.from_pretrained(model_dir, trust_remote_code=True)
            cfg._attn_implementation = "eager"
            m = AutoModelForSequenceClassification.from_pretrained(
                model_dir, config=cfg, trust_remote_code=True, dtype=torch.float32)
        self.model = m.to(self.device).eval()
        for p in self.model.parameters():   # grads flow to inputs_embeds only (IG)
            p.requires_grad_(False)
        self.num_labels = int(m.config.num_labels)
        self.pad_id = int(getattr(m.config, "pad_token_id", 0) or 0)

        self.tok = Tokenizer.from_file(os.path.join(model_dir, "tokenizer.json"))
        self.tok.enable_truncation(max_length=max_len, strategy="longest_first")

        # decision.json — SAME semantics as e12 lean_infer.decide (the container's rule)
        dp = os.path.join(model_dir, "decision.json")
        d = json.load(open(dp)) if os.path.exists(dp) else {}
        self.head = d.get("head", "softmax3" if self.num_labels >= 2 else "ordinal")
        # ordinal_regime (the shipped rule): cuts are chosen by the row's OWN category root.
        # Without this the UI would draw argmax labels while the container serves regime cuts —
        # i.e. the explanation would contradict the model it claims to explain.
        self.regimes = d.get("regimes") or {}
        self.fashion_roots = set(d.get("fashion_roots", []))
        self.technical_roots = set(d.get("technical_roots", []))
        self.has_cuts = ("cuts" in d) or bool(self.regimes)
        self.cuts = [float(c) for c in (d.get("cuts") or d.get("global") or [0.5, 1.5])]
        labels = d.get("labels")
        if labels is None:
            i2l = m.config.id2label if isinstance(m.config.id2label, dict) else {}
            i2l = {str(k): v for k, v in i2l.items()}
            real = i2l and not any(str(v).startswith("LABEL_") for v in i2l.values())
            labels = [i2l[str(i)] for i in range(len(i2l))] if real else LABELS
        self.labels = list(labels)

    # ------------------------------------------------------------ forward pieces
    def _forward(self, ids=None, embeds=None, tt=None, mask=None, output_attentions=False):
        # token_type_ids MUST be a fresh contiguous tensor: NewEmbeddings mul_(0)s it in place.
        out = self.model(input_ids=ids, inputs_embeds=embeds,
                         token_type_ids=None if tt is None else tt.contiguous().clone(),
                         attention_mask=mask,
                         output_attentions=output_attentions, return_dict=True)
        return out.logits, (out.attentions if output_attentions else None)

    def encode(self, query: str, doc: str):
        """(query, doc) -> tokenizers.Encoding (pair encoding, per-sequence offsets)."""
        return self.tok.encode(query, doc)

    def _tensors(self, enc):
        ids = torch.tensor([enc.ids], dtype=torch.long, device=self.device)
        tt = torch.tensor([enc.type_ids], dtype=torch.long, device=self.device)
        mask = torch.ones_like(ids)
        return ids, tt, mask

    @torch.inference_mode()
    def score_pairs(self, pairs: list[tuple[str, str]], chunk: int = 16) -> np.ndarray:
        """Batched scoring for counterfactual variants. Mirrors serving's _assemble padding
        (pad ids, zero type ids, 0/1 mask). Returns logits (N, num_labels)."""
        if not pairs:
            return np.zeros((0, self.num_labels), np.float32)
        encs = self.tok.encode_batch(pairs)
        out = np.zeros((len(encs), self.num_labels), np.float32)
        for s in range(0, len(encs), chunk):
            sub = encs[s:s + chunk]
            L = max(len(e.ids) for e in sub)
            ids = np.full((len(sub), L), self.pad_id, dtype=np.int64)
            tt = np.zeros_like(ids)
            mk = np.zeros((len(sub), L), dtype=np.int64)
            for i, e in enumerate(sub):
                n = len(e.ids)
                ids[i, :n] = e.ids
                tt[i, :n] = e.type_ids
                mk[i, :n] = 1
            lg, _ = self._forward(ids=torch.from_numpy(ids).to(self.device),
                                  tt=torch.from_numpy(tt).to(self.device),
                                  mask=torch.from_numpy(mk).to(self.device))
            out[s:s + len(sub)] = lg.float().cpu().numpy()
        return out

    # ------------------------------------------------------------ decision
    def regime_of(self, category: str) -> str:
        # byte-identical to lean_infer._roots_from_csv (t1 -> root -> strip -> lower)
        root = NZ.t1(str(category or "")).split("/")[0].strip().lower()
        if root in self.fashion_roots: return "fashion"
        if root in self.technical_roots: return "technical"
        return "other"

    def cuts_for(self, category=None):
        """The two cuts the CONTAINER would use for this row (regime table, else global)."""
        if self.head == "ordinal_regime" and self.regimes and category is not None:
            g = self.regime_of(category)
            c = self.regimes.get(g)
            if c: return [float(c[0]), float(c[1])], g
        return list(self.cuts), "global"

    def decide(self, logits, category=None) -> dict:
        """One row of logits -> label + scalar + cuts + probs. The LABEL follows the serving
        rule byte-for-byte (lean_infer.decide): softmax3 -> argmax over logits; ordinal ->
        logits[0] against decision.json cuts. The SCALAR is x = p_semi + 2*p_rel (the ordinal
        expected-relevance readout e15 evaluates), drawn against the cuts on the number line."""
        lg = [float(v) for v in logits]
        if self.head == "ordinal_regime":
            x = np.asarray(lg) - max(lg); p = np.exp(x) / np.exp(x).sum()
            probs = {self.labels[i]: float(p[i]) for i in range(len(p))}
            rank = [RANK.get(self.labels[i], i) for i in range(len(p))]
            s = float(sum(p[i] * rank[i] for i in range(len(p))))     # x = p_semi + 2*p_rel
            (c1, c2), reg = self.cuts_for(category)
            label_id = 0 if s < c1 else (1 if s < c2 else 2)
            kind = "expected_relevance"
            rule = f"ordinal skaler, kategori rejimine ({reg}) göre iki eşik — konteynerdeki kuralın aynısı"
        elif self.head == "ordinal":
            # MUST match lean_infer.decide's "ordinal" branch, which uses _ordinal_scalar(logits)
            # = p_semi + 2*p_rel -- NOT logits[0]. Reading the raw logit here made the UI label
            # disagree with the container on essentially every row the moment a checkpoint shipped
            # head=="ordinal" (BERTurk does; TY-ecomm shipped ordinal_regime, which is why this
            # never surfaced). verify_ui_parity.py is the guard and must be re-run after ANY
            # change to either decider or to decision.json.
            x = np.asarray(lg) - max(lg); p = np.exp(x) / np.exp(x).sum()
            probs = {self.labels[i]: float(p[i]) for i in range(len(p))}
            rank = [RANK.get(self.labels[i], i) for i in range(len(p))]
            s = float(sum(p[i] * rank[i] for i in range(len(p))))
            c1, c2 = self.cuts
            label_id = 0 if s < c1 else (1 if s < c2 else 2)
            kind = "expected_relevance"
            rule = "ordinal skaler x = p_kısmen + 2·p_ilgili, iki global eşiğe karşı — konteynerdeki kuralın aynısı"
        else:
            x = np.asarray(lg) - max(lg)
            p = np.exp(x) / np.exp(x).sum()
            label_id = int(np.argmax(lg))            # serving parity: argmax, NOT the scalar
            probs = {self.labels[i]: float(p[i]) for i in range(len(p))}
            rank = [RANK.get(self.labels[i], i) for i in range(len(p))]
            s = float(sum(p[i] * rank[i] for i in range(len(p))))   # = p_semi + 2*p_rel
            c1, c2 = self.cuts
            kind = "expected_relevance"
            rule = "softmax3 argmax (sunucudaki lean_infer.decide ile birebir aynı kural)"
        return {
            "label": self.labels[label_id],
            "label_tr": TR_LABEL.get(self.labels[label_id], self.labels[label_id]),
            "label_id": label_id,
            "labels": self.labels,
            "logits": lg,
            "probs": probs,
            "scalar": float(s),
            "cuts": [float(c1), float(c2)],
            "cuts_source": ("decision.json" if self.has_cuts
                            else "varsayılan sıra-orta noktaları (0.5, 1.5) — decision.json'da cut alanı yok"),
            "scalar_kind": kind,
            "decision_rule": rule,
            "distance_to_cuts": [float(s - c1), float(s - c2)],
        }

    def _target_fn(self, label_id: int):
        """IG objective — attribute the quantity that ACTUALLY decides this row.

        ordinal / ordinal_regime -> x = p_semi + 2*p_rel, the scalar the container slices with
                          the cuts. Every checkpoint in this repo is a 3-logit classifier, so the
                          plain-ordinal head decides with the SAME scalar, only against global
                          cuts (lean_infer.decide calls _ordinal_scalar for both). Attributing a
                          single class logit would explain a number that does not make the
                          decision — the shipped BERTurk (head="ordinal") hit exactly that: the
                          old branch returned logits[:,0], the IRRELEVANT-class logit. Computed
                          differentiably (softmax dot rank) so completeness holds end to end.
        softmax3       -> the predicted class logit (argmax IS the rule there)."""
        if self.head in ("ordinal", "ordinal_regime"):
            rank = torch.tensor([RANK.get(l, i) for i, l in enumerate(self.labels)],
                                dtype=torch.float32)
            def _scalar(lg):
                return (torch.softmax(lg.float(), dim=-1) * rank.to(lg.device)).sum(-1)
            return _scalar, "decision_scalar:p_semi+2*p_rel"
        return (lambda lg: lg[:, label_id]), f"class_logit:{self.labels[label_id]}"

    # ------------------------------------------------------------ integrated gradients
    def integrated_gradients(self, enc, label_id: int, steps: int = 16, step_batch: int = 4,
                             coarse_points: int = 17, retry_rel_gap: float = 0.08) -> dict:
        """Adaptive-quadrature IG on the word-embedding layer.

        Baseline: <unk> word embedding at every non-special position (this checkpoint zeroes
        the <pad> row via padding_idx, an extreme off-manifold point; <unk> is the model's own
        trained "unknown token"); <s>/</s> keep their embeddings so their attribution is 0.

        Pass 1 (forward-only, cheap): g(a) = target(x0 + a*delta) on a uniform grid of
        `coarse_points` (endpoints double as f_baseline / f_input). Pass 2: `steps` gradient
        evaluations allocated across the coarse intervals proportional to |dg| with a 15%
        uniform floor, weighted-midpoint quadrature. This model's path is plateau -> cliff ->
        plateau; adaptive placement reaches the accuracy of a ~128-step uniform grid at a
        fraction of the cost. If the honest completeness gap still exceeds
        max(0.02, retry_rel_gap*|Δf|), up to two retries double the gradient budget and the
        best-gap pass is kept. The gap is REPORTED on every call, never renormalised away."""
        ids, tt, mask = self._tensors(enc)
        baseline_ids = ids.clone()
        # The baseline token must be the tokenizer's REAL unknown token. This was hardcoded to 3
        # (XLM-R's <unk>); on BERTurk id 3 is [SEP] and [UNK] is 1, so every attribution was
        # measured against a baseline of "sentence separator everywhere", which is meaningless.
        unk_id = None
        for cand in ("[UNK]", "<unk>", "<|endoftext|>"):
            tid = self.tok.token_to_id(cand) if hasattr(self.tok, "token_to_id") else None
            if tid is None and hasattr(self.tok, "convert_tokens_to_ids"):
                tid = self.tok.convert_tokens_to_ids(cand)
                if tid is not None and tid < 0: tid = None
            if tid is not None:
                unk_id = int(tid); self.unk_token = cand; break
        if unk_id is None:
            unk_id = int(getattr(getattr(self, "hf_tok", None), "unk_token_id", 0) or 0)
            self.unk_token = "<fallback id %d>" % unk_id
        for i, sid in enumerate(enc.sequence_ids):
            if sid is not None:
                baseline_ids[0, i] = unk_id

        emb = self.model.get_input_embeddings()
        with torch.no_grad():
            x = emb(ids)            # (1, L, H)
            x0 = emb(baseline_ids)
        delta = x - x0
        target_fn, target_name = self._target_fn(label_id)

        # ---- pass 1: coarse forward-only scan of g(alpha) ----
        C = max(coarse_points, 5)
        grid = torch.linspace(0.0, 1.0, C, device=self.device)
        g_vals = []
        with torch.no_grad():
            for chunk in grid.split(max(step_batch * 2, 8)):
                k = chunk.shape[0]
                e = x0 + chunk.view(k, 1, 1) * delta
                logits, _ = self._forward(embeds=e, tt=tt.repeat(k, 1), mask=mask.repeat(k, 1))
                g_vals.extend(target_fn(logits).tolist())
        g_vals = np.asarray(g_vals, dtype=np.float64)
        f_base, f_in = float(g_vals[0]), float(g_vals[-1])

        def alloc(budget: int) -> list[tuple[float, float]]:
            """(alpha, weight) samples: per coarse interval, weighted midpoints, count ~ |dg|."""
            seg = np.abs(np.diff(g_vals))
            tv = seg.sum()
            mass = seg + (0.15 * tv + 1e-9) / len(seg)      # 15% uniform floor
            n = np.floor(budget * mass / mass.sum()).astype(int)
            rem = budget - int(n.sum())
            if rem > 0:                                      # largest fractional parts first
                frac = budget * mass / mass.sum() - n
                for j in np.argsort(-frac)[:rem]:
                    n[j] += 1
            width = 1.0 / len(seg)
            out = []
            for j, nj in enumerate(n):
                if nj <= 0:
                    continue
                for m in range(nj):
                    out.append((float((j + (m + 0.5) / nj) * width), float(width / nj)))
            return out

        def integrate(samples: list[tuple[float, float]]) -> torch.Tensor:
            acc = torch.zeros_like(x)
            order = sorted(samples)
            with torch.enable_grad():
                for s in range(0, len(order), step_batch):
                    sub = order[s:s + step_batch]
                    k = len(sub)
                    a = torch.tensor([p[0] for p in sub], dtype=torch.float32, device=self.device)
                    w = torch.tensor([p[1] for p in sub], dtype=torch.float32, device=self.device)
                    e = (x0 + a.view(k, 1, 1) * delta).detach().requires_grad_(True)
                    logits, _ = self._forward(embeds=e, tt=tt.repeat(k, 1), mask=mask.repeat(k, 1))
                    obj = (w * target_fn(logits)).sum()      # rows come back pre-scaled by w
                    (g,) = torch.autograd.grad(obj, e)
                    acc += g.sum(0, keepdim=True)
            return (delta * acc).sum(-1)[0]                  # weights already integrate to 1

        # Bounded escalation: up to MAX_RETRY extra passes, doubling the gradient budget each
        # time, keeping the BEST-gap pass (more steps is not monotonically better on a cliff).
        # Only rows that fail the check pay for this, so typical latency is unchanged.
        tol = max(0.02, retry_rel_gap * abs(f_in - f_base))
        used_steps = int(steps)
        attr = integrate(alloc(used_steps))
        gap = abs((f_in - f_base) - float(attr.sum()))
        retries = 0
        MAX_RETRY = 2
        while gap > tol and retries < MAX_RETRY:
            retries += 1
            n2 = int(steps) * (2 ** retries)
            a2 = integrate(alloc(n2))
            g2 = abs((f_in - f_base) - float(a2.sum()))
            if g2 < gap:
                attr, gap, used_steps = a2, g2, n2
        retried = retries > 0

        attr_np = attr.detach().float().cpu().numpy()
        return {
            "attributions": attr_np,
            "target": target_name,
            "steps": used_steps,
            "coarse_points": C,
            "adaptive": True,
            "retried": retried,
            "retries": retries,
            "completeness_tol": float(tol),
            "completeness_ok": bool(gap <= tol),
            "baseline": getattr(self, "unk_token", "unk"),
            "f_input": f_in,
            "f_baseline": f_base,
            "sum_attributions": float(attr_np.sum()),
            "completeness_gap": float(gap),
        }

    # ------------------------------------------------------------ attention alignment
    @torch.inference_mode()
    def attention_alignment(self, enc, last_k: int = 2) -> dict:
        """Query-token x product-token attention. Primary matrix: mean over heads of the last
        `last_k` layers (raw attention — corroborating view only). Secondary: attention rollout
        (Abnar & Zuidema 2020) through all layers with residual mixing (0.5·A + 0.5·I)."""
        ids, tt, mask = self._tensors(enc)
        logits, atts = self._forward(ids=ids, tt=tt, mask=mask, output_attentions=True)
        per_layer = [a[0].float().mean(0) for a in atts]    # NL x (L, L), heads averaged
        last = torch.stack(per_layer[-last_k:]).mean(0)

        eye = torch.eye(last.shape[0], device=last.device)
        roll = None
        for A in per_layer:
            Ah = 0.5 * A + 0.5 * eye
            Ah = Ah / Ah.sum(-1, keepdim=True)
            roll = Ah if roll is None else Ah @ roll

        q_ix = [i for i, s in enumerate(enc.sequence_ids) if s == 0]
        p_ix = [i for i, s in enumerate(enc.sequence_ids) if s == 1]
        sub = lambda M: [[round(float(M[r, c]), 5) for c in p_ix] for r in q_ix]  # noqa: E731

        # Late layers of a fine-tuned BERT attend STRUCTURALLY (every query token looks at the same
        # key/boundary positions), so their rows are near-identical and carry no word alignment.
        # Pick, per request, the layer whose query rows differ most from each other: mean pairwise
        # (1 - cosine) between row distributions over the product columns. This is a display choice,
        # not a claim about the model; the score is returned so the UI can say when rows are alike.
        def distinct(M) -> float:
            R = M[q_ix][:, p_ix].clamp_min(0)
            R = R / R.sum(-1, keepdim=True).clamp_min(1e-9)
            if R.shape[0] < 2: return 0.0
            Rn = R / R.norm(dim=-1, keepdim=True).clamp_min(1e-9)
            cos = Rn @ Rn.T
            n = R.shape[0]
            return float((1.0 - cos).sum() / (n * (n - 1)))
        scores = [distinct(A) for A in per_layer]
        best_i = int(max(range(len(scores)), key=lambda i: scores[i])) if q_ix and len(q_ix) > 1 else len(per_layer) // 2
        return {
            "logits": logits.float().cpu().numpy()[0],
            "query_index": q_ix,
            "product_index": p_ix,
            "matrix_best": sub(per_layer[best_i]),
            "best_layer": best_i + 1,
            "distinct_best": round(scores[best_i], 4),
            "distinct_last": round(distinct(last), 4),
            "distinct_rollout": round(distinct(roll), 4),
            "distinct_per_layer": [round(x, 4) for x in scores],
            "matrix_last": sub(last),
            "matrix_rollout": sub(roll),
            "layers": f"best layer {best_i + 1} by row distinctness; mean(heads) over last {last_k} layers; rollout over all {len(per_layer)}",
        }

    # ------------------------------------------------------------ counterfactuals
    def counterfactual_segments(self, query: str, fields: dict, base: dict,
                                top_k: int = 4, max_attr_segments: int = 30) -> dict:
        """Leave-one-segment-out on the PRODUCT side + single-word deletions on the QUERY side,
        all re-scored by the same model in one batched pass (N+1 forwards).

        Product segments = the "|" template fields (removal: text fields -> "", gender/
        age_group -> "unknown" [the pipeline's own missing token], attributes -> "unknown"
        [the v2 empty sentinel]) and each individual "key: value" attribute pair (removal:
        drop the pair, rejoin; empty -> sentinel). `fields` are the NORMALISED fields."""
        variants = []   # (kind, removed_desc, query2, fields2)
        removal = {"title": "", "brand": "", "category": "",
                   "gender": "unknown", "age_group": "unknown", "attributes": AC.EMPTY_SENTINEL}
        for f, void in removal.items():
            if fields.get(f, "") not in ("", void):
                f2 = dict(fields); f2[f] = void
                variants.append(("field", TR_FIELD.get(f, f) + " alanı", query, f2))

        attrs = fields.get("attributes", "")
        n_attr_skipped = 0
        if attrs not in ("", AC.EMPTY_SENTINEL):
            segs = [s for s in NZ.SEG.split(attrs) if s.strip()]
            if len(segs) >= 2:
                if len(segs) > max_attr_segments:
                    n_attr_skipped = len(segs) - max_attr_segments
                    segs_used = segs[:max_attr_segments]
                else:
                    segs_used = segs
                for i, sgm in enumerate(segs_used):
                    rest = [x for j, x in enumerate(segs) if not (j == i)]
                    f2 = dict(fields)
                    f2["attributes"] = ", ".join(rest) if rest else AC.EMPTY_SENTINEL
                    name = sgm if len(sgm) <= 70 else sgm[:67] + "…"
                    variants.append(("attr", "“" + name + "”", query, f2))

        n_product = len(variants)
        words = query.split()
        if len(words) >= 2:
            for i, w in enumerate(words):
                q2 = " ".join(words[:i] + words[i + 1:])
                variants.append(("query_word", w, q2, fields))

        pairs = [(q2, build_doc(f2)[0]) for _k, _n, q2, f2 in variants]
        logits = self.score_pairs(pairs)
        rows = []
        _orig_cat = fields.get("category")
        for (kind, name, q2, _f2), lg in zip(variants, logits):
            d = self.decide(lg, category=_orig_cat)
            rows.append({
                "kind": kind,
                "removed": name,
                "label": d["label"],
                "label_tr": d["label_tr"],
                "scalar": round(d["scalar"], 4),
                "delta_scalar": round(d["scalar"] - base["scalar"], 4),
                "flips": d["label"] != base["label"],
            })

        def pack(sub):
            by_delta = sorted(sub, key=lambda r: -abs(r["delta_scalar"]))
            return {"flips": [r for r in by_delta if r["flips"]][:top_k],
                    "top_deltas": by_delta[:top_k],
                    "n_variants": len(sub)}

        out = {"base_label": base["label"], "base_label_tr": base["label_tr"],
               "product": pack(rows[:n_product]),
               "query_words": pack(rows[n_product:])}
        if n_attr_skipped:
            out["product"]["n_attr_skipped"] = n_attr_skipped
        return out

    # ------------------------------------------------------------ Turkish summary (templates)
    def summarize_tr(self, base: dict, fields_payload: list, tokens: list, cf: dict,
                     words: list | None = None) -> str:
        """One-paragraph Turkish rationale assembled from the numbers above. Pure templates."""
        def pct(v):
            return ("%" + f"{100 * v:.1f}").replace(".", ",")

        lbl = base["label_tr"]
        c1, c2 = base["cuts"]
        s = base["scalar"]
        if base["probs"] is not None:
            conf = pct(base["probs"].get(base["label"], 0.0))
            head = f"Model bu sorgu-ürün çiftini {conf} olasılıkla “{lbl}” olarak etiketledi."
        else:
            head = f"Model bu sorgu-ürün çiftini “{lbl}” olarak etiketledi."
        if s >= c2:
            pos = f"Skaler skor {s:.2f}, “ilgili” eşiği e₂={c2:.2f} üzerinde."
        elif s >= c1:
            pos = f"Skaler skor {s:.2f}, e₁={c1:.2f} ile e₂={c2:.2f} eşikleri arasında (kısmen ilgili bölgesi)."
        else:
            pos = f"Skaler skor {s:.2f}, “ilgisiz” eşiği e₁={c1:.2f} altında."

        # Direction matters. When IG attributes the DECISION SCALAR (ordinal_regime), a positive
        # attribution pushes the score UP (toward "relevant"), not "toward the predicted class".
        # For an irrelevant verdict the decisive evidence is therefore the most NEGATIVE fields —
        # listing positive ones would name exactly the evidence that argued against the verdict.
        scalar_target = self.head in ("ordinal", "ordinal_regime")
        lid = base["label_id"]
        sign = 0 if not scalar_target else (-1 if lid == 0 else (1 if lid == 2 else 0))
        cand = [f for f in fields_payload if f["field"] not in ("sep", "special")]
        real = [t for t in ((words or []) if (words or []) else tokens)
                if not t.get("special") and t.get("field") != "sep"
                and str(t.get("text", "")).replace("▁", "").replace("##", "").strip(" |")
                and str(t.get("text", "")).replace("▁", "").replace("##", "").strip(" |:").lower() not in ("unknown",)]

        def _fmt(items, key, k=2, tok=False):
            if tok:
                return ", ".join("“" + t["text"].replace("▁", "").replace("##", "").strip(",.;:()|") + "”" for t in items[:k])
            return " ve ".join(TR_FIELD.get(f["field"], f["field"]) for f in items[:k])

        drivers = ""
        if sign > 0 or not scalar_target:
            top = sorted([f for f in cand if f["total"] > 0], key=lambda f: -f["total"])
            ttop = sorted([t for t in real if t["ig"] > 0], key=lambda t: -t["ig"])
            lead = ("Skoru “ilgili” yönünde yukarı çeken alanlar" if scalar_target
                    else "Karara en çok katkı veren alanlar")
        elif sign < 0:
            top = sorted([f for f in cand if f["total"] < 0], key=lambda f: f["total"])
            ttop = sorted([t for t in real if t["ig"] < 0], key=lambda t: t["ig"])
            lead = "Skoru “ilgisiz” yönünde aşağı çeken alanlar"
        else:   # "kısmen ilgili": the verdict sits BETWEEN the cuts, so both directions decided it
            top = sorted(cand, key=lambda f: -abs(f["total"]))
            ttop = sorted(real, key=lambda t: -abs(t["ig"]))
            lead = "Skoru en çok hareket ettiren alanlar"
        if top:
            drivers = f" {lead}: {_fmt(top, 'total')}."
            if sign == 0 and scalar_target and len(top) >= 2:
                floor = max(0.02, 0.1 * max(abs(f["total"]) for f in top))
                up = [f for f in top if f["total"] > floor]; dn = [f for f in top if f["total"] < -floor]
                if up and dn:
                    drivers = (f" Skor iki eşiğin arasında kaldı: {_fmt(up, 'total', 1)} yukarı, "
                               f"{_fmt(dn, 'total', 1)} aşağı çekti.")
            if ttop:
                drivers += f" Öne çıkan belirteçler: {_fmt(ttop, 'ig', 3, tok=True)}."

        cfs = ""
        pflips = cf["product"]["flips"] if cf else []
        qflips = cf["query_words"]["flips"] if cf else []
        if pflips:
            r = pflips[0]
            what = r["removed"] if r["kind"] == "field" else r["removed"] + " niteliği"
            cfs = (f" Karşı-olgusal: {what} çıkarılırsa etiket “{r['label_tr']}” olurdu — "
                   f"karar bu bilgiye dayanıyor.")
        elif qflips:
            r = qflips[0]
            cfs = f" Karşı-olgusal: sorgudan “{r['removed']}” silinirse etiket “{r['label_tr']}” olurdu."
        elif cf and (cf["product"]["n_variants"] or cf["query_words"]["n_variants"]):
            cfs = (" Hiçbir tek alan, nitelik veya sorgu kelimesinin çıkarılması etiketi "
                   "değiştirmiyor; karar birden fazla kaynağa yayılmış ve kararlı.")
        return head + " " + pos + drivers + cfs

    # ------------------------------------------------------------ full explanation

    # ---------------------------------------------------------------- faithfulness + robustness
    def faithfulness(self, query: str, fields: dict, words: list, base: dict, k: int = 6) -> dict:
        """Deletion curve: drop the top-attributed WORDS in order and re-score after each.

        Completeness (already reported) is an AXIOMATIC check -- attributions sum to
        f(input)-f(baseline) by construction. It says nothing about whether the ranking is real.
        This is the EMPIRICAL check: if the words IG calls important are important, removing them
        in order should collapse the decision scalar fast. A flat curve would mean the ranking is
        decorative. Compared against removing RANDOM words, so the reader sees the contrast.
        """
        prod = [w for w in words if w.get("side") == "product" and w.get("field") not in ("sep", "special")
                and str(w.get("text", "")).strip()]
        if len(prod) < 3: return {}
        order = sorted(prod, key=lambda w: -abs(w.get("ig", 0.0)))[:k]
        rng = random.Random(0)
        rnd = rng.sample(prod, min(k, len(prod)))
        def curve(seq):
            pairs, drop = [], set()
            for w in seq:
                drop.add((w["seq"], w["start"], w["end"]))
                # rebuild the doc with the dropped spans blanked, character-exact
                d0 = build_doc(fields)[0]
                chars = list(d0)
                for (_sq, a, b) in drop:
                    for i in range(a, min(b, len(chars))): chars[i] = " "
                pairs.append((query, "".join(chars)))
            return pairs
        seqs = curve(order) + curve(rnd)
        lg = self.score_pairs(seqs)
        n = len(order)
        def scal(v):
            e = np.exp(np.asarray(v) - max(v)); pr = e / e.sum()
            return float(pr[1] + 2 * pr[2])
        s0 = base["scalar"]
        top = [s0] + [scal(lg[i]) for i in range(n)]
        ran = [s0] + [scal(lg[n + i]) for i in range(len(rnd))]
        drop_top = abs(top[0] - top[-1]); drop_rnd = abs(ran[0] - ran[-1])
        return {"k": n, "top_words": [w["text"] for w in order], "curve_top": top,
                "curve_random": ran, "drop_top": drop_top, "drop_random": drop_rnd,
                "ratio": (drop_top / drop_rnd) if drop_rnd > 1e-9 else None}

    def robustness(self, query: str, fields: dict, base: dict) -> list:
        """Turkish-specific perturbations of the QUERY, re-scored by the model.

        Real shoppers drop diacritics, glue compounds and typo. We MEASURED a hole here: on
        public rows whose query is a glued compound the shipped model scores 74.4% against 85.8%
        elsewhere. This strip shows, per row, whether the decision survives those rewrites --
        invariance is the property we want, and where it breaks we would rather show it than hide it.
        """
        q = query.strip()
        TRMAP = str.maketrans("çğıöşü", "cgiosu")
        variants = [("aksansız", q.translate(TRMAP))]
        if " " in q:
            variants.append(("boşluksuz", "".join(q.split())))
        # one-letter typo: drop the middle LETTER (never the space, which would duplicate "boşluksuz")
        letters = [i for i, ch in enumerate(q) if ch.isalnum()]
        if len(letters) > 4:
            i = letters[len(letters) // 2]
            variants.append(("tek harf eksik", q[:i] + q[i + 1:]))
        variants.append(("büyük harf", q.upper()))
        variants = [(n, v) for n, v in variants if v and v != q]
        if not variants: return []
        # Every variant goes through the SAME normalisation the serving path applies to the real
        # query (T1: NFC + U+0307 strip + lower-case). A variant that normalises back to the base
        # query is reported as neutralised by preprocessing, not scored -- scoring the raw string
        # would show a "brittleness" the container never has.
        base_norm = NZ.t1(q)
        scored, out = [], []
        for nm, v in variants:
            vn = NZ.t1(v)
            if vn == base_norm:
                out.append({"name": nm, "query": v, "label": base["label"], "label_tr": base["label_tr"],
                            "scalar": base["scalar"], "same": True, "delta": 0.0, "normalized_away": True})
            else:
                scored.append((nm, v, vn))
        if scored:
            doc = build_doc(fields)[0]
            lg = self.score_pairs([(vn, doc) for _n, _v, vn in scored])
            for (nm, v, vn), row in zip(scored, lg):
                d = self.decide(row, category=fields.get("category"))
                out.append({"name": nm, "query": vn, "label": d["label"], "label_tr": d["label_tr"],
                            "scalar": d["scalar"], "same": d["label"] == base["label"],
                            "delta": d["scalar"] - base["scalar"], "normalized_away": False})
        order = {n: i for i, (n, _v) in enumerate(variants)}
        out.sort(key=lambda r: order.get(r["name"], 99))
        return out

    def explain(self, query: str, fields: dict, ig_steps: int = 16, ig_step_batch: int = 4,
                counterfactuals: bool = True) -> dict:
        """The whole rationale payload for one (query, product) pair. `fields` are the
        NORMALISED product fields; `query` is (possibly juror-edited) free text, already
        t1-normalised by the caller."""
        doc, spans = build_doc(fields)
        enc = self.encode(query, doc)
        timings = {}

        t = time.time()
        att = self.attention_alignment(enc)
        base = self.decide(att["logits"], category=fields.get("category"))
        timings["forward_attention_ms"] = round((time.time() - t) * 1000)

        t = time.time()
        ig = self.integrated_gradients(enc, base["label_id"], steps=ig_steps,
                                       step_batch=ig_step_batch)
        timings["ig_ms"] = round((time.time() - t) * 1000)

        tokens = []
        for i, (tstr, sid, off) in enumerate(zip(enc.tokens, enc.sequence_ids, enc.offsets)):
            if sid is None:
                field = "special"
            elif sid == 0:
                field = "query"
            else:
                field = _assign_field(off, spans)
            tokens.append({
                "i": i,
                "text": tstr,
                "seq": sid,
                "field": field,
                "start": int(off[0]) if sid is not None else -1,
                "end": int(off[1]) if sid is not None else -1,
                "ig": round(float(ig["attributions"][i]), 6),
                "special": sid is None,
            })

        # Sub-word pieces ("pe", "jo", "30") are not something a juror can read. Aggregate them
        # back into WHOLE WORDS over the original character offsets: IG is additive, so a word's
        # attribution is exactly the sum of its pieces — no approximation is introduced.
        words = _words_from_tokens(tokens, {0: query, 1: doc})

        field_totals = {}
        for tk in tokens:
            field_totals[tk["field"]] = field_totals.get(tk["field"], 0.0) + tk["ig"]
        field_order = ["query"] + DOC_FIELDS + ["sep", "special"]
        fields_payload = [{"field": f, "total": round(field_totals.get(f, 0.0), 6)}
                          for f in field_order if f in field_totals]

        cf = None
        if counterfactuals:
            t = time.time()
            cf = self.counterfactual_segments(query, fields, base)
            timings["counterfactual_ms"] = round((time.time() - t) * 1000)

        payload = {
            "prediction": base,
            "tokens": tokens,
            "words": words,
            "ig": {
                "target": ig["target"],
                "steps": ig["steps"],
                "coarse_points": ig.get("coarse_points"),
                "adaptive": ig.get("adaptive", False),
                "retried": ig.get("retried", False),
                "baseline": ig.get("baseline"),
                "f_input": round(ig["f_input"], 6),
                "f_baseline": round(ig["f_baseline"], 6),
                "sum_attributions": round(ig["sum_attributions"], 6),
                "completeness_gap": round(ig["completeness_gap"], 6),
                "fields": fields_payload,
            },
            "attention": {
                "query_tokens": [tokens[i]["text"] for i in att["query_index"]],
                "product_tokens": [tokens[i]["text"] for i in att["product_index"]],
                "product_fields": [tokens[i]["field"] for i in att["product_index"]],
                # Map every attention row/column onto an index into `words`, so the UI can show
                # attention over WORDS and FIELDS instead of a raw token x token grid. At realistic
                # sizes that grid is ~5 x 90 cells of 4px with rotated, truncated sub-word labels —
                # unreadable, and it does not answer the question a juror actually asks, which is
                # "what did this query term attach to, and how strongly".
                "query_word_idx": _token_to_word_idx(att["query_index"], tokens, words),
                "product_word_idx": _token_to_word_idx(att["product_index"], tokens, words),
                "words": words,
                "matrix_last": att["matrix_last"],
                "matrix_rollout": att["matrix_rollout"],
                "matrix_best": att["matrix_best"],
                "best_layer": att["best_layer"],
                "distinct_best": att["distinct_best"],
                "distinct_last": att["distinct_last"],
                "distinct_rollout": att["distinct_rollout"],
                "distinct_per_layer": att["distinct_per_layer"],
                "layers": att["layers"],
            },
            "doc": doc,
            "normalized": fields,
            "n_tokens": len(tokens),
            "truncated": len(tokens) >= self.max_len,
            "timings_ms": timings,
        }
        if cf is not None:
            payload["counterfactuals"] = cf
        # --- empirical faithfulness of the IG ranking, and Turkish robustness -------------
        try:
            t = time.time(); payload["faithfulness"] = self.faithfulness(query, fields, words, base)
            timings["faithfulness_ms"] = round((time.time() - t) * 1000)
        except Exception:
            payload["faithfulness"] = {}
        try:
            t = time.time(); payload["robustness"] = self.robustness(query, fields, base)
            timings["robustness_ms"] = round((time.time() - t) * 1000)
        except Exception:
            payload["robustness"] = []
        # --- do the two INDEPENDENT attribution methods agree? -----------------------------
        # IG is gradient-based; the counterfactual panel is occlusion-based (actually re-scoring
        # the model without each field). They share no machinery, so their agreement is real
        # evidence rather than a restatement. Spearman over the fields both methods scored.
        try:
            occ = {r["removed"].removesuffix(" alanı"): -r["delta_scalar"]
                   for r in (cf or {}).get("product", {}).get("top_deltas", [])
                   if r.get("kind") == "field"}
            igf = {TR_FIELD.get(f["field"], f["field"]): f["total"] for f in fields_payload}
            common = [k for k in occ if k in igf]
            if len(common) >= 3:
                import statistics as _st
                def rank(d, keys):
                    order = sorted(keys, key=lambda k: d[k])
                    return {k: i for i, k in enumerate(order)}
                ra, rb = rank(occ, common), rank(igf, common)
                n = len(common)
                d2 = sum((ra[k] - rb[k]) ** 2 for k in common)
                rho = 1 - 6 * d2 / (n * (n * n - 1))
                payload["method_agreement"] = {"n_fields": n, "spearman": round(rho, 3)}
        except Exception:
            pass
        payload["summary_tr"] = self.summarize_tr(base, fields_payload, tokens, cf, words)
        return payload
