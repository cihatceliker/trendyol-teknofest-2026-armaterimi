# Explainability — the technical account (locked 2026-08-28)

Everything on the page is computed by the **shipped checkpoint** (`ckpt_gf2_berturk_6ep_bf16`,
BERTurk-128k, 12 layers, 184,347,651 params) with the **same preprocessing** as the container
(`normalize.t1–t4` + `attr_clean.apply_v2` + the `title | brand | category | gender age | attributes`
template) and the **same decision rule** (`decision.json`: ordinal scalar, two global cuts). There is no
second model and no LLM. `verify_ui_parity.py` asserts UI label == container label (300/300).

## 0. One request, in order

1. Normalize the (possibly edited) query and fields exactly as the container does.
2. Tokenize the pair as one sequence: `[CLS] query [SEP] product-template [SEP]` (cross-encoder).
3. One forward pass with attention outputs → logits → softmax → `p_irr, p_semi, p_rel`.
4. Decision: `x = p_semi + 2·p_rel`, cut at `e1 = 0.9518`, `e2 = 1.0619` (fitted on DEV only).
5. Integrated Gradients on `x` (not on a class logit) at the embedding layer.
6. Occlusion counterfactuals: one piece removed at a time, model re-run (one batched forward).
7. Robustness variants of the query, normalized, re-scored.
8. Faithfulness (deletion) test and IG-vs-occlusion agreement.
9. Template summary from the numbers. Total ≈ 0.3–0.9 s on the GPU.

## 1. The decision (Karar)

Classes are ordinal (irrelevant < semi < relevant → ranks 0, 1, 2). `x` is the expected rank, a number
in [0, 2]. Two thresholds turn it into a label; the number line shows the score, the cuts and the
distance to the nearest cut. This is the calibration step argmax lacks (+0.0084 public F1) and it is
one-dimensional, which is why it can be drawn. Why not a regression head: MSE cannot express
"either relevant or irrelevant, not semi", assumes equidistant classes, and fine-tunes worse than CE;
our scalar is the regression target learned with the better loss.

## 2. Word contributions (Integrated Gradients)

- **What:** for each input token, the integral of the gradient of `x` along the straight path from a
  baseline input to the real input, at the word-embedding layer. Axiomatic (Sundararajan et al. 2017).
- **Baseline:** every token replaced by the tokenizer's `[UNK]` embedding (the model's own "no
  information" token); `[CLS]`/`[SEP]` kept. The model scores this blank pair high (≈1.5–1.9, the
  relevant region) — its prior when nothing contradicts relevance. **Consequence:** red = a word that
  pulled the score down from that prior (counter-evidence); blue = pushed it up. On semi/irrelevant
  verdicts most words are red by construction, because the score had to come down. The baseline
  score is printed in the footer so this reading is checkable.
- **Completeness:** Σ attributions = f(input) − f(baseline). Printed per request (footer + modal);
  never re-normalized. Tolerance `max(0.02, 4%·|Δf|)`.
- **Adaptive quadrature:** the integrand on this checkpoint is plateau → cliff → plateau; uniform
  16-step IG missed the cliff. We pre-scan the path at 17 points, allocate the gradient budget per
  interval ∝ |Δg| (15% uniform floor), and if the completeness gap exceeds tolerance we double the
  budget, at most twice, keeping the best pass. `steps` in the modal is the budget actually used.
- **Words vs sub-word pieces:** WordPiece splits rare words (`mes ##si`, `un ##kn ##own`). IG is
  additive, so a word's value is the exact sum of its pieces; the default view is words, the checkbox
  shows pieces. Fragmentation is a property of every BERT model; the 128k vocabulary splits 9–18%
  fewer tokens than 32k Turkish BERTs. The three-piece `unknown` sentinel costs ~1% tokens; changing
  it needs a retrain (not worth it).
- **Field bars:** the same values summed per field. The query has its own group because in a
  cross-encoder the query tokens are half of the input and carry evidence (an unanswered query word
  is negative evidence). The bars sum to Σ attributions.

## 3. Counterfactuals (occlusion)

Each product field, each `key: value` attribute (first 30) and each query word is removed one at a
time; the model is re-run on the remaining text; label and score change are reported. Rows that flip
the label are highlighted — "the decision rests on this". **Reading rule:** removing a piece removes
its evidence, so the score drifts toward the blank-input prior (~1.9): removing the field that held
the strongest counter-evidence makes a row look *more* relevant. That is measurement, not a bug. If
nothing flips, the decision is distributed across several pieces.

## 4. Spelling robustness

Query variants — without diacritics, without spaces, one middle letter dropped, upper-cased — are
passed through the container's own normalization (`t1`: NFC, U+0307 strip, lower-case) and
re-scored. A variant that normalizes back to the original is reported as neutralized by
preprocessing, not scored. Upper-case Turkish `I` lower-cases to `i` (dotless ı lost) exactly as in
the container; the row shows the string the model saw. The glued-compound and typo rows show the
model's real sensitivity; it is shown, not hidden.

## 5. Query → product map (attention, supporting)

"Son katmanlar": raw attention averaged over heads of the last 2 layers, query rows × product
columns, summed to words, each row normalized to 100%. "Tüm katmanlar": attention rollout
(Abnar & Zuidema 2020) through all 12 layers with 0.5·A + 0.5·I residual mixing. Colour band =
field of the column word. Always introduced as supporting: attention is where the model looked,
not why it decided (Jain & Wallace 2019; Wiegreffe & Pinter 2019). The primary explanation is IG.

## 6. Technical checks (modal)

- **Faithfulness (deletion):** delete the top-k IG words vs k random words and re-score. Reported
  with a verdict: if both changes are < 0.05 the score is saturated and the test is not
  discriminative on that row; else top ≫ random means the ranking is real.
- **Method agreement:** Spearman between IG's field ranking and occlusion's field ranking. If no
  field removal moves the score by > 0.05, the comparison is declared not meaningful for that row.
  The two methods answer different questions (path from blank input vs. leave-one-out with everything
  else present), so disagreement on redundant rows is expected.

## 7. What is *not* claimed

Attention is not causal. IG explains the score relative to a blank baseline, not relative to "the
average product". Counterfactuals are single-piece removals, not all subsets. The summary is a
template. None of the panels changes the prediction; the label shown is the container's label.
