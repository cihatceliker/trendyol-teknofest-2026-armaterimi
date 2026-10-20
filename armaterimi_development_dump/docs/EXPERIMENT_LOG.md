# Stage-2 experiment log

One dated entry per session/decision. Newest first. Numbers are DEV unless stated.

## 2026-08-25 17:40 — TY-ecomm 10-LAYER: same accuracy, +12.5% speed (new ship candidate)

x1_tyecomm_10L (bottom-10 of the pretrained stack, identical recipe/data/seed): DEV OOF 0.8469
two-boundary / 0.8494 scalar / **0.8571 with regime cuts** (regime still pays here: +0.0075, vs only
+0.0017 on BERTurk). AUC(S|I) 0.9112 — BETTER than the 12L champion (0.9037); R|rest 0.9742.
90k bench: 26.95s = **3,340 SPS** in-container vs 12L's 30.3s / 2,967.
**So: -0.0016 macro-F1 (one seventh of the 0.012 noise floor) for +12.5% throughput.**
Speed sweep (both models, byte-checked): batch 256 + MAXLEN 128 is the best config —
TY-12L 2,910->3,133 SPS, BERTurk 4,931->5,224. MAXLEN 192 gains nothing (mean len is 105 and
bucketing already handles the tail); the +7.7% comes from truncating the long tail at 128.
Revised finalist table (DEV OOF best rule / in-container 90k SPS):
  TY-ecomm 12L  0.8587 / 2,967      TY-ecomm 10L 0.8571 / 3,340      BERTurk 0.8373 / 5,073
Reading: 10L dominates 12L (same accuracy, more speed) and narrows the BERTurk speed ratio from
1.56x to 1.42x while keeping a 0.020 accuracy edge and 2x the margin above the 0.80 floor.
PENDING: x2 aux S|I head; Y = MAXLEN-128 accuracy verification with cuts REFIT at that length
(the earlier 128 numbers reused 256-fitted cuts, so the +7.7% is not bankable yet); then ONE
HOLDOUT read on whichever candidate is frozen.

## 2026-08-25 16:10 — BERTurk on the FULL recipe: 0.8350 (the head-to-head was unfair before)

c1_berturk_3ep had trained on 678,549 rows; the TY-ecomm champion on 813,269 (incl. the old store).
Retrained BERTurk on the SAME 813,269: **DEV OOF 0.8350 two-boundary / 0.8373 scalar** (was 0.8205),
AUC S|I 0.9030, R|rest 0.9596. The old-store rows are worth **+0.015** to BERTurk.
Regime cuts on it: +0.0017 (0.8336 -> 0.8353) — much weaker than on TY-ecomm (+0.0081); its best
readout stays the scalar at 0.8373.
**Revised finalist gap: 0.8587 vs 0.8373 = 0.021 (was 0.035); BERTurk floor margin 0.037 (was 0.024).**
Both of my arguments for TY-ecomm weaken accordingly: on the scoring math it is now near a coin flip
when the field is slow, and still TY-ecomm when the field is fast (speed MaxNorm compresses, accuracy
rank-mult does not). The 10-layer TY-ecomm run therefore becomes the decisive experiment.
v2 BERTurk-4ep auto-skipped as planned.

## 2026-08-25 16:00 — report filled against measured numbers; 3h GPU queue running

FINAL_REPORT_DRAFT.md: 10 placeholder blocks replaced with real results — HOLDOUT 0.8646 (read #3, on
the shipped container output; DEV half 0.8633, all-7500 0.8639, so no in-sample inflation), true-90k
speed 31.1s = 2,892 SPS in-container (~2,390 official clock), jury GPU rehearsal PASS under real
flags, image 9.56 GB, lean GTE port parity max|dlogit|=0.0, explainability integrated + tunnel-verified
and compliant with the organizer 2026-08-25 clarification (own-model signals, no meta-explainer LLM).
Added: mermaid pipeline diagram, the frozen training recipe (813,269 rows with the exact mix + the
"public never trained on" statement + n=3 seed sensitivity), and a component-removal ablation table
built from the measured data ladder (0.8149 -> 0.8217 -> 0.8441 -> 0.8529 -> 0.8553 -> 0.8587;
argmax 0.8434 as the decision-rule contrast).
GPU queue (3h, detached + monitored): v1 BERTurk on the FULL 813k recipe (its 0.8236 was measured on
only 678k -> unfair head-to-head) -> W batch x MAXLEN sweep on both finalists (byte-checked) ->
x1 TY-ecomm 10-LAYER + 90k bench (the decision-changer: 12L 0.8553@2,892 vs 8L 0.8393@1,938) ->
x2 aux S|I output head. Dropped v2 BERTurk-4ep (4 epochs measured worse every time).
INT8 status: torchao 0.9.0 installed and working, but quantize_ needs an nn.Module and lean_gte holds
raw tensors -> must be measured on the HF module first, then transplanted via torch._int_mm if it pays.

## 2026-08-25 16:20 — HOLDOUT read #3, on the SHIPPED CONTAINER OUTPUT

Challenged on where 0.8639 came from. It is the organizer's own evaluate.py over all 7,500 public
rows from the container's submission.csv — but the two cuts were fitted on the DEV half, so half of
it is in-sample. Split it:
  all 7,500          0.8639
  DEV half   (3,750) 0.8633   <- cuts fitted on these
  **HOLDOUT half (3,750) 0.8646  <- never used to fit anything: the honest number**
No in-sample inflation (the untouched half scores HIGHER), so the fitted cuts generalise and 0.8639
is not flattered. Best honest estimate for the shipped artifact: **~0.865**.
HOLDOUT read log: #1 0.8235, #2 0.8385/0.8438 (older checkpoints), **#3 0.8646 (shipped container)**.
Note the DEV OOF estimate for this rule was 0.8587 — the shipped cuts beat it because they are fitted
on all 3,750 DEV rows rather than 4/5 folds, i.e. less cut variance. **FREEZE PLAN CORRECTED (user, 2026-08-25): do NOT refit the cuts on all 7,500.** Cuts are fitted on
DEV only; HOLDOUT stays a pure test set. Refitting would consume the only clean measurement we have,
and it is unnecessary — the DEV-fitted cuts already generalise (HOLDOUT 0.8646 >= DEV 0.8633).
DELIVERABLE NOW: public 0.8639 / HOLDOUT-half 0.8646, 31.12s in-container at true 90k = 2,892 SPS
(~2,390 on the official clock).

## 2026-08-25 16:00 — first TRUE 90k timing + worker-side tokenisation (+12.5% SPS)

Built a real 90,000-row input (public rows x12, unique row_ids) instead of extrapolating from 7.5k.
**Measured in-container at 90k: 35.01s = 2,571 SPS** (previous projection was 1,900-2,100 — we were
pessimistic). Breakdown: tokenize 9.9s (28%), inference 22.6s (65%), read/normalise 0.2s (hidden in
workers), write 0.8s.
**FINDING: the `tokenizers` Rust parallelism is dead in the parent after ANY fork.** Standalone
parallel encode of 90k = 3.7s; in-pipeline = 9.9s; no-fork serial run = 6.2s; and a worker-count
sweep (2/4/6/10) left it flat at ~9-10s, so it is the fork, not CPU contention. The pool is still
worth it (it hides 11.4s of normalisation), so the fix is to tokenise INSIDE the workers.
**Implemented** `preload_tokenizer` (loaded pre-fork, inherited copy-on-write) + `_norm_tok_chunk`
(workers normalise AND tokenise, return flat numpy arrays + lengths — small enough for the 64MB
/dev/shm) + `_Enc` shim (only .ids/.type_ids are ever used). WTOK=1 is the default; WTOK=0 and
LEAN_SERIAL_READ=1 remain as fallbacks.
**Result: data path 10.78s -> 6.24s, total 35.01s -> 31.12s, 2,571 -> 2,892 SPS (+12.5%),
output byte-identical** (cmp clean vs the previous path at 90k). Image rebuilt (337876c2), jury test
PASS, public F1 unchanged at 0.863942, wall 13s. parity_datapath re-run: token ids + type_ids
identical on all 7,500 rows.
**Also measured and REJECTED: streaming the data path into inference.** Only +1.5% (my generator
serialises tokenize->infer), and it changes batch composition, which flips 0.1% of labels (90/90,000,
all semi<->irrelevant / semi<->relevant) because padding moves logits by float-noise and the regime
bands are as narrow as 0.037. Default left OFF (STREAM=0); code kept behind the flag.
NOTE for the report: label output is reproducible for a FIXED batching, but ~0.1% of rows sit within
float-noise of a cut — worth stating rather than claiming bit-determinism across configurations.

## 2026-08-25 15:05 — decision.json fixed: image now serves per-regime cuts, not argmax

The shipped image was serving `{"head":"softmax3"}` (argmax) — measured 0.8469 public F1 — while the
same weights with fitted cuts score ~0.855-0.86. Wrote the verified per-regime table into
ckpt_r1_oldstore/decision.json (`ordinal_regime`; old file kept as decision.json.argmax_bak),
rebuilt (image 14d94bd6, 9.56 GB) and re-ran the jury test.
**Shipped public macro-F1 0.8469 -> 0.8639 (+0.017); wall unchanged at 13s** (the regime lookup is a
single-column CSV pass). CAVEAT: cuts were fitted on DEV = half of that public set, so 0.8639 is
partly in-sample; the honest OOF estimate for the rule is 0.8587 (regime replicated 5/5 checkpoints).
FREEZE: cuts stay DEV-fitted (do NOT refit on all 7,500 — HOLDOUT must remain a pure test set).
Options table for the user compiled from every measured run (backbones, data increments, failures).

## 2026-08-25 14:40 — TWO-HEAD + Stage-1: CLOSED (architecture, not data, is the problem)

| run | 2-boundary | scalar | AUC R|S | AUC S|I | AUC R|rest |
| baseline 3-class (r1_oldstore) | **0.8553** | 0.8543 | **0.9663** | 0.9037 | 0.9775 |
| two-head + all 139,147 Stage-1  | 0.8465 | 0.8466 | 0.9555 | **0.9126** | 0.9728 |
| two-head + 96k confident only   | 0.8458 | 0.8451 | 0.9530 | 0.9116 | 0.9713 |

Dropping the noisy uncertain band changed nothing (-0.0007; R|S even slightly worse), so the
regression is NOT the Stage-1 label noise — it is the two-head decomposition itself. Both runs move
identically regardless of data: S|I +0.009, R|S -0.012, net negative. Reading: head B becomes an S/I
specialist (hence the S|I gain) but head A loses the three-way contrast that teaches where relevant
ends and semi begins; extra relevant-vs-rest examples cannot fix a missing contrast.
Pre-band FN measurement that predicted (wrongly) that t3 would recover: confident-band positives are
99.88% correct and negatives 88.7%, vs uncertain-band positives at 56.4% (coin flip) — the noise was
real, it just was not the binding constraint.
**DECISION: ship the 3-class single head (0.8553). Stage-1 data + two-head closed.**
SALVAGE: S|I 0.9126 is the best S/I separation ever measured here (baseline 0.9037) and S/I owns 2x
the headroom — worth one run of 3-way softmax PLUS a separate auxiliary S|I output head (distinct
from the flat AUX_SI margin-on-same-logits experiment).
Also settled today (calibration): cuts ARE the calibration (+0.021 over argmax; argmax over-predicts
semi 43.6% vs 33.3% truth and every one of the 179 rows the cuts rescue was an argmax "semi").
Global temp/prior recalibration adds only +0.004. Conditional-by-category replicates 5/5 checkpoints:
regime-3 +0.0058 (sd 0.0038), root-15 +0.0070 (sd 0.0042); regime x query-length looked like +0.0138
on one checkpoint but is 4/5 with higher variance -> REJECTED as selection noise.

## 2026-08-25 11:00 — CORRECTED Stage-1 head-A build (139,147 rows) + two-head chain

MISTAKE CORRECTED: the first e21 build added a catalog-resolution filter nobody asked for (item
title had to exist in the Stage-2 pool), which cut the set to 43,837 rows / 3.6% of training. The
instruction was ONE filter: take all unique mix2 pairs, drop the ones already in Stage-2, use the
rest for head A. Stage-1 items.csv carries all six fields, so no resolution step is needed — the
Stage-1 text goes through the same normalize.py + attr_clean v2 transform, which is exactly what
strips the snapshot-A markers (`color detail:` etc.).
**Corrected chain:** mix2 299,476 rows -> 198,950 unique pairs -> drop 56,502 already in Stage-2 ->
drop 643 public-query collisions -> drop ~2,658 post-normalization duplicates = **139,147 rows**
(85,176 relevant / 53,971 not-relevant, 36,585 queries; confident 95,885 / uncertain 43,262).
Mapping re-verified on the 56,502 dropped overlaps: Stage-1 relevant -> Stage-2 relevant 99.77%;
Stage-1 not-relevant -> semi 82.91%.
**Wiring confirmed in the run log:** headA 952,416 rows (813,269 Stage-2 + 139,147 Stage-1),
headB 548,150 rows (Stage-2 semi+irrelevant only — Stage-1 abstains, since Stage 1 never
distinguished semi from irrelevant).
Queue: t2 (all Stage-1) -> t3 (confident band only) -> t1 (architecture control, no Stage-1).
Watch AUC(R|S): 0.9601 now vs the 12B's 0.9839 — that is the boundary these 139k rows target.

## 2026-08-25 10:45 — TWO-HEAD architecture + Stage-1 complementary labels (user design)

**Design (user):** head A = relevant-vs-rest, head B = semi-vs-irrelevant CONDITIONAL on not-relevant.
Stage-2 rows train both heads; Stage-1 rows train head A ONLY (a Stage-1 negative is a complementary
label — semi-or-irrelevant, unresolved — so head B abstains). p_rel=s(A), p_semi=(1-s(A))s(B),
p_irr=(1-s(A))(1-s(B)) → the ordinal scalar and two cuts carry over unchanged; 2 logits vs 3, so
serving speed is untouched.
**Why:** oracle on our own DEV preds — oracle R-gate +0.059, oracle S/I +0.120; and our AUC(R|S) is
0.960 vs the 12B's 0.984 on the same rows, i.e. the relevant boundary is provably improvable and had
never been targeted (the one aux-loss run pushed S/I and was flat).
**e21 data build:** source = first_stage mix2.parquet (299,476 rows = the FINAL Stage-1 Asure run2
training set). Filters: item title must exist in the Stage-2 catalog (57.7%) AND the item's whole
field record is taken FROM THE STAGE-2 POOL (never Stage-1 text — that would re-import the snapshot
artifact); drop 1,276 public-query collisions; drop 111,281 pairs already organizer Stage-2 rows;
hard labels only (target>=0.5), confident band {0.05,0.08,0.95,0.97} vs uncertain {0.28,0.72}.
Result **43,837 rows** (confident 30,849 / uncertain 12,988) over 21,152 queries.
**Mapping validated in-builder as a regression assert** (111,281 overlapping pairs):
Stage-1 relevant → Stage-2 relevant **99.85%**; Stage-1 not-relevant → semi **82.79%** / relevant 17.2%.
**Code:** TWO_HEAD/S1_W/HEAD_B_W in train_student.py (masked BCE pair, pos-weighted, label-smoothed);
STAGE1=/STAGE1_BANDS= source in build_student_data.py; two-head probs in eval_student.py.
**e23 per-regime cuts:** fitter written; r1_oldstore global 0.8506 → per-regime **0.8587 (+0.0081)**.
lean_infer.decide() extended: `two_head` probs + `ordinal_regime` (cuts by the row's own category
root — per-row, no batch stats; single-column CSV pass only when active). Verified offline: regime
rule matches the reference exactly, two-head probs sum to 1 and p_rel==sigmoid(A), and softmax3 /
global-ordinal back-compat unchanged.
**Running:** T1 two-head (arch only) ETA 12:28 → T2 +Stage-1 confident → T3 +uncertain.
Accept only on >0.012 DEV OOF vs the recipe mean 0.848, with the gain visible in AUC(R|S).

## 2026-08-24 20:20 — freeze tooling built and code-validated (DEV half only)

e15 scripts/fit_ship_cuts.py: full-public forward via the EXACT serving stack (lean_gte, 7,500 rows in
2.2s on GPU), fits the two ordinal cuts by macro-F1, writes decision.json + fit manifest with backups.
Discipline guard: --half full refuses without FREEZE_CONFIRM env (must follow tomorrow's HOLDOUT read).
Dry-run on DEV half: 0.8568 at cuts [0.9623, 1.0925] (in-sample; consistent with 0.8553 OOF).
Tomorrow's freeze = 4 commands: eval HOLDOUT -> fit_ship_cuts --half full --write -> prepare+build ->
test_like_jury (final shipped F1+SPS) -> FREEZE manifest + fill report/deck placeholders.

## 2026-08-24 20:10 — night hardening: hostile inputs, integration final, GPU explain verified

Hostile-input audit (shipped serve_gte, CPU): 8 adversarial rows (all-empty, 20k-char, emoji/unicode
oddities, CSV quote/newline escapes, malformed attribute strings, boilerplate-only) — exit 0, schema
valid, row order preserved. No crash class found.
Integration agent final: new image 7cf8221f (9.56GB) reproduces the solution-mode sha BYTE-IDENTICALLY
(570bb850) at wall 13s; ngrok 3.39.11 sha cross-verified vs the apt repo; token absent from all layers.
GPU explain-mode (jury flags + --gpus all + CSV bind-mount): tunnel URL t+10s, model cuda-ready 5.9s,
full explain (IG+attention+counterfactual+TR summary) in 0.6s through the tunnel. All modes verified.

## 2026-08-24 20:10 — seed-44: variance is real; expectations recalibrated

Freeze recipe, three seeds: 42=0.8553, 43=0.8514, 44=0.8367 → mean 0.8478, sd ≈0.0095, range 0.019.
Honest reading: the shipped ckpt (seed 42) is the best ARTIFACT we own, but the recipe's expected value
is ~0.848±0.010 and 0.8553 is its lucky tail — private-set expectation anchors at ~0.845-0.855, which
matches the HOLDOUT scalar readings (0.8438). No decision changes (we ship the checkpoint, not the
recipe; its 2ep sibling was HOLDOUT-verified). Report/deck variance claims must quote n=3.

## 2026-08-24 20:05 — explainability INTEGRATED into the image (both jury modes)

Image 7cf8221f996e, 9.56GB (+140MB): e20 app at /app/explain, sha-pinned vendored ngrok 3.39.11,
pandas stack pinned, baked E20_* env. Explainability mode verified END-TO-END under jury flags:
public URL t+9s, /readyz t+12s, hand-edited query flips label through the tunnel, clean teardown.
Ngrok token proven absent from logs/ENV/all 9.56GB of layers (byte grep). Solution-mode retest
auto-fires when the GPU frees (must reproduce sha 570bb850). The image is now DELIVERABLE-COMPLETE
except the freeze items (ordinal decision.json + final ckpt confirmation).

## 2026-08-24 19:30 — presentation outline drafted (docs/PRESENTATION_OUTLINE.md)

14 slides TR + timing + 60s demo script + 10-question jury Q&A, numbers as FREEZE placeholders mirroring
the report. Transcript findings that matter: presentation format is FREE (details on-site 5-6 Sep);
**the live explainability GUI demo is mandatory and its quality is scored at the presentation** (review
period only checks the ngrok link works + arbitrary row_id + hand-edited query); explanations MUST come
from the predicting model itself (narrator-LLM rejected — our IG/counterfactual design complies);
last year original demos differentiated points. Deck stays Turkish.
In flight: seed-44 variance run (GPU), explainability-into-image integration (agent).

## 2026-08-24 19:00 — final image state for the night

predict.py CUDA fail-fast now a ctypes cuInit probe (~50ms) instead of a serial torch import (~4s dead
time; torch loads overlapped on the serve build thread). Jury runs: wall 17s (old) -> 14s -> **13s**;
in-container 6.1s/7.5k (model-ready 3.9s critical path, infer 2.17s), public argmax F1 0.846869 stable
across every image variant (byte-exact ports). Overhead decomposition: nvidia hook 3.7s + create 0.3s +
harness/mounts ~2.5s (structural, all teams). Honest official-clock projection @90k: **~45-48s ≈
1,900-2,100 SPS at 0.8553** (with ordinal cuts: ~0.855-class F1, cuts pending at freeze).
Backup v2 uploaded (evening artifacts). Accuracy levers dry; speed at diminishing returns in-container.
TOMORROW = FREEZE: (i) HOLDOUT read of r1 (DEV-cut form), (ii) cuts on all 7.5k -> decision.json ordinal,
(iii) rebuild + jury rerun = final shipped F1+SPS, (iv) FREEZE manifest + report placeholders, then
explainability-into-image (phase 7) + report polish + reproducibility bundle.

## 2026-08-24 18:45 — serve_gte BAKED into the image: in-container 7.1s -> 4.3s

predict.py takes SERVE_SCRIPT (baked ENV, default serve_gte.py, serve_hf fallback); lean_gte+serve_gte
copied into src; image rebuilt (5c32fd4d). Jury-flag run: light imports 0.008s, data path 1.97s,
model ready 2.12s (fully overlapped), infer 3,477 rows/s, in-container total 4.29s/7.5k, wall 14s;
public argmax F1 identical to the 12th digit (0.846869) — the port is byte-exact in-container too.
Official-clock projection @90k: ~10s container + ~31s in-container ≈ 41s → **~2,200 SPS at 0.8553**.
Remaining fixed cost = docker/nvidia-hook overhead ~9.7s (structural, same for all teams).

## 2026-08-24 18:20 — official-clock evening: jury runs, fast image, container-overhead discovery

J (old image, jury flags): in-container 8.1s/7.5k, wall 17s. K: bare GPU benches of the rewritten path —
champion serve_hf 47.7s@90k (1,885 SPS; serial-read A/B 52.7s => parallel data path worth ~5s),
BERTurk lean 32.3s (2,788). Image rebuilt with fast scripts + champion (9.42GB, build 1s cached);
SECOND jury run: imports 0.007s, in-container total 7.088s/7.5k (infer 3,519 rows/s),
**public macro-F1 from the shipped image: 0.8469 with the argmax decision.json** (cuts pending, ~+0.01).
**DISCOVERY: wall 17s vs in-container 7.1s => ~9.9s docker/container/nvidia-hook overhead INSIDE the
official runtime window** (both images, both runs). At 90k that is ~18% of runtime. K3 queued: same image
twice back-to-back to split cold vs warm overhead. Real official-clock projection @90k with fast path:
~10s container + ~45s in-container ≈ 55s → ~1,650 SPS champion / ~40s → ~2,250 BERTurk.

## 2026-08-24 18:15 — P-chain: recipe is robust, training budget is dry -> FREEZE CANDIDATE SET

p1 seed-43 replicate of the freeze recipe: 0.8514 / scalar 0.8562 / AUC(S|I) 0.9058 vs seed-42
0.8553 / 0.8543 / 0.9037 — seed spread ~0.004: the recipe is solid, champion honest value 0.853±0.004.
p2 continued-FT (+1ep @ 5e-6 from ckpt_r1_oldstore): 0.8541 — no gain; the accuracy squeeze is exhausted.
DECISION: freeze candidate = ckpt_r1_oldstore (seed 42, DEV 0.8553). Remaining accuracy lever is only the
decision rule at freeze (cuts on all 7.5k public into decision.json — the pinned FREEZE-CRITICAL item).
Evening continues: J jury-timed run (old image) -> K fast-path benches + rebuild + jury rerun -> K2 serve_gte bench.

## 2026-08-24 16:02 — modernbert-tr-reranker: great per-param accuracy, disqualifying speed

149M Turkish reranker (distilled from Qwen3-Reranker-8B), 3ep on the R1 data: DEV 0.8518 two-boundary /
AUC(S|I) 0.9043 — ties the 306M champion's S/I AUC at half the params. BUT measured 663 SPS (135.6s @90k):
22 deep layers + an HF ModernBERT path that is ~4x slower than the champion on our serving stack.
OUT for shipping (win math needs >=3,000-class SPS); noted in the report as a falsification with numbers.
Champion unchanged: TY-ecomm r1_oldstore 0.8553 @ ~1,758 baseline SPS (fast-path benches tonight).

## 2026-08-24 16:15 — explainability app DONE (e20) + a freeze-critical catch

e20_explain_ty (agent): champion-adapted explainability — IG with ADAPTIVE quadrature (uniform IG
provably fails on this checkpoint: plateau-cliff integrand, 90% completeness gaps at 8-32 steps; fixed
via two-pass adaptive budget, gaps 0.2-3.2%), product-side leave-one-segment-out counterfactuals,
decision number-line, Turkish summary; preprocessing imported from the sources of truth and asserted
byte-identical to serving; stdlib HTTP server (no fastapi in venv); 34/34 smoke PASS, warm explain
2.3-5.2s CPU. Docker phase-7 needs: app + normalize/e18 files + pandas pin + vendored ngrok.
**FREEZE-CRITICAL:** ckpt decision.json is `{"head":"softmax3"}` (argmax ~0.832) — the +0.02 scalar+cuts
rule is NOT in the artifact. At freeze: fit cuts on all 7,500 public rows and write
`{"head":"ordinal","cuts":[c1,c2]}` into the shipped model dir. Tonight's jury runs time correctly but
score with argmax — expect ~0.83 from evaluate.py there; not a regression.

## 2026-08-24 15:45 — GTE no-transformers port: BIT-EXACT

e12 scripts/lean_gte.py + serve_gte.py + verify_parity_gte.py (agent): torch+safetensors-only forward of
the champion arch (RoPE half-rotation w/ fixed-NTK base correction, fused qkv, gated MLP up|gate split,
tanh-pooler head). Parity vs transformers: max|d|=0.0, argmax 1.0, full-public CSVs byte-identical
(sha match). Startup: model_ready 6.37->3.44s, first_batch 9.90->7.01s (-2.9s); import 2.7s->0.1s.
CPU rows/s slightly lower (SDPA mask kernel; diagnostic only) — GPU bench queued as K2 behind K.
Champion combined projection with all speed work: ~2,600-2,800 SPS at 0.8553 — inside the WIN_SCENARIOS
target band pending measurement.

## 2026-08-24 15:25 — report rewritten; small reconciliations

docs/FINAL_REPORT_DRAFT.md fully rewritten to the current solution (old 12B draft kept as .bak):
14 organizer headings preserved, 21 explicit FREEZE placeholders + checklist, 15-item falsification
section. Reconciliations recorded for freeze: (a) judge spend ≈ $105 total ($45 pre-08-22 + $60.41
measured via OpenRouter usage 130.24->190.65); (b) the adversarial organizer-baseline reads 0.738-0.751
across runs — same protocol, sampling noise of the baseline itself; cite as ~0.74-0.75 with run labels.

## 2026-08-24 15:20 — serving speed rewrite (agent, parity-gated)

e12 lean_infer.py + serve_hf.py rewritten (originals .bak): process-parallel CSV read+normalize
(fork workers, no torch in workers, /dev/shm-safe), lazy imports (serve_hf time-to-work 5.9s->0.16s),
model build on background thread, direct auto_map class load (-1.2s). Data path @90k ~17.9s->~11.9s.
Parity: full-public datapath byte-identical (ids+tokens); graph parity 4.8e-06; e2e CSVs byte-identical
across all modes. Projected: champion ~2,400-2,600 SPS, BERTurk lean ~2,800-3,000. GPU A/Bs queued (K)
behind tonight's chains: bench -> docker rebuild with fast scripts -> second jury-timed run.
Evening GPU pipeline: m2 reranker retry -> P(p1 seed43, p2 continued-FT) -> J(jury run, old image)
-> K(benches + rebuild + jury rerun). Agents in flight: GTE lean port (no-transformers champion path),
explainability app adaptation (e20, champion).

## 2026-08-24 14:45 — backup live; depth pruning rejected; truncation sweep

Kaggle backup CONFIRMED: private dataset celiker/trendyol-stage2-backup (champion ckpt 1.2GB, all label
parquets, label store, scripts, docs — Kaggle auto-extracted the tarball; files listing verified).
L8 (8-layer): 0.8393/0.8481 @ 1,938 SPS vs 12L 0.8553/0.8543 @ 1,758 — +10% speed for -0.6..-1.6 F1
points; layers own only ~1/3 of runtime on this path -> DEPTH PRUNING REJECTED, L6 cancelled by user.

## 2026-08-24 afternoon — Docker deliverable built + win-scenario math

**docs/WIN_SCENARIOS.md** written (from stage-1 35.4 vs leader 40): we beat any slow-LLM leader with
what we have; the threat is a fast-tiny-model team via the speed MaxNorm denominator (+0.01 F1 ~ +0.35
pts, +500 SPS vs fast leader ~ +2 pts). TARGET: >=0.85 F1 at >=3,000 SPS; subjective 30 pts are 1/3 of
the remaining score.
**e19_docker_ty/** (agent-built, CPU-only): `armaterimi_solution` image 9.42 GB, base pytorch 2.10-cu128
digest-pinned, pips asserted in-build, frozen organizer scripts sha-verified, checkpoint swap via
MODEL_SRC + prepare_context.sh, CPU smoke passed under jury flags (read-only rootfs, uid 1000:1001,
offline trust_remote_code, header row_id,real_label correct), secrets layer-grep clean.
RISKS for the GPU dress rehearsal (deferred to freeze): CUDA 12.8 userspace under judge's 13.0 driver
unproven until first GPU run (fallback: rebuild on nvidia/cuda 13.0.2 base); explainability stub fails
closed but is NOT the phase-7 UI; test = `cd e19_docker_ty && bash test_like_jury.sh` (refuses if GPU busy).
GPU pipeline queued: L8 -> L6 -> M(maxlen sweep + modernbert-tr-reranker) -> N(Qwen3-1.7B sandwich, overnight).
Backup tarball 1.1 GB at /home/jupyter/backup_stage2_0824.tar.gz — Kaggle upload blocked on user kaggle.json.

## 2026-08-24 13:31 — R1: the pre-v5 label store pays

TY-ecomm 3ep on champion data + old 136,489 judged pairs (813,269 rows total): **DEV 0.8553**
two-boundary / **0.8543 scalar** / **AUC(S|I) 0.9037** — best on all three readouts (champion c2:
0.8529 / 0.8426 / 0.8964). Two-boundary delta is inside seed noise but scalar +0.012 and S/I AUC
+0.007 agree in direction; the old hard-band data does NOT poison the model at this strength
(the 2026-08-21 D1b fear is resolved — that test predated v2 text, extras, and judged v5 data).
Data recipe for the freeze: organizer + allI + old-I + contrast + extras + old-S/R = build_oldstore.
Seed noise calibration (q2, seed 43): 0.8499 vs 0.8529 — single-run deltas <0.015 are weather.

## 2026-08-24 midday — decision-rule study (user challenge: "cuts might overfit; we can't see the 90k")

Zero-parameter alternatives measured on the champion's existing prob files (no new HOLDOUT read):
argmax 0.8322 | argmax/train-priors 0.8354 | +temperature 0.8359 — the model over-calls semi (44% vs 33%)
and no global recalibration fixes it; value-cuts (+0.021, transfer optimism measured 0.003) stay justified.
**RANK-THIRDS rule** (private set is DOCUMENTED 30k/30k/30k: label by scalar quantiles of the test batch
itself, no fitted values): DEV 0.8467 vs value-cuts 0.8529, but on HOLDOUT **0.8461 vs 0.8385** — beats the
fitted cuts on unseen data because it is immune to scalar location/scale drift. Mix sensitivity: 31/34.5
mix costs 0.001; 35/32 costs 0.011. FREEZE PLAN (superseded by user constraint 2026-08-24: the 90k is NOT processed as one batch — NO batch-level postprocessing, every label from the row alone): serve per-row scalar + FIXED cuts fitted on all 7,500 public rows. Rank-thirds stays as an offline diagnostic only. Aux S/I loss (q1): flat (0.8444, AUC S|I unchanged) -> q3
auto-skipped. Queue: q2 seed-43 -> R1 old-store (136,489 pre-v5 judged pairs, all classes) -> L8/L6 depth
truncations (backbones/tyecomm_{8,6}L, bottom-K of pretrained, verified) each with an SPS bench.

## 2026-08-24 09:15 UTC — C-chain + first real speed measurements

**C-chain (champion data 678,549 rows, v2 text, one training at a time):**
  * c1 BERTurk-128k 3ep -> DEV **0.8205** (scalar 0.8233). Extras lift BERTurk +0.018 too -> the
    extras gain is a property of the DATA, not of one backbone.
  * c2 TY-ecomm-embed 3ep -> DEV **0.8529** (scalar 0.8426). NEW CHAMPION, crosses 0.85 on DEV.
    (2ep was 0.8441; +0.009 is under the noise floor but 3ep is the best point measured.)

**Serving fixes before benchmarking:** `e12_lean_infer/scripts/lean_infer.py` still inlined the OLD
18-key drop list -> would have been silent train/serve skew. Now carries the 60-key v2 list + the
boilerplate-value filter (compat keys keep "belirtilmemiş"), verified **byte-identical to the
training transform on all 7,500 public rows**; HF-vs-lean forward parity re-confirmed (max|dlogit|
1.05e-5, argmax agreement 1.0). New `scripts/serve_hf.py` = same read/normalise/tokenise/bucket/
decide/write pipeline imported FROM lean_infer, only the model forward differs (for the GTE "New"
architecture, which the hand-rolled BERT loop cannot run). `bench_e2e.py` takes BENCH_SCRIPT=.

**Measured end-to-end (cold subprocess -> finished CSV, projected to 90k):**
| model | 90k wall | SPS | startup | pure inference | DEV |
| BERTurk (lean) | 36.6s | **2,459** | 4.7s | 6,100 rows/s | 0.8205 |
| TY-ecomm (HF)  | 51.2s | **1,758** | 10.7s | 3,600 rows/s | 0.8529 |
Speed gap is **1.4x, not 3x**. FACT worth acting on: at 90k, BERTurk spends 4.7s startup + ~17s
read/normalise/tokenise + 14.7s inference — **~46% of runtime is the single-threaded Python data
path, not the model**. That is the biggest and most model-independent speed lever available.

**Cascade (BERTurk decides confident rows, TY-ecomm the uncertain band) — REJECTED by measurement:**
0% routed 0.827 @0.69x | 50% routed 0.847 @0.93x | 100% 0.850 @1.00x. Running BERTurk over every
row eats the saving; no sweet spot. The model choice is binary.

## 2026-08-24 morning — night queue results; EXTRAS ADOPTED; duplicate-launch incident

**INCIDENT.** The night queue ran TWICE in parallel (bash PIDs started 21:49 and 21:51): the tool call the
user interrupted had already fired `setsid nohup run_night.sh` before the interrupt landed, then the
rewritten script was launched. Bash re-reads a script by byte offset, so the first chain picked up the new
rung list mid-file — both chains ran the SAME tags into the SAME output dirs on one GPU. Effects: ~2x
slower, same-tag ckpt/eval overwrites (8 result lines, 5 surviving JSONs), and n2_vf_sandwich died of CUDA
OOM because the other chain held 17.5 GB. Collision-tainted artifacts moved to
`outputs_frozen_v2/tainted_dup_run/` (e5large, mdeberta, berturk_4ep, tyecomm_4ep — all <=0.819, none a
champion, not re-run). Rule going forward: one training at a time, guarded by a `pgrep` check in the chain
script; never `pkill -f` a pattern that appears in your own command line (self-killed the shell twice).

**RESULT — judge-semi/relevant extras are worth ~+0.02 and are now part of the standard recipe.**
Two clean, unique-tag runs (TY-ecomm-embed, v2 text, hard labels):
  * n1_g2_extras: all-judged-I 152.4k + contrast + 29.5k extras = 678,549 rows, 2 ep -> **DEV 0.8441**
  * n1_best_extras: final 143.9k + contrast + extras = 672,057 rows, 3 ep -> DEV 0.8403
  vs 0.8217 for the same backbone/epochs without extras.
**HOLDOUT read #2** (06:50 UTC) on n1_g2_extras: **0.8385 two-boundary / 0.8438 scalar** (read #1 was
0.8235 on the 0.8217 model). DEV-HOLDOUT gap -0.006, inside the 0.012 floor -> DEV still honest.
NOTE for the freeze: on HOLDOUT the plain scalar beat the two-boundary rule (0.8438 vs 0.8385) while on DEV
the order reversed — fit the final decision rule on DEV + the full public set, not DEV alone.
Other backbones (contended, suspect but all well below): mdeberta ~0.812-0.819, e5-large ~0.806-0.810,
BERTurk 4ep ~0.804-0.808, TY-ecomm 4ep ~0.808 (4 epochs overfits; 3 > 4).
**Standard training recipe from here:** organizer 250k R/S + `labels_allI.parquet` (all judged-I) +
`labels_contrast_plus_extras.parquet` (contrast + judged S/R extras), ATTR_CLEAN=v2 -> 678,549 rows
(`outputs_frozen_v2/build_g2_extras/`).

## 2026-08-23 21:42 UTC — HOLDOUT read #1 (milestone; daily-read protocol starts)

Champion by DEV: f6_tyecomm_v2_3ep (TY-ecomm-embed, v2 text, 3 ep; DEV 0.8217 / scalar 0.825).
**HOLDOUT 0.8235 two-boundary / 0.8280 scalar** (AUC S|I 0.9028). HOLDOUT >= DEV -> no DEV overfit.
Protocol: one read/day, DEV-chosen champion only, HOLDOUT never selects. G-ladder rung 1 before the
user stopped the queue: all-judged-I (152.4k) 0.8190 ~= shaped 143.9k 0.8149 (within noise; shaped set kept).

## 2026-08-23 afternoon — model phase on the frozen set; attribute cleaning v2 adopted

All runs: hard labels, organizer 250k/250k + e17 irrelevant_final 143.9k + contrast 886 (643,163 rows), 2 ep, DEV
two-boundary OOF. BERTurk-128k **0.7920** (was 0.7766 on the unjudged v4 set: +0.015). TY-ecomm-embed
(306M, cross-encoder head) **0.7994**. Vision-Flash text tower (Qwen3-0.6B, full FT, LR 1e-5) 0.7660 — 4x slower,
weakest S/I boundary; second-round candidate only (sandwich input, v2 text, LoRA).
**e18_attr_clean v2** (analysis + implementation by two agents): drop 60 attribute keys (18 leak keys + 42 zero
query-landing keys: care/washing, warranty, origin, sustainability, packaging, legal, manufacturer identity...) and
boilerplate values (belirtilmemiş / mevcut değil / care sentences; compatibility keys keep belirtilmemiş; yok/var
kept). -34% attribute chars, -15% BERTurk tokens, -18% Qwen tokens (truncation@256 7.7%->1.2% on public);
also unifies the ""/"unknown" empty-attribute snapshot tell. Paired TY-ecomm run: **0.8149** vs 0.7994 (+0.0155,
R-vs-S AUC 0.872->0.909). ADOPTED: ATTR_CLEAN=v2 is the text transform for training and inference from here
(builder/eval flag; lean_infer must apply attr_clean.apply_v2 identically).

## 2026-08-23 — e17_mine_v5 FROZEN (mining + labelling done)

**Judge spend today:** canary $7.36 (21,878 rows) + top-up $3.71 (12,000) + remainder ≈ $45 (144,910) — all
rows labelled except 24 unparseable. Blind 300-row audit (3 independent readers, rubric only): precision
**0.987**, 0 flips to relevant, 98.3% unanimous; every recipe ≥ 0.955.
**Judge verdicts on 182,000 negatives:** irrelevant 152,425 | semi 17,590 | relevant 12,778 (keep by
recipe: M1 .986, M2 .995, M6 .972, M7 .998, M4 .998, M5 .997, M3b .935, M3a .892, M8 .623, M10 .662).
**Frozen files (`e17_mine_v5/outputs/`, FREEZE.manifest.json):** `irrelevant_final.parquet` **145,848**
(post-judge constrained re-selection; fashion 33%, M10 31% / M1 22% / M6 19% / M2 15% / M8 8%),
`irrelevant_reserve.parquet` 6,577 (judged-I not selected — dose ablation), `judged_extras_quarantine.parquet`
29,551 (judge-S/R — ablation arm only), `contrast_final.parquet` 902 (device-model matches, in by default).
**Post-judge gates (enforced):** adversarial 0.7185 vs organizer baseline 0.748+0.02; Asure hardness
frac>0 0.186 (DEV-I 0.234, band .18–.28), median −1.5 vs −1.0; hygiene clean. Label store now
~323k judged pairs. Pre-judge pilot (rule labels): DEV 0.752, real-I recall 0.72.
Lessons: detach long jobs with setsid (session close killed the scorer); never pgrep -f a pattern
that appears in your own command line; the 0.65 adversarial bar was below the train→test floor.

## 2026-08-22 (evening session) — mining+labelling rebuilt as e17_mine_v5

**Finding:** e16 v4's 61.8k "clean" set was never judged — `clean_labels_ruleconf.parquet` stamps
`irrelevant` by rule (0 judge calls, ~11% expected contamination, M8 ~40%); e1_clean/e2_clean
student numbers (0.777/0.767) were trained on assumed labels. Not a frozen set.

**Decisions (user):** hard labels only, no distillation; Asure-12B = reference bar (e11 S/I LoRA
finished: DEV scalar OOF 0.8724), not a deliverable; judge everything mined (~$61 of $69.76 left),
keep judge-S/R rows (quarantined file, ablation arm); target >=150k judged irrelevants.

**e17_mine_v5 design (README.md):** ranked candidate lists per query; GLOBAL ITEM CAP (3) enforced
at the generator's emit point (post-hoc cap had discarded 65k rows); per-query cap
clip(1.25*n_RS, 8, 30); department profile = DEV-I roots, scarcest root filled first; normalized
item identity for dedup/reuse/store (5,556 pool rows collapse after normalize.py); device-model
contrast rows for templated queries; gates ENFORCED and sha-bound; Asure hardness scores named
by selection sha; judge in two disjoint slices with a stratified 20k canary + blind 300-row audit
before the remainder; post-judge constrained re-selection; FREEZE manifest.

**Gate readings so far (DEV-I 1,250):** rule-recipe-only selection (M2 56%): adversarial AUC
0.7977, zero_all 0.552 vs 0.191, brand_in_query 0.168 vs 0.401 -> FAIL. Added M10
"retrieval near-miss" (tf-idf top-80 minus own R/S rows minus forbidden leaves) to reproduce the
organizer's retrieve-then-judge mechanism; re-mining.

## 2026-08-21 (takeover session)

**Strategy reset (scoring math verified from ORGANIZER_REQUIREMENTS §2):** Stage-1 banked us
35.4/40 vs the leader's 40 → we must gain >4.6 pts over the leader across the remaining 60.
Speed axis (MaxNorm×10, no rank-mult) realistically swings ~9.5 pts; accuracy axis ~2.7.
⇒ Primary submission = 12B teacher stack distilled into a fast Turkish cross-encoder
(BERTurk-class, target ≥0.85 macro-F1 at 1,500-5,000 effective SPS); 12B two-boundary rule
(0.8339) kept as the floor-clearing fallback. Full effort on report + explainability +
presentation (30 absolute pts).

**Mined-irrelevant repair (e08_repair):**
- Hard-band labeling finished 07:16: 87,763 rows, $29.47, 80.7% judged irrelevant.
- Gate re-run on COMPLETE cache: adversarial AUC 0.9101 (partial-cache 0.9064 confirmed, worse).
- e08 match (judged-I only): AUC 0.9158→0.8280, ESS 7.5k — insufficient support. Diagnosis:
  the hard band (margin>0) is anti-correlated with the organizer profile (their I median
  margin −1.0, 63% zero-title-overlap, 62% gender-known).
- Pilot on matched-30k: DEV-I accuracy 0.374→0.4664, macro-F1 0.6477→0.6773 — transfer tracks
  the gate. Mechanism confirmed.
- **e08b match over ALL 684k candidates (+ margin as covariate): AUC 0.6248, ESS 91,810.**
  50k selection frozen (pool_matched_selection.parquet); origin mix A/B = 49.9/49.9 (no
  snapshot leak). 40,599 unjudged rows sent to gemini-3.7-flash (~$15, running).
- Queued $81 label run of label_selection_v1 (217k) CANCELLED — superseded by the matched 50k.

**In flight:** matched-selection labeling → chain_after_labels.sh (finalize → gate on judged-I
→ 4-backbone bake-off: BERTurk-128k / BERTurk-cased / tybert / XLM-R-base, pre-registered
winner-by->0.012 rule). vLLM+LoRA smoke test (E0 serving mechanism; found df1164d3 snapshot is
config-only — d3a9651e holds the weights, same config blob). Two-boundary cut freeze on DEV
(e09_fallback). E0 docker scaffold (agent). E3 S/I-LoRA trainer build (agent). 300-row hand
audit of matched judged-I vs rubric (3 agents; B2 E2 kill rule: precision ≥0.85).

**Plan:** today: bake-off + E1 verbalizer probe + E0b throughput measure + E3 data prep;
tonight: E3 S/I LoRA (12-18h); tomorrow: teacher scalars at scale from E3 model → student v3
distillation; day 6: student iteration + docker + single HOLDOUT read; day 7: freeze + zip.

### Session results, 08:00–09:15

- **Matched-selection labeling done**: 50,000 rows, 95.8% judged irrelevant → 47,889 training-I
  ($14.7 new spend; total judge spend ≈ $45). Gate on the judged set: **AUC 0.6264** (from
  0.9101). 300-row independent hand audit (3 Claude agents vs rubric): **88.3% precision**,
  0 flips to relevant → E2 kill gate PASSED.
- **Backbone bake-off** (125k balanced rows, 2 ep, hard labels): BERTurk-128k **0.7358** >
  tybert 0.7265 ≈ BERTurk-cased 0.7249 (within noise) ≫ XLM-R-base 0.6885. DEV-I transfer
  0.6048 (vs 0.4664 on match-30k, 0.3744 hard-band) — transfer tracks the gate. **Backbone =
  dbmdz/bert-base-turkish-128k-cased.**
- **E1 verbalizer probe** (pre-registered 4 variants, DEV, bf16, normalized text): binary
  sandwich AUC(S|I) 0.9114 / OOF 0.7989; 3-level digits 0.9172/0.8034; +constraint text
  0.9150/**0.8120**; query-last worse. No variant clears +0.010 AUC(S|I) → **prompt branch
  dropped per gate**. (Side-finding: normalized text+bf16 beats NF4+clean_attrs base scalar.)
- **vLLM+LoRA serving validated**: pearson 0.998 vs e01 scalars, identical decisions at cuts.
  Snapshot gotcha: df1164d3 is config-only; d3a9651e holds the weights.
- **E0 fallback**: container dry run measured LoRA pass 49.7 r/s, base 58.7 r/s → 90k ≈ 57 min.
  Cuts frozen ON THE CONTAINER PATH (DEV bootstrap-avg): CUT_R=2.3260, CUT_I=−2.4574,
  **DEV OOF 0.8373** (better than the e01-scalar 0.8328). Docker scaffold complete + image
  build in progress. logprobs param fix applied (must equal len(logprob_token_ids)).
- **E3 S/I LoRA LAUNCHED 08:55** (300k rows: 126k R/126k S/47.6k judged-I, soft targets,
  BUDGET=9000, r32 all-7-linears, verbalizer-init head). Smoke validated: mem 30.3/39.5 GiB;
  10-step smoke checkpoint already scores DEV **AUC(S|I) 0.9381** (> the 0.930 ship gate),
  OOF scalar 0.8162. Pre-registered primary readout: OOF two-boundary macro-F1 vs 0.8328
  reference, gate > 0.8448 AND AUC(S|I) ≥ 0.930. ETA ~12-15h.
- **E0a lean inference built** (no-transformers BERT loop, parity 2.7e-06, import 1.7 s) and
  **E13 explainability app built** (IG + attention + number-line + counterfactuals, verified
  endpoints on CPU).
- **E14 written** (teacher-at-scale trick: merge LoRA + graft head into unused-token rows of
  the tied lm_head → vLLM serves teacher probs at ~55 r/s; bias applied offline). Run tomorrow
  on the E3 checkpoint; validation = DEV probs vs eval_dev's HF path.
- In progress (agents): E0 docker image build; report skeleton (14 headings, TR) +
  development-dump plan; e15 student distillation trainer (S1/S2/S3 ladder).

### Session results, 09:15–09:45

- **E0 image BUILT + verified**: 36.2 GB, cold build 10m16s, base digest-pinned
  (rebuild byte-identical), secrets provably absent, cuts/frozen-scripts verified in-image.
  Remaining: containerized GPU timed run; explainability serve integration.
- **Report draft written** (docs/FINAL_REPORT_DRAFT.md, 14 verbatim headings, TR) +
  docs/DEVELOPMENT_DUMP_PLAN.md. Gap list recorded there; hand audit materialized as
  e08_repair/outputs/hand_audit_300.{parquet,json} (265/300 = 0.8833).
- **e15 student trainer built + CPU-verified**; S1 set assembled: 545,944 rows
  (organizer 498,374 + judged-I 47,570, public-overlap dropped). Row identity for teacher
  attachment = full 7-field text tuple. Ladder S1→S2→S3 pre-registered (climb on >0.012).
- **E3 throughput confirmed**: 1,365 tok/s, loss 0.594 falling at step 100/3767, ETA 16.8h
  → done ~02:00. **Overnight chain armed** (e14/scripts/chain_overnight.sh, monitored):
  on completion → eval_dev → merge_and_graft (vision tower kept for vLLM) → DEV agreement
  gate (vLLM-grafted vs HF, abort if max|Δp|≥0.05) → teacher probs over S1 546k + candidates
  630k. GPU busy until ~09:00 tomorrow; student ladder runs after.

## 2026-08-26 — serving speed, the real jury clock, and the frontier

**SPS is WALL CLOCK** (`run_solution.sh` start -> submission written). Everything in the window
counts. Measured decomposition of the champion's 34.0 s run:
`4.3 s docker --gpus all (NVIDIA hook) | 0.02 s predict boot | 27.7 s server | ~2.0 s tail`.
The hook alone is 3.6 s (`docker run --entrypoint /bin/true`: 0.69 s without `--gpus`, 4.33 s with)
and is not ours to fix. **Physical ceiling for ANY GPU container: 90000/6.3 s ~ 14,000 SPS** — the
speed denominator cannot run away from us, which bounds the downside of the fast play.

**Serving fixes, all output-preserving (0/90,000 labels changed, verified per model):**
- `_is_boiler` re-derived all ten folded constants on every call: 8.6M `_fold()` calls per 90k file
  where ~1.4M are real work. Hoisted -> serial normalise **11.15 s -> 3.22 s**; pooled data path
  5.43 -> 3.98 s. Byte-identical on public 7.5k AND synth 90k.
- Pool IPC narrowed int64/int64 -> int32/int8 (~151 MB -> ~85 MB per run).
- **Rayon oversubscription**: each pool worker's Rust tokenizer opens its OWN rayon pool, so 10
  workers = ~120 threads on 12 cores and the model-load thread starved (`model_ready` 5.93 s vs a
  2.0 s uncontended cost). `OMP_NUM_THREADS=1 RAYON_NUM_THREADS=1 LEAN_WORKERS=8` -> +5% to +17%
  depending on model, bit-exact. Baked into the Dockerfile.
- `BATCH_ROWS` was still **128/32768** in the image, smaller than anything benched -> 256/65536.
  (512/131072 is 1.2% faster but moves 22/90,000 labels on the cut band; not taken.)
- cuInit "~50 ms" probe measured **1.394 s** of serial dead time -> cheap device-node gate up
  front, real probe joined after serving: **1.394 s -> 0.019 s**.
- Dropped a redundant 90k input re-read in predict.py; validation no longer re-parses the output.

**INT8: measured and REJECTED.** `torch._int_mm` vs bf16 at our shapes on A100: qkv 1.15x,
attn-out 1.20x, mlp-up 1.16x, mlp-down 1.62x -> **1.29x on a layer's GEMMs**, not 2x (skinny K=768,
launch/memory bound). Best case 20.5 -> 16.9 s before quantisation overhead; ~+5-7% wall for a
hand-rolled int8 lean path plus F1 risk. Not worth it.

**Three shipped-pipeline bugs found while benching non-TY candidates:**
1. `prepare_context.sh`, `build_docker.sh` and the Dockerfile model-layer check ALL required
   `modeling.py`+`configuration.py`, which exist only for remote-code GTE checkpoints -> the
   pipeline could only ever build TY-ecomm. Now conditional on `auto_map`.
2. `predict.py` loaded the serve script under the hardcoded module name `"serve_hf"`, so
   multiprocessing died with `PicklingError: attribute lookup _norm_tok_chunk on serve_hf failed`
   — every plain-BERT candidate crashed when the data-path pool started. Registered under its real
   name now.
3. `SERVE_SCRIPT=serve_gte.py` is baked; GTE cannot load BERT weights. predict.py now picks the
   server from the checkpoint's own `config.json`.

**IN-CONTAINER frontier (jury clock, all fixes in, 2 reps each):**
| model | DEV OOF (2-boundary) | seeds | in-container SPS |
|---|---|---|---|
| TY-ecomm 12L | 0.8587 | 1 | 2,652 |
| TY-ecomm 8L  | 0.8474 (0.8530/0.8436/0.8455, sd 0.0041) | 3 | 3,351 |
| BERTurk 12L  | 0.8372 (0.8373/0.8370, sd 0.0003) | 2 | 4,401 |
| tiny2 8L/512 6ep | 0.8257 | 1 | 5,625 |
| tiny2 8L/512 3ep | 0.8205 (0.8218/0.8191) | 2 | 5,625 |
Old recorded in-container numbers (2,967 / 3,340 / 5,073) were on the UNTUNED image and do not
reproduce. Dead: t3a grown-10L 0.8127, t3b base-uncased 0.8302, TY-6L 0.7991, tiny1/3/4.

**Epoch schedule**: 3 epochs was tuned on the 306M champion. tiny2 at 6 epochs = 0.8257,
**+0.0052** over its 3-epoch mean (4x the seed sd) at identical speed. BERTurk at 6 epochs is
running — until it lands the tiny2-vs-BERTurk comparison is CONFOUNDED (different schedules).

**MAXLEN=128 with cuts REFIT at 128** (settles the old "+7.7%" claim, which used 256-fitted cuts):
TY-12L -0.0066/+7.1% | TY-10L -0.0142/+7.8% | BERTurk -0.0046/+2.5% | tiny2 +0.0008/+2.2% |
TY-8L -0.0036/+6.3%. Only TY-8L is arguably worth it; not taken.

## 2026-08-27 — the çocuk leak, the pick, and the freeze

**LEAK FOUND (biggest single win of the project).** A subagent sweep for high-precision regex
rules turned up, incidentally, that the organizer's `train.csv` makes `gender == 'çocuk'` a
PERFECT predictor of semi-relevant: **16,634 rows, 0 relevant**. Two-snapshot artifact — R rows
and S rows come from different catalogue exports. At test those rows are 44% irrelevant / 29%
relevant / 27% semi, and the model scored **69.0% there vs 88.1% elsewhere on 9.7% of rows**.
Same disease as `unisex` (22,516 rel / 2 semi), which T4 already neutralised — **we had fixed one
value of that field and never audited the rest**. Full sweep of every gender/age_group value
found no third case (age_group max shift 0.10 vs gender-çocuk 0.51).

| | no fix | inference patch | RETRAIN on fixed data |
|---|---|---|---|
| TY-ecomm 12L | 0.8587 | +0.0069 | **0.8643** |
| TY-ecomm 8L | 0.8474 | — | **0.8581** |
| BERTurk 12L 6ep | 0.8428 | +0.0120 | **0.8546** |
| turkish-medium 8L | 0.8221 | — | **0.8337** |

Bigger than every serving optimisation combined, and it is a DATA fix.
`brand == 'genel markalar'` (11,243 rows, 0 relevant) has the same signature but was
**deliberately not fixed**: brand has no neutral sink (0 empty, 0 'unknown' across 56,085
brands), so any rewrite feeds the model an out-of-distribution token, and the measured effect
(8 discordant DEV predictions) was inside the noise floor.

**REGEX-SKIP LINE: DEAD.** Four independent angles. `(query,category)` memo scored 41.7% coverage
at 0.9912 on train and fires on **0.00%** of real test rows — train and test are FULLY
query-disjoint (0 of 3,750 DEV queries appear in train's 31,134). Lexical overlap tops out at
0.97% coverage at 0.98 precision; numerics cap at 4.7% of rows and lose money at usable
precision. Structural reason: **semi-relevant occupies both tails** — 32% of zero-overlap rows
and 42% of perfect-containment rows are semi.

**TENSORRT: MEASURED AND REJECTED.** Built a real fp16 engine for the BERTurk pick (ONNX export
+ TRT 11 strongly-typed): **0.99x**. fp16 and bf16 run at identical tensor-core rates on A100,
and our lean forward already has SDPA, no framework overhead and 0.4% padding waste, so there was
nothing for TRT to recover. INT8 ceiling bounded from measured raw `_int_mm` at our shapes
(1.32x): **+8..+10% at absolute best**, crediting zero quantisation cost — our own fused attempt
landed at 0.68x because the quantise/rescale traffic exceeded the GEMM.
NOTE: TRT 11 removed `BuilderFlag.FP16`; precision comes from the ONNX graph (strongly typed).
And `torch._int_mm` needs a COLUMN-MAJOR rhs — row-major measures 0.29x and looks like a refutation.

**PICK: BERTurk-12L, 6 epochs, gender-fixed** (`ckpt_gf2_berturk_6ep`). Best worst-case across the
leader archetypes; 0.0097 below the best F1 at 66% more throughput.

**THREE BUGS CAUGHT AT FREEZE, all would have shipped:**
1. `decision.json` was the 20-byte `{"head":"softmax3"}` stub — every retrain writes it and the
   cut fitter only ever PRINTED its numbers. Shipping it throws away the whole ordinal-cut gain.
2. Setting head to `"ordinal"` silently broke the explainability UI: `lean_infer` reads
   `_ordinal_scalar(logits)` = p_semi+2*p_rel, `model_adapter` read `lg[0]`. TY-ecomm shipped
   `ordinal_regime`, so that branch had never run. UI would have contradicted the CSV on ~99% of
   rows — a direct §6.4 violation.
3. IG baseline hardcoded `unk_id = 3` (XLM-R `<unk>`); on BERTurk id 3 is `[SEP]`, `[UNK]` is 1.
   Every attribution was measured against "separator everywhere".

**FROZEN NUMBERS** (from inside the submitted image): public macro-F1 **0.857943456149**
(organizer `evaluate.py`), DEV 0.8557 / **HOLDOUT 0.8602** — holdout ABOVE dev, i.e. the cuts are
not overfitted to the fitting half. In-container **4,386 SPS**. Reproducibility **100%,
byte-identical across three runs** (requirement: >99%).

## 2026-08-28 — final-day audit (Fable takeover) and ship

Full re-audit from COMPETITION_RULES.md up. Findings and fixes:
- **§8 VIOLATION CAUGHT**: the dev dump excluded all generated data. Rules require shipping the
  mined negatives / pseudo-labels themselves. Added: irrelevant_final (145,848), reserve, extras,
  contrast, judge label tables, mining manifest+gates, and the exact 813,269-row training table,
  plus a DATA_README. Verified 0 public-query overlap in every shipped file.
- **§3.4 disclosure**: README now names the LLM judge (gemini-3.7-flash via OpenRouter, dev-only)
  and the backbone with revision fea3225... and MIT license; rejected backbones listed.
- **Speed honesty**: the 17.8 s / ~5,030 SPS figure only reproduces after many consecutive runs
  (hot docker overlay). Fresh-state runs are ~20.0 s = ~4,500 SPS with identical stage timings.
  All docs now quote ~4,500 single-run with 5,030 as the warm footnote. Stage breakdown proven
  unchanged (read 4.8 / model_ready 5.2 / infer 9.5 / docker 3.95).
- Report: "(TASLAK)" removed from the title; all 14 §7 headings verified; docx regenerated.
- **Explainability e2e through the organizer's own script with a real ngrok tunnel**: URL live in
  <60 s, page/API 200 from the public internet, explain 0.9 s, token absent from image, truth
  overlay absent (dev-only flag), clean teardown (404 after container kill).
- **JUDGE-PATH test**: extracted the final zip to a clean dir, built the image FROM THE ZIP alone
  (4 s cached), ran the 90k: exit 0, 20.70 s, output byte-identical to every verified run this
  week (cbacea4b68a5b99b), public F1 0.857943456149 from the zip-built image.
- Model re-questioned with fresh eyes and CONFIRMED: BERTurk-128k-cased 6ep genderfix. Overnight
  n2 grid: 3 baseline seeds mean 0.8546 sd 0.0013; 9ep -4.4sd, LR2e-5 -2.4sd, uncased -2.7sd,
  typo aug -1.3sd, none better. Recipe is at a measured local optimum.
FINAL ZIP: /home/jupyter/armaterimi_submission.zip  374 MB  sha256 e499a4bc184a0188...

## 2026-08-28 — final solidification round (post-audit fixes) and RE-SHIP
- XAI BLOCKER fixed: _target_fn for head="ordinal" (the SHIPPED head) attributed logits[:,0]
  (the irrelevant-class logit) instead of the decision scalar — the ordinal_regime fix from
  08-26 had not covered the plain-ordinal branch. Now both heads attribute p_semi+2*p_rel;
  verified live (ig.target=decision_scalar), smoke ALL PASS, parity 300/300.
- Report: 13+5 corrections — the graded docx no longer describes the rejected GTE system
  anywhere (PUBLIC-F1 0.8639->0.857943456149, ordinal_regime->ordinal global cuts, 9.56GB->8.69GB
  image, 306M->184.3M weights, TY-era seed stats -> shipped-recipe 0.8546 sd 0.0013, HOLDOUT
  ledger relabeled + Okuma 4 added, judge spend ~105 USD, çocuk 8.9%/43-30-27, boru hattı x3
  removed, draft-note block and every orphan-producing FREEZE region cleaned).
- UI: 14 text/code fixes (WordPiece '##' pieces view, subtitle grammar, suffix hardcodes,
  sep-token filter in summaries, stale ckpt defaults in smoke/dev_server/parity).
- normalize.py T2/T3 docstring now states the true train-only discovery; e18 README status
  ADOPTED + 60 keys; make_submission ships e04_label/label.py (judge caller) per §8.
- README §9.3: weights sha256, CPU row, checkpoint/resume-behavior statement added.
- FINAL ZIP re-cut and judge-path re-verified from the zip alone: build OK, 90k exit 0,
  20.01 s / 4,497 SPS, output cbacea4b68a5b99b (unchanged all week), report 41.7 KB in zip.
  /home/jupyter/armaterimi_submission.zip  374 MB  sha256 d7c16196e2d62f52...

## 2026-08-28 — report: full end-to-end read and rewrite (by hand)
Read all ~780 lines. Beyond number fixes, the NARRATIVE still carried the TY-ecomm era in
several load-bearing places; all rewritten for the shipped BERTurk:
- Model Selection: one confused table (same model twice, mixed eras) -> Tur 2 (early, pre-fix,
  backbone ranking) + Tur 3 (final frontier, same data, in-container SPS) + uncased A/B note.
- Model Architecture: BERTurk was described as "çok dilli, e-ticaret verisiyle ön-eğitimli"
  (TY-ecomm's description) -> correct BERTurk facts (Turkish-only, 128k WordPiece, 184.3M,
  revision, license, vocab efficiency).
- Training Strategy: shipped recipe was stated as 3ep/BS32/LR2e-5 (TY) twice -> 6ep/BS64/LR3e-5;
  duplicate composition table removed; stale TY seed sentence removed; epoch optimum stated as
  backbone-specific (TY 3, BERTurk 6, 9 regresses).
- Validation: decision-rule stress test now carries the shipped model's numbers (argmax 0.8495 ->
  cuts 0.8579; no-public-access counterfactual 0.8501; split-robustness 0.850±0.014 root-disjoint).
- Inference: wall-clock breakdown now the BERTurk 20.0s run (was TY's 33.94s labeled "şampiyon");
  LEAN_WORKERS=10 with the nproc=1 rationale; capitalized "Boru hattı" removed.
- Explainability: section rewritten to answer the five §7 questions with the CURRENT panel set
  (decision-scalar IG w/ adaptive quadrature + [UNK] baseline + completeness, occlusion
  counterfactuals, faithfulness curve, method agreement, word-level attention, Turkish
  robustness strip, preprocessing diff, template summary, UI/container parity guard).
- What worked/didn't + Ablation: TY-as-"worked" reframed, 4 HOLDOUT readings, epoch item
  qualified, ablation E reversed-pruning wording, table B header truthful, new table F for the
  shipped model, recalibration semi rate 37%; historical HTML checklist block deleted.
docx regenerated (6,919 words), probes clean. Zip re-cut sha 8606b191...

## 2026-08-28 — report: final round on the RENDERED docx (grader's view)
Converted docx -> plain text and read all 7,499 words as a grader would. Fixed: one table row
broken by a literal '|' in "AUC(S|I)" (-> ∣), a "+ sıcaklık" line rendering as a stray bullet,
a code span wrapped across lines ("gender:    çocuk"), Bulgu-7 empty-attributes count corrected
to the verified 2,247 / 100% semi, cuts-vs-argmax figure reconciled (+0,0073 DEV OOF / +0,0084
public), Tur-3 header (turkish-medium also 6 ep), and decimal style unified to Turkish comma
outside code spans (231 numbers; version strings and the verbatim evaluate.py figure kept).
Rendered docx: 0 markup artifacts, 0 stray bullets, all probes OK.
FINAL ZIP: /home/jupyter/armaterimi_submission.zip  sha256 afebc4224c48f115...  (docx in zip ==
final docx, unzip -t clean). Docker context unchanged since the 08:07 freeze (hash cbacea4b...).

## 2026-08-28 — report switched to ENGLISH (docs/FINAL_REPORT_EN.md); zip re-cut
Why: user's viewer showed the Turkish docx as mojibake (docx bytes verified valid UTF-8 — a
viewer codepage issue, but a real risk class); rules/announcements never specify a report
language and the organizers' own technical docs are English (transcript 31:51). Rewritten from
the verified Turkish numbers, ~30% less content: no Asure S/I fine-tune anywhere (12B zero-shot
gate use kept), TY-ecomm stated as an embedding model (GTE) + our own 3-class head, Stage-2 only.
Rendered-docx audit: 14 H2, 10/10 tables, Title block, 0 markup/list artifacts; every number
cross-checked against the Turkish source (only new figures: INT8 1.32×/0.68×, TRT 0.99×).
FINAL ZIP: /home/jupyter/armaterimi_submission.zip  sha256 86fbaa5c23a34909...  (docx sha a445b151...,
unzip -t clean). Turkish docx kept as scratchpad/armaterimi_FINAL_REPORT_TR_backup.docx.
Docker context unchanged (90k hash cbacea4b... still valid).

## 2026-08-28 — report replaced by the team's GPT-edited version (4,000 words); audit-provenance FIX
Team rewrote the English report (recent_version_report.docx). Audit found ONE factual error that
had propagated from our own earlier drafts/README: the "blind 300-row audit, 3 independent
readers" was three Claude Code agents (LOG 08-23, e17 audit_votes_*.csv), NOT humans; the GPT
text had escalated it to "three human reviewers" (x2). Corrected everywhere: report (§1 table,
§3, §4, §9, §12), README §11 (Claude now disclosed under §3.4 as a development-only audit
reader), and the audit vote files + verdict now ship in generated_data/audit_300/ so the claim
is checkable. Other in-place edits (python-docx, formatting preserved): judge benchmark = 1,200
DEV rows vs organizer labels (not "rubric-labeled"); TY-ecomm = embedding model + our own head;
added no-public counterfactual 0.8501, wall-clock anatomy/14k ceiling, decision-scalar target +
UI parity, attention-supporting + brittleness limits, 4 "did not work" bullets (recipe variants,
batch-quantile rule rejected, genel markalar unfixed, near-miss recipe). F1 rounding unified to
0.8579 in prose (12-digit once in §9). All other GPT numbers verified (incl. checksum suffix e191,
tokenizers 0.22.1 / safetensors 0.6.2 = image pins, 0.768 = 0.748+0.02).
FINAL ZIP: sha256 efcef59ca5705679...  docx sha a1410db4...  unzip -t clean. Image unchanged.
Backups in scratchpad: recent_version_report_ORIG.docx, armaterimi_FINAL_REPORT_EN_long_backup.docx.
Round 2 on the team docx (full rendered read): rebuilt the two inserted §4 rows as clones of an
existing row (Compact style/tcPr identical); brittleness note reduced to the verified UI case
(celikbileklik → semi, "celik bileklik" → relevant; dropped unverified "single-character
deletions"); §13 external-services row lists Claude too. FINAL ZIP sha256 0f1479fe4dd61ead...,
docx sha eb2a450f..., unzip -t clean, audit_300/ present (8 entries). Image unchanged.
Round 3 (user items): "(if any)" dropped from 3 headings; rejected-approaches paragraph expanded;
extras +0.0224 now explained as boundary examples AND decorrelation of class from export source
(single-source R/S rows through the negatives' pipeline) — stated as interpretation, no isolating
ablation; §6 says BERTurk is a BERT encoder used as a CROSS-encoder (not bi-encoder) and that
the cuts are global, DEV-only OOF, frozen in decision.json (provenance re-verified); v1/v2
attribute cleaning defined at first mention and in §7. 4,391 words.
FINAL ZIP sha256 d1128671e16459fd...  docx sha dbe5559a...  unzip -t clean. Image unchanged.
