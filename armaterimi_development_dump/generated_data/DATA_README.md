# Generated data (rules §8)

| file | rows | role |
|---|---|---|
| student_train_final.parquet | 813,269 | The exact training table of the submitted model: organizer relevant/semi + our judged irrelevant + judge-relabeled extras, after normalization and the gender-artifact fix (T4); its sha256 prefix e2b5c75ee265 is recorded in train_meta_gf2_berturk_6ep.json (shipped in scripts/e15_student outputs metadata). |
| irrelevant_final.parquet | 145,848 | Frozen mined-negative set. Produced by scripts/e17_mine_v5 (10 rule recipes + tf-idf retrieval near-miss), labeled by the LLM judge (google/gemini-3.7-flash via OpenRouter; validated 0.924 macro-F1 vs a hand rubric; blind 3-reader audit: 0.987 precision). |
| irrelevant_reserve.parquet | 6,577 | Reserve negatives, judged but not used in training. |
| judged_extras_quarantine.parquet | 29,551 | Mined pairs the judge relabeled as semi/relevant; the usable slice feeds the training table. |
| contrast_final.parquet | 902 | Deliberately adversarial near-positive pairs, used for validation only. |
| labels_*.parquet | — | Raw judge outputs backing the tables above. |
| selected_v5_final.manifest.json | — | Mining selection manifest with input hashes. |
| gates_v5_final.json | — | Pre-spend distribution gates (adversarial AUC vs the organizer's own train/test shift). |
| audit_300/ | 300 pairs × 3 votes | Blind cross-model audit of the frozen negatives: the 300 sampled pairs (no judge label), three independent Claude reader agents' votes (rubric only, blind to the judge and to each other), the scored table and the verdict (majority-vote precision 0.987, 98.3% unanimous). Scorer: scripts/e17_mine_v5/scripts/audit_v5.py. |

Reproduction code: scripts/e17_mine_v5 (mining + selection), scripts/e04_label/scripts/label.py
(the LLM-judge caller + rubric used for the pseudo-labels), scripts/e15_student (training-table
assembly + training). No organizer public-test rows or labels appear in any of these files.
