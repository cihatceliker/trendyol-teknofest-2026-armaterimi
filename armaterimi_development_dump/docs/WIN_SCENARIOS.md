# Win scenarios — what we must ship to take 1st (2026-08-24)

Bank: Stage-1 = **35.4/40**; the stage-1 leader has 40.0 → deficit **4.6** over the remaining 60.

Scoring (verified, ORGANIZER_REQUIREMENTS §2 + S3 transcript):
`final-set = (F1/best_F1) × rank_mult × 20` (rank_mult 1.00/0.95/0.90/…), on the private 90k.
`speed = (SPS/best_SPS) × 10`, **no rank multiplier**, `SPS = 90000 / runtime` where runtime =
`./run_solution.sh` start → submission file fully written, on their identical-spec machine.
Report 10, explainability+presentation ~20 (subjective; assume roughly a wash between top teams).

## Margins vs leader archetypes (subjective axes equal, us = various ship options)

| leader ships → | 12B LLM (.88, 60 SPS) | mid encoder (.855, 1500) | tiny bert (.82, 5000) | balanced (.86, 2000) |
|---|---|---|---|---|
| us: today (.850, 1750) | **+3.4** | −4.3 | −9.4 | −7.1 |
| us: +speedwork (.855, 2600) | **+3.6** | **+0.6** | −7.6 | −3.4 |
| us: quality push (.87, 1750) | **+3.8** | −1.8 | −9.0 | −4.6 |
| us: 8L-class (.845, 3400) | **+3.5** | −0.2 | −6.2 | −1.8 |

## The reading

1. **We beat every slow-LLM leader already.** If the stage-1 leader ships a 12B, any of our options wins.
2. **The threat is speed, not accuracy.** A team shipping a ~5,000-SPS tiny model takes 4–9 pts from us
   via the MaxNorm denominator. Our F1 edge cannot buy that back (+0.01 F1 ≈ +0.35 pts; +500 SPS vs a
   fast leader ≈ +1.5–2.5 pts).
3. **Therefore: maximize F1 subject to SPS ≥ ~3,000, and chase the fastest-runtime engineering wins**
   (startup, data path, MAXLEN, depth) before model-quality wins below ~+0.01.
4. Being #1 on F1 flips the rank multiplier (~1 pt) AND sets the denominator — worth holding.
5. Unknowable: other teams' choices. The robust play is the Pareto point: **≥0.85 F1 at ≥3,000 SPS**,
   which loses only to (unlikely) "tiny model + excellent F1" combinations and wins or ties everything else.

Sensitivity notes: subjective axes (30 pts) are assumed equal; every point we take there is a full point
of margin — the report/explainability work is not optional polish, it is 1/3 of the remaining score.
