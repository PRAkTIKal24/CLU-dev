# PREREG — w21 spoke/words-full-test (target `maze-t-symmetric`, runs 4–6/9)

Written 2026-09-23 BEFORE any TEST number, after the instrument check passed
(`repro/REPRO_COMPARE.txt`: w19 WORDS-FULL VAL CLU n-way 0.8019 / 0.5689 / 0.8034 reproduced
bit-for-bit, max |Δ| = 0 over the CLU + 4 controls × 4 rows × 3 seeds; derangement [1, 0]).

## What is measured (all frozen — a measurement, not an experiment)
* Condition **WORDS-FULL**: the 318-wide `[bias|main|pair]` design to EVERY arm (CLU and
  mlp / lstm / transformer); cont_mlp gets the words as numbers (w19 C2, unchanged).
* CLU **Config A1 UNCHANGED** (latent 128, K 2, 6 atoms, ridge 1.0).
* Rivals matched on read-path floats (target 4,584) by the standing rule; trained at 8,000 AND
  80,000 steps. The 10×-read-floats column (8,000 steps) is printed, **never a bar, never R\***.
* 4 charter-C controls, learned BLIND-SEQUENTIAL / BLIND-COLOUR, WORDS lookup floor.
* Banks, hold-outs, judge, codec, rival recipe: w15/w16/w19, imported.
* Command, per seed s ∈ {0, 1, 2} (all or none):
  `python scripts/w19_symmetric.py s <out>/test --cond WORDS-FULL --leg test`
  (deletion ON by default on the test leg; both modes).
* Only code change vs w19 tip 0455961: `deletion_rows` also prints the PRE-deletion CLU read
  on the same surviving-colour rows (Advisor ruling (d)). Does not touch any non-deletion number.

## THE LABEL RULE (the Head's; REPLACES the w19 margin bars — no numeric margin bar)
Primary row `nway` (`exact`, unseen n-way combinations); means over 3 seeds.
* **R\*** = the matched rival with the highest 3-seed mean over every kind
  (mlp, lstm, transformer, cont_mlp) AND both budgets (8k row, 80k column).
* Per-seed paired difference Δ_s = CLU_s − R\*_s; Δ̄ = mean over seeds.
* **"One seed's spread" σ** = sample standard deviation (ddof = 1) of the per-seed paired
  difference Δ_s over the 3 seeds. (Declared choice: the paired difference is the
  quantity the label is about, and it removes the shared per-seed data ceiling — e.g. seed 1's
  0.5689 cap on VAL — which would otherwise inflate a per-arm spread.)
* **Store gate** = w19 SB2 unchanged: every charter-C control mean ≤ CLU − 0.15.
* **WIN** iff Δ̄ > 0 AND Δ_s > 0 on ≥ 2 of 3 seeds AND the store gate holds.
* else **COMPARABLE** iff |Δ̄| ≤ σ.
* else **LOSS** iff Δ̄ < 0.
* else (Δ̄ > σ but the majority clause or the store gate fails) **COMPARABLE**, with the
  failing clause named.
* Printed beside the label, NEVER changing it: σ computed from the CLU's own per-seed
  spread and from R\*'s; the per-seed best matched rival (max over all matched arms at that
  seed); the 10×-read-floats diagnostic.
* The same rule is printed for `seen`, `pair`, `value` (secondary; those rows carry no claim
  beyond w19's reading: seen = sanity, pair = blind falsifier, value = extrapolation).
* The w19 PREREG fields (B1–B4, `label` STORE/INTERFACE/LOSS/NOTHING) still appear in
  `test/BARS-WORDS-FULL-test.json` because `scripts/w19_bars.py` is reused unchanged; they are
  printed as the w19 record and are **superseded** by `HEAD_LABEL` in the same file.

## Mechanism (named with the label)
Candidates declared now: (i) *the shared pair block carries composition* — a matched network
given the same 318 design matches or beats the store read (w19 VAL reading); (ii) *store
advantage* — the CLU is ahead with controls collapsed; (iii) *seed-level data ceiling* —
arms tie at the hold-out's cap. The label is printed with whichever the numbers show; no
re-tuning, re-selection or extra seed to move it.

## Also printed
per-step exactness AND mean first-error step (CLU and best rival; no arm validates a rollout
— stated); each rival's hidden width beside its matched float count; deletion (both modes):
store-scoped exactness, downdate-vs-fitted-without agreement on surviving rows, CLU
adaptation time, rival retrain-without cost, PRE-deletion composition beside post-deletion.

## What would prove the reading wrong
A TEST result with the CLU ahead of R\* on the mean and on ≥ 2 seeds with the controls
collapsed (WIN) would overturn w19's VAL reading that the pair block, not the store, carries
the w16 margin.
