# PREREG — w21 hub/maze-t-clu-improve (target maze-t-symmetric, scored runs 7, 8, 9 of 9)

Written BEFORE any TEST number exists for this config. Branch w21/maze-t-clu-improve (base w19 tip 0455961).

## 1. The frozen config (the ONLY thing that changed vs w19)
`prereg/CLU-REL-A10.json`, sha256 `0d4ec5b9fba55bd6f943a91fc8afa20027d79c5edea3531a47117c86f08ee5c6`:
K 2 · n_atoms 10 · ridge 1.0 · latent 128 (codec recipe unchanged) · design =
`[bias | main (27) | 0.5 x pair (290) | eq (6) | eq x door one-hot (42) | eq x dir one-hot (30)]`, width 396.
eq(x, y) = 1 iff colour words x, y (pocket, key, door-closed-colour, tint) both name a colour and it is the SAME colour (the "colour coding" item of the brief's OPEN list). pair_scale 0.5 == ridge 4 on the pair block (hierarchical shrinkage).
Read floats = 10 x 128 atoms + 396 x 20 address book = **9,200** (A1: 4,584). Code: `scripts/w21_clu.py` (TransitionCluX / CluFactory), hooked into `scripts/w19_symmetric.py` via `--clu_cfg`.
VAL (gate, free, closed form): n-way 0.9029 / 0.9141 / 0.9943, mean 0.9371 vs A1 0.8019 / 0.5689 / 0.8034 (0.725) -> KEPT. 69 candidates logged in LOG.md.

## 2. FROZEN and unchanged
Banks (TEST base+300k), hold-outs, judge (`exact`), w9 state encoder, codec, rival architectures and recipes (8,000 / 80,000 steps), read-path-float matching rule (rivals re-matched AUTOMATICALLY to 9,200), per-condition input (BOTH = observation (+) 318 design for every rival; the CLU reads the true words), learned blind arms (A1 design, K2/6 atoms, UNCHANGED — they are falsifiers, not the candidate), floors, controls' recipes (across-wells a DERANGEMENT, asserted and printed).

## 3. Runs
- Scored: condition **BOTH**, leg **test**, seeds **0, 1, 2** = runs **7, 8, 9 / 9**, deletion rows ON, rel diagnostic ON. Ledger row appended BEFORE each launch. All 3 reported whatever they show; all or none.
- Free, printed: BOTH VAL and WORDS-FULL VAL, 3 seeds each, same config. ⛔ The VAL rival comparison does NOT gate the scored runs (the gate was §1's CLU-vs-A1 gate; declared now, before any VAL-rival number, so nothing is cherry-picked).

## 4. The label (the Head's rule; replaces w19's numeric margin bars) — on unseen n-way, `exact`, TEST
R* = best MATCHED rival over every kind and BOTH budgets (8k row, 80k column), by mean; best-matched-per-seed = max over the same set on each seed.
- **WIN**: CLU mean > R* mean AND CLU > best-matched-per-seed on >= 2 of 3 seeds AND the store gate holds.
- **LOSS**: CLU mean < R* mean - s.
- **COMPARABLE**: otherwise (|CLU - R*| within s, i.e. within one seed's spread), where s = sample std over the 3 seeds of the per-seed margin (CLU - best-matched-per-seed).
- Store gate (as w19 SB2): every charter-C control mean <= CLU - 0.15. If it fails, the claim is INTERFACE or nothing, said so.
- Printed with it (never bars): learned blind max vs CLU - 0.05 (SB3); WORDS lookup floor vs CLU - 0.10 (SB4); the 10x-read-floats diagnostic; the '+rel' diagnostic (every rival kind given the CLU's colour-relational columns appended to its frozen input, matched floats, 8k and 80k) — this names whether the margin is the STORE or the RELATIONAL CODING.
- Also printed: seen / pair / value rows, per-step exactness, rollout validity (limitation), deletion rows (store-scoped exactness bar 1e-5), byte ledger with convention.

## 5. What would prove this wrong
- CLU TEST n-way not above R* on the mean / on a majority of seeds -> COMPARABLE or LOSS.
- Controls not collapsing -> no store claim.
- '+rel' rivals matching the CLU -> the margin is the relational coding (an input feature), not the store; it would be reported as such.
