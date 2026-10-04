# PREREG — w16 hub/maze-wm, Task 2: CMAZE-O (observation generation)

Written 2026-09-23 BEFORE any w16 CMAZE-O number exists (the AE prep jobs print only
pool sizes and AE training loss). Task 2 cap = 6 scored runs. Instrument: w15 CMAZE-O
harness imported unmodified (bank, pool, hold-outs, AE recipe 6,000 steps, OCluRead,
controls, rivals, lookup floor, BLIND arms, exact judge `valid = exact_objects AND legal`,
deletion study). Runner `scripts/w16_cmaze_o.py`. tolerance_used = null.

## Legs
* One frozen AE per seed, trained ONCE on TRAIN items (reconstruction only), cached, shared
  by every arm and every config.
* **VAL** = the w15 rows (rng 10,000+seed). **TEST** (scored) = the w15 row recipe on a
  DISJOINT rng stream (20,000+seed), same bank / hold-out patterns / stratification,
  excluding every train request and every VAL-row request (receipts in each JSON:
  0 in train, 0 in VAL rows, measured at prep). `seen` is train items on both legs — a
  sanity row, never a claim.
* Seeds 0, 1, 2 — all or none.

## Controls on this instrument
`landscape_removed`, `landscape_shuffled_channels`, `landscape_shuffled_across_wells`,
`random_atoms`. ⚠ The atom-ROW `landscape_shuffled` is a proved identity for a re-solved
signed-span read (w15-O FINDINGS §3.6a, read gap 1e-6); it is printed in `full` runs and
never used as evidence either way — its channel-permuting version is the one that can move.

## Config A — Shapes recipe unchanged, ONE config pinned across seeds
ward_balanced label-free wells, `[bias|main|pair]` head (pair_block = True), signed-span
read, frozen shared AE, ridge 1.0, **rank/width ≤ 0.30 printed**.
Grid: K ∈ {1,2,4,8,16} × n_atoms ∈ {1,2,4,6,8,12,16,24,32,38}, rank/width ≤ 0.30.
Selection (VAL, 3-seed means, `valid`): eligible iff every control ≤ 0.5 × CLU on
`unseen_nway` (if none eligible, all eligible and the row is labelled INTERFACE); best per
K by `unseen_nway`; K* = smallest K within 0.01 of the max; best at K*, ties fewest floats.
(0.01 not 0.02: CMAZE-O's scale is ~0.1, declared.)

## VAL bars (VAL `full` run of the pinned config, 3-seed means; ALL must hold)
* VB0 viability: CLU `unseen_nway` ≥ 0.05.
* VB1 composition: CLU `unseen_nway` ≥ best matched rival (mlp / cont_mlp / lstm /
  transformer) + 0.03, and ahead of the best rival on every seed.
* VB2 store gate on `unseen_nway`: each of the four controls ≤ 0.5 × CLU.
* VB3 blind: every BLIND column ≤ CLU − 0.03 on `unseen_nway`.
* VB4 floor: matched-bytes LOOKUP ≤ CLU − 0.03 on `unseen_nway`.

## SCORED bar (TEST, 3 runs): SB0–SB4 = VB0–VB4 on TEST `unseen_nway`.
All pass → STORE row. SB2 fails but SB0/1/3/4 pass → INTERFACE row (reported as such,
not a store win). `seen`, `unseen_2way`, value rows reported with per-row labels.

## Amended configs (only if A fails; each seeded only when its VAL passes)
* **B** = A + a task-shaped interface given IDENTICALLY to every arm (defined in an
  amendment before its VAL numbers).
* **C** = K/rank re-sweep on the maze latent + pair-block (address-book) ablation as the
  address-book dial (`pair_block=False`, P 2,097 → 69) + item/field store ablation.
  Selection over the pair_block=False grid by the same rule; same bars.

---
## AMENDMENT 1 — before any CMAZE-O number
* The across-wells control uses a DERANGEMENT (PREREG-T1 AMENDMENT 1: the w15 control's
  random permutation can leave wells in place, and at K = 2 is the identity half the time).
* K = 1 is REMOVED from the grid: with one well the across-wells control is undefined
  (there is no other well to deal to), so the store gate cannot be evaluated there.
  Grid is K ∈ {2, 4, 8, 16}.

---
## AMENDMENT 2 — Config A verdict and Config C definition, BEFORE any Config C number
**Config A (CMAZE-O) FAILS VAL.** VAL sweep (`t2/sweep_seed*.json`), rank/width ≤ 0.30:
the best CLU `unseen_nway` over the whole grid is 0.0208 (K 2, 16 atoms, pair block) —
below VB0 (≥ 0.05). The selection rule's pick is that config; its CLU read is a
deterministic function of the cached AE, bank and pool, so a VAL `full` run would print the
identical CLU number: VB0 cannot pass. DECLARED; no scored run.
**Config C** (VAL sweep `sweepC`, same bars VB0–VB4, same selection rule, 3-seed means):
* **C-field** — the item/field store: the store and head are fitted in the observation
  FIELD (item tensor, 2,475 dims) instead of the AE latent, read out by the parameterless
  `decode_state` (logit = read − 0.5). No AE in the read path. Grid K ∈ {2,4,8,16} ×
  n_atoms ∈ {4,8,16,32,64}, rank/width ≤ 0.30, K·n_atoms ≤ 512. Rivals under C-field are
  given the IDENTICAL output space (they regress the field, same decode); the lookup floor
  stores field items.
* **C-rank** — the latent store ABOVE rank/width 0.30: (K, n_atoms) ∈ {(2,32), (4,16),
  (8,8), (8,12), (4,24)}. Reported; VB2 is expected to be binding (w15-O).
* Config C's pick = the best C-field config by the selection rule if any C-field config is
  eligible, else the best eligible C-rank config, else (none eligible) the best C config by
  `unseen_nway`, labelled INTERFACE.

---
## AMENDMENT 3 — Config C2, BEFORE its VAL `full` number
Config C (C-field) WON on TEST (3 runs). Task 2 has 3 runs of its cap left. **Config C2 =
the C-rank pick** (latent store above rank/width 0.30: K 2, 32 atoms, rank/width 0.50,
read floats 138,304 — the selection rule's pick over C-rank, `t2/SELECT-C-rank.json`,
store-gate eligible on the VAL sweep). Question it answers on a scored estimand: with the
SAME store recipe and the SAME rows, how much of the C-field win is the READ SPACE (AE
latent vs observation field)? Same bars VB0–VB4 / SB0–SB4.

---
## AMENDMENT 4 — rival training budget, after the diagnostic, BEFORE any further scored run
The C-field training-budget diagnostic (`t2/C_rivaldiag_val_seed*.json`) shows the w15
rival recipe (3,000 steps) UNDER-TRAINS the rivals on CMAZE-O: the matched MLP at 30,000
steps reads VAL `unseen_nway` 0.99 vs the CLU's 0.70. Therefore, for EVERY further CMAZE-O
config, VB1/SB1 are evaluated against rivals trained **30,000 steps** (10× the recipe),
same matched parameters, same inputs, same output space. C2's recipe-rival VAL pass
(`t2/BARS-C2-val.json`) is NOT sufficient; C2 is re-run on VAL as tag `C2r`.
The Config-C (C-field) TEST verdict is re-labelled: pre-registered bars met under the
recipe, **rival margin WITHDRAWN** by this diagnostic (reported, not re-scored).
