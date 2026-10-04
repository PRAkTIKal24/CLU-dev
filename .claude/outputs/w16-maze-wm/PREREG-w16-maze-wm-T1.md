# PREREG — w16 hub/maze-wm, Task 1: CMAZE-T (transition)

Written 2026-09-23 BEFORE any w16 number exists. Target `maze-wm` (18 scored runs,
hub-owned); Task 1 cap = 6 scored runs. Instrument: w15 CMAZE-T harness
(`/c/chlu/outputs/w15-cmaze-t-harness/`), engine/solver/judge/banks/hold-outs/codec/encoder/
controls/rivals/floors imported unmodified (merged at `b1ca534`). Runner:
`scripts/w16_cmaze_t.py`.

## Legs
* **VAL** (free): generator block `base+200k .. base+300k`, 45 mazes, `base = 2,000,000 + 400,000·seed`.
* **TEST** (one scored run per seed): block `base+300k .. base+400k`, 45 mazes — reserved by
  w15, never built before this wave.
* Train block `base .. base+200k`, 90 mazes, identical for both legs. Seeds 0, 1, 2 — all or none.
* Hold-outs from `make_holdout(4, seed)` (data-free), identical on VAL and TEST.
  ⚠ Declared: the held-out word COMBINATIONS are the same on VAL and TEST; the MAZES are
  disjoint. Selecting a config on VAL held-out rows is therefore hyper-parameter selection on
  a split that shares the combination set; TEST is the estimand.

## Config A — the Shapes recipe unchanged, ONE config pinned across all three seeds
Label-free wells (`ward_balanced_partition`), tensor head over the five transition factors
(`request_design` `[bias | main | pair]`), signed-span latent-mixing read (`head_read`, no
feature argument), shared frozen codec (train codes + enumerated alphabet) and frozen state
encoder, ridge 1.0 (pinned default). **rank/width = n_atoms / (latent/K) ≤ 0.30, printed.**

**Grid** (VAL sweep, `sweep` mode): latent ∈ {64, 128, 256} × K ∈ {2, 4, 8, 16} ×
n_atoms ∈ {1, 2, 3, 4, 6, 8}, every point with rank/width ≤ 0.30.

**Selection — the crossover rule on VAL** (means over seeds 0/1/2):
1. A config is *eligible* iff the store gate holds on VAL `nway`: each of the four charter-C
   controls ≤ CLU − 0.15. If no config is eligible, all are eligible and the row is labelled
   INTERFACE.
2. For each K take its best eligible config by VAL `nway` CLU exact.
3. **K\*** = the smallest K whose best VAL `nway` is within 0.02 of the maximum over K
   (the crossover: past K\* extra wells buy < 0.02 composition for more ledger).
4. Config A = the best eligible config at K\*; ties by fewest read floats.

## VAL bars (Config A passes VAL iff ALL hold; VAL `full` run, means over 3 seeds)
* **VB1 composition**: CLU `nway` ≥ best matched rival (mlp/lstm/transformer) `nway` + 0.10,
  AND CLU > best matched rival on `nway` on every seed.
* **VB2 store gate** on `nway`: every charter-C control ≤ CLU − 0.15.
* **VB3 falsifier**: max(learned BLIND-COLOUR, learned BLIND-SEQUENTIAL) `nway` ≤ CLU − 0.05.
* **VB4 floor**: cross-maze WORDS lookup `nway` ≤ CLU − 0.10.

## SCORED bar (TEST, 3 seeds = 3 runs)
The same four conditions (SB1–SB4) on TEST `nway`, means over 3 seeds, SB1 including the
every-seed clause. A WIN = SB1–SB4 all hold → claimed as a **STORE** row on `nway`.
If SB1, SB3, SB4 hold and SB2 fails → reported as an **INTERFACE row**, not a store win.
Other rows (`seen`, `pair`, `value`, rollout) are REPORTED with a per-row claim label
(store / interface / nothing) by the same control gate; they are not bars. `seen` is a
sanity row (cross-maze WORDS floor ≈ 1.0 there — retrieval).

## Amended configs (only if A fails; each seeded only when its VAL passes)
* **Config B** = A + task-shaped interface: two relation bits `[pocket == door colour,
  pocket == tint colour]` appended to the request. ⛔ EVERY arm gets the IDENTICAL bits
  (rivals in their observation, lookup floors in their key, CLU/blind-sequential in the
  request; BLIND-COLOUR is denied them because they ARE the colour relation — declared).
  Same bars; SB/VB4 is expected to be the binding one (w14: the interface lifts the lookup).
* **Config C** = K/rank re-sweep on the maze latent + item/field store ablation + address-book
  rank as a dial. Same bars.

## Standing rules
Rivals matched on read-path floats (atoms + address book); memory ledger printed separately,
never matched. Lookup = printed floor. Seed count a column. Bytes with convention named
(deployed = 4 × (atom + address-book floats), float32; sufficient statistic float64 beside,
never netted). CMAZE-T never shares a table with CMAZE-1/2/2b. tolerance_used = null.

---
## AMENDMENT 1 — 2026-09-23, after the Config-A VAL run, BEFORE any scored run
**Defect found:** the w15 `landscape_shuffled_across_wells` control deals well blocks with
`rng.permutation(K)`. At K = 2 that is the identity with probability 1/2; on the Config-A
(K = 2) VAL run it WAS the identity on seeds 0 and 1 — control == CLU to every digit
(0.8019 / 0.5689 on `nway`) — and the 3-seed mean (0.489) cleared the gate only because
seed 2 happened to swap. At any K a random permutation leaves ~1 well in place, contrary
to the control's own docstring ("every well's atom block is dealt to a DIFFERENT well").
**Fix:** the across-wells control uses a DERANGEMENT (a permutation with no fixed point;
at K = 2 the swap), drawn from the same rng `seed + 991`. Implemented in
`scripts/w16_cmaze_t.py::make_control`; the w15 module is not edited. The derangement is
printed in every `full` JSON.
**Consequence:** the VAL sweep and the selection are RE-RUN with the fixed control
(`t1/sweep_none_am1_seed*.json`, `t1/SELECT-none_am1.json`); the pre-amendment sweep,
selection and Config-A VAL run are preserved unmodified and quoted in LOG.md. No bar moves.

---
## AMENDMENT 2 — 2026-09-23, after the A1 scored runs (WIN), BEFORE any Config B number
Task 1 has 3 runs of its 6-run cap left. Config B (defined above, unchanged) is run on VAL
to spend them on the question w14 left open for this instrument: does the task-shaped
interface lift the CLU, or the rivals and floors, more? Selection = the same crossover rule
over the iface=match sweep (with the AMENDMENT-1 control). Bars = VB1–VB4 / SB1–SB4
unchanged. Additionally REPORTED (not a bar): per-arm lift B − A1 on `nway` and `pair`
for CLU, every rival, the lookup floors — the w14 comparison.

---
## AMENDMENT 3 — Config C, BEFORE any Config C number
Config B FAILED VAL (VB3). Config C = the K/rank re-sweep on the maze latent + the item
store ablation, same bars, same crossover selection (AMENDMENT-1 control):
* **C-code** (item store): store + head fitted directly in the 21-dim next-state CODE
  space, decoded by the parameterless per-block argmax (no codec in the read path);
  K ∈ {2,3,4,7} × n_atoms ∈ {1,2,3}, rank/width ≤ 0.30.
* **C-512**: codec latent 512, K ∈ {2,4,8,16} × n_atoms ∈ {1,2,3,4,6,8,12,16}, rank/width ≤ 0.30.
Config C's pick = the selection rule over the union (C-code ∪ C-512). If its pick is a
config already covered by A1's grid it would be A1 again; it is then DECLARED as adding
nothing and not seeded.

---
## AMENDMENT 4 — rival training budget + Config C pick, BEFORE Config C's VAL `full` number
The CMAZE-O diagnostic showed a recipe can under-train rivals. On CMAZE-T the same check
(`t1/A1_rivaldiag_val_seed*.json`: matched rivals at 80,000 steps = 10× the recipe) leaves
A1's margin intact (best matched rival at 10× steps: n-way ≤ 0.392 vs CLU 0.7247), so the
A1 TEST verdict stands. For every FURTHER CMAZE-T config VB1/SB1 are evaluated against
matched rivals trained **80,000 steps** (the 10×-floats capacity diagnostic keeps 8,000).
**Config C pick** (selection over C-code ∪ C-512, `t1/SELECT-none_am1_C.json`): latent
512, K 2, 8 atoms, rank/width 0.031, read floats 9,184. It is NOT A1's config (latent 512
was outside A1's grid) but its VAL-sweep `nway` equals A1's per seed (the design plateau).
It is run on VAL; if it passes, its 3 scored runs are a replication of the store row at a
4× wider codec latent and 2× the read floats, against 10×-trained rivals.
