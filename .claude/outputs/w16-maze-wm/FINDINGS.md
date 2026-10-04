# w16 hub/maze-wm — FINDINGS

Branch `w16/maze-wm` (base `base/iter8 = 148f3b7` + the two w15 harness branches merged,
`b1ca534`). Output dir `/c/chlu/outputs/w16-maze-wm/`. `chlu.__file__` asserted inside the
worktree before any number (LOG.md). Local CPU only, 4 cores. **n_seeds = 3 (0, 1, 2) on
every row; `tolerance_used` null everywhere.** Every table cell below is generated into
`TABLES.md` by `scripts/w16_tables.py`; the machine-readable record is `MAZE-WM.json`
(`scripts/w16_assemble.py`). ⛔ CMAZE-T and CMAZE-O never share a table.

---

## 0. The answer

| | task / config | scored runs | verdict by the pre-registered rule | what survives scrutiny |
|---|---|---|---|---|
| 1 | **CMAZE-T · Config A1** (Shapes recipe unchanged, pinned: latent 128, K 2, 6 atoms, rank/width 0.094) | 3 (1–3/18) | **WIN — STORE row on unseen n-way** | ⭐ survives: matched rivals trained 10× longer still ≤ 0.42 vs CLU 0.72 |
| 2 | **CMAZE-O · Config C (C-field)** (store in the observation field, K 8, 64 atoms) | 3 (4–6/18) | WIN by the rule (0.693 vs 0.167) | ⛔ **rival margin WITHDRAWN**: the matched MLP trained 10× longer reads 0.99 on VAL n-way. Store gate holds (controls 0.000); the READ SPACE is the lever |
| 3 | **CMAZE-T · Config C-512** (latent 512, K 2, 8 atoms), vs 10×-trained rivals | 3 (7–9/18) | **WIN — STORE row on unseen n-way** (0.725 vs 0.453) | replication of A1; ⚠ seed 1 random_atoms 0.464 vs CLU 0.567 (per-seed margin 0.10) |
| — | CMAZE-T Config B (match-bit interface) | 0 | FAILED VAL (VB3) | interface lifts the rival 2× more than the CLU |
| — | CMAZE-O Config A (latent, rank/width ≤ 0.3) | 0 | FAILED VAL (VB0) | the AE latent cannot be carried by a ≤ 0.3 store |
| — | CMAZE-O Config C2 (latent, rank/width 0.5) | 0 | FAILED VAL (VB1, 10×-trained rivals) | |
| — | Task 3 (CMAZE-2b route) | 0 | DECLARED NOT-RUN | gated on "no win in Tasks 1–2"; Task 1 won |

**Budget: 9 of 18 scored runs spent, 9 DECLARED NOT-RUN** (§6).

The one-paragraph reading: **the CMAZE-T transition result is the maze section's store
row** — a single pinned design, three seeds, held-out TEST mazes, beats every
parameter-matched rival (also when those rivals are trained 10× longer and at 10× the
floats), the four charter-C controls collapse on every seed, the learned composition-blind
arms fail, and the lookup floor is 0.000. ⚠ It has a ceiling set by the request design, not
the store (n-way plateaus at 0.7247 across ~15 store configs; seed 1 at 0.569 in every
config). **CMAZE-O yields a clean negative with its mechanism named**: in the AE latent the
store cannot be both load-bearing and alive (link: CONTENT/READ); moved into the
observation field the store is load-bearing (controls 0.000) and composes (0.69 on unseen
n-way, above the AE's own 0.48 ceiling) — but a converged matched MLP given the same field
interface composes better (0.99), so the claim there is an INTERFACE finding, not a
store-over-rivals win (link: INTERFACE). Together with Config B on CMAZE-T this is the w14
pattern for the third time: **a task-shaped interface lifts the rivals more than it lifts
the store.**

---

## 1. Method (the spending rule, as run)

* Every config measured on **VAL first** (free); 3 scored seeds only after VAL bars pass;
  ledger row written **before** each scored run (`/c/chlu/SCORED-LEDGER.md`, rows
  `maze-wm 1/18 … 9/18`, each naming its PREREG).
* PREREGs: `PREREG-w16-maze-wm-T1.md` (+ AMENDMENTS 1–4), `PREREG-w16-maze-wm-T2.md`
  (+ AMENDMENTS 1–4); sha256 + UTC of every version in `*.sha256`.
* One config **pinned across seeds** (w15's per-seed Config-A selection was its dominant
  spread). K chosen by the **crossover rule on VAL**: among store-gate-eligible configs,
  the smallest K whose best VAL n-way is within 0.02 (T1) / 0.01 (T2) of the max.
* Primary row = **unseen n-way** (the row the instruments exist for). Other rows reported
  with per-row claim labels.

### 1.1 ⚠ A defect found in the shipped control, and fixed before spending (AMENDMENT 1)
`landscape_shuffled_across_wells` (w15, both harnesses) dealt well blocks with
`rng.permutation(K)`. At K = 2 that is the identity half the time — and on Config A's VAL
run it **was** the identity on seeds 0 and 1: control == CLU to every digit (0.8019,
0.5689; `t1/A_val_seed*.json`). The 3-seed mean (0.489) cleared the gate only because seed
2 swapped. At any K a random permutation leaves ~1 well in place. Fix: a **derangement**
(`scripts/w16_cmaze_t.py::make_control`, same in `w16_cmaze_o.py`); the w15 files are not
edited; the derangement is printed in every JSON. The VAL sweep, selection and VAL run were
re-done (`*_am1*`, `A1_*`); the superseded files are kept. Deranged, the control reads
0.2504 on VAL n-way (was 0.489). ⭐ w15's own CMAZE-T numbers at K = 4/8 carry the same
partial-identity risk.

---

## 2. Task 1 — CMAZE-T (transition)

### 2.1 Selection
VAL sweep 53 configs × 3 seeds (`t1/sweep_none_am1_seed*.json`), selection
`t1/SELECT-none_am1.json`: 25/53 store-gate eligible; best VAL n-way per K = 0.7247 (K 2),
0.7318 (K 4), 0.7247 (K 8), 0.7247 (K 16) → **K\* = 2 → Config A1 = latent 128, K 2,
6 atoms, rank/width 0.094, read floats 4,584.**

### 2.2 ⭐ SCORED: Config A1 on TEST (runs 1–3/18) — `t1/BARS-A1-test.json`
Unseen n-way, `exact`, n = 761 / 742 / 686:

| arm | mean | per seed |
|---|---|---|
| **CLU (A1)** | **0.7246** | 0.800 / 0.566 / 0.808 |
| rival: transformer (matched) | 0.4035 | 0.394 / 0.395 / 0.421 |
| rival: mlp (matched) | 0.2194 | 0.353 / 0.000 / 0.305 |
| rival: lstm (matched) | 0.1413 | 0.005 / 0.104 / 0.315 |
| control: landscape_removed | 0.0000 | 0 / 0 / 0 |
| control: landscape_shuffled (channels) | 0.2248 | 0.231 / 0.124 / 0.319 |
| control: across_wells (deranged) | 0.2435 | 0.415 / 0.209 / 0.106 |
| control: random_atoms | 0.1655 | 0.271 / 0.019 / 0.207 |
| arm: BLIND-SEQUENTIAL (learned) | 0.3418 | 0.239 / 0.334 / 0.452 |
| arm: BLIND-COLOUR (learned) | 0.0716 | 0.000 / 0.102 / 0.112 |
| floor: lookup cross-maze WORDS | 0.0000 | 0 / 0 / 0 |

Bars: SB1 PASS (margin 0.321, ahead every seed) · SB2 PASS (every control ≤ CLU − 0.15, bar
0.575; also on each seed) · SB3 PASS (learned blind max 0.342 vs 0.675) · SB4 PASS.
**→ STORE row on CMAZE-T unseen n-way.** (Full rows incl. rivals at 10× floats, planner
columns and floors: TABLES.md.) Replay 0 disagreements, 0 impossible rows, disjointness
pass on all three TEST banks.

**Other rows, labelled** (TEST means): `seen` 0.968 — sanity/retrieval (WORDS lookup ≈ 1.0 there);
`pair` 0.410 vs best rival 0.361 — the control gate passes by **0.0001** (across-wells 0.2596 vs
bar 0.2597, knife-edge) **but the learned BLIND-SEQUENTIAL arm reads 0.557 > CLU 0.410**: the
falsifier fires on this row → **NOTHING is claimed on `pair`** (the untrained pair cell drags the
read; w15 §5.6); `value` 0.465 **< best rival 0.511 and < learned blind-seq 0.485 — nothing
claimed on `value`**.
**Rollout validity 0.000 for every learned arm** (per-step exactness CLU 0.925 vs transformer
0.818; 0.93^43 ≈ 0.04 — arithmetic, as w15 said).

**Deletion on the winner** (TABLES.md): store-scoped exactness 7.7e-16 … 1.2e-15 (S_k rel),
deployed atoms Δ = 0.0; the downdated read equals the fitted-without read on every row; CLU
adaptation 0.03–0.13 s, **light and data-free** (A, Mx, S_k, t_k, n_k downdated, same closed
form re-solved — not "no changes"); the rival deletes only by retrain-without, 13.8–14.9 s
and 8,000 gradient steps. Frozen codec and state encoder declared OUTSIDE the deletable object.

### 2.3 ⭐ The training-budget objection, put and survived — `t1/A1_rivaldiag_val_seed*.json`
VAL, rivals at **80,000 steps** (10× recipe): matched mlp 0.392 / 0.000 / 0.369, lstm
0.001 / 0.010 / 0.352, transformer 0.418 / 0.388 / 0.389; **at 10× the read floats AND 10×
steps**: mlp 0.432 / 0.376 / 0.451, lstm 0.415 / 0.437 / 0.460, transformer 0.455 / 0.412 /
0.454 — every one below the CLU's 0.802 / 0.569 / 0.803 (closest: seed 1, lstm 0.437 vs
0.569). The CMAZE-T margin is not a starvation artefact.

### 2.4 Config B — FAILED VAL (VB3), declared — `t1/BARS-B-val.json`
Match bits `[pocket==door, pocket==tint]` given IDENTICALLY to every arm. CLU n-way
0.9258, pair 0.8443 — but the learned BLIND-SEQUENTIAL arm reads **0.8782 > bar 0.8758**.
Named link: **INTERFACE** — the bits are the interaction. The w14 comparison (VAL n-way):
CLU 0.7247 → 0.9258 (**+0.20**); best matched rival 0.3891 → 0.7902 (**+0.40**). The
interface lifts the rival twice as much as the store.

### 2.5 Config C — C-code weak; C-512 → §3
`t1/SELECT-none_am1_C.json`. C-code (item store directly in the 21-dim code, parameterless
decode): best seed-level n-way ≤ 0.46 — the code is too small to hold a rank-limited store.
C-512 pick: latent 512, K 2, 8 atoms (rank/width 0.031); VAL n-way **identical** to A1 per
seed (0.802 / 0.569 / 0.803).

### 2.6 ⚠ The ceiling is the request design, not the store
VAL n-way = 0.7247 with the SAME per-seed triple for ~15 configs spanning latent 64–512,
K 2–16, rank/width 0.03–0.25 (`t1/SELECT-none_am1*.json → table`). The `[bias|main|pair]`
head fixes which held-out 4-tuples are readable; the store decides only whether they are
read cleanly. That is why seed 1 caps at 0.569 everywhere, and why the 2-way row (an
untrained pair CELL) is weak while n-way (trained pairs) is strong (w15 §5.6).

---

## 3. CMAZE-T Config C-512 on TEST (runs 7–9/18)

`t1/BARS-C512-test.json`. Config C-512: latent 512, K 2, 8 atoms, rank/width 0.031, read
floats 9,184; **matched rivals trained 80,000 steps** (PREREG-T1 AMENDMENT 4). Unseen n-way,
`exact`, n = 761 / 742 / 686:

| arm | mean | per seed |
|---|---|---|
| **CLU (C-512)** | **0.7251** | 0.800 / 0.567 / 0.808 |
| rival: transformer (matched, 80k steps) | 0.4533 | 0.468 / 0.430 / 0.462 |
| rival: mlp (matched, 80k steps) | — | 0.403 / 0.368 / 0.394 |
| rival: lstm (matched, 80k steps) | — | 0.411 / 0.394 / 0.380 |
| control: landscape_removed | 0.0140 | 0.030 / 0.000 / 0.012 |
| control: landscape_shuffled (channels) | 0.1997 | 0.227 / 0.232 / 0.140 |
| control: across_wells (deranged) | 0.3798 | 0.565 / 0.000 / 0.574 |
| control: random_atoms | 0.4481 | 0.558 / 0.464 / 0.322 |
| arm: BLIND-SEQUENTIAL (learned) | 0.4145 | 0.350 / 0.442 / 0.452 |
| arm: BLIND-COLOUR (learned) | 0.0341 | — |
| floor: lookup cross-maze WORDS | 0.0000 | 0 / 0 / 0 |

Bars (3-seed means, as pre-registered): SB1 PASS (margin 0.272 over 10×-trained rivals, ahead
on every seed) · SB2 PASS (max control 0.448 ≤ 0.575) · SB3 PASS (0.414 ≤ 0.675) · SB4 PASS.
**→ by the rule, a second STORE row on CMAZE-T n-way — a replication of A1 at a 4× wider codec
latent, against rivals trained 10× longer.** ⚠ **Weaker than A1 per seed:** on seed 1
random_atoms reads 0.464 against the CLU's 0.567 (0.103 < 0.15), and the controls' means sit
0.13–0.28 above A1's. The wider latent lets random atoms decode more (w15's
Johnson–Lindenstrauss saturation, now at low rank/width because the latent is wide). **A1 is
the headline store row; C-512 is its replication, with this caveat attached.**
Other rows: `pair` 0.485 < learned blind-seq 0.637 → nothing; `value` 0.496 ≈ best rival
0.494 and < blind-seq 0.501 → nothing; rollout validity 0.000 (per-step CLU 0.925 vs
transformer 0.897). Replay 0 disagreements on all seeds.
**Deletion**: store-scoped exactness 8.1e-16 … 1.5e-15, atoms Δ 0.0; downdated read == fitted-
without on every cell (surviving-colour n-way 0.437 / 0.430 / 0.683); CLU adaptation 0.95–2.7 s
(data-free) vs rival retrain-without **1,002–1,646 s and 80,000 gradient steps** (surviving-colour
n-way 0.045–0.279).

---

## 4. Task 2 — CMAZE-O (observation generation)

### 4.1 Config A — FAILED VAL (VB0), declared
`t2/sweep_seed*.json`, `t2/SELECT-A.json`. AE ceiling reproduces w15 exactly (seen
0.575 / 0.753 / 0.928). At rank/width ≤ 0.30 the CLU is dead: best unseen n-way **0.0208**
(K 2, 16 atoms), every other config ≤ 0.011. **Named link: CONTENT/READ** — a ≤ 0.3 × 128
= 38-dim latent subspace cannot carry a decodable picture through this AE. C-rank (latent
above 0.3, `t2/SELECT-C-rank.json`) reaches only 0.137 at rank/width 0.5; w15's 0.688 gave
0.116 with controls not collapsing.

### 4.2 Config C = C-field — scored (runs 4–6/18) — `t2/BARS-C-test.json`
Store and head fitted in the **observation field** (2,475-dim item tensor), read out by
the parameterless `decode_state`; no AE in the read path; every rival regresses the
IDENTICAL field. Selection `t2/SELECT-C-field.json`: K\* = 8, 64 atoms (rank/width 0.207 by
the sweep convention; per-well ledger in TABLES.md), read floats 1,232,064.

TEST unseen n-way `valid` (n = 400/seed): **CLU 0.6933** (0.637 / 0.730 / 0.713) · best
recipe rival (mlp) 0.1667 · all four controls **0.0000** · blind max 0.060 · lookup 0.000 ·
pipeline ceiling 1.000 (AE-latent ceiling on the same TEST rows, for reference: 0.330 / 0.492 / 0.542). 2-way 0.521 vs 0.099
(controls 0.000). By the pre-registered rule: SB0–SB4 PASS.

**Deletion** (colour 2, TEST): value deletes 5,530 / 5,250 / 5,959 of 8,000 rows, conjunction
2,240 / 1,435 / 2,168: relative ΔS = 0.0 on every seed; after-deletion == refit-without in every
cell; deleted-colour side of unseen n-way → **0.000** (from 0.58–0.80); other colours move
0.673→0.623, 0.679→0.764, 0.678→0.661 (value) — the refit-without object moves identically;
CLU adaptation 1.8–3.3 s (light, data-free) vs MLP retrain-without 27.7–29.3 s.

### 4.3 ⛔ The rival margin is WITHDRAWN — `t2/C_rivaldiag_val_seed*.json`
Same matched rivals, same field output, **30,000 steps** (10× the w15 recipe), VAL
unseen n-way: **mlp 0.988 / 0.990 / 0.990**, transformer 0.748 / 0.643 / 0.628, lstm
0.470 / 0.483 / 0.418, cont_mlp 0.215 / 0.220 / 0.255 — vs CLU 0.640 / 0.730 / 0.730. The
recipe under-trained the rivals. **No CMAZE-O config can pass VB1 against a converged
matched MLP** (it would need ≥ 1.02). What survives: (i) the store gate (controls 0.000 —
a STORE row relative to its controls), (ii) the read-space effect for the CLU (field
0.69 vs latent ≤ 0.14 on the same rows), (iii) exact deletion. **Named link: INTERFACE** —
the field output space is the lever, and it lifts a plain MLP further than the store.
PREREG-T2 AMENDMENT 4 then made 30k-step rivals mandatory; **C2 (latent, rank/width 0.5)
FAILS VAL** against them: CLU 0.088 / 0.150 / 0.173 vs rival 0.315 / 0.413 / 0.463
(`t2/C2r_val_seed*.json`).

---

## 5a. Ledgers — every byte figure with its convention named (n_seeds = 3, identical per seed)

Convention: **deployed bytes = 4 × (atom floats + address-book floats), float32** — the object
that produces a read; the **float64 sufficient statistic** (what buys exact deletion) is printed
beside it, never netted in, never matched. Rivals are matched on **read floats** (naive vs naive).

| instrument · config | read floats | atoms + head | deployed B (f32) | statistic B (f64) | rank/width |
|---|---|---|---|---|---|
| CMAZE-T · A1 | 4,584 | 768 + 3,816 | 18,336 | 34,320 | 0.094 per well |
| CMAZE-T · C-512 | 9,184 | 4,096 + 5,088 | 36,736 | 530,448 | 0.031 per well |
| CMAZE-O · C-field | 1,232,064 | 158,400 + 1,073,664 | 4,928,256 | 3,092,584 | ≤ 0.207 (ledger 0.174–0.182) |

Matched-rival parameter counts and residuals per seed: TABLES.md (CMAZE-T) and the
`rivals` block of each CMAZE-O JSON (all `matched: true`, within 2 %).

## 5. Negatives, each with its failing link

| negative | link that fails | evidence |
|---|---|---|
| CMAZE-T `pair` row | **interface** (request design: a held-out pair cell is untrained; one-hot words carry no equality) | CLU 0.410 < learned blind-seq 0.557; control gate passes by 0.0001; Config B's match bits lift it to 0.844 |
| CMAZE-T `value` row | **content** (a never-trained colour has no trained cells) | CLU 0.465 < rival 0.511 and < blind-seq 0.485 on TEST |
| CMAZE-T rollout validity | **read** (per-step error compounds over 43 steps) | 0.000 every learned arm; per-step 0.925 |
| CMAZE-T Config B | **interface** (the bits ARE the interaction) | blind-seq 0.878 ≥ CLU − 0.05 |
| CMAZE-O Config A | **content / read** (AE latent not carried by a ≤ 0.3 store) | best n-way 0.021 |
| CMAZE-O C-field vs rivals | **interface** (field output lifts a converged MLP to 0.99) | §4.3 |
| CMAZE-O C2 | **read** (latent pipeline caps the CLU) | 0.137 vs 0.40 |

---

## 6. Budget — 18 runs accounted

| runs | what | status |
|---|---|---|
| 1–3 | T1 CMAZE-T A1, TEST seeds 0/1/2 | SPENT — WIN (store row) |
| 4–6 | T2 CMAZE-O C-field, TEST seeds 0/1/2 | SPENT — rule-WIN, rival margin withdrawn by diagnostic |
| 7–9 | T1 CMAZE-T C-512, TEST seeds 0/1/2 | SPENT — WIN by the rule (replication; per-seed caveat) |
| 10–12 | T2 remaining cap | **DECLARED NOT-RUN**: no CMAZE-O config passes VAL against converged rivals (A: VB0; C2: VB1; any field config would need n-way ≥ 1.02 to clear the 0.99 MLP). Seeding a VAL-failing config is forbidden. |
| 13–18 | Task 3 (CMAZE-2b route, CLU vs NN rivals, w14 merge shared) | **DECLARED NOT-RUN**: the brief opens Task 3 "ONLY if Tasks 1–2 yield no win"; Task 1 yielded a store win that survives the training-budget objection. T1 is at its 6-run cap. |

---

## 7. What did NOT run
* Task 3 (above). * T2 runs 10–12 (above). * A 10×-steps re-scoring of the C-field TEST
leg: not a scored run — the VAL diagnostic already decides it (§4.3). * Deletion rows for
Config B / C2 / C2r (VAL only, not winners; `DELETION: DECLARED_NOT_RUN` in the JSONs).
* GPU/cluster; any seed beyond 3. * Any edit to w8_envscout / w9 / w15 harness modules
(imported; the control fix lives in the w16 runners). * A CMAZE-O value hold-out
comparison: `value_holdout` is 0.000 for every arm incl. the field ceiling-bypassing CLU —
a judge-integrity row, as in w15.

## 8. What would have proved it wrong
* T1: a converged/over-provisioned rival reaching the CLU on n-way — **tested, did not
  happen** (≤ 0.45). A control reaching CLU − 0.15 — did not on TEST, on any seed. A
  learned blind arm reaching CLU − 0.05 — did not (0.34). The WORDS lookup — 0.000.
* T2: a converged rival reaching the CLU — **it happened** (MLP 0.99), and the rival
  claim is withdrawn accordingly.
