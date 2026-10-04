# FINDINGS — w15/cmaze-o-harness

**CMAZE-O, the compositional observation-generation instrument.**
Spoke `w15/cmaze-o-harness`, branch `w15/cmaze-o-harness`, base `base/iter8 = 148f3b7`.
**Scored runs spent: ZERO.** Every number below is **VAL**.
**Acceptance artefact: `HARNESS-READY.json`, `"status": "ready"`.**

Every number in this file carries the artefact path it came from. Nothing lives only in
prose.

---

## 0. The one-line answer

CMAZE-O is built, judged with **no tolerance**, replayed by an **independent second
checker with 0 disagreements over 105,408 judged states**, and shipped with a **seeded
3-seed VAL smoke of the Config-A single design against every rival, the matched-bytes
lookup floor, four charter-C controls and five BLIND columns, on five rows**.

The instrument separates: the CLU is ahead of **all four matched-parameter rivals and the
lookup floor on 3 of 3 seeds on every row**. The **lookup floor reads 0.0000 on both
unseen rows**, so the task is *not* memorisable at this pool size.

It also says one thing against itself, loudly, and ships anyway:
**the charter-C controls do not collapse.** At the span fraction Config-A deploys
(0.688), `landscape_shuffled_across_wells` and `random_atoms` retain most of the CLU's
performance. By COMMON.md that makes this an **INTERFACE claim, not a STORE claim** —
and the interface here is 201,312 of 202,848 read floats (99.24 %). That is recorded in
`HARNESS-READY.json → instrument_warnings` and is the first thing wave 16 should act on.

---

## 1. What CMAZE-O is

| | |
|---|---|
| **request** | nine words — layout id × agent colour × key colour × door colour × tinted-floor colour × agent cell × key cell × door cell × goal cell |
| **answer** | the rendered grid state |
| **judge** | exact placement of every object **AND** legality of the state under the engine's rules. `valid = exact_objects AND legal` |
| **tolerance** | `tolerance_used` is `null` on **every** row of **every** artefact; checked by `HARNESS-READY.json → ACCEPTANCE.4` which walks the whole smoke document |
| **engine** | `w8_envscout.maze_colour`, **imported unmodified** — sha256 `9eacdda17d87cac0ad5ced6ccbb6fa43ebc1508c0460d0cc183e545d066beaac`, the same hash w14's `REQUEST-LAYER-READY.json` recorded for CMAZE-2b |
| **notation** | **n = 9** factors jointly composed in the vocabulary; **n = 4** in the n-way row; **K = 8** wells |

Files: `chlu/exp_w15_cmaze_o_{env,splits,encoder,arms,pipeline,deletion,verify,readiness}.py`,
`scripts/w15_cmaze_o_{valsmoke,span,probes,report,finalise}.py`.

### 1.1 The design decisions, and the measurement behind each

A cell cannot be a raw `(r, c)` — a coordinate is not a word and does not compose. So the
15×15 grid carries a fixed 3×3 arrangement of 5×5 **zones**, and each zone designates
**four distinct cells, one per object slot**, per layout. That makes **`layout × zone`** a
real interaction: the same zone word resolves to a different cell in every layout. It is
the interaction the BLIND columns are blind to, and it is what the instrument is built
around.

| decision | measurement that forced it | artefact |
|---|---|---|
| `braid = 0.08`, not CMAZE-1's adopted `0.40`; door on the **balanced articulation point** of the floor graph | at braid 0.40 the door separates the agent from the goal in **0.0208** of requests and RULES A/B are wallpaper; at 0.25 → 0.0693, 0.15 → 0.1621, **0.08 → 0.2914** | `artefacts/braid_probe.json` |
| the item tensor carries **colour as a shared field** (11 channels), not a per-slot colour block (22 channels) | with a per-slot block, "a colour never seen in THIS slot" is an output channel that was never once positive, so every slot-value hold-out is structurally zero and says nothing about any arm | `chlu/exp_w15_cmaze_o_env.py` channel layout |
| the four object channels are trained with a **spatial softmax over the 225 cells**, the colour field with a **masked per-cell softmax** | per-cell BCE gave an autoencoder ceiling of `exact_objects = 0.0025`; with the spatial softmax the object **cells** went to 1.0000 but colours stalled at ~0.36; with the masked colour softmax as well, `exact_objects` went to 0.3375 at only 1,500 steps and 0.7522 at 6,000 | `artefacts/ae_loss_probe.json`, `artefacts/ae_probe_v{1,2,3}.log` |
| the pool is **stratified so 25 % of every row has a load-bearing door** | a load-bearing door is a ~5 % event under a uniform draw and the breakout would have had no n; realised train fractions 0.281 / 0.275 / 0.274 | `valsmoke/seed{0,1,2}.json → pool_stats` |

### 1.2 The engine's rules are load-bearing, and it is measured

Admissibility of a request (is the state the nine words name a legal state?) broken out by
the three conditions that decide it, n = 4,000 uniform draws:

| door cuts agent from goal | `door colour == key colour` | `tint colour == key colour` | n | admissible |
|---|---|---|---|---|
| no | — | — | 2,855 | **1.0000** |
| yes | no | no | 743 | **0.0000** — RULE A |
| yes | no | yes | 173 | **0.0000** — RULE A |
| yes | yes | no | 179 | **0.5307** |
| yes | yes | yes | 50 | **0.0800** — RULE B |

Source: reproduced by `chlu.exp_w15_cmaze_o_env.admissible` over the same draw; the
structural cut rate is **0.28625**. RULE A and RULE B are each visibly load-bearing on
whether a state exists at all. Over the whole pool, **74.5 %** of uniformly drawn
9-word requests are admissible, so feasibility is a real constraint and not a formality.

**`door_load_bearing` is reported, never required.** Requiring it would force
`door colour == key colour` and collapse two words of the vocabulary into one.

### 1.3 The sentinel-key device, declared

A CMAZE-O state carries exactly one key, but `Instance` needs a home for all five colours.
The four absent colours get homes on **WALL cells**. This is inert and it is asserted on
real states rather than argued: `sentinels_are_inert` replays the engine's own optimal
route and counts how often the walker stands on a sentinel cell — **0 sentinel-cell
occupancies over 600 states replayed**, 200 per seed × 3 seeds
(`artefacts/sentinel_inert.json`). A CMAZE-O state's
legality never mentions the sentinels: "a key never on a wall" is a statement about the
state's one key.

---

## 2. Acceptance — all six criteria, checked not asserted

`HARNESS-READY.json`, `"status": "ready"`.

| # | criterion | result | where |
|---|---|---|---|
| 1 | independent replay of every judged state by a second independently-driven checker | **105,408 states replayed, 0 disagreements**, 35,136 per seed; layout-bank rebuild mismatches **0** | `HARNESS-READY.json → ACCEPTANCE.1`, `valsmoke/VERIFY.json` |
| 2 | no impossible request shown to a non-refuser arm | **30,588 requests checked, 0 impossible** (10,196 per seed) | `ACCEPTANCE.2` |
| 3 | hold-out disjointness from train proven by construction | **pass on all 3 seeds**; 0 exact-request duplicates on every row; every hold-out word and every 3-way sub-tuple has train support > 0; n-way factor-set train coverage **0.9740 / 0.9693 / 0.9595** of 1,728 combinations | `ACCEPTANCE.3` |
| 4 | `tolerance_used: null` on every row | **0 non-null found** anywhere in the smoke document | `ACCEPTANCE.4` |
| 5 | seeded VAL smoke, Config-A vs every rival, 3 seeds, all rows | **3 seeds × 5 rows × 16 arms**, complete | `ACCEPTANCE.5`, `valsmoke/VAL-SMOKE.json` |
| 6 | PREREG with bars before any number | `PREREG.md` + `PREREG.timestamp.txt` = **2026-09-22T21:53:56Z** | `ACCEPTANCE.6` |

**What "independently driven" means.** `chlu/exp_w15_cmaze_o_verify.py` imports neither
`judge_state` nor `legality` nor `render_state` nor `OState.engine_instance` nor
`floor_degree`. It reads the packed bits of every state the pipeline judged, rebuilds the
layout bank **from the seed** and cross-checks it against the saved arrays, re-derives the
reference state from the nine words with its own slot-table lookup, writes the corridor
test as an explicit neighbour loop rather than the pipeline's shifted-array convolution,
assembles the `Instance` itself, and calls `w8_envscout.maze_colour.solve` directly. The
only shared surface is the **engine**, which is the point: both drivers must agree with
CMAZE-1, not with each other's re-implementation of it.

**What "by construction" means for the hold-outs.** Train is drawn uniformly and then
*filtered* against every hold-out pattern, so no held-out pattern can enter it; each
held-out row is drawn with its defining words fixed. A pattern whose word or (n−1)-way
sub-tuple turns out to have no train support is **dropped before any evaluation request is
drawn** — it would be an unknown word, not a composition hold-out. On all three seeds
**6 of 6 two-way patterns and 8 of 8 n-way tuples survived** that test
(`valsmoke/seed*.json → pool_stats.holdout_patterns_dropped_for_no_train_support`).

**Charter B.3 (request-only read).** `assert_head_request_only` is run on every arm of
every row: `head_read` has no feature argument by signature, and the read is bit-identical
under an arbitrary feature perturbation in the caller's scope
(`valsmoke/seed*.json → request_only_assertion`). **`out_of_range` design rows = 0 on all
three seeds** — the colour grid runs over all five words, so the value hold-out is a real
unseen *word*, not a value clamped onto the nearest trained one.

**Charter C.4 (is the read a lookup table in disguise?).** Nearest-stored-item distance
over typical inter-item distance on the `unseen n-way` row: **0.4721 / 0.4472 / 0.4194**
— "not a lookup table by C.4" on all three seeds
(`valsmoke/seed*.json → nearest_item_audit_unseen_nway`).
---

## 3. What the VAL smoke says

All tables in §7. Read against the bars in `PREREG.md`.

### 3.1 The instrument is viable — B1 passes

`AE-CEILING` is the arm that **encodes the true picture** and decodes it: it is the best
any arm on this pipeline can do, and it is printed on every row.

`valid` on `seen`: **0.7522 ± 0.1767** over 3 seeds (per seed 0.5750 / 0.7533 / 0.9283).
Pre-registered bar B1 was ≥ 0.50. **Pass.**

The ceiling is **row-dependent and must be read that way**: the autoencoder was trained by
reconstruction on the TRAIN items only, so a held-out request's picture is a picture it has
never seen. `unseen 2-way` ceiling **0.2946**, `unseen n-way` ceiling **0.4767**. Every arm
is capped by its row's ceiling, and the CLU's fraction of its own ceiling is

| row | CLU | ceiling | CLU / ceiling |
|---|---|---|---|
| seen | 0.2728 | 0.7522 | 0.36 |
| unseen 2-way | 0.0657 | 0.2946 | 0.22 |
| unseen n-way (n=4) | 0.1158 | 0.4767 | 0.24 |

(`valsmoke/VAL-SMOKE.json → table`.) The seed sd on the ceiling (0.1767 on `seen`) is
large; three seeds is three seeds and the spread is printed rather than smoothed.

### 3.2 The CLU separates from every rival and from the floor — on 3 of 3 seeds

The pre-declared reporting rule (PREREG §"What is NOT a bar") was: the CLU **separates**
from a rival only if it is ahead on ≥ 2 of 3 seeds on that row. Realised:

| comparator | seen | unseen 2-way | unseen n-way |
|---|---|---|---|
| MLP, matched params | **3/3** | **3/3** | **3/3** |
| continuous MLP, matched params | **3/3** | **3/3** | **3/3** |
| LSTM, matched params | **3/3** | **3/3** | **3/3** |
| transformer, matched params | **3/3** | **3/3** | **3/3** |
| matched-bytes LOOKUP floor | **3/3** | **3/3** | **3/3** |

(computed from `valsmoke/VAL-SMOKE.json → table.<arm>.<row>.valid.per_seed`.)

All four rivals are reported, always, on every row; ⛔ the best is never reported alone.
The **strongest** rival is the transformer (0.1817 / 0.0412 / 0.0792); the **weakest** is
the continuous MLP (0.0006 / 0.0000 / 0.0000), which is the expected and informative
result — the nine words are categorical (a layout id, a zone id), and a rival that reads
them as min-max normalised *numbers* cannot represent `layout × zone` at all. It is in the
set because it is the only family that could in principle answer an off-grid word; on
CMAZE-O there are no off-grid words, and it pays for the representation with everything.

**Naive vs naive.** Every rival is matched to the CLU's **deployed read floats**
(202,848) within 2 %: mlp 203,210 · cont_mlp 202,588 · lstm 202,028 · transformer 204,572,
all `matched_within_2pct = true` on all three seeds
(`valsmoke/VAL-SMOKE.json → per_seed_ledgers.<seed>.rivals`). The rivals get the same
interpolated one-hots the CLU's design is built from but **without the pair-interaction
block**. The float64 sufficient statistic is declared separately and **never charged to
them** — they are not asked to delete. **The lookup table is a printed floor, never a
rival.**

### 3.3 The task is NOT memorisable — B3 passes

The matched-bytes lookup floor holds 1,584 whole latents (202,752 floats against the CLU's
202,848) and answers with the mean of the training items whose nine words agree most.

| row | LOOKUP | CLU |
|---|---|---|
| seen | 0.1506 | 0.2728 |
| unseen 2-way | **0.0000** | 0.0657 |
| unseen n-way (n=4) | **0.0000** | 0.1158 |

Bar B3 was "lookup < CLU − 0.02 on `unseen n-way`". Realised gap **0.1158**. **Pass.** The
task file's falsifier — *if the lookup floor reaches the CLU on unseen n-way, the task is
memorisable and the instrument cannot support a store claim* — **did not fire**. That is
the instrument's strongest result: an unseen 4-way combination cannot be answered by
keeping the items.

### 3.4 The BLIND columns are all below the CLU — B4 passes

| BLIND column | what it is blind to | seen | unseen 2-way | unseen n-way |
|---|---|---|---|---|
| layout-free cells | `layout × zone` | 0.0589 | 0.0320 | 0.0625 |
| colour marginal | the four colour words | 0.0061 | 0.0034 | 0.0000 |
| fully factorised marginal | both | 0.0011 | 0.0000 | 0.0000 |
| no layout word | the layout word | 0.1117 | 0.0530 | 0.0100 |
| RANDOM state | everything | **0.0000** | **0.0000** | **0.0000** |

The CLU is ahead of every blind column on `seen` on 3/3 seeds. On `unseen 2-way` the
`no-layout` column is ahead of the CLU on 2 of 3 seeds — that column renders the **modal
layout's** picture with the right zones and colours, and gets the whole request right
whenever the modal layout *is* the requested layout (1 in 12 by chance, plus whatever the
stratification adds). It is reported, not hidden: on `unseen n-way` it falls to 0.0100
against the CLU's 0.1158, which is the row where the blindness actually costs.

The `legal` column separates the blind arms much more sharply than `valid` does:
`layout-free cells` is legal only **0.4161** of the time and the `fully factorised
marginal` only **0.3656**, because putting an object at a layout-independent cell puts it
on a wall. That is the legality clause doing its job on arm output, which is exactly what
it is for.

### 3.5 The value hold-outs

**Strict (`value_holdout`, colour 5 never seen anywhere): 0.0000 for every arm, including
the AE ceiling.** This was pre-declared as the expected outcome and it is shipped as a
**judge-integrity check**, not a comparison: a colour that was never once rendered cannot
be rendered, and an arm scoring above zero on it would be reading something it should not
have. Nothing did.

**Slot (`value_slot_holdout`, colour 4 on keys/doors/tints in training but never on an
agent): AE ceiling 0.0092, CLU 0.0008, every rival 0.0000.** The shared-colour-field
representation was chosen precisely so this row *could* be non-zero, and it is — barely.
The honest reading is that **the frozen decoder, not the read, is the binding constraint
here**: the ceiling itself is 0.0092, so the row has almost no headroom for any arm to
compete in. It is reported as a **near-floor row**, and the recommendation to wave 16 is
in §6.

### 3.6 The charter-C controls do NOT collapse — B2 FAILS, and this is the finding

| control | seen | unseen 2-way | unseen n-way | collapses? |
|---|---|---|---|---|
| `landscape_removed` | **0.0000** | **0.0000** | **0.0000** | yes |
| `landscape_shuffled` (atom ROWS) | 0.2728 | 0.0657 | 0.1158 | **no — identical to the CLU** |
| `landscape_shuffled_across_wells` | 0.1733 | 0.0421 | 0.0650 | no |
| `random_atoms` | 0.1500 | 0.0404 | 0.0667 | no |

Bar B2 required every control at ≤ 0.10 × the CLU. Only `landscape_removed` meets it.
**By COMMON.md this is an INTERFACE claim, not a STORE claim, and this FINDINGS says
which: it is an interface claim.** It is recorded in
`HARNESS-READY.json → instrument_warnings` as three
`CHARTER_C_CONTROL_DID_NOT_COLLAPSE` entries and the harness **ships anyway**.

Two separate things are going on, and they should not be confused.

**(a) `landscape_shuffled` is a PROVED NULL for this read.** It agrees with the CLU to
every printed digit on all five rows and all three seeds — not approximately, identically.
That is not a measurement failure, it is an identity. The read is
`x̂_k = Φ W_k G_k` and `W_k` is **re-solved against the deployed atoms**. Permuting the
ROWS of `G_k` by `P` gives `P G_k`; the ridge objective
`‖Φ W (P G) − X‖² + λ‖W‖²` is minimised by `W* Pᵀ`, because `‖W Pᵀ‖_F = ‖W‖_F`; so
`Φ (W* Pᵀ)(P G) = Φ W* G` and the read is bit-identical. **Any signed-span read whose
address book is re-solved against the atoms is invariant to atom ordering.** The
row-permutation control can therefore never be evidence of anything, and it is stated once
here so it is never mistaken for evidence again. A fifth control,
`landscape_shuffled_channels`, permutes the **channels** inside the well instead and can
move; it is shipped in `chlu.exp_w15_cmaze_o_arms.CONTROLS_SHIPPED` and measured in the
span sweep (§4).

**(b) `random_atoms` and `landscape_shuffled_across_wells` do not collapse because the
CLU on CMAZE-O is almost entirely ADDRESS BOOK.** The ledger says it plainly:

| | floats | share |
|---|---|---|
| atoms (the store) | 1,536 | **0.76 %** |
| address book (the head) | 201,312 | **99.24 %** |
| total deployed read | 202,848 | 100 % |

and the span fraction is **0.688** — `n_atoms = 12` over wells of width 16. That is
precisely the regime `w9_maze/arms.py` warns about in its own docstring: *"a store cannot
span more of the latent space without moving toward saturation, where `rank == d_k`, the
projection is the identity, the content carries no information and the random-atom control
stops collapsing."* CMAZE-O at Config-A is in that regime. §4 is the sweep that locates
where it stops being.
---

## 4. The span sweep — where does the store start to matter?

`valsmoke/SPAN-SWEEP.json`, `valsmoke/span.log`. 3 seeds, VAL, same frozen autoencoders as
the smoke. `n_atoms ∈ {2, 6, 12}` over wells of width 16, `K = 8`.

**The autoencoder is reproduced, not re-trained differently.** `train_ae` draws batches
from `np.random.default_rng(0)` and initialises from `jax.random.PRNGKey(seed)`, so
rebuilding it from the same seed and step count gives the same frozen model. The check is
in the numbers: at `n_atoms = 12` the sweep's CLU reads **0.2728** on `seen` and **0.1158**
on `unseen n-way` — identical to the VAL smoke to four decimal places.

### 4.1 The sweep

`valid`, 3-seed mean; the bracketed figure is the control as a **fraction of the CLU**.

| n_atoms | span | CLU read floats | CLU seen | `landscape_removed` | `landscape_shuffled_channels` | `landscape_shuffled_across_wells` | `random_atoms` |
|---|---|---|---|---|---|---|---|
| 2 | 0.062 | 33,808 | 0.0006 | 0.0000 (0.00×) | 0.0000 (0.00×) | 0.0000 (0.00×) | 0.0000 (0.00×) |
| 6 | 0.312 | 101,424 | 0.0267 | 0.0000 (0.00×) | 0.0083 (**0.31×**) | 0.0111 (**0.42×**) | 0.0067 (**0.25×**) |
| 12 | 0.688 | 202,848 | 0.2728 | 0.0000 (0.00×) | 0.1389 (**0.51×**) | 0.1733 (**0.64×**) | 0.1500 (**0.55×**) |

and on `unseen n-way`: CLU 0.0000 / 0.0125 / 0.1158, `random_atoms` 0.0000 / 0.0075 /
0.0667.

**The reading.** Lowering the span *does* make the controls matter more — the random-atom
fraction falls from 0.55× at span 0.688 to 0.25× at 0.312 — but it does **not** produce a
point where the controls collapse *and* the CLU still works, because the CLU falls with
them: 0.2728 → 0.0267 → 0.0006. The charter-C bar (≤ 0.10×) is never met at any point on
this grid. **There is no span on CMAZE-O at this read where a store claim becomes
available.**

One confound is declared rather than buried: `n_atoms` moves the **read floats** as well
as the span, because the address book is `K · P · M`. The three points are 33,808 /
101,424 / 202,848 floats, so the sweep is a joint sweep of span and capacity, and it is
capacity that most plausibly explains the CLU's own fall. Separating them needs the
pair-block ablation (`P: 2,097 → 69`), which is **DECLARED NOT-RUN** (§8).

### 4.2 The row-permutation identity, measured

`read_gap_vs_CLU` is `max |read_CLU − read_control|` over the evaluation rows:

| control | n_atoms 2 | 6 | 12 |
|---|---|---|---|
| `landscape_shuffled` (atom **rows**) | **1.9e-06** | **3.8e-06** | **5.7e-06** |
| `landscape_shuffled_channels` | 2.2e+01 | 2.0e+01 | 1.8e+01 |
| `landscape_shuffled_across_wells` | 2.2e+01 | 2.0e+01 | 1.8e+01 |
| `random_atoms` | 2.1e+01 | 1.9e+01 | 1.5e+01 |
| `landscape_removed` | 2.2e+01 | 2.3e+01 | 2.3e+01 |

Seven orders of magnitude apart. The row-permutation control's gap is float32 round-off;
it is the identity of §3.6(a), measured. Every other control moves the read by ~15–23 in
the latent, and their differing *scores* are therefore real differences, not noise.
---

## 5. The deletion rows

Full tables in §7. Deleted colour **c = 2**, pinned in
`chlu/exp_w15_cmaze_o_deletion.py::DELETE_COLOUR`, not selected.

* **value deletion** — every training state carrying colour 2 in **any** slot:
  **5,530 / 5,250 / 5,959 of 8,000 rows (69.1 % / 65.6 % / 74.5 %)**. That is a very large
  deletion and it is the task file's literal wording; the fraction is printed rather than
  softened.
* **conjunction deletion** — colour 2 **∧** tinted floor, i.e. every training state whose
  *tinted-floor* colour is 2: **2,240 / 1,435 / 2,168 rows (28.0 % / 17.9 % / 27.1 %)**.

### 5.1 Store-scoped exactness — B6 passes

The claim is store-scoped: *the store after deletion == the store fitted from scratch
without those rows*, on the same partition. Measured, not asserted:

| deletion | max abs ΔS | relative ΔS | Δt | Δn |
|---|---|---|---|---|
| value, seeds 0/1/2 | 2.04e-10 / 1.75e-10 / 1.75e-10 | 2.72e-15 / 7.74e-16 / 1.60e-15 | **0.0** | **0** |
| conjunction, seeds 0/1/2 | 1.16e-10 / 2.33e-10 / 1.75e-10 | 5.91e-16 / 4.35e-16 / 5.95e-16 | **0.0** | **0** |

Pre-registered bar B6 was `max abs diff ≤ 1e-6` relative. Realised worst case
**2.72e-15**. **Pass** — this is float64 round-off in `S = Σ x xᵀ`, not an approximation.

And it shows up in the **behaviour**, which is the check that matters: in every one of the
twelve post-deletion cells, **`CLU after deletion` equals `CLU refit-without` to four
decimal places** (§7). The exact downdate does not approximate the refit-without object;
it *is* it.

### 5.2 Forgetting, and what survives

| deletion | row | side | CLU as-trained | CLU after deletion | CLU refit-without | MLP as-trained | MLP retrain-without |
|---|---|---|---|---|---|---|---|
| conjunction | seen | **deleted colour** | 0.2518 | **0.0000** | **0.0000** | 0.1113 | 0.0000 |
| conjunction | seen | other colours | 0.2845 | 0.3173 | 0.3173 | 0.1271 | 0.1396 |
| conjunction | unseen n-way | **deleted colour** | 0.1133 | **0.0000** | **0.0000** | 0.0422 | 0.0000 |
| conjunction | unseen n-way | other colours | 0.1063 | 0.1044 | 0.1044 | 0.0566 | 0.0608 |

The conjunction deletion is **complete forgetting**: 0.0000 on every row of the deleted
side, and the other colours are untouched (on `seen` they *improve*, 0.2845 → 0.3173,
because the ridge no longer spends capacity on the deleted tint colour).

The value deletion — which removes two thirds of the training pool — forgets to
**0.0814 / 0.0148 / 0.0219** rather than to zero on the deleted side. That is not residual
memory of colour 2: with colour 2 absent from every surviving row, its design column is
unsupported and the read simply ignores that word, so a request whose *only* colour-2 slot
is one of four can still get the other seven facts right. The refit-without object scores
exactly the same, which is the proof that nothing survived.

### 5.3 The cost of deleting, printed

| | CLU | rival (matched-parameter MLP) |
|---|---|---|
| mechanism | exact downdate of `S, t, n` **and** of the head's `A = ΦᵀΦ`, `Mx = ΦᵀX`, then the **same closed form re-solved** | **retrain-without** — there is no other deletion available to it |
| adaptation, seconds | 3.88 / 3.48 / 2.19 (value) · 3.25 / 2.02 / 3.49 (conjunction) | 15 / 17 / 20 (value) · 17 / 16 / 12 (conjunction) |
| speed-up | — | **4× – 9×** |
| data touched | **none** — no pool row is read, no item is kept | the whole surviving pool, 3,000 steps |

**Post-deletion adaptation is light and data-free — it is not "no changes".** The head is
re-solved; what makes it data-free is that it is re-solved against statistics that were
*already retained* and were downdated by the deleted rows' own contribution. The
`refit_from_scratch` timing is printed beside it in `valsmoke/seed*.json` so the
closed-form re-solve is never confused with a free operation.

**The frozen encoder and decoder are declared OUTSIDE the deletable object.** They were
trained by reconstruction on the whole training pool and are **not** re-trained for either
arm after deletion; the rival's retrain-without keeps the same frozen encoder, so the
comparison is like for like. Deleting rows from the store does not and cannot remove them
from the encoder, and this FINDINGS does not claim it does.

### 5.4 The memory ledger, printed separately and never matched

| object | value (all 3 seeds) | byte convention |
|---|---|---|
| CLU deployed read | **202,848 floats = 811,392 B** | 4 bytes/float, float32; atoms + address book — what turns nine words into a latent |
| — of which atoms | 1,536 floats | float32 |
| — of which address book | 201,312 floats | float32 |
| CLU float64 sufficient statistic | **9,792 B** | float64; buys **exact deletion**; declared beside the deployed bytes, **never netted in, never matched to a rival** |
| LOOKUP floor | 1,584 items held | matched to the CLU's read floats |
| rivals | 203,210 / 202,588 / 202,028 / 204,572 params | params × 4 B, float32; all matched within 2 % |
| shared frozen AE | 2,292,115 params | declared **outside** the deletable object; identical for every arm, so it cancels from every comparison |

Seed count is a column on every table. ⛔ The inflated-rival row does not exist: no rival
was ever charged the float64 statistic.
---

## 6. What would have proved this wrong, and what did

The task file names one falsifier and COMMON.md names two more. All three were run.

| falsifier | fired? | evidence |
|---|---|---|
| **the lookup floor reaches the CLU on `unseen n-way`** → the task is memorisable and the instrument cannot support a store claim | **NO** — lookup 0.0000 vs CLU 0.1158 on 3/3 seeds; C.4 nearest-item ratio 0.42–0.47, "not a lookup table" | §3.3, `valsmoke/VAL-SMOKE.json` |
| **the charter-C controls do not collapse** → it is an interface claim or nothing | **YES** — only `landscape_removed` collapses | §3.6, §4, `HARNESS-READY.json → instrument_warnings` |
| **a BLIND column matches the CLU** → the composition is not load-bearing | **partially** — `BLIND-no-layout` is ahead on `unseen 2-way` on 2/3 seeds, and falls to 0.0100 vs 0.1158 on `unseen n-way` | §3.4 |
| **the AE ceiling near zero** → the table measures the pipeline, not the read | **NO** — ceiling 0.7522 on `seen`, above its pre-registered bar of 0.50; but it is **row-dependent** (0.2946 on `unseen 2-way`) and every arm is capped by it | §3.1 |

**The honest verdict this spoke ships: CMAZE-O supports an INTERFACE claim. It does not
support a STORE claim as configured.** The instrument is sound — the judge is exact, the
replay is clean, the hold-outs are disjoint, the floor does not reach the arm — and the
arm that wins is winning through its **address book**, which is 99.24 % of its deployed
read.

### What wave 16 should do with this

1. **Do not read a store claim off CMAZE-O at Config-A.** Every table here is an
   interface table. `HARNESS-READY.json → instrument_warnings` says so in machine-readable
   form; it is not buried in prose.
2. **The address book is the object to shrink, not the store.** `K·P·M` with `P = 2,097`
   (nine factors, full pair block) is what makes the head 201,312 floats. A pair-block
   ablation is one flag (`HeadSpec(pair_block=False)`, which drops `P` from 2,097 to 69)
   and would move the ledger by a factor of 30. That ablation is **DECLARED NOT-RUN** here
   (§8) and is the single highest-value next measurement.
3. **Never use `landscape_shuffled` (atom rows) as evidence.** It is an identity, proved
   analytically and measured at 4.8e-07 read gap (§4.2). Use
   `landscape_shuffled_channels`, which is now shipped in
   `chlu.exp_w15_cmaze_o_arms.CONTROLS_SHIPPED`.
4. **Raise the autoencoder before raising the arms.** The ceiling on `unseen 2-way` is
   0.2946; three quarters of what an arm could get right on that row is unavailable to it
   because the frozen decoder has not seen that picture. More AE steps, or an AE trained on
   a wider pool of *pictures* than the arms are trained on *requests*, would lift every row
   at once.
5. **`value_slot_holdout` is a near-floor row** (ceiling 0.0092). Either accept it as an
   instrument check like the strict value row, or give the decoder a colour-field
   augmentation so the ceiling has headroom. As it stands no arm can be measured on it.


---

## 7. The tables

### `valid` — VAL, 3 seeds, mean ± sd over seeds

Source: `valsmoke/VAL-SMOKE.json` → `table.<arm>.<row>.valid`. `tolerance_used` is `null` on every cell.

| arm | seen | unseen 2-way | unseen n-way (n=4) | value hold-out (strict) | value hold-out (slot) |
|---|---|---|---|---|---|
| AE ceiling (encode the true picture, decode, judge) | 0.7522 ± 0.1767 | 0.2946 ± 0.0574 | 0.4767 ± 0.1203 | 0.0000 ± 0.0000 | 0.0092 ± 0.0014 |
| **CLU — Config-A single design** | 0.2728 ± 0.0913 | 0.0657 ± 0.0315 | 0.1158 ± 0.0396 | 0.0000 ± 0.0000 | 0.0008 ± 0.0014 |
| MLP, matched params | 0.1211 ± 0.0301 | 0.0210 ± 0.0119 | 0.0533 ± 0.0270 | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 |
| continuous MLP, matched params | 0.0006 ± 0.0010 | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 |
| LSTM, matched params | 0.0094 ± 0.0042 | 0.0000 ± 0.0000 | 0.0033 ± 0.0029 | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 |
| transformer, matched params | 0.1817 ± 0.0289 | 0.0412 ± 0.0287 | 0.0792 ± 0.0447 | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 |
| matched-bytes LOOKUP **floor** (never a rival) | 0.1506 ± 0.0428 | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 |
| control: landscape_removed | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 |
| control: landscape_shuffled | 0.2728 ± 0.0913 | 0.0657 ± 0.0315 | 0.1158 ± 0.0396 | 0.0000 ± 0.0000 | 0.0008 ± 0.0014 |
| control: landscape_shuffled_across_wells | 0.1733 ± 0.0353 | 0.0421 ± 0.0162 | 0.0650 ± 0.0115 | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 |
| control: random_atoms | 0.1500 ± 0.0347 | 0.0404 ± 0.0101 | 0.0667 ± 0.0267 | 0.0000 ± 0.0000 | 0.0008 ± 0.0014 |
| BLIND: layout-free cells | 0.0589 ± 0.0054 | 0.0320 ± 0.0119 | 0.0625 ± 0.0303 | 0.0608 ± 0.0063 | 0.0508 ± 0.0029 |
| BLIND: colour marginal | 0.0061 ± 0.0079 | 0.0034 ± 0.0029 | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 |
| BLIND: fully factorised marginal | 0.0011 ± 0.0019 | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 |
| BLIND: no layout word | 0.1117 ± 0.0060 | 0.0530 ± 0.0067 | 0.0100 ± 0.0066 | 0.0992 ± 0.0115 | 0.0817 ± 0.0080 |
| RANDOM state | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 |

Row sizes per seed: seen n=[600, 600, 600], unseen 2-way n=[396, 396, 396], unseen n-way (n=4) n=[400, 400, 400], value hold-out (strict) n=[400, 400, 400], value hold-out (slot) n=[400, 400, 400].

### `exact_objects` — VAL, 3 seeds, mean ± sd over seeds

Source: `valsmoke/VAL-SMOKE.json` → `table.<arm>.<row>.exact_objects`. `tolerance_used` is `null` on every cell.

| arm | seen | unseen 2-way | unseen n-way (n=4) | value hold-out (strict) | value hold-out (slot) |
|---|---|---|---|---|---|
| AE ceiling (encode the true picture, decode, judge) | 0.7522 ± 0.1767 | 0.2946 ± 0.0574 | 0.4792 ± 0.1227 | 0.0000 ± 0.0000 | 0.0092 ± 0.0014 |
| **CLU — Config-A single design** | 0.2728 ± 0.0913 | 0.0665 ± 0.0312 | 0.1192 ± 0.0416 | 0.0000 ± 0.0000 | 0.0008 ± 0.0014 |
| MLP, matched params | 0.1211 ± 0.0301 | 0.0210 ± 0.0119 | 0.0533 ± 0.0270 | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 |
| continuous MLP, matched params | 0.0006 ± 0.0010 | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 |
| LSTM, matched params | 0.0094 ± 0.0042 | 0.0000 ± 0.0000 | 0.0042 ± 0.0014 | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 |
| transformer, matched params | 0.1817 ± 0.0289 | 0.0412 ± 0.0287 | 0.0792 ± 0.0447 | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 |
| matched-bytes LOOKUP **floor** (never a rival) | 0.1506 ± 0.0428 | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 |
| control: landscape_removed | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 |
| control: landscape_shuffled | 0.2728 ± 0.0913 | 0.0665 ± 0.0312 | 0.1192 ± 0.0416 | 0.0000 ± 0.0000 | 0.0008 ± 0.0014 |
| control: landscape_shuffled_across_wells | 0.1739 ± 0.0362 | 0.0429 ± 0.0154 | 0.0692 ± 0.0151 | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 |
| control: random_atoms | 0.1528 ± 0.0323 | 0.0404 ± 0.0101 | 0.0692 ± 0.0267 | 0.0000 ± 0.0000 | 0.0008 ± 0.0014 |
| BLIND: layout-free cells | 0.0589 ± 0.0054 | 0.0320 ± 0.0119 | 0.0625 ± 0.0303 | 0.0608 ± 0.0063 | 0.0508 ± 0.0029 |
| BLIND: colour marginal | 0.0061 ± 0.0079 | 0.0034 ± 0.0029 | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 |
| BLIND: fully factorised marginal | 0.0011 ± 0.0019 | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 |
| BLIND: no layout word | 0.1133 ± 0.0058 | 0.0530 ± 0.0067 | 0.0125 ± 0.0109 | 0.1000 ± 0.0109 | 0.0817 ± 0.0080 |
| RANDOM state | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 |

Row sizes per seed: seen n=[600, 600, 600], unseen 2-way n=[396, 396, 396], unseen n-way (n=4) n=[400, 400, 400], value hold-out (strict) n=[400, 400, 400], value hold-out (slot) n=[400, 400, 400].

### `legal` — VAL, 3 seeds, mean ± sd over seeds

Source: `valsmoke/VAL-SMOKE.json` → `table.<arm>.<row>.legal`. `tolerance_used` is `null` on every cell.

| arm | seen | unseen 2-way | unseen n-way (n=4) | value hold-out (strict) | value hold-out (slot) |
|---|---|---|---|---|---|
| AE ceiling (encode the true picture, decode, judge) | 0.9744 ± 0.0084 | 0.8561 ± 0.0617 | 0.9292 ± 0.0146 | 0.8508 ± 0.0362 | 0.8983 ± 0.0435 |
| **CLU — Config-A single design** | 0.8600 ± 0.0117 | 0.8089 ± 0.0511 | 0.8575 ± 0.0347 | 0.8275 ± 0.0043 | 0.8092 ± 0.0238 |
| MLP, matched params | 0.8228 ± 0.0019 | 0.8232 ± 0.0397 | 0.8250 ± 0.0278 | 0.8050 ± 0.0000 | 0.8083 ± 0.0184 |
| continuous MLP, matched params | 0.7317 ± 0.0144 | 0.7332 ± 0.0780 | 0.7442 ± 0.0260 | 0.6817 ± 0.0923 | 0.6892 ± 0.0191 |
| LSTM, matched params | 0.8350 ± 0.0209 | 0.7837 ± 0.0316 | 0.8208 ± 0.0302 | 0.7850 ± 0.0090 | 0.8017 ± 0.0624 |
| transformer, matched params | 0.8317 ± 0.0033 | 0.7685 ± 0.0501 | 0.7950 ± 0.0304 | 0.7808 ± 0.0397 | 0.7975 ± 0.0295 |
| matched-bytes LOOKUP **floor** (never a rival) | 0.9072 ± 0.0042 | 0.8779 ± 0.0039 | 0.8850 ± 0.0263 | 0.8775 ± 0.0156 | 0.8608 ± 0.0138 |
| control: landscape_removed | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 |
| control: landscape_shuffled | 0.8600 ± 0.0117 | 0.8089 ± 0.0511 | 0.8575 ± 0.0347 | 0.8275 ± 0.0043 | 0.8092 ± 0.0238 |
| control: landscape_shuffled_across_wells | 0.8067 ± 0.0202 | 0.7938 ± 0.0884 | 0.7917 ± 0.0126 | 0.7917 ± 0.0038 | 0.7608 ± 0.0270 |
| control: random_atoms | 0.8106 ± 0.0063 | 0.7803 ± 0.0291 | 0.8092 ± 0.0411 | 0.7717 ± 0.0063 | 0.7250 ± 0.0150 |
| BLIND: layout-free cells | 0.4161 ± 0.0570 | 0.4251 ± 0.1672 | 0.4475 ± 0.0565 | 0.4183 ± 0.0379 | 0.4083 ± 0.0397 |
| BLIND: colour marginal | 0.8150 ± 0.1603 | 0.8561 ± 0.1247 | 0.8258 ± 0.1510 | 0.8250 ± 0.1516 | 0.8158 ± 0.1595 |
| BLIND: fully factorised marginal | 0.3656 ± 0.0801 | 0.4007 ± 0.1735 | 0.4283 ± 0.0841 | 0.3792 ± 0.0674 | 0.3658 ± 0.0751 |
| BLIND: no layout word | 0.8633 ± 0.0613 | 0.8653 ± 0.0722 | 0.8342 ± 0.0884 | 0.8392 ± 0.0713 | 0.8517 ± 0.0690 |
| RANDOM state | 0.7606 ± 0.0243 | 0.7483 ± 0.0439 | 0.7583 ± 0.0363 | 0.7367 ± 0.0189 | 0.7533 ± 0.0151 |

Row sizes per seed: seen n=[600, 600, 600], unseen 2-way n=[396, 396, 396], unseen n-way (n=4) n=[400, 400, 400], value hold-out (strict) n=[400, 400, 400], value hold-out (slot) n=[400, 400, 400].

### `exact_layout` — VAL, 3 seeds, mean ± sd over seeds

Source: `valsmoke/VAL-SMOKE.json` → `table.<arm>.<row>.exact_layout`. `tolerance_used` is `null` on every cell.

| arm | seen | unseen 2-way | unseen n-way (n=4) | value hold-out (strict) | value hold-out (slot) |
|---|---|---|---|---|---|
| AE ceiling (encode the true picture, decode, judge) | 0.9994 ± 0.0010 | 0.8822 ± 0.0675 | 0.9342 ± 0.0275 | 0.7417 ± 0.1039 | 0.9108 ± 0.0406 |
| **CLU — Config-A single design** | 0.9706 ± 0.0127 | 0.8687 ± 0.0910 | 0.8867 ± 0.0636 | 0.9075 ± 0.0222 | 0.8400 ± 0.0597 |
| MLP, matched params | 0.9922 ± 0.0092 | 0.9663 ± 0.0168 | 0.9858 ± 0.0029 | 0.9850 ± 0.0195 | 0.9783 ± 0.0161 |
| continuous MLP, matched params | 0.3928 ± 0.0712 | 0.5008 ± 0.1048 | 0.5417 ± 0.2532 | 0.3025 ± 0.1345 | 0.3967 ± 0.0569 |
| LSTM, matched params | 0.9956 ± 0.0025 | 0.9781 ± 0.0139 | 0.9858 ± 0.0101 | 0.8183 ± 0.0593 | 0.9108 ± 0.0539 |
| transformer, matched params | 0.9956 ± 0.0025 | 0.9672 ± 0.0219 | 0.9808 ± 0.0113 | 0.8983 ± 0.1227 | 0.9742 ± 0.0128 |
| matched-bytes LOOKUP **floor** (never a rival) | 0.9994 ± 0.0010 | 1.0000 ± 0.0000 | 0.9975 ± 0.0043 | 1.0000 ± 0.0000 | 1.0000 ± 0.0000 |
| control: landscape_removed | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 |
| control: landscape_shuffled | 0.9706 ± 0.0127 | 0.8687 ± 0.0910 | 0.8867 ± 0.0636 | 0.9075 ± 0.0222 | 0.8400 ± 0.0597 |
| control: landscape_shuffled_across_wells | 0.8111 ± 0.0728 | 0.6877 ± 0.1494 | 0.7550 ± 0.1070 | 0.7350 ± 0.0978 | 0.6758 ± 0.0906 |
| control: random_atoms | 0.7561 ± 0.0415 | 0.6423 ± 0.0077 | 0.6875 ± 0.0667 | 0.6550 ± 0.0611 | 0.6050 ± 0.0346 |
| BLIND: layout-free cells | 1.0000 ± 0.0000 | 1.0000 ± 0.0000 | 1.0000 ± 0.0000 | 1.0000 ± 0.0000 | 1.0000 ± 0.0000 |
| BLIND: colour marginal | 1.0000 ± 0.0000 | 1.0000 ± 0.0000 | 1.0000 ± 0.0000 | 1.0000 ± 0.0000 | 1.0000 ± 0.0000 |
| BLIND: fully factorised marginal | 1.0000 ± 0.0000 | 1.0000 ± 0.0000 | 1.0000 ± 0.0000 | 1.0000 ± 0.0000 | 1.0000 ± 0.0000 |
| BLIND: no layout word | 0.0989 ± 0.0082 | 0.0505 ± 0.0051 | 0.0000 ± 0.0000 | 0.0867 ± 0.0101 | 0.0733 ± 0.0101 |
| RANDOM state | 0.0711 ± 0.0100 | 0.0682 ± 0.0158 | 0.0742 ± 0.0052 | 0.0842 ± 0.0095 | 0.0883 ± 0.0201 |

Row sizes per seed: seen n=[600, 600, 600], unseen 2-way n=[396, 396, 396], unseen n-way (n=4) n=[400, 400, 400], value hold-out (strict) n=[400, 400, 400], value hold-out (slot) n=[400, 400, 400].

### Memory ledger — printed separately, never matched (COMMON.md)

Source: `valsmoke/VAL-SMOKE.json` → `per_seed_ledgers.<seed>`. Seed count is a column. Byte convention is named on every figure.

| object | seed 0 | seed 1 | seed 2 | seeds | byte convention |
|---|---|---|---|---|---|
| CLU deployed read floats (atoms + address book) | 202,848 | 202,848 | 202,848 | 3 | 4 bytes/float, float32 |
| … of which atoms | 1,536 | 1,536 | 1,536 | 3 | float32 |
| … of which address book | 201,312 | 201,312 | 201,312 | 3 | float32 |
| CLU deployed bytes | 811,392 | 811,392 | 811,392 | 3 | float32 |
| CLU float64 sufficient statistic (bytes) | 9,792 | 9,792 | 9,792 | 3 | float64 — buys exact deletion, **never netted in, never matched** |
| CLU span fraction (n_atoms / d_k) | 0.688 | 0.688 | 0.688 | 3 | — |
| LOOKUP floor items held | 1,584 | 1,584 | 1,584 | 3 | matched to the CLU's read floats |
| rival mlp params (matched) | 203,210 (matched) | 203,210 (matched) | 203,210 (matched) | 3 | params × 4 bytes, float32 |
| rival cont_mlp params (matched) | 202,588 (matched) | 202,588 (matched) | 202,588 (matched) | 3 | params × 4 bytes, float32 |
| rival lstm params (matched) | 202,028 (matched) | 202,028 (matched) | 202,028 (matched) | 3 | params × 4 bytes, float32 |
| rival transformer params (matched) | 204,572 (matched) | 204,572 (matched) | 204,572 (matched) | 3 | params × 4 bytes, float32 |
| shared frozen AE params (encoder + decoder) | 2,292,115 | 2,292,115 | 2,292,115 | 3 | declared OUTSIDE the deletable object; identical for every arm |

### Deletion rows — VAL, colour c = 2

Source: `valsmoke/VAL-SMOKE.json` → `deletion.<seed>`; `valsmoke/seed<N>.json` → `deletion`.

| deletion | seed | rows deleted | store-scoped max abs ΔS | rel. ΔS | Δn | CLU adaptation (s) | rival retrain-without (s) | speed-up |
|---|---|---|---|---|---|---|---|---|
| value | 0 | 5530 (69.1%) | 2.037e-10 | 2.723e-15 | 0 | 3.88 | 15 | 4× |
| value | 1 | 5250 (65.6%) | 1.746e-10 | 7.740e-16 | 0 | 3.48 | 17 | 5× |
| value | 2 | 5959 (74.5%) | 1.746e-10 | 1.604e-15 | 0 | 2.19 | 20 | 9× |
| conjunction | 0 | 2240 (28.0%) | 1.164e-10 | 5.908e-16 | 0 | 3.25 | 17 | 5× |
| conjunction | 1 | 1435 (17.9%) | 2.328e-10 | 4.353e-16 | 0 | 2.02 | 16 | 8× |
| conjunction | 2 | 2168 (27.1%) | 1.746e-10 | 5.952e-16 | 0 | 3.49 | 12 | 4× |

#### Post-deletion composition on the OTHER colours (`valid`)

| deletion | row | side | CLU as-trained | CLU after deletion | CLU refit-without | MLP as-trained | MLP retrain-without | n per seed |
|---|---|---|---|---|---|---|---|---|
| value | seen | other_colours | 0.2621 | 0.3663 | 0.3663 | 0.1285 | 0.1953 | 190+208+162 |
| value | seen | deleted_colour | 0.2764 | 0.0814 | 0.0814 | 0.1181 | 0.0380 | 410+392+438 |
| value | unseen 2-way | other_colours | 0.0495 | 0.0307 | 0.0307 | 0.0354 | 0.0246 | 116+65+149 |
| value | unseen 2-way | deleted_colour | 0.0682 | 0.0148 | 0.0148 | 0.0125 | 0.0081 | 280+331+247 |
| value | unseen n-way (n=4) | other_colours | 0.0725 | 0.0490 | 0.0490 | 0.0490 | 0.0691 | 211+101+67 |
| value | unseen n-way (n=4) | deleted_colour | 0.1125 | 0.0219 | 0.0219 | 0.0497 | 0.0242 | 189+299+333 |
| conjunction | seen | other_colours | 0.2845 | 0.3173 | 0.3173 | 0.1271 | 0.1396 | 441+495+446 |
| conjunction | seen | deleted_colour | 0.2518 | 0.0000 | 0.0000 | 0.1113 | 0.0000 | 159+105+154 |
| conjunction | unseen 2-way | other_colours | 0.0654 | 0.0617 | 0.0617 | 0.0236 | 0.0246 | 317+254+326 |
| conjunction | unseen 2-way | deleted_colour | 0.0590 | 0.0000 | 0.0000 | 0.0094 | 0.0000 | 79+142+70 |
| conjunction | unseen n-way (n=4) | other_colours | 0.1063 | 0.1044 | 0.1044 | 0.0566 | 0.0608 | 350+250+250 |
| conjunction | unseen n-way (n=4) | deleted_colour | 0.1133 | 0.0000 | 0.0000 | 0.0422 | 0.0000 | 50+150+150 |

---

## 8. What did NOT run — DECLARED NOT-RUN

Every item here is a deliberate scope decision, not a failure and not a zero.

| not run | why | what it would have cost |
|---|---|---|
| **pair-block ablation** (`HeadSpec(pair_block=False)`) | out of scope for "build the instrument"; it is the highest-value next measurement and is named in §6 | ~1 h wall — no new AE needed if the AE is cached |
| **TEST split / any scored run** | the task file says **scored runs: ZERO**. No `/c/chlu/SCORED-LEDGER.md` row was written, because COMMON.md scopes that discipline to spokes that spend one | — |
| **`n_atoms` sweep on the held-out `unseen 2-way` and value rows** | the span sweep evaluates `seen` and `unseen n-way` only, to keep it inside one wall-clock budget alongside the smoke | ~2× the sweep's 32 min |
| **more than 12 layouts** | the layout bank is a declared instrument parameter; 12 keeps `layout × zone` at 432 combinations, which 8,000 training requests cover ~18× | a larger bank needs a larger pool and a longer AE |
| **a refusal / certificate arm** | CMAZE-O is a RESULT instrument: no impossible request is ever written or shown, so there is nothing for a refuser to refuse. The DISPLAY-row machinery is CMAZE-2b's (`w9_maze/evaluate.py::certificate`), not this spoke's | — |
| **CMAZE-1 / CMAZE-2 / CMAZE-2b comparison rows** | ⛔ COMMON.md: they never share a table with CMAZE-O, and none is in this document | — |
| **a `cont_mlp` off-grid row** | the continuous MLP exists to answer off-grid words; CMAZE-O's vocabulary is closed and has none, so the row would be empty | — |
| **seed count above 3** | 3 seeds, all or none, as pre-registered. ⛔ No "9/9 seeds" anywhere | — |

---

## 9. Reproduction

```bash
cd /c/chlu/worktrees/CHLU-w15-cmaze-o-harness
source scripts/w15_cmaze_o_env.sh          # env_w3.sh + the Windows PYTHONPATH fix
python scripts/w15_cmaze_o_probes.py   $CHLU_O_OUT/artefacts
python scripts/w15_cmaze_o_valsmoke.py "C:/chlu/outputs/w15-cmaze-o-harness/valsmoke" --seeds=0,1,2
python scripts/w15_cmaze_o_span.py     "C:/chlu/outputs/w15-cmaze-o-harness/valsmoke" --seeds=0,1,2
python scripts/w15_cmaze_o_finalise.py "C:/chlu/outputs/w15-cmaze-o-harness" "C:/chlu/worktrees/CHLU-w15-cmaze-o-harness"
```

**Environment note, recorded because it silently voids numbers.** `/c/chlu/env_w3.sh` sets
`PYTHONPATH="$CHLU_ROOT:$_H/.venv/Lib/site-packages"` with a POSIX `:` separator. The
interpreter on this box is **Windows CPython 3.11**, which splits `PYTHONPATH` on `;` —
so the whole string is one bogus entry, `equinox` is unimportable, and `chlu` resolves
only because `python -c` puts the cwd on the path. `scripts/w15_cmaze_o_env.sh` sources
`env_w3.sh` (which is not this spoke's file to edit) and then re-exports the **same two
paths** joined with `;` in Windows form. With that, the required assertion gives

```
OK C:\chlu\worktrees\CHLU-w15-cmaze-o-harness\chlu\__init__.py
jax 0.9.0   numpy 2.4.1   equinox 0.13.4
equinox from C:\Users\PratikJawahar\chlu\chlu-bundle-v2\.venv\Lib\site-packages
```

`chlu.__file__` resolves **inside** the worktree and the pinned wave-2 versions are the
ones running. No venv was built; no `uv sync` was run.
