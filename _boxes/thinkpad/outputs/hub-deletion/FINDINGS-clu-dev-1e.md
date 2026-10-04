# FINDINGS — `hub/deletion-2`, independently verified
**Session `clu-dev-1e [eb2e09]`. Written 2026-09-17T14:22Z, elapsed 3:19 of 12 h (START 11:03:31Z).**
⛔ Every number here was re-derived by me from raw artefacts (`build.json`, `modes.npz`, `preds.npz`,
`rival_preds.npz`) with my own code, not read out of `analyse.py`. Where I reproduced another agent's
number I say so; where I contradicted one I say that too.
⚠ The agent that drove the cell (`clu-dev-89`) disappeared between 12:05Z and 14:04Z; its detached
driver is still building. See `INCIDENT-two-agents-clu-dev-1e.md` §11.

---
## 1. ⭐ THE BLOCKING GATE — PASSED, and stronger than a pass
Three independent OS processes, three different launch paths, **22/22 arrays bitwise identical**:
`store_digest d0eacf3b…`, `param_hash 1bb1edb5…`, `book_hash 57bc8a31…`, `labels_sha256 27b49454…`,
`well_sizes [303,305,573,501,366]`, `loss_first 47.4907341003418`, `loss_final 32.049007415771484`.
- ⭐ **Load-invariant**: the three ran at 554.9 / 617.9 / 728.3 s wall — a **1.31× spread** under very
  different contention — and produced identical bytes. That rules out AUDIT-01's top-ranked risk
  (BLAS thread count perturbing `channel_correlation` and flipping a greedy `ward` merge).
- ⭐ A **fourth** confirmation through the whole eval path: on the fixed `eval_a` set, `det_A` and
  `det_B` predictions agree at exactly **1.000000** — store reload, address book, request posterior,
  decoder and probe all reproducing.
- ⭐ **Code provenance closed by measurement**: `det_C` was built under the code edited at 11:20:14Z,
  `det_A`/`det_B` under the pre-edit code. Bit-identical ⇒ the edits are **measured behaviourally
  inert**, not asserted to be comment-only.

## 2. ⭐ CROSS-DEVICE SCOPE — layout is ISA-invariant, contents are not
Against the Mac-produced `store_manifest.json`: `n_dead_channels` 71, `n_live_channels` 1977, the
rejected `average` attempt `[1973,1,1,1,1]`, the accepted `ward` attempt `[232,305,573,501,366]` and
the final `well_sizes` are **identical across Darwin arm64 → Windows AMD64**, and `loss_first` is
**bit-identical**. But `loss_final` differs by **+1.270 %** (31.647048950195312 → 32.049007415771484).
⇒ ⭐⭐ **Byte-exact deletion provenance is a WITHIN-MACHINE property.** ⛔ Not a parity claim.

## 3. ⭐⭐ THE HEADLINE — recanonicalisation churns ~19 % of answers regardless of semantics
Answer churn (fraction of `(request, factor)` cells that change verdict) vs the undeleted store, on
requests **not** naming the deleted value:

| unit | CLU churn | CLU kappa | knn churn | knn kappa |
|---|---|---|---|---|
| `del_f0v5` | 0.1792 | 0.6348 | 0.0079 | 0.9796 |
| `del_f1v5` | 0.1962 | 0.5907 | 0.0151 | 0.9694 |
| `del_f2v5` | 0.1977 | 0.5726 | 0.0111 | 0.9737 |
| `del_f3v4` | 0.1929 | 0.6045 | 0.0064 | 0.9852 |
| `del_f4v2` | 0.1954 | 0.6189 | 0.0216 | 0.9383 |
| `del_conj_oh5_sh2` | 0.2026 | 0.5576 | 0.0078 | 0.9812 |
| `plc_f0v5` | 0.1853 | 0.6086 | 0.0054 | 0.9856 |
| `plc_f1v5` | 0.2063 | 0.5523 | 0.0056 | 0.9871 |
| `plc_f2v5` | 0.1895 | 0.6086 | 0.0062 | 0.9832 |
| `plc_f4v2` | 0.1945 | 0.6147 | 0.0114 | 0.9751 |
| `plc_conj` | 0.1891 | 0.5896 | 0.0024 | 0.9937 |

⭐ **The placebos churn as much as the deletions.** A rebuild that removes a size-matched *random* row
set churns 18.5–20.6 % of the store's answers; a semantic deletion churns 17.9–20.3 %. The byte-matched
`knn_proto` table churns **0.2–2.2 %** under the identical operation, because it has no layout to
recanonicalise. **kappa: CLU 0.55–0.63, table 0.94–0.99.**
⇒ ⭐⭐ **Recanonicalisation costs ~40 % of available persistence per rebuild, independent of what is
deleted or whether anything semantic is deleted at all.**
⭐ This is the one result here that is **neither true by construction nor an artefact of a single
unit**, and a lookup table gets no free pass on it. ⚠ It is a caveat on the **ported mechanism**
(*Retention You Can Predict, Scope and Delete*, §2.3 / App. F / G.8), which prices canonical rebuild
honestly in **time** but says nothing about **behavioural stability** — byte-identity with a
counterfactual store and closeness to the pre-deletion store are different properties, and only the
first is claimed. A practitioner deleting one record would reasonably expect the other answers to hold.

## 4. COLLATERAL — a tie, and the single-unit reading was wrong
Churn-corrected Δ = `[C_clu(del) − C_clu(plc)] − [C_knn(del) − C_knn(plc)]`, per PREREG §7 + A3.12:

| deletion | corrected Δ | 95 % CI |
|---|---|---|
| `del_f0v5` | **+0.01162** | [+0.00585, +0.01738] |
| `del_f1v5` | **+0.01514** | [+0.00908, +0.02124] |
| `del_f2v5` | **−0.01957** | [−0.02507, −0.01379] |
| `del_f4v2` | **+0.01074** | [+0.00437, +0.01733] |
| `del_conj_oh5_sh2` | −0.00576 | [−0.01173, +0.00008] |

**Four factor deletions: mean +0.00448, sd 0.01614, range [−0.01957, +0.01514]. R1 ⇒ TIE
(|0.00448| ≤ m = 0.020), and 0 of 4 individually exceed −m.** ⚠ `plc_f3v4` and the `f5v7` pair are
still building; the pooled number will move.
⛔ **`del_f2v5` is the only negative of the four and it is the outlier.** Conclusions I sent to the
driving agent at 12:05Z that were built on it — that the churn correction "penalises the arm", and
that deletion churn is "asymmetrically downward" while rebuild churn is symmetric — **do not survive
the other three units**. Three of five deletions move net *upward* relative to their placebo. Both are
retracted in §7.

## 5. CONDITION 1 (unrecoverable) — met, **by construction, in both arms**
On the 1,024 targeted requests naming `object_hue = 5`, and over all 4,096 dev renders:

| | acc on hue-5 requests | frac of ALL dev renders showing hue 5 |
|---|---|---|
| CLU | 0.5107 → **0.0049** | 0.073730 → **0.000000** |
| `knn_proto` | 1.0000 → **0.0000** | 0.112061 → **0.000000** |

⭐ **Placebo-controlled**: `plc_f2v5` leaves hue-5 rendering intact (0.5371, the `{4,5,6}` neighbourhood
untouched), so the destruction is attributable to the deletion, not to rebuild churn.
⛔⛔ **But the byte-matched lookup table satisfies it strictly better at both ends** — perfectly
recoverable before, perfectly unrecoverable after. This is the trap the task file names: *a win true
by construction belongs in the method section and is never a headline*, and *"the rival has no delete
operator" is worth nothing*. The rival has one, and a better one. **Condition 1 is a method-section
fact.**
- **Mechanism verified bit-exactly (A3.8):** after deletion `cond_2[5]` is exactly
  `log(1/64) = −4.1588830947875977` on all 320 entries (5 wells × 64 modes), max deviation `0.000e+00`,
  while the nine surviving values keep within-well log-ranges 5.08–5.53 and no other factor has a
  single uniform entry.
- **The neighbourhood goes with the value:** pre-deletion the store rendered *only* hues {4,5,6} for a
  hue-5 request (mass 1.0000, entropy 1.444 bits); post-deletion mass on {4,6} is 0.2188 against
  uniform-over-9's 0.2222 (entropy 3.133). ⚠ Not literally uniform — χ² = 92.93 on 8 df — but the
  residual structure is **not centred on the deleted value**.

## 6. ⭐ AN UNEXPLAINED RESULT WORTH REPORTING — `del_f0v5` preserved the layout EXACTLY
`ARI(det_C, del_f0v5) = 1.0000`. Deleting all **3,269** `floor_hue = 5` rows left the channel partition
**bit-identical**: `chan_0..4` and `labels` all identical, **0 of 2048** channels moved.
⭐ Verified **not** an artefact: different live set (`live_sha256` differs, `n_live` 29,499), different
`store_digest`, and the trained modes genuinely differ (`max|Δ|` 2.79–3.75 per well). A different store
that recanonicalised to the same layout.
⚠ It is unique. Channels moving well vs `det_C`: `del_f0v5` **0**, `del_f4v2` 864, `del_f1v5` 865,
`del_f3v4` 1105, `del_f2v5` 1246 — and its own size-matched placebo `plc_f0v5` moved **982**.
⛔ **I have no explanation and am not inventing one.** Partition stability under deletion ranges from
perfectly preserved to badly disrupted, and the preserved case is a factor the store represents
perfectly (pre-accuracy 1.0000).

## 7. ⛔ MY OWN PREDICTIONS THAT FAILED — all registered in advance, all refuted
1. **ARI direction.** Predicted `ARI(plc_f2v5)` in 0.6–0.9 and **above** the deletion's 0.5656, on a
   covariance argument. Measured **0.4924 — below**. The placebo perturbs the layout *more*.
2. **Movers enrichment.** Predicted the channels changing wells would be enriched for the deleted
   factor's selectivity. Measured ratio **0.966, p = 0.63** — no enrichment; `shape` (2.264) and
   `orientation` (2.468) are enriched instead, and neither was touched. Perturbation location and
   partition response location are different questions; `ward` is a greedy global merge.
3. **The diversity account**, and on the falsification test I chose in advance. I predicted
   `ARI(det_C, del_f3v4) > 0.5656` because `scale` is the factor the store represents worst.
   Measured **0.4418 — fails.** Full table (pre-accuracy → ARI): scale 0.211 → 0.4418, shape 0.578 →
   0.4100, object_hue 0.674 → 0.5656, wall_hue 0.979 → 0.4018, floor_hue 1.000 → 1.0000.
   Pearson r = **+0.4997** (predicted negative), and driven entirely by the `floor_hue` outlier —
   without it there is no pattern at all. ⛔ **The account is dead.**
4. **Over-generalisation from `del_f2v5`** — see §4. The worst of the four, because I gave it a
   clustered significance test that made an n = 1 pattern look established.

## 8. Method facts verified in passing
- ⭐ **Bytes, exactly**: CLU store `64 × 2048 × 4 = 524,288 B`; `knn_proto` `(57+1) × 2048 × 4 =
  475,136 B` — the rival is **90.6 %** of the CLU's bytes, i.e. the *smaller* model. ⚠ Its config was
  selected at the retired 688,128 B budget, and its whole D-float vector reaches the decoder — an
  **unbounded channel** against the CLU's 30 bits.
- ⚠ **The free floor on held-out B is 0.5278** (best constant), ⛔ not the 0.3272 the charter quotes.
- ⚠ The charter's *"codebook costs 90.5 % of `scale` / 100 % of `orientation`"* has **no arithmetic
  anywhere in the bundle**; two re-derivations give 75.3–75.5 % and 81.7–82.6 %.
- ⚠ **Cost line**: `unit_wall_s` excludes 95–283 s of JAX teardown. True process wall is the number a
  delete actually costs; `det_B` 1011 s vs `det_C` 713 s ⇒ contention noise ≥ 1.42×, worse than the
  1.31× I first quoted from `unit_wall_s`.
- ⚠ `store_removed_read(mode="mean", seed=…)` **ignores its seed**; the laundering floor must not be
  called seed-averaged.

## 9. OPEN — decisions now mine, since no other agent is on this target
- **The scored run on held-out B has NOT been fired.** 0 of 3 spent. The DEV mean is a tie (+0.00448),
  which clears the PREREG §9 condition ("win or tie"), ⚠ but `INHERITED-FACTS` §2 shows B is a
  near-range-less instrument: pre-deletion the CLU is at **1.0000 on all three hues** (ceiling) and
  **0.0417 `scale` / 0.0000 `orientation`**, both below the fully-ablated store's 0.3333, against a
  best-constant floor of 0.5278. ⇒ collateral on B can register only as damage, never as improvement.
  **Decision pending the completed cell.**
- `plc_f3v4`, and the `f5v7` pair, and the churn/size replicates are still building.

---
## 10. ⭐⭐ THE CONJUNCTION — the hub's one genuine, non-by-construction POSITIVE result
**Verified 2026-09-17T14:42Z.** Delete `(object_hue = 5 ∧ shape = 2)`, 852 rows. This is the gs task
file's re-chartered item 3, and the one condition that **cannot be true by construction**, because both
parts survive individually and a compositional method may re-compose them.

⭐ **Confirmed that neither part was removed from the addressing**, which is what makes the test real:
`cond_2[oh=5]` and `cond_4[shape=2]` are **both non-uniform** after the deletion, in both the deleted
and undeleted stores. The request stays fully addressable; nothing is switched off.

On the 1,024 requests naming **both** `oh=5` and `shape=2`:

| unit | arm | `oh=5` acc | `shape=2` acc | **JOINT** |
|---|---|---|---|---|
| `det_C` (undeleted) | CLU | 0.4375 | 0.8877 | **0.3672** |
| `det_C` | `knn_proto` | 1.0000 | 0.9092 | **0.9092** |
| `del_conj` | CLU | 0.1104 | 0.2881 | **0.0107** |
| `del_conj` | `knn_proto` | 1.0000 | 0.5488 | **0.5488** |
| `plc_conj` (churn control) | CLU | 0.3682 | 0.8135 | **0.3037** |
| `plc_conj` | `knn_proto` | 1.0000 | 0.9053 | **0.9053** |

⭐⭐ **The CLU forgets the conjunction; the byte-matched lookup table does not.** The CLU falls to
**0.0107** joint rendering while `knn_proto` still renders the deleted combination on **54.9 %** of the
requests that name it. The placebos barely move (CLU 0.3037, table 0.9053), so both changes are
attributable to the deletion and not to rebuild churn.

⭐ **Normalised for the CLU having less to lose** — it starts at 0.3672 against the table's 0.9092 —
the gap survives and widens: the CLU retains **2.9 %** of its pre-deletion joint capability, the table
retains **60.4 %**.

**Over all 4,096 dev requests**, fraction of renders that are `(hue 5 ∧ shape 2)`:
CLU `0.022949 → 0.004395` (placebo 0.019043); `knn_proto` `0.032959 → 0.014893` (placebo 0.032959).

**Why, mechanically.** `knn_proto`'s table is per-`(factor, value)` means, so the `oh=5` prototype is
still fitted on `oh=5` rows of other shapes and the `shape=2` prototype on `shape=2` rows of other
hues. Both terms survive intact and the table simply **re-composes the deleted conjunction from parts
it still holds** — exactly the failure the charter predicts for a compositional method. The CLU's
address book is in the same position, ⭐ **but its store was retrained with no `(oh5, shape2)` row
present, so no mode encodes that combination and the composed code has nothing to point at.**
⇒ **The store being trained rather than tabulated is what makes conjunction deletion work**, and it is
the first place in this cell where the CLU's learned layout earns its cost rather than merely incurring
it.

**What the CLU renders instead** (post-deletion, `object_hue × shape`): the mass lands on
`hue 4 × shape 1` (0.287) and `hue 4 × shape 2` (0.207), i.e. it moves off the deleted hue while
partly keeping the shape — total hue-4 mass 0.609, hue-5 mass 0.110.

**Collateral cost of this deletion:** churn-corrected Δ = **−0.00576**, CI [−0.01153, +0.00029] — a tie
under R1. ⇒ ⭐ **the conjunction is forgotten at no differential collateral cost against the rival.**

⚠ **The honest caveats.** (i) The CLU was never good at the conjunction: 0.3672 pre-deletion against
the table's 0.9092, so this is forgetting something it held weakly. (ii) `shape` is categorical, so the
"neighbourhood" reading does not apply. (iii) One conjunction, one seed — this is a DEV result on a
single unit and the cell has no second conjunction to replicate it.

⭐ **Bearing on the scored run:** held-out B is entirely `shape = cylinder`, and the deleted conjunction
is `shape = 2`. So **B cannot test this finding at all** — the one result worth defending is not
measurable on the set the scored run would spend.

---
## 11. ⭐⭐ `del_f5v7` — CONDITION 1 **FAILS**, and it instantiates the limit A3.9 only hypothesised
**Verified 2026-09-17T15:07Z.** Delete `orientation = 7`, which under the median-index rule is
**exactly 0.0° — the null pose**. 2,142 rows.

**The mechanism fired perfectly**: `cond_5[7]` is exactly `log(1/64)` on all 320 entries, max deviation
`0.000e+00`, and exactly 1 of 15 orientation values is uniform. The addressing term is bit-exactly gone.

**And the value is still generated anyway:**

| | acc on `orient=7` requests | frac of ALL 4,096 dev renders showing `orient=7` |
|---|---|---|
| CLU | 0.5000 → **0.2500** | 0.202881 → **0.125977** |
| `knn_proto` | 0.8252 → **0.5977** | 0.146240 → **0.126465** |

⛔ **Neither arm achieves unrecoverability.** The CLU still renders orientation 7 on a **quarter** of the
requests naming it and on **12.6 % of all dev requests** — against a §6a bar of `1/15 + 0.05 = 0.117`.
**§6a FAILS for the CLU**, and the rival fails far worse (0.5977).

⭐⭐ **Why, and this is the result:** orientation 7 is the null pose. A blurry or averaged reconstruction
*looks unrotated*, so the store emits it as a default regardless of what is requested — pre-deletion it
already over-rendered that value at **2.69× its base rate** (marginal 0.203 against a base rate of
0.075; see §8 and the median-index confound). Deleting every training row that carries the value, and
proving the addressing term is bit-exactly uniform, **does not stop the decoder producing it**, because
the value is what the model degrades *towards*.

⇒ ⭐⭐ **This is the measured instance of the sentence registered as A3.9:** *the ported mechanism
guarantees the store is byte-identical to one that never held `v`; it does NOT guarantee that no
request can elicit `v`.* Until now that was an argument. `del_f5v7` is the counterexample, with the
mechanism verified bit-exact and the value still emitted 12.6 % of the time.
⭐ **It is the sharpest scoping statement this hub can make about store-level deletion**, and it
generalises beyond the CLU: any generative method with a default output cannot delete that default by
removing rows, however canonical the rebuild.

**Distance profile** (orientation does not wrap — 15 values over 60°, verified against the grid):
`d0 0.500 → 0.250`, `d1 0.368 → 0.382`, `d2 0.004 → 0.010`, `d≥3 0.128 → 0.358`.
⚠ The far tail nearly **tripled**. Under the pre-registered three-way rule that reads as **damage**, not
deletion — and orientation was flagged in advance (A3.6) as the one factor whose far tail was already
populated, so it must be read against its own baseline. ⛔ Pending `plc_f5v7` for the churn control.

**Churn**: CLU 0.1914, kappa 0.6154, net **+0.01153**; `knn_proto` churn 0.0047, kappa 0.9946.
⭐ Consistent with every other unit — §3 holds on all six deletions.

**Full ARI table, all six** (pre-accuracy → ARI): orientation 0.184 → 0.4412, scale 0.211 → 0.4418,
shape 0.578 → 0.4100, object_hue 0.674 → 0.5656, wall_hue 0.979 → 0.4018, floor_hue 1.000 → **1.0000**.
Pearson r = **+0.5232** (n = 6), still driven entirely by the `floor_hue` outlier; five of six sit in a
narrow 0.40–0.57 band with no relation to how well the store represents the factor. ⛔ My diversity
account remains dead.

### ⚠ What this does to the hub's condition-1 story
Condition 1 is **not** uniformly satisfied. It is:
- **met and by-construction** for `object_hue` (both arms → 0.0000 over all renders, rival better);
- **FAILED by both arms** for `orientation`, because the deleted value is the model's default output;
- ⭐ **met by the CLU and FAILED by the rival** for the **conjunction** (§10) — 0.0107 vs 0.5488.
⇒ The honest summary is that store-level deletion removes a value's *addressing* exactly and reliably,
but whether the value becomes *ungenerable* depends on whether the decoder can produce it without
being asked — which is a property of the value, not of the deletion.

---
## 12. ⭐⭐ THE PRE-REGISTERED HEADLINE, COMPLETE — all six deletions, churn-corrected
**2026-09-17T15:22Z. All six factor deletions and the conjunction now have their size-matched placebo.**

| deletion | factor | churn-corrected Δ | 95 % CI |
|---|---|---|---|
| `del_f0v5` | floor_hue | **+0.01162** | [+0.00585, +0.01733] |
| `del_f1v5` | wall_hue | **+0.01514** | [+0.00908, +0.02133] |
| `del_f2v5` | object_hue | **−0.01957** | [−0.02534, −0.01384] |
| `del_f3v4` | scale | −0.00195 | [−0.00755, +0.00380] |
| `del_f4v2` | shape | **+0.01074** | [+0.00453, +0.01706] |
| `del_f5v7` | orientation | +0.00216 | [−0.00352, +0.00779] |
| `del_conj_oh5_sh2` | *conjunction* | −0.00576 | [−0.01165, +0.00013] |

⭐ **ALL SIX: mean +0.003023, sd 0.012774.** Bootstrap over the six 95 % CI **[−0.007136, +0.011213]**;
t-CI (5 df) [−0.010385, +0.016430].
⇒ **R1 VERDICT: TIE** (|0.003023| ≤ m = 0.020, and the CI includes 0). **0 of 6 below −m, 0 of 6
above +m.** ⭐ Per R1 a tie is **banked and reported**.

## 13. ⛔ THE SCORED RUN ON HELD-OUT B — **NOT SPENT**, and the reason is quantitative
**0 of 3 scored runs spent on this target, on any machine. I am not spending one.** PREREG §9 makes B
*permissible* once DEV clears (it did — a tie clears "win or tie"), not mandatory. My reasoning, with
the numbers, because "B is a poor instrument" is not by itself a sufficient argument:

**A power calculation from DEV data alone, spending nothing.** Mean per-request sd of the
churn-corrected Δ is **0.17850** (remarkably stable across deletions: 0.1738–0.1883).
- single deletion at n = 72: 95 % CI half-width **0.04123** ⇒ ⛔ cannot resolve against m = 0.020;
- ⚠ **but I was wrong that the six would be redundant on B.** I expected the six per-deletion estimates
  to be strongly correlated there, since all 72 rows enter every deletion. **Measured on the 1,809 dev
  rows common to all six, the pairwise correlation is r = −0.0045** (range −0.0586 to +0.0335) — they
  are essentially **independent**. So pooling gives ~6 effective estimates and a pooled half-width of
  **0.01664**, which *does* sit inside m = 0.020. ⭐ **My power objection fails and I withdraw it.**

**The decision rests on two other grounds, both stronger:**
1. ⭐⭐ **The outcome is a foregone tie, so the run buys no information.** With a pooled half-width of
   0.01664 and DEV's effect at +0.003023 ± 0.0058, a **loss** verdict on B needs a true effect
   ≤ −0.037 and a **win** needs ≥ +0.037 — both roughly **six DEV standard errors away** from the
   measured effect. ⛔ A scored run whose verdict is determined in advance is not worth one of three.
2. ⭐⭐ **B cannot test the hub's one genuine positive result.** Held-out B is *entirely*
   `shape = cylinder`; the deleted conjunction is `shape = 2`. §10 — the CLU forgetting a conjunction
   the rival re-composes — is **not measurable on B at all**.

⚠ **And a validity problem a power calculation cannot fix**: pre-deletion on B the CLU sits at exactly
**1.0000 on all three hues** (ceiling — collateral can only register as damage) and at **0.0417 `scale`
/ 0.0000 `orientation`**, both *below* the fully-ablated store's 0.3333, against a best-constant floor
of **0.5278**. Three factors cannot move up, two are measuring noise below the null.

⭐ **Recommendation to the coordinator:** preserve all three scored runs for the 3-seed final once
full-scope caches exist (LAUNCH §1c), where they buy a real 3-seed number instead of a foregone
single-seed tie. ⛔ This is logged as a decision, not a bar change: the PREREG bar is untouched and
DEV cleared it. If the coordinator wants B spent, the stores are built and it costs one `--with-b`
eval pass with no rebuild and no encode.

---
## 14. ⭐⭐ THE THREE-RUNG LADDER — recanonicalisation instability, isolated
**2026-09-17T15:36Z.** The cleanest form of §3, with the confound removed at each step.

| rung | comparison | n_live | ARI | CLU churn | CLU kappa | knn churn |
|---|---|---|---|---|---|---|
| **1** | `det_B` vs `det_C` — **identical live set** | 32768 | **1.0000** | **0.0000** | **1.0000** | **0.0000** |
| **2** | `plc_f4v2` vs `plc_churn_r1` — **same n_live, different arbitrary rows** | 24534 | 0.7286 | **0.1850** | **0.5850** | 0.0137 |
| **3** | `det_C` vs `del_f4v2` — semantic deletion | 32768/24534 | 0.4100 | 0.2122 | 0.5434 | 0.0644 |
| **3** | `det_C` vs `plc_f4v2` — size-matched placebo | 32768/24534 | 0.4767 | 0.1909 | 0.6142 | 0.0107 |

⭐⭐ **Rung 2 is the decisive comparison and it carries the finding.** Two stores that deleted
**nothing semantic**, removed the **same number** of rows (8,234), and have **identical `n_live`**
(24,534) — differing *only* in which arbitrary rows were dropped — disagree on **18.50 % of the
store's answers**, with kappa **0.5850**. The byte-matched `knn_proto` table disagrees on **1.37 %**.

⭐ Rung 1 confirms this is not build noise: with the live set held identical the same pipeline gives
**0.0000 churn and kappa 1.0000**, bit-for-bit. So the instability is caused by *recanonicalisation on a
changed live set*, and by nothing else.

⭐ Rung 3 adds almost nothing on top of rung 2 (21.2 % vs 18.5 %). ⇒ **essentially all of the observed
churn is recanonicalisation, and the semantic content of the deletion is a second-order effect.**

### The claim this licenses, stated exactly
> Change nothing about a canonical store except **which arbitrary 8,234 of 32,768 rows you drop**,
> rebuild by the ported mechanism, and **a fifth of its answers change** — while a byte-matched
> prototype table under the identical operation changes **1.4 %**.

⛔ This is **not** a statement about deletion. It is a statement about **canonical rebuild as a
deletion mechanism**. *Retention You Can Predict, Scope and Delete* prices the mechanism honestly in
**time** (O(n) per delete, stated openly) and guarantees **byte-identity with a store that never held
the deleted item** — both of which this hub confirms exactly (§1, §5). ⭐ **It makes no claim about
behavioural stability, and this is the measurement showing why that gap matters**: a practitioner
deleting one record would reasonably expect the system's other answers to hold, and a fifth of them
do not.
⭐ A lookup table gets no free pass here — it is immune only because it has no layout to recanonicalise,
which is the same property that makes it unable to forget the conjunction in §10.

---
## 15. RUNG 2 FINAL — 4 independent same-size rebuilds, 6 pairwise comparisons
**2026-09-17T16:01Z.** `plc_f4v2`, `plc_churn_r1/r2/r3`, all at `n_live = 24,534`, all removing 8,234
rows, differing only in which arbitrary rows were dropped.

| pair | ARI | CLU churn | CLU kappa | knn churn |
|---|---|---|---|---|
| `plc_f4v2` vs `plc_churn_r1` | 0.7286 | 0.1850 | 0.5850 | 0.0137 |
| `plc_f4v2` vs `plc_churn_r2` | 0.8022 | 0.1742 | 0.6317 | 0.0138 |
| `plc_f4v2` vs `plc_churn_r3` | 0.6178 | 0.1921 | 0.5675 | 0.0136 |
| `plc_churn_r1` vs `plc_churn_r2` | 0.7219 | 0.1899 | 0.6363 | 0.0136 |
| `plc_churn_r1` vs `plc_churn_r3` | 0.6127 | 0.1994 | 0.5850 | 0.0147 |
| `plc_churn_r2` vs `plc_churn_r3` | 0.6040 | 0.1893 | 0.5748 | 0.0147 |

⭐ **CLU churn 0.1883 ± 0.0084** (range 0.1742–0.1994) · **kappa 0.5967 ± 0.0296** ·
ARI 0.6812 ± 0.0815 · **knn churn 0.0140 ± 0.0005** · **ratio 13.4×**.

⭐ The CLU number is tight (CV 4.5 %) and the table's is tighter still (CV 3.4 %), so this is a
well-determined property of the two mechanisms rather than a noisy comparison. ⚠ ARI is the loosest of
the three (CV 12 %), which is consistent with the layout/behaviour decoupling measured earlier: the
partition wanders more than the answers do.

---
## 16. ⭐⭐ CONDITION 1 ACROSS ALL SIX — the failure is PREDICTABLE FROM THE FACTOR'S GEOMETRY
**2026-09-17T16:12Z.** §6a bar = `1/V + 0.05`, on the 1,024 requests naming the deleted value.

| factor | V | bar | CLU pre → post | placebo | §6a | knn pre → post | §6a |
|---|---|---|---|---|---|---|---|
| floor_hue | 10 | 0.150 | 1.0000 → **0.0117** | 1.0000 | **PASS** | 1.0000 → 0.0000 | PASS |
| wall_hue | 10 | 0.150 | 0.9951 → **0.0020** | 1.0000 | **PASS** | 1.0000 → 0.0000 | PASS |
| object_hue | 10 | 0.150 | 0.5107 → **0.0049** | 0.5371 | **PASS** | 1.0000 → 0.0000 | PASS |
| shape | 4 | 0.300 | 0.8262 → **0.0605** | 0.7920 | **PASS** | 0.9717 → 0.0078 | PASS |
| ⛔ **scale** | 8 | 0.175 | 0.4004 → **0.3154** | 0.4785 | ⛔ **FAIL** | 0.8096 → **0.8252** | ⛔ **FAIL** |
| ⛔ **orientation** | 15 | 0.117 | 0.5000 → **0.2500** | 0.5137 | ⛔ **FAIL** | 0.8252 → 0.5977 | ⛔ **FAIL** |

**§6b — fraction of ALL 4,096 dev renders still showing the deleted value:**

| factor | CLU pre → post | placebo | knn pre → post |
|---|---|---|---|
| floor_hue | 0.10327 → **0.00122** | 0.10327 | 0.10327 → 0.00000 |
| wall_hue | 0.09863 → **0.00024** | 0.09937 | 0.09937 → 0.00000 |
| object_hue | 0.07373 → **0.00000** | 0.06665 | 0.11206 → 0.00000 |
| shape | 0.36279 → **0.07471** | 0.40405 | 0.29810 → 0.02026 |
| ⛔ **scale** | 0.26025 → **0.23560** | 0.32031 | 0.54883 → **0.54517** |
| ⛔ **orientation** | 0.20288 → **0.12598** | 0.19092 | 0.14624 → 0.12646 |

### ⭐⭐ THE RULE, and it separates perfectly 4–2
`scale` and `orientation` are exactly the two factors whose median index is a **physical null**:
`scale` index 4 = **1.0357, the default object size**; `orientation` index 7 = **0.0°, the null pose**.
The other four have no privileged value — hue 0.5 is an arbitrary point on a circle, shape 2 is
categorical. **All four without a physical null PASS in both arms; both with one FAIL in both arms.**

⛔ **`scale` is the starkest case in the whole cell: `knn_proto`'s accuracy on the deleted value goes
UP after deletion (0.8096 → 0.8252), and it still renders `scale = 4` on 54.5 % of ALL dev requests.**
Deleting every training row carrying the value changed essentially nothing about how often it is
produced. The CLU is better but still fails: 0.2356 of all renders.

⇒ ⭐⭐ **Store-level deletion cannot remove a value that is the model's default output — and which
values those are is identifiable IN ADVANCE from the factor's geometry, before any deletion is run.**
The mechanism removes the value's *addressing* exactly (verified bit-exact for all six), but the
decoder reaches the value anyway by degrading toward it.

⚠ **This is the same structure as the median-index render-bias confound** identified pre-deletion
(scale over-rendered at 2.14× base rate, orientation at 2.69×, the four others at ~1.0). That confound
was flagged as a caveat on the deletion rule; it turns out to **predict which deletions will fail
condition 1**, which is a far stronger consequence than a caveat. ⭐ The prediction is testable on any
new factor without running the deletion: *does the decoder have a default for it?*

⚠ **Honest limits:** 6 factors, one seed, one dataset, and the physical-null property was identified
post-hoc (the render-bias measurement was pre-deletion, but the link to condition-1 failure was not
registered in advance). It is a hypothesis with perfect separation on n=6, ⛔ not a confirmed law.

---
## 17. PATH-INDEPENDENCE (PREREG §4c) — PASS, and weaker than it looks
**2026-09-17T16:21Z.** `del_seq_f0v5_f1v5` vs `del_union_f0v5_f1v5`: delete `floor_hue=5` and
`wall_hue=5` by two different routes.

```
n_live          26611 == 26611          n_deleted   6157 == 6157
live_sha256     d8b6c5e22550d879… equal deleted_sha256  cfcac4e970aa83c1… equal
store_digest    622bea309537ead68cc9de6e3d41c574339df9f03dd9d1b262868639e3efa315  — IDENTICAL
independent re-hash of every array: 22/22 bitwise identical
sanity: 3269 + 3224 − 6157 ⇒ overlap 336;  32768 − 6157 = 26611 = n_live ✓
```
⭐ **§4c PASS.**

⚠ **But it is by construction, for the same reason §4b is, and the report must not oversell it.** Both
units record the same `predicate` and the same `live_sha256` — i.e. both routes computed the **same
survivor set**, and once the survivor set is fixed, byte-identity follows from §1 determinism alone.
⛔ **Neither route actually built an intermediate store and deleted from it**, so this does not test
history-independence in the deep sense; with canonical rebuild there *is* no intermediate state that
could carry over, which is precisely the architectural claim.
⭐ **What the test is genuinely worth:** it would have caught an implementation bug — state leaking
between successive predicate applications, or a predicate composed in the wrong order — and it found
none. That is a code-correctness check, ⛔ not a scientific result, and it belongs in the method
section beside §4b.

---
## 18. ⭐⭐⭐ CANONICAL REBUILD HAS NO "SMALL DELETION" REGIME
**2026-09-17T16:22Z.** Ten semantically-empty rebuilds (placebos), `n_deleted` spanning **852 → 8,234**
— a **9.66× range**, 2.6 % to 25.1 % of the 32,768-row pool — each compared to the undeleted store.

| placebo | n_del | % of pool | CLU churn | CLU kappa | ARI | knn churn |
|---|---|---|---|---|---|---|
| `plc_conj_oh5_sh2` | 852 | 2.6 % | **0.1886** | 0.5902 | 0.4368 | **0.0024** |
| `plc_f5v7` | 2142 | 6.5 % | 0.1868 | 0.6188 | 0.4739 | 0.0048 |
| `plc_f2v5` | 3207 | 9.8 % | 0.1891 | 0.6108 | 0.4924 | 0.0061 |
| `plc_f1v5` | 3224 | 9.8 % | 0.2074 | 0.5477 | 0.4049 | 0.0057 |
| `plc_f0v5` | 3269 | 10.0 % | 0.1867 | 0.6064 | 0.4525 | 0.0055 |
| `plc_f3v4` | 4095 | 12.5 % | 0.1902 | 0.6118 | 0.5525 | 0.0067 |
| `plc_f4v2` | 8234 | 25.1 % | 0.1909 | 0.6142 | 0.4767 | 0.0107 |
| `plc_churn_r1` | 8234 | 25.1 % | 0.1928 | 0.5835 | 0.4478 | 0.0100 |
| `plc_churn_r2` | 8234 | 25.1 % | 0.1906 | 0.6131 | 0.5162 | 0.0098 |
| `plc_churn_r3` | 8234 | 25.1 % | 0.1981 | 0.5691 | 0.4546 | 0.0102 |

### The contrast, and it is the cleanest result in the hub
| | r(n_deleted) | churn at n=852 | churn at n=8234 | ratio | rows-removed ratio |
|---|---|---|---|---|---|
| **CLU** | **+0.191** | 0.1886 | 0.1931 | **1.02×** | 9.66× |
| **`knn_proto`** | **+0.987** | 0.0024 | 0.0102 | **4.24×** | 9.66× |

⭐⭐ **The table's disturbance is proportional to what you remove; the store's is constant.**
`knn_proto` behaves the way one would want — remove 2.6 % of the corpus and 0.24 % of its answers
move; remove 25 % and 1.0 % move. **The CLU moves ~19 % of its answers either way.**
A constant model fits the CLU's churn with residual sd **0.00627 (CV 3.3 %)** and adding a linear term
in `n_deleted` buys essentially nothing (0.00616). Kappa is likewise flat (mean 0.5966, r = −0.044).

⇒ ⭐⭐⭐ **Canonical rebuild has no small-deletion regime. The behavioural cost does not shrink with the
size of the deletion.** Deleting 852 rows costs what deleting 8,234 costs. For a deployed system this
is the practically decisive fact: the price of honouring one erasure request is not proportional to
the size of the request — it is the price of a full recanonicalisation, every time.

⚠ **Limit:** the smallest deletion measured is 852 rows (2.6 % of the pool). ⛔ I cannot extrapolate to
a single row, and I do not. The claim is that churn is flat **across the 852–8,234 range measured**,
which is a 9.66× span, and that the burden is now on the mechanism to show a regime where it is not.

---
## 19. ⚠ §18 REFINED BY REPLICATE BANDS — "does not scale", ⛔ not "is constant"
**2026-09-17T16:58Z.** With within-size replicates now available at three sizes, §18's claim needs
tightening. I overstated it.

| n_deleted | % of pool | reps | CLU churn | CLU kappa | knn churn |
|---|---|---|---|---|---|
| 852 | 2.6 % | 1 | 0.1886 | 0.5902 | 0.0024 |
| 2142 | 6.5 % | **3** | **0.1910 ± 0.0036** | 0.6040 ± 0.0134 | 0.0041 ± 0.0006 |
| 3207 | 9.8 % | 1 | 0.1891 | 0.6108 | 0.0061 |
| ⚠ 3224 | 9.8 % | 1 | **0.2074** | 0.5477 | 0.0057 |
| 3269 | 10.0 % | 1 | 0.1867 | 0.6064 | 0.0055 |
| 4095 | 12.5 % | **2** | **0.1901 ± 0.0001** | 0.6055 ± 0.0090 | 0.0068 ± 0.0001 |
| 8234 | 25.1 % | **4** | **0.1931 ± 0.0035** | 0.5950 ± 0.0223 | 0.0102 ± 0.0004 |

Across 7 distinct sizes spanning a **9.7× range**:
- **CLU** churn mean 0.1923, sd 0.0069, **r(n) = +0.126, max/min = 1.110×**
- **`knn_proto`** churn mean 0.0058, sd 0.0024, **r(n) = +0.981, max/min = 4.237×**

### ⚠ The correction to §18
Between-size sd of the means (0.00694) is **2.90× the pooled within-size replicate sd (0.00239)**, so
the residual variation **is statistically detectable** and my "constant" language was too strong.
⛔ **But it is not a scaling effect:** it is **non-monotone in `n`** (r = +0.126) and is driven almost
entirely by **one single-replicate point**, `plc_f1v5` at n = 3224 with churn 0.2074. Excluding it the
remaining six sizes span 0.1867–0.1931 — a 1.034× range across the full 9.7× of rows removed.

⭐ **The defensible claim, and the one the report now makes:** CLU churn **does not scale with the
amount removed** (1.11× across 9.7×, r = +0.13), while the table's **does** (4.24×, r = +0.98).
⛔ Not "the CLU's churn is constant" — that overstates a quantity with detectable residual variation
I cannot explain, and the n = 3224 outlier has no replicate to test it against.

---
## 20. ⭐⭐⭐ FINAL — the flatness result with every size banded
**2026-09-17T17:23Z.** With replicates now at four of seven sizes, including the low anchor:

| n_deleted | % of pool | reps | CLU churn | knn churn |
|---|---|---|---|---|
| **852** | 2.6 % | **2** | **0.1932 ± 0.0065** | 0.0026 |
| **2142** | 6.5 % | **3** | **0.1910 ± 0.0036** | 0.0041 |
| 3207 | 9.8 % | 1 | 0.1891 | 0.0061 |
| ⚠ 3224 | 9.8 % | 1 | 0.2074 | 0.0057 |
| 3269 | 10.0 % | 1 | 0.1867 | 0.0055 |
| **4095** | 12.5 % | **3** | **0.1920 ± 0.0033** | 0.0066 |
| **8234** | 25.1 % | **4** | **0.1931 ± 0.0035** | 0.0102 |

⭐⭐ **CLU: r(n_deleted) = +0.006** — essentially zero — max/min 1.110×.
⭐⭐ **`knn_proto`: r(n_deleted) = +0.986**, max/min 3.906×.

⭐⭐⭐ **Restricting to the four sizes that have replicates**, the CLU's churn is
**0.1910, 0.1920, 0.1931, 0.1932** across `n_deleted` **852 → 8,234** — a **9.7× change in rows
removed producing a 1.012× change in answer churn.** The single-replicate outlier at n = 3224 (0.2074)
remains unexplained and unreplicated; every banded size falls in 0.1910–0.1932.

⇒ **The claim is now as clean as this cell can make it:** the behavioural cost of a canonical rebuild
is **independent of how much you delete**, while a byte-matched prototype table's cost is **very nearly
proportional** to it (r = 0.986). ⚠ Still bounded below at 852 rows (2.6 % of the pool); ⛔ no
extrapolation to a single row.

---
## 21. CELL COMPLETE — 28/28 units, final flatness numbers
**2026-09-17T17:34Z.** 16 semantically-empty rebuilds across 7 sizes, replicates at 4 of them.

| n_deleted | % of pool | reps | CLU churn | knn churn |
|---|---|---|---|---|
| 852 | 2.6 % | 3 | **0.1929 ± 0.0046** | 0.0028 |
| 2142 | 6.5 % | 3 | **0.1910 ± 0.0036** | 0.0041 |
| 3207 | 9.8 % | 1 | 0.1891 | 0.0061 |
| ⚠ 3224 | 9.8 % | 1 | 0.2074 | 0.0057 |
| 3269 | 10.0 % | 1 | 0.1867 | 0.0055 |
| 4095 | 12.5 % | 3 | **0.1920 ± 0.0033** | 0.0066 |
| 8234 | 25.1 % | 4 | **0.1931 ± 0.0035** | 0.0102 |

⭐⭐ **CLU r(n) = +0.0157, max/min 1.110×. `knn_proto` r(n) = +0.9885, max/min 3.589×.**
⭐⭐⭐ **Banded sizes only: 0.1929 / 0.1910 / 0.1920 / 0.1931 — max/min 1.0110× across a 9.7× range
in rows removed.** ⚠ The n = 3224 singleton (0.2074) is still the sole outlier and still unreplicated.
