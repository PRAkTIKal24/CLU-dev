# "A BENCHMARK IS NEEDED" — the statement, drafted from the w5/w6 negatives

**w10 `spoke/physics-section-evidence`, 2026-09-21.** For the paper-writer to lift.
⭐ **Written as a REQUIREMENT LIST.** ⛔ **Not** as an argument that this programme deserves a
benchmark, and not as a complaint about the field.

Every requirement below is followed by **the measurement that forces it**, with its path. Sources:
`evidence/outputs/_thinkpad/w5/{w5-gym1,w51-gym2}/LOG.md`,
`evidence/outputs/_thinkpad/w6/{w6-potentials,w6-retention,w6-compose-physics}/LOG.md`,
`evidence/outputs/{w5,w6}-advisor-report.md`, `evidence/outputs/_thinkpad/w{5,6}/SCORED-LEDGER.md`.

⛔ **Declared scope.** `evidence/claims_matrix.md` and `evidence/negative_results.md` end at the
C2W11 fold and contain **no wave-5 or wave-6 entry**; they are not sources here. The Campaign-3
benchmark scouts (`c3-benchmark-scout.md`, `scout-benchmark-targeting.md`,
`hepa-benchmark-scout.md`) survey a **different campaign and a different object** and are **declared
NOT USED**. ⚠ **Nothing below asserts that no such benchmark exists in the literature** — no scout in
this evidence base looked, at this wave, for one. The statement is a **specification of what would
make these claims falsifiable**, nothing more.

---

## 1. What the programme measures today

| axis | metric | instrument |
|---|---|---|
| **composition** | `distinct_valid_novel_tuples` per corner | 49 held-out Montero n-way corners + 60 pair-tiled sub-corners, disjoint, **paired by corner**, every row in `test_rows` and 0 in any fit pool (asserted), cache generation 2, 3 seeds |
| **cost** | an **asserted byte pin** | store **524,288 B, tolerance 0**, `matches_budget` asserted on the deployed object before each leg; the read's 189–195 kB declared **beside** the store, never netted in |
| **refusal** | the pair `(reject_invalid, accept_unseen_valid)` | scored by the frozen `gen_scoreboard.score_refusal_row`; faults to 0 unless threshold, setting and `deployed_bytes` match across the pair |
| **deletion** | bytes changed / bytes changed **outside** the permitted range, **plus** surviving composition | `assert_no_refit` on every fitted segment; byte-level image diff over 2,054,866 B |

⭐ **The positive result this has to support:** `pca_tensor` (per-well leading principal subspace of
the fit pool, **unsupervised — no label enters the content**) beats a parameter-matched trained
learner `mlp_180k` on **both** target rows, **all three seeds**, beyond a margin declared before the
first held-out number: row 4 **+10.143 / +8.673 / +7.796** (bootstrap 2.5 % quantile +6.632 / +5.816
/ +5.082), row 3 **+4.033 / +2.483 / +2.417** (+2.117 / **+1.033** / +1.200). It also beats `lstm`,
`mlp`, `transformer`, `cont_mlp` and `table` on both rows, every seed, every lower bound above +1.0.
⚠ **Row 3 seed 1 clears by 0.033** and is not called comfortable.
*(`evidence/outputs/w5-advisor-report.md` §1.)*

---

## 2. What is currently unscorable, and the measurement that says so

### N1 — An inductive-bias row has no comparator, and an 8-byte rule ties it
Rows 5 and 6 score **(1.0000, 1.0000)** for both the 524,288 B field store and an **8-byte rule**,
on all three seeds — pre-registered as the expected outcome and reported first. `table_absence`
**is** `bitmap8`, asserted in code, at a byte multiple of **59,392×**. The surviving refusal claim is
expressed **by a flag** (`coupling_hole`), not by geometry, and a separate **8 + 166 B** pair-deletion
table matches it exactly.
⛔ **A benchmark that scores refusal as a rate cannot distinguish a store from a bitmap.**
*(w5 `hub/refuse`, 3/3 spent; `w5-advisor-report.md` §3.)*

### N2 — Composition and exact forgetting live on different objects
- The content that **wins** composition (principal subspaces) **forfeits charter §B.4 exact per-item
  deletion by construction** — a principal direction is not a written item. Priced: keeping items
  instead costs **≈ 18.6 tuples/corner** on row 4.
- The object that **forgets exactly and for free** — the coupling hole — refuses `red ∧ cylinder` at
  **1.0000** for **ONE byte**, **0 bytes changed outside** the permitted range, **no refit**, and
  costs **exactly 0.0000** composition, while **all 101 other never-held conjunctions are
  untouched**, zero seed spread. But **eviction** of `red` on the same store **fails the composition
  bar: −2.09 / −0.91 / −6.74** against a declared ±1.0, because *an item store loses composition when
  it loses atoms, whichever atoms they are.*

⛔ **This is the paper's central open problem and it is measured rather than argued. No benchmark in
use scores both axes on one object.**
*(w5 `hub/delete`, 1/3 spent, 2 returned; `w5-advisor-report.md` §1 §2 §6.1.)*

### N3 — A lifetime is not a deletion primitive, and the gap is invisible to a refusal metric
Deletion-by-decay on the field store, 109 held-out corners, 3 seeds:
- **Bar 1 (its own falsifier) PASSES:** at the last horizon before the items die, decay's composition
  cost differs from instant eviction's by **+0.0562 / +0.0337 / +0.0112** against a declared margin of
  **1.0**. ⛔ Its dial declaration is falsified in its own words, and it pre-registered that.
- **Bar 2 DOMINATED:** `t_pay(50 %)` = **7.992 / 5.994 / 7.992** half-lives, `t_pay(97 %)` =
  **15.984 / 7.992 / 11.988**, against `t_refuse` = **19.943** (measured **= derived**).
- ⭐ **The mechanism:** the composition channel reads the atom weight
  `lifetime · exp(−d²/2σ²)` **continuously**; the refusal channel reads a **liveness bit** at
  `LIFETIME_EPS = 1e-6`. **97.31 % / 96.30 % / 99.83 %** of the price is charged by the continuous
  quantity while the discrete one has not yet flipped.
- ⭐ Sharpest row: seed 2 at `t = 11.988` has already paid **100.0 %** of instant eviction's cost
  (−6.7416) with refusal still **0.0000**, and holds that for a further **8 half-lives**.

⛔ **A benchmark whose deletion metric is a refusal rate would score this store as not-yet-deleted
for eight half-lives after it has paid the entire price of deletion.**
*(w6 `hub/retention`, 2/3 spent, 3/3 returned; `w6-retention/LOG.md` §2 §4.)*

### N4 — A dynamics result cannot enter the scoreboard at all
w6 `hub/potentials` charter §C.1, in its own words: ⛔ *"DOES NOT APPLY and is not given a
manufactured pass. There is no lookup comparator for 'does a closed form predict a rollout', and
this hub claims no row against any rival."*
⭐ **The programme's cleanest theory-meets-measurement result is structurally unable to occupy a
taxonomy row.** That is a property of the claim, not a gap in effort.
*(`w6-potentials/LOG.md` §7.)*

### N5 — Two purpose-built instruments failed their own adoption gates, on two different grounds
**GYM-1** (1-D): matched-bytes fragment stitching reaches **0.8520 on 41,657 of 997,896 bytes**, and
a lookup table at **0.9797** beats **both** trained, rules-visible, 250 k-parameter baselines
(**0.3700**, **0.4173**) by **2.3×**. Cause proved over **25.3 M configurations**: in 1-D, composing
a speed limit with a distance **is** repeating a stored segment — the obstacle bent **0 of
25,261,855** solvable configurations.

**GYM-2** (2-D): 2-D did exactly what was asked — the obstacle now bends **41.85 %** (36,810,755 of
87,959,520) of solvable paths, and whole-trajectory retrieval collapsed **0.7027 → 0.0003**. ⛔ **It
still fails: TW-B = 0.9825 against a pre-registered 0.60, and the lookup beats the better trained
baseline by 5.7× (0.9825 vs 0.1733).**
*(`w5-gym1/LOG.md` §1 §2; `w51-gym2/LOG.md` §0 §1 §2 §4.)*

---

## 3. ⭐ THE REQUIREMENT LIST

A benchmark that would make this programme's claims **falsifiable** must have all ten of the
following. Each is forced by a number already paid for.

---

**R1 — At least one rule must be able to render a candidate ILLEGAL, not merely more expensive.**

*Forced by* GYM-2's rule-blind oracle battery:

| oracle | blind to | `valid_and_reached` | faults |
|---|---|---|---|
| optimal-plan control | nothing | 1.0000 | 0 |
| O1 obstacle-blind | the obstacle field | **0.0387** | 5,673 collisions |
| O2 speed-blind | the speed limit | **0.4213** | 3,533 speed violations |
| **O3 composition-blind** | **only the interaction** | **1.0000** | **0** |

Deleting the obstacle costs 96.1 points and relaxing the speed limit costs 57.9 points, and **their
interaction is worth 0.0000 on the metric.** O3 differs from the true optimum on 8.69 % of 1,347
route decisions and costs 0.164 extra steps — with **zero** faults.
⭐ **The mechanism, general to every dimension and every gridworld:** a speed limit is an **upper
bound**, so composing it with an obstacle can only ever change a route's **cost**, never its
**feasibility**. ⛔ *"'Compose the speed limit with the obstacle to find a different route' cannot be
scored by `valid_and_reached`, `all_steps_valid`, `reached_goal` or `held_at_goal` — by any
feasibility-or-goal-reaching metric at all. Going to 2-D did not fix this and no number of dimensions
will."*
**Admissible rules:** a *minimum* speed, a fuel or step budget, a one-way constraint, a time window.
⚠ **This is a statement about the rule algebra, not about the metric and not about the dimension.**
*(`w51-gym2/LOG.md` §3, §11.)*

---

**R2 — Score with an OPTIMALITY GAP, not a feasibility-or-goal-reaching metric.**

*Forced by* the addendum in which the spoke tested its own recommendation. 300 held-out episodes,
mean optimal cost 13.84; `within +s` = fault-free, parked on the goal, within `s` extra steps of
provably optimal:

| policy | `valid_and_reached` | mean extra steps | **+0** | +2 | +8 |
|---|---|---|---|---|---|
| T4 coarse, K ≤ 3 (the tripwire) | 0.9700 | 1.25 | **0.3900** | 0.8467 | 0.9700 |
| O3 composition-blind | 1.0000 | 0.22 | **0.9367** | 0.9733 | 0.9933 |
| LSTM | 0.1100 | 1.82 | 0.0767 | 0.0900 | 0.1033 |
| transformer | 0.1733 | 5.02 | 0.0800 | 0.1200 | 0.1433 |

✅ It **rescues the tripwire**: TW-B falls **0.9825 → 0.3900**, below the 0.60 line — stitched plans
are valid and they reach, but they are *sloppy*.
⛔ It **does not rescue R1**: O3 only falls to **0.9367**, still above the 0.90 pass line.
⛔ It **does not rescue the table-vs-model gap**: **0.3900 vs 0.0800 = 4.9×**, still.
⭐ **Take R2 because it is strictly better, and do not expect it to buy R1.**
*(`w51-gym2/LOG.md` §11.)*

---

**R3 — The state space must be large relative to `4 · n_params / frag_bytes`.**

*Forced by* GYM-2's budget arithmetic. Budget **1,001,188 B** = 4 × 250,297 parameters. A
`min_frag = 6` macro-action costs 5 bytes, so the budget buys **200,238** of them against an
environment of `G²(2V+1)² = 400 × 81 =` **32,400** states — **5.078 macro-actions per state**.
⭐ *"A table with a useful macro-action at every state IS a planner, whatever the gym looks like."*
The sweep confirms it exactly:

| budget (B) | fragments | macro-actions **per state** | K ≤ 3 |
|---|---|---|---|
| 1,001,188 (matched) | 164,524 | **5.078** | **0.9750** |
| 300,000 | 56,244 | 1.736 | 0.9150 |
| 100,000 | 20,000 | 0.617 | 0.8050 |
| 30,000 | 6,000 | 0.185 | **0.4900** |
| 3,000 | 600 | 0.019 | **0.0000** |

The coarse band crosses 0.60 at ≈ **46,000 B ≈ 11,000 parameters ≈ 0.28 macro-actions per state** —
**18× less than the matched budget**. ⭐ **At 250 k parameters and 5-byte macro-actions the
environment needs of order 7 × 10⁵ states — a 100×100 grid at these velocities, not a 20×20 one.**
⛔ Under-sizing the state space is not a flaw in one gym; **it is a flaw every small gridworld
shares**, and it is why GYM-1 and GYM-2 failed the same gate for what looked like different reasons.
*(`w51-gym2/LOG.md` §4.)*

---

**R4 — Run a matched-bytes STITCHING tripwire, with a `min_frag` sweep, BEFORE adoption.**

*Forced by* GYM-1: ⛔ **T1 exact-key lookup alone scored 0.0000 and would have certified a dead gym
as sound.** Exact-key lookup is not a tripwire. In GYM-2 whole-trajectory retrieval died (T1
**0.0000**, T2 **0.0003**, T3 **0.0003**, against GYM-1's T3 = 0.7027) while **coarse recombination
still fired at 0.9825**. And tightening `min_frag` makes the table **better** at low K — 0.8000 at
min_frag 6, **0.8450** at min_frag 8 with K = 2 — because the floor forces a fixed budget to buy long
macro-actions. ⛔ **There is no `min_frag` at which this band is safe**; at min_frag 10 the budget is
not even binding (66.8 % used).
*(`w5-gym1/LOG.md` §1 and §6 item 3; `w51-gym2/LOG.md` §4.)*

---

**R5 — Make a rule-blind oracle a mandatory gate.**
*Forced by* R1's table: it is nearly free and it is decisive. **If a policy that cannot see a rule
scores 1.0, that rule is untested and no claim may rest on it.**
*(`w5-gym1/LOG.md` §6 item 4; `w51-gym2/LOG.md` §1 §3.)*

---

**R6 — Measure held-out diversity BEFORE adopting, and hard-fail on it.**
*Forced by* GYM-2's gate **D**: N3-rel median **2.0** against a required ≥ 4, hard-fail ≤ 2 — ⛔
**FAIL**. ⭐ *"If the median held-out path tiles from two stored fragments, no split will save it."*
*(`w51-gym2/LOG.md` §1 §5; `w5-gym1/LOG.md` §6 item 2.)*

---

**R7 — The validity check must be EXACT, zero tolerance.**
*Forced by* gate **V** on both gyms: **no tolerance used anywhere**, and in GYM-1 the batched
environment was checked field-for-field against the scalar reference on 300 episodes with **0
mismatches**; episode-level validity is computed **before** any repair. ⭐ **This is what makes a
downstream refusal claim non-approximate.** A benchmark with a tolerance in its validity predicate
cannot carry an exact-refusal or exact-deletion claim at all.
*(`w5-gym1/LOG.md` §3; `w51-gym2/LOG.md` §1.)*

---

**R8 — It must contain a DELETION request, scored on the SAME object and in the SAME currency as
composition.**
*Forced by* N2. The programme's central open problem — that the content which composes cannot
forget exactly, and the object that forgets exactly loses composition when it evicts — is **only
visible if one benchmark scores both axes on one object in one unit.** Today the price is quoted in
the composition unit (**≈ 18.6 tuples/corner**) precisely because no benchmark supplies one.
*(`w5-advisor-report.md` §1 §6.1.)*

---

**R9 — It must separate a CONTINUOUS degradation channel from a DISCRETE liveness channel.**
*Forced by* N3: **96.30–99.83 %** of the composition price arrives through a continuous weight that
is already at `1.5e-5` while the discrete liveness bit has not flipped. ⛔ A benchmark with one
binary "is it deleted" column reports **0 % deleted** through the entire window in which the store
pays **100 %** of the cost of deletion. It must expose both, on the same horizon axis.
*(`w6-retention/LOG.md` §4.)*

---

**R10 — It must not be able to score a DYNAMICS prediction as a taxonomy row.**
*Forced by* N4, and by this section's own scope limit. The Curvature–Retention Relationship predicts
a **settle half-life**; it says nothing about how many valid novel tuples a store composes. ⛔ **A
benchmark that let a dynamics prediction imply a scoreboard number would manufacture exactly the
claim this evidence base exists to prevent.** The requirement is negative and it is binding: the
scoreboard's row definitions must make a half-life prediction **inadmissible**, not merely
unrewarded.
*(`w6-potentials/LOG.md` §7; `w6-advisor-report.md` §6.2.)*

---

## 4. One paragraph, if the section only has room for one

> The programme scores composition as distinct valid novel tuples on disjoint held-out n-way corners
> at an asserted byte pin, refusal as a `(reject, accept)` pair, and deletion as bytes changed plus
> surviving composition. On those instruments it has a result — an unsupervised principal-subspace
> content beats a parameter-matched trained learner on both target rows and three seeds — and it has
> a measured open problem: the content that composes cannot delete exactly, and the object that
> deletes exactly loses composition when it evicts. **Neither half of that is scored by any
> instrument we were able to build.** Two purpose-built gyms failed their own pre-registered
> adoption gates, on two different and quantified grounds: in one dimension an obstacle bends no
> path at all (0 of 25,261,855 configurations), and in two dimensions, where it bends 41.85 % of
> them, a matched-bytes stitching table still scores 0.9825 against a 0.60 threshold and beats the
> better trained baseline by 5.7×. The oracle that is blind only to the *interaction* between the
> two rules still scores 1.0000. **What is missing is not effort; it is a benchmark with ten
> specific properties** — at least one rule that can make a candidate illegal rather than merely
> expensive; an optimality-gap metric; a state space large relative to `4·n_params/frag_bytes`; a
> matched-bytes stitching tripwire swept over fragment length; a mandatory rule-blind oracle; a
> pre-adoption diversity gate; an exact zero-tolerance validity predicate; a deletion request in the
> same currency as composition; separate continuous and discrete degradation channels; and row
> definitions under which a dynamics prediction is inadmissible rather than merely unrewarded.
> **Each of the ten is forced by a number this programme has already paid for.**
