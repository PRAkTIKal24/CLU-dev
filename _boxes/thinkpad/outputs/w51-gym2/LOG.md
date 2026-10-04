# w5.1-gym2 — LOG

AUTO-HUB RETRY of `w5/gym1`, kind **R-B (mechanism/instrument)**. ⛔ **ZERO scored runs.
⛔ NO claim about the CLU.** This spoke builds a measuring device and gates it.

- **START_UTC**: 2026-09-20T00:44:45Z · allowance 8 h → hard stop 2026-09-20T08:44Z
- Worktree `C:/chlu/worktrees/CHLU-w51-gym2`, branch `w5.1/gym2`, base = parent tip `bfee159`
- Outputs → `C:/chlu/outputs/w51-gym2` only. 3 threads, system python 3.11.9, jax 0.9.0 CPU.

---

## 2026-09-20T00:45Z — read parent, inherited what is re-usable

Read `/c/chlu/outputs/w5-gym1/LOG.md` (561 lines) and all of `chlu/gym1/*.py` (2,458 lines).
Parent verdict, taken as **reading, not truth**: GYM-1's matched-bytes trajectory-lookup tripwire
fires (T4 stitching 0.9797 on 41,657 of 997,896 bytes; lookup beats both 250k trained rules-visible
baselines 0.3700/0.4173 by 2.3x). Cause claimed structural: on a line, composing a speed limit with
a distance *is* repeating a stored constant-velocity segment.

**Machinery I inherit verbatim (parent section 6.5), ported to 2-D unchanged in substance:**

1. Zero-tolerance integer validity predicates, reported independently per fault kind.
2. Unmasked rollout + `verify_unmasked_rollout()` adversarial check (drive a deliberately
   illegal policy; prove the env applies it verbatim rather than repairing it).
3. One-step supervision with `N_HOLD = 4`, survival and at-goal-over-time curves at 2x horizon.
4. Provably optimal ground truth by exhaustive backward BFS over the full state graph.
5. Batched env checked field-for-field against the scalar reference before any number is trusted.
6. Pre-registration with a written falsification threshold, fixed before the instrument is built.

**Thing I will NOT inherit as-is — and why.** The parent's N3 (`episodes.fragment_cover`) tiles the
held-out path against a corpus of **absolute** position sequences, while its own T4 tripwire harvests
**translation-invariant** macro-actions (start velocity + velocity profile). Those are different
corpora and the T4 one is strictly stronger. N3-absolute therefore *over-states* diversity relative
to the table that actually attacks the gym. GYM-2 measures N3 **both** ways and treats the
translation-invariant number as the binding one.

---

## 2026-09-20T00:52Z — PRE-REGISTRATION (written before GYM-2 exists; no numbers seen)

### P0. What GYM-2 must be

At least 2-D with momentum, because an obstacle cannot bend a path in 1-D. The wave-7 question is
"**compose the speed limit with the obstacle to find a different route**", so the gym must be one in
which *the optimal route's shape depends on the speed limit*. A speed limit is an upper bound, so a
larger `V` never makes a route infeasible — the composition must therefore be a **time** trade-off:
a short twisty route that can only be crawled versus a long open route that can be cruised. If that
trade-off is absent, GYM-2 fails for the same reason GYM-1 did, one dimension up.

### P1. Order of reporting (fixed now, so it cannot be chosen after the fact)

Diversity -> tripwire -> oracles -> baselines. Diversity and the tripwire are reported **before any
baseline is trained**. If a gate fails I report and stop; I do not harden past a fired tripwire.

### P2. GATE D — path diversity. Reported FIRST.

Measured on >= 1,500 held-out episodes against the whole training corpus, exact substring tests.

| | statistic | PASS requires |
|---|---|---|
| N1 | held-out episode key novel | = 1.000 (a leak check, not a gate) |
| N2a | held-out **position** sequence never a contiguous substring of any training path | >= 0.95 |
| N2b | held-out **action** sequence never a contiguous substring of any training path | >= 0.90 |
| N3-rel | median fewest **translation-invariant** training fragments (each >= `MIN_FRAG_GATE`) tiling the held-out path | **>= 4** |

`MIN_FRAG_GATE` is fixed as `floor(median held-out optimal path length / 2)` — computed from the
split, not chosen after seeing N3. **GATE D fails if N3-rel median <= 2** ("the median held-out path
tiles from two stored fragments" — the parent's own stop condition).

### P3. GATE T — the matched-bytes trajectory-lookup tripwire. THE gate.

Budget = 4 bytes x the parameter count of the matched baselines, exactly as the parent. Every
strategy is scored by **replaying what it retrieves in the real environment**, under the same metrics
a neural model gets. Strategies T1 exact key, T2 nearest-neighbour absolute, T3 nearest-neighbour
relative, T4 fragment stitching as a curve over `K` (fragment allowance) **and** `min_frag`.

⛔ **T1 alone is not a tripwire** (0.0000 on GYM-1). The headline is the best over the whole grid,
split into three registered bands:

- **TW-A, whole-trajectory retrieval** = max(T1, T2, T3, T4@K=1). **FIRES if > 0.50.**
- **TW-B, coarse recombination** = max over T4 with `min_frag >= MIN_FRAG_GATE` and `K <= 3`.
  **FIRES if > 0.60.** (The parent's GYM-1 scored **0.970** in this band: 3 chunks of >= 12 steps.)
- **TW-C, fine recombination** = T4 at `min_frag = 3`, `K` unbounded. Reported, **not a gate**, with
  the reason stated in advance: with a 9-symbol action alphabet, storing *every* 3-step velocity
  profile from *every* start velocity costs far less than the byte budget, so this band degenerates
  into "is the ground truth a shortest path", which is true by construction in any gym whose teacher
  is BFS. I will **verify that degeneracy explicitly** (is the harvested min_frag=3 fragment set
  behaviourally complete?) on GYM-1 *and* GYM-2 rather than assert it.
- **TW-D, the parent's section 4 comparison.** **FIRES if the best non-degenerate lookup (TW-A or
  TW-B) >= the better trained rules-visible baseline's `valid_and_reached`.** A gym on which a
  trivial table outscores every trained model cannot show that a trained model composed anything.

⛔ **If GATE T fires: report and stop. No hardening.**

### P4. GATE O — rule-blind oracles. MANDATORY. A rule that survives blinding is untested.

Each oracle plans optimally with one rule deleted or relaxed, then **replays into the true
environment** and is scored by the same metrics. **PASS requires `valid_and_reached` < 0.90.**

- **O1 obstacle-blind** — plan with every obstacle deleted.
- **O2 speed-blind** — plan under `V = V_MAX`, replay under the episode's true `V`.
- **O3 composition-blind (the decisive one for wave 7)** — keep the *route* (homotopy class /
  corridor choice) that is optimal at `V = V_MAX`, then re-plan **optimally under the true `V`
  restricted to that route**. If O3 >= 0.90 *and* its cost equals the unrestricted optimum, the
  route does not depend on the speed limit and **the composition wave 7 wants to test is empty**,
  even though O1 and O2 both fail loudly. O3 is the 2-D analogue of the parent's wall-blind oracle
  and it is the gate GYM-1 would have failed one dimension up.

Alongside O3, an **exhaustive structural sweep** (the analogue of the parent's 25.3 M-configuration
result): over every sampled layout x every `V` x every `(start, v0, goal)`, the fraction of solvable
configurations whose optimal **route choice** changes with `V`, and the fraction whose optimal
**cost** changes by more than pure re-timing. Exhaustive, not sampled.

### P5. GATE H — headroom. Baselines: LSTM and transformer, rules visible, matched parameters.

Both see `V` **and the full obstacle map** at every step. Parameter match tolerance **2 %**, counts
printed. **PASS requires** the better baseline's held-out `valid_and_reached` to lie in
**(0.05, 0.95)**. >= 0.95 ⇒ no headroom, wave 7 must know before it is scoped. <= 0.05 ⇒ nothing is
measurable on this gym either.

### P6. GATE V — validity is exact

PASS requires every predicate to be integer arithmetic with **zero tolerance** — including 2-D
swept-cell collision, which must be an exact rational test, not a sampled line trace. If any
predicate needs a tolerance, I say so; an approximate validity predicate makes every downstream
refusal and deletion claim approximate.

### P7. Declared NOT-RUN

Scored run: **NOT RUN** (mandated). Any claim about the CLU: **NOT MADE**. GPU: not contemplated.

---
## 2026-09-20T01:28Z — GYM-2 built; PRE-REGISTRATION CORRECTION (before any number)

**Environment** `chlu/gym2/grid2d.py`. 20x20 grid, agent state `(x, y, vx, vy)`, 9 actions
`(ax, ay) in {-1,0,1}^2`, `|ax|,|ay| <= 1`. Speed limit `|vx| <= V and |vy| <= V`, `V in {1,2,3,4}`.
Three exact fault predicates, zero tolerance, reported independently: speed-limit violation,
obstacle collision, out-of-bounds. Collision is **swept-cell**: a cell is hit iff the closed segment
between the two cell centres meets its closed unit square, decided once at import with
`fractions.Fraction` (`SWEEP`, 81 displacements). Consequence stated rather than hidden: a pure
diagonal step grazes the corners of both orthogonal neighbours, so a diagonal move needs all four
cells free -- the standard no-corner-cutting rule, exact, not a safety margin. Out-of-bounds is
tested on **every** swept cell, so leaving the grid mid-step and returning is still recorded.
Ground truth = exhaustive backward BFS over the full `(x,y,vx,vy)` graph (up to 32,400 states).

**Layouts** `chlu/gym2/layouts.py`. A solid block of columns `[cx, cx+tw)` x rows `[0, hb)` with a
**one-cell-wide serpentine tunnel** bored through it, and an **open detour** over the top. Because
the passage is one cell wide and corner-cutting is a collision, the tunnel cannot be taken fast; the
detour can. Short-and-crawlable vs long-and-cruisable is the only shape a speed-limit composition
can have, and it is the thing 1-D could not express. Route attribution is exact, not heuristic: block
`block_top` and the tunnel is the only crossing; block `block_tunnel` and the detour is; two extra
BFS runs give `cost_tunnel` and `cost_detour`.

**First probe that the mechanism is real** (14 layouts x 3 start rows = 41 configurations, all four V):
**12/41 = 29 % flip their optimal route as V rises**, always tunnel -> detour, e.g. lid 8
`T T D D` with tunnel costs `[22,16,15,15]` against detour costs `[29,16,14,14]`. The tunnel cost
saturates by V=2 while the detour keeps falling -- exactly the intended trade-off. This is a probe on
fixed starts; the exhaustive sweep is the reported number.

### ⚠ Correction to P2, made before any diversity number was computed

As written, GATE D required `N3-rel >= 4` with `MIN_FRAG_GATE = floor(median path length / 2)`.
That is **arithmetically impossible**: a path of length `Lp` tiled by fragments of length
`>= Lp/2` can never need more than 2 of them. The gate as drafted could only ever fail. Corrected,
with the same intent and the same stop condition:

- **N3-rel is measured at `min_frag = 3`** (the parent's `MIN_FRAG`, so the numbers are directly
  comparable to its 1.51 mean). **PASS requires median >= 4; GATE D FAILS if median <= 2.**
- At the coarse length `MIN_FRAG_GATE` the meaningful statistic is not a count but
  **`coverable_frac`** -- the fraction of held-out paths that can be tiled *at all* out of stored
  fragments that long. Reported, and directly comparable to TW-B, but **not** a gate.
- N2 is reported three ways: **N2a** absolute position sequence, **N2b** action sequence, and
  **N2c** the *behaviour* string `v0 a0 v1 a1 ... vT` -- the translation-invariant object a
  stitching table actually stores. N2c is the binding one. PASS: N2a >= 0.95, N2b >= 0.90,
  N2c >= 0.90.

No other part of the pre-registration is touched, and no tripwire, oracle or baseline number has
been looked at.

## 2026-09-20T01:03Z — the split is built and clean

`splits.pkl`, `splits_summary.json`. 190 layouts (150 training-visible + 40 reserved novel),
balanced across the three tunnel-length buckets (64/63/63). **60,000 training / 3,000 held /
1,500 held-on-novel-layouts / 1,000 refusal** episodes. Optimal path length mean 14.7, median 13,
max 36; 89.8 % of episodes require crossing the block, so the route decision is actually posed.

Split checks, all asserted rather than inspected: key overlap train/held **0**, key overlap
train/held-novel-layout **0**, (V, bucket) **pairing** overlap **0**, every `V` present in
training **true**, every bucket present in training **true**, layout overlap between training and
the novel-layout set **0**. 8 training pairings, 4 held-out pairings `(1,2) (2,0) (3,1) (4,2)`.

One deviation, logged: `sample_refusal` as inherited could not work here, because the layout
generator **rejects any draw whose free space is not fully connected**, so an unreachable goal
essentially never occurs. The refusal split is therefore built on **sealed** variants (both
crossings closed, two components). A sealed map is an ordinary input to the gym -- the obstacle
field is an observation, not a constant -- and it is the one place where reading the obstacle is
the only way to know the goal must be refused.

---

## 2026-09-20T01:04Z — ⛔ GATE D (path diversity) — REPORTED FIRST, and it FAILS

1,500 held-out episodes against the whole 60,000-episode training corpus. Exact substring tests
throughout; no index approximation, no candidate cap.

| | statistic | value | required | |
|---|---|---|---|---|
| N1 | episode key novel | **1.000** | = 1.000 | ✅ the split is clean; this is not a leak |
| N2a | absolute **position** sequence novel | **0.988** | >= 0.95 | ✅ |
| N2b | **action** sequence novel | **0.964** | >= 0.90 | ✅ |
| N2c | **behaviour** string novel (start velocity + actions) | **0.965** | >= 0.90 | ✅ |
| N3-rel | median translation-invariant fragments (min_frag 3) tiling the path | **2.0** | >= 4 | ⛔ **FAIL** |

⛔ **N3-rel median = 2.0 hits the pre-registered HARD-FAIL condition (`<= 2`)** — the parent's own
stop condition, in its words: *"If the median held-out path tiles from two stored fragments, no
split will save it."*

Distribution of N3-rel: `{1: 52, 2: 1053, 3: 345, 4: 11}`, mean 2.216, uncoverable 2.6 %.
N3-abs (the parent's absolute-position version, for comparability): mean 2.738, median 3.0,
uncoverable 15.1 %.

Two things this says that are worth more than the pass/fail:

1. ⭐ **The 2-D gym is genuinely more diverse than GYM-1 on every surface measure and still fails.**
   GYM-1's held-out *action* sequences were novel only 3.3 % of the time; GYM-2's are novel
   **96.4 %** of the time, and its position paths **98.8 %**. Going to two dimensions with a
   9-symbol alphabet did exactly what it was supposed to do to *surface* novelty. It did almost
   nothing to *fragment* novelty: N3-rel went from 1.51 to 2.22 — one extra fragment.
2. ⭐ **The parent's N3 over-states diversity, as predicted before the run.** On the same episodes
   N3-abs = 2.74 and N3-rel = 2.22, and uncoverability collapses from 15.1 % to 2.6 %. Tiling
   against *absolute* positions credits the gym with novelty that a translation-invariant table
   does not have to pay for. Any future gym must report N3 the translation-invariant way.

At the coarse length `MIN_FRAG_GATE = 6` (= median path length 13 / 2), **43.5 %** of held-out
paths can be tiled at all out of stored >= 6-step fragments. That is a bound on exact-path
reproduction, not on goal-reaching, which is why the tripwire is still the binding gate.

**Held out on layouts that appear NOWHERE in training** — the strictly harder set — gives
essentially the same answer: N1 1.000, N2a 0.991, N2b 0.968, N2c 0.968, **N3-rel median 2.0**
(mean 2.266, hist `{1: 48, 2: 980, 3: 393, 4: 19}`), coarse coverable 45.9 %.

⭐⭐ **That is the important control.** The fragment-tiling failure is *not* layout memorisation:
withholding the obstacle map entirely moves N3-rel by 0.05 of a fragment. What a stitching table
re-uses is the **velocity-profile alphabet**, which is a property of the dynamics, not of the map.

> **GATE D verdict: FAIL** (`N3rel_pass: false`, `N3rel_fail_hard: true`, `GATE_D_PASS: false`).

---

## 2026-09-20T01:05Z — GATE T launched (budget from the matched baselines)

LSTM `d=64, h=200` = **250,289** parameters; transformer `d=64, 3 layers, 4 heads, d_ff=400` =
**250,297**. Relative gap **0.0032 %** against a pre-registered 2 % tolerance. Tripwire budget =
4 bytes x 250,297 = **1,001,188 bytes**, the same rule the parent used.

First result in: **T1 exact key = 0.0000** on 994,411 of 1,001,188 bytes, exactly as on GYM-1.
⛔ This is the third independent demonstration that **exact-key lookup is not a tripwire**: it is
0.0000 on a gym that fails and 0.0000 on a gym that fails differently, and it would certify both.

---

## 2026-09-20T01:16Z — ⭐ the exhaustive structural sweep: 2-D FIXES what 1-D could not express

190 layouts x 4 speed limits x 40 goals, **exhaustive in (start cell x start velocity)** —
one backward BFS gives the exact cost from every state at once. **109,440,000 configurations**,
87,959,520 of them solvable. (The parent's 1-D sweep covered 25,261,855.)

**S1 — does the obstacle bend the path?**

| | 1-D (GYM-1, parent) | 2-D (GYM-2) |
|---|---|---|
| solvable configurations examined | 25,261,855 | **87,959,520** |
| ... made strictly more expensive by the obstacle | **0** | **36,810,755 = 41.85 %** |
| ... made infeasible by the obstacle | — | 21,480,480 |

> ⭐ **This is the one thing the retry was told to fix, and it is fixed.** On a line an obstacle
> could never raise the optimal cost of a solvable configuration — not once in 25.3 M. In this
> gym it raises it in **41.85 %** of them. The obstacle is load-bearing, not decorative.

**S2 — does the SPEED LIMIT bend the ROUTE?** Over the 16,377,426 configurations where both the
tunnel and the detour are open at every V:

| | value |
|---|---|
| optimal route changes with V | **1,882,092 = 11.49 %** |
| ... at V1->V2 | 1,154,277 |
| ... at V2->V3 | 671,114 |
| ... at V3->V4 | 103,483 |

and the route mix moves monotonically in the predicted direction — strict-tunnel
3,445,145 -> 2,173,688 -> 1,565,127 -> **1,514,206** and strict-detour
4,140,994 -> 5,221,065 -> 5,836,735 -> **5,902,182** as V goes 1 -> 4. The crossover sits at low
V, exactly as the "crawl the short twisty way versus cruise the long open way" trade-off predicts.
Flip rates are flat across the three tunnel-length buckets (12.8 % / 10.4 % / 11.2 %), so the
effect is the mechanism and not one corner of the generator.

Caveat stated rather than hidden: ties (`cost_tunnel == cost_detour`, 35.7 M cases) are resolved
to "tunnel" at every V, so a counted flip means the tunnel went from *at least as good* to
*strictly worse*. Tie-to-tie never counts as a flip.

> **So GYM-2 poses the wave-7 question for real.** "Compose the speed limit with the obstacle to
> find a different route" is a non-empty instruction here in a way it provably was not in 1-D.
> Whether the gym can *measure* an answer is a separate question, and that is GATE T.

---

## 2026-09-20T01:34Z — ⛔⛔ GATE T FIRES. TW-B = 0.9825 against a pre-registered 0.60.

Budget **1,001,188 bytes** = 4 x 250,297. (Accounting note against myself: the transformer as
actually built has `max_len = 104`, giving 248,761 parameters, so the true matched maximum is the
LSTM's **250,289** and the budget should have been 1,001,156. The table therefore got **32 bytes
more** than the rule allows — 0.003 % — which makes the tripwire marginally *stronger*, the safe
direction. Relative parameter gap 250,289 vs 248,761 = **0.61 %**, inside the 2 % tolerance.)

**Whole-trajectory retrieval is DEAD in 2-D — the one thing that clearly improved:**

| strategy | held | held, novel layouts | bytes used / 1,001,188 |
|---|---|---|---|
| T1 exact key | **0.0000** | 0.0000 | 994,411 |
| T2 nearest neighbour, absolute | **0.0003** | 0.0033 | 994,411 |
| T3 nearest neighbour, relative | **0.0003** | 0.0660 | 994,411 |

Against GYM-1's T3 = **0.7027**. In 2-D a stored trajectory replayed from anywhere else simply
collides, so **TW-A = 0.0050 against a threshold of 0.50: PASS, by two orders of magnitude.**
⛔ And T1 is **0.0000 for the second gym running** — the third demonstration that exact-key lookup
is not a tripwire and would have certified both gyms.

**Fragment stitching, `valid_and_reached` on 400 held-out episodes across 135 layouts:**

| min_frag | fragments | bytes | K=1 | K=2 | K=3 | K=4 | K=6 |
|---|---|---|---|---|---|---|---|
| 3 | 182,016 | 1,001,182 | 0.0025 | 0.7475 | 0.9650 | 0.9950 | 0.9975 |
| 4 | 178,599 | 1,001,182 | 0.0025 | 0.7500 | 0.9625 | 0.9950 | 0.9975 |
| **6** = `MIN_FRAG_GATE` | 164,524 | 1,001,183 | 0.0050 | **0.8000** | **0.9825** | 0.9950 | 0.9950 |

> ⛔⛔ **TW-B (coarse recombination, `min_frag >= 6`, `K <= 3`) = 0.9825.
> Pre-registered falsification threshold: 0.60. THE TRIPWIRE FIRES.**
> **Two** stored macro-actions of at least six steps each already solve **80.00 %** of held-out
> pairs; **three** solve **98.25 %**.

Note which direction the curve runs: **raising the minimum fragment length makes the table
BETTER at K=2** (0.7475 -> 0.7500 -> 0.8000). The floor forces the fixed byte budget to buy long
macro-actions instead of a pile of 3-steppers, and long macro-actions are what a 13-step path
needs two of. There is no min_frag at which this band is safe; the band gets worse as it is
tightened.

Self-consistency check that passed, as on GYM-1: **`hit_rate` equals `valid_and_reached` to four
decimals at every (min_frag, K)**, so every plan the exact footprint test accepted also replayed
cleanly in the real environment. The tripwire is not scoring itself.

### ⚠ My pre-registered explanation of TW-C was WRONG, and I am reporting that

P3 predicted TW-C (min_frag 3, K unbounded) would be near-1.0 *because the fragment table would
be behaviourally complete* — every legal 3-step macro-action present — making stitching literally
BFS. I said I would verify that rather than assert it. **Verified, and it is false.** Coverage of
the legal 3-step macro-action alphabet, exact counts:

| V | legal 3-step macro-actions | present in the table | coverage |
|---|---|---|---|
| 1 | 1,681 | 536 | **0.319** |
| 2 | 9,025 | 3,620 | **0.401** |
| 3 | 22,201 | 7,476 | **0.337** |
| 4 | 41,209 | 8,971 | **0.218** |

The table holds between a fifth and two fifths of the alphabet and **still reaches 0.9975**. So
TW-C is not saturated because the table is complete; it is saturated because **a third of the
macro-action alphabet is already enough to re-plan.** That is a worse fact for the gym than the
one I predicted, and it removes the excuse I had pre-written for the TW-C row.

⛔ **Per the dial declaration and the task's stop rule, I stop hardening GYM-2 here.** What
follows is characterisation and the remaining *mandatory* gates (the rule-blind oracles, and the
baselines that TW-D needs), not an attempt to make the gym pass.

---

## 2026-09-20T01:47Z — ⛔⭐ GATE O FAILS on O3, and this is the most useful thing in the spoke

Each oracle plans optimally with one rule (or the interaction) removed, then is **replayed into
the true environment** and scored on exactly the model's metrics. 1,500 held-out episodes.
Adversarial checks passed first: `verify_unmasked_rollout.ok = true`, batched-vs-scalar
environment **0 mismatches on 300 episodes**, optimal-plan control **1.0000** by construction.

| oracle | what it is blind to | `valid_and_reached` | faults it causes | gate |
|---|---|---|---|---|
| O1 obstacle-blind | the obstacle field | **0.0387** | 5,673 collisions | ✅ the obstacle is tested |
| O2 speed-blind | the speed limit | **0.4213** | 3,533 speed violations | ✅ the speed limit is tested |
| **O3 composition-blind** | **only their interaction** | **1.0000** | **0** | ⛔ **FAIL** |

O3 keeps the route that is optimal at `V = V_MAX`, then re-plans **optimally under the episode's
true V, restricted to that route**. Both rules are fully respected; only the knowledge that
*which route is cheaper depends on how fast you may go* is removed. O3 stats:

- route decisions posed: **1,347**
- the V_MAX route differs from the true optimum: **117 = 8.69 %**
- mean extra steps versus the true optimum: **0.164**
- strictly costlier at all: **5.79 %**

> ⭐⭐ **The mechanism, and it is general.** A speed limit is an **upper bound**, so raising it can
> never make a route infeasible. Composing a speed limit with an obstacle can therefore only ever
> change the optimal route's **cost** — never its **feasibility**. Picking the wrong route costs
> you 0.164 steps on average and **zero** faults. So
>
> ⛔ **"compose the speed limit with the obstacle to find a different route" cannot be scored by
> any feasibility-or-goal-reaching metric, in any number of dimensions.** `valid_and_reached`,
> `all_steps_valid`, `reached_goal`, `held_at_goal` — the whole family is structurally blind to it.

This is strictly stronger than the parent's finding. On GYM-1 the wall-blind oracle scored 1.0
because the obstacle did nothing at all. Here the obstacle does a great deal (deleting it costs
96.1 points) and the speed limit does a great deal (relaxing it costs 57.9 points), and the
**interaction between them is still worth 0.0000 on the metric.** Two dimensions bought a real
obstacle, as the structural sweep shows (41.85 % of paths bent) — it did not buy a *measurable*
composition.

---

## 2026-09-20T02:01Z — baselines (rules visible, matched parameters), and ⛔ TW-D fires

| | params | steps | seconds | final one-step loss |
|---|---|---|---|---|
| LSTM `d=64, h=200` | **250,289** | 6,000 | 680 | 0.3531 |
| transformer `d=64, 3L, 4H, d_ff=400` | **248,761** | 6,000 | 882 | 0.3280 |

Relative parameter gap **0.61 %** (pre-registered tolerance 2 %). Both see `V`, the **whole**
20x20 obstacle grid as a learned linear projection, and a 7x7 egocentric window on it, at every
step. 1,120,631 supervised one-step targets.

**Exact validity, 1,500 held-out episodes, free-running and unmasked, horizon 48:**

| | LSTM | transformer |
|---|---|---|
| **valid_and_reached** | 0.1127 | **0.1733** |
| all steps valid | **0.3900** | 0.3020 |
| goal reached | 0.1127 | **0.2347** |
| held at goal | **0.1127** | **0.0087** |
| speed-limit violations | 83 | 306 |
| obstacle collisions | 8,772 | 9,605 |
| out-of-bounds | 83 | 505 |
| episodes with any collision | 59.2 % | 59.5 % |
| first fault: speed / obstacle / out | 12 / 877 / 26 | 41 / 868 / 138 |

Novel layouts cost about 2 points (LSTM 0.0887, transformer 0.1293).

> ⛔⛔ **TW-D FIRES. The matched-bytes lookup table (TW-B = 0.9825) beats the better trained,
> rules-visible, 250k-parameter baseline (0.1733) by 5.7x.** The parent's factor was 2.3x.
> The gym got harder for models and no harder for the table.

**GATE H PASSES** — 0.1733 lies in the pre-registered (0.05, 0.95) window. There is headroom;
that is not the problem.

### ⭐ Persistence at one step reproduces the parent's finding on a different gym

Survival = fraction still fault-free through step t; at-goal = fraction parked on the goal at t.
Holding the goal past step 4 is never supervised.

| | LSTM | transformer |
|---|---|---|
| survival @ t=48 | **0.3900** | 0.3020 |
| survival @ t=96 | **0.3900** | **0.1627** |
| parked on goal @ t=24 | 0.1053 | **0.1993** |
| parked on goal @ t=36 | 0.1120 | 0.0607 |
| parked on goal @ t=48 | **0.1127** | **0.0087** |
| parked on goal @ t=96 | 0.1127 | 0.0113 |

The transformer **reaches** the goal twice as often as the LSTM (0.2347 vs 0.1127) and **holds**
it at one thirteenth the rate. Its at-goal curve is non-monotonic — arrive, then drift off — and
its survival halves when the horizon merely doubles, entirely outside anything it was supervised
on. The LSTM is flat to t=96. ⭐ **Goal-reach ranks the transformer first; persistence reverses
it. This protocol transfers, and it is the part of the parent worth keeping.**

### The rules are used here, unlike in GYM-1

In-distribution ablations (permute the rule across episodes so every token stays a value the
model has seen, but no longer describes its own episode):

| | true | obstacle map shuffled | delta | V shuffled | delta |
|---|---|---|---|---|---|
| LSTM | 0.1127 | 0.0613 | **-0.0514** | 0.0767 | **-0.0360** |
| transformer | 0.1733 | 0.0987 | **-0.0746** | 0.1193 | **-0.0540** |

On GYM-1 destroying the wall information cost the LSTM 0.0140 and *helped* the transformer by
0.0020 — noise. Here both models lose 30-45 % of their score when either rule is scrambled.
**Both rules are genuinely read.** It is only their interaction that is unmeasurable (GATE O).

**Refusal split** (1,000 episodes on sealed maps: the goal is unreachable but a fault-free
behaviour exists): LSTM `all_valid` **0.1350** with 10,467 collisions; transformer **0.0850**
with 16,977. Neither was trained on unsolvable episodes, so this is expected; it is reported
because it confirms the collision metric is live rather than structurally unable to fire.

**Caveat that never binds, measured:** the velocity embedding clips model *input* at `|v| <= 8`.
Largest `|v|` reached in any rollout: **6** (LSTM) and **7** (transformer); fraction of episodes
ever exceeding the clip: **0.0000**.

---

## 2026-09-20T02:13Z — ⭐ the frontier between "lookup-proof" and "learnable": the curves never cross

The task's fourth falsification case asks for this measured rather than asserted. 480 held-out
episodes, 120 per length bucket.

**F1 — as a function of optimal path length:**

| path length | N3-rel mean | N3-rel median | tileable at min_frag 6 | **T4 coarse, K<=3** | LSTM | transformer |
|---|---|---|---|---|---|---|
| 10-12 | 2.07 | 2.0 | 0.217 | **1.000** | 0.092 | 0.125 |
| 13-15 | 2.29 | 2.0 | 0.550 | **0.983** | 0.067 | 0.150 |
| 16-19 | 2.34 | 2.0 | 0.683 | **1.000** | 0.117 | 0.192 |
| 20-36 | 2.44 | 2.0 | 0.875 | **0.800** | 0.167 | 0.183 |

> ⛔ **Lengthening the task does not close the gap.** Over a 3x range of path length the lookup
> table moves from 1.000 to 0.800 while the best model moves from 0.125 to 0.183. The curves are
> separated by a factor of 4-8 everywhere and are not converging. **There is no usable window on
> this axis.** Note also that N3-rel median stays pinned at **2.0** for *every* length bucket,
> including 20-36-step paths: a 30-step held-out path still tiles from two stored behaviours.

**F2 — as a function of how much training data the table may harvest** (coarse band, K<=3, same
byte budget, 160 episodes):

| harvested from | fragments | bytes used | T4 coarse K<=3 |
|---|---|---|---|
| 2,500 episodes | 45,162 | 294,221 | **0.931** |
| 10,000 episodes | 132,095 | 853,397 | **0.981** |
| 40,000 episodes | 175,768 | 1,001,187 | **0.900** |

> ⛔ **The tripwire is not data-bound.** Fragments from **2,500** training episodes, using under
> **30 %** of the byte budget, already reach 0.931 — within 5 points of the full table. Nothing
> about the size of the training corpus is what makes this fire, so no amount of split
> engineering or data curation addresses it. (The 40 k row being *lower* than the 10 k row is an
> artefact of my own cheapest-first pool truncation: with more source episodes the candidate pool
> fills with more short fragments and crowds out long ones. Reported rather than smoothed.)

## 2026-09-20T02:05Z — the structural sweep's route flip, with ties removed

The 11.49 % headline resolved cost ties to "tunnel". The **strict** version — the tunnel is
strictly cheaper at some V *and* strictly more expensive at another, independent of tie-breaking —
over the same 190 layouts x 40 goals, exhaustive in (start cell x start velocity):

**1,518,313 of 16,405,110 = 9.26 %.** The effect is not a tie-breaking artefact.

---

## 2026-09-20T02:14Z — ⭐⭐ WHY the tripwire fires: the byte budget buys more macro-actions than the environment has states

This is instrument characterisation, not hardening: GYM-2 is condemned and stays condemned;
nothing about the gym, the split or the metric is changed. What is swept is the **tripwire's own
operating point**, because that is the number wave 7 needs before it picks an environment.

The arithmetic. A fragment of length `min_frag = 6` costs `frag_bytes = 5`, so a budget of B buys
`B/5` macro-actions. The environment has `G^2 (2V+1)^2 = 400 x 81 = 32,400` distinct
(cell, velocity) states. At the parameter-matched budget the table can afford **5.08 macro-actions
per state**. A table with a useful macro-action available at every state *is* a planner, whatever
the gym looks like.

Coarse band (`min_frag = 6`), 200 held-out episodes, budget swept:

| budget (bytes) | fragments | macro-actions **per state** | K=1 | K=2 | **K<=3** |
|---|---|---|---|---|---|
| **1,001,188** (matched) | 164,524 | **5.078** | 0.0100 | 0.7850 | **0.9750** |
| 300,000 | 56,244 | 1.736 | 0.0000 | 0.4900 | **0.9150** |
| 100,000 | 20,000 | 0.617 | 0.0000 | 0.2300 | **0.8050** |
| 30,000 | 6,000 | 0.185 | 0.0000 | 0.0400 | **0.4900** |
| 10,000 | 2,000 | 0.062 | 0.0000 | 0.0050 | **0.1550** |
| 3,000 | 600 | 0.019 | 0.0000 | 0.0000 | **0.0000** |
| 1,000 | 200 | 0.006 | 0.0000 | 0.0000 | **0.0000** |

> ⭐ **The coarse band crosses the 0.60 falsification line at a budget of roughly 46,000 bytes —
> about 11,000 parameters — which is about 0.28 macro-actions per state.** At the
> parameter-matched budget the table has **18x more** than that.

Turned into a spec wave 7 can apply before building anything:

> ⭐⭐ **A matched-bytes stitching tripwire can only gate an environment whose state space is large
> compared with `4 * n_params / frag_bytes`.** For 250 k parameters and 5-byte macro-actions that
> is 200,238 entries, so the environment needs of order **7 x 10^5 states** for the tripwire to
> be informative — a 100x100 grid at these velocities, not a 20x20 one. **GYM-2 has 32,400.**
> ⛔ Under-sizing the state space is not a flaw in *this* gym; it is a flaw that any small
> gridworld shares, and it is why both GYM-1 and GYM-2 failed the same gate for what look like
> different reasons.

## 2026-09-20T02:10Z — re-verifying the parent on the parent's own data

The brief says to treat the parent's numbers as reading, not truth. Run against
`w5-gym1/splits.pkl` with the parent's own code:

- **R3 reproduces its headline coarse band exactly.** min_frag 12, K<=3, 1,000 held-out episodes,
  budget 997,896: K=1 **0.2410**, K=2 **0.7770**, **K=3 = 0.9700** on 26,319 bytes / 4,602
  fragments. The parent reported **0.970**. Verified.
- ⚠ **One small correction to the parent.** It stated "T4's hit rate equals its `valid_and_reached`
  to four decimals at every K". In this coarse band it does not: hit 0.789 vs vr 0.777 at K=2,
  and hit 0.985 vs vr 0.970 at K=3. About 1.5 % of the plans its fast legality test accepted
  **faulted on replay**. Its reported score is the *replay* score, so the headline is unaffected
  and if anything conservative — but the self-consistency claim does not hold at min_frag 12.
  My GYM-2 stitcher does satisfy it: `hit_rate == valid_and_reached` to four decimals at all
  fifteen (min_frag, K) settings.
- **R1: GYM-1's min_frag=3 table is NOT behaviourally complete either** — coverage of the legal
  3-step macro-action alphabet is 0.411 / 0.497 / 0.547 / 0.576 / 0.592 for V = 2..6, on 8,359
  fragments and 41,657 bytes (reproducing the parent's 41,657 exactly). So the TW-C degeneracy
  argument I pre-registered is wrong for *both* gyms, and the parent's K=10 row is not merely
  "the table is a complete planner". Half the alphabet is enough.
- **R2: GYM-1's diversity, measured the way GYM-2's was**, so the two can be ranked on one
  statistic:

| | GYM-1 (1-D) | GYM-2 (2-D) |
|---|---|---|
| N2c behaviour string novel | 0.389 | **0.965** |
| N3-rel mean | 1.389 | **2.216** |
| N3-rel median | **1.0** | **2.0** |
| N3-rel uncoverable | 0.0007 | **0.026** |

  GYM-2 is genuinely and measurably more diverse on every one of these. It still fails.

---

# FINAL REPORT — w5.1/gym2

**INSTRUMENT ONLY.** ⛔ This spoke makes **no claim about the CLU**, and **spent no scored run**.
It built a second measuring device, gated it before adopting it, and reports what the gates said.

## 0. Verdict

> ⛔⛔ **GYM-2 is not an instrument either. The matched-bytes trajectory-lookup tripwire FIRES:
> TW-B = 0.9825 against a pre-registered falsification threshold of 0.60, and the lookup table
> beats the better trained rules-visible 250 k-parameter baseline by 5.7x.
> Wave 7 must not be scoped on this gym.**
>
> ⭐ **But the reason is NOT the reason GYM-1 failed, and the difference is the deliverable.**
>
> ⭐⭐ **The single most load-bearing thing these two spokes establish** (section 11, where I test
> my own recommendation and it comes out half right): to test "compose rule A with rule B to find
> a different route", at least one rule must be able to render a route **ILLEGAL**, not merely
> slower. A speed limit is an upper bound and can only ever change a route's cost, so no metric
> and no number of dimensions makes it composable. That is a statement about the rule algebra.
> Two dimensions did exactly what the parent asked for — the obstacle now bends **41.85 %** of
> solvable paths where in 1-D it bent **0 of 25,261,855** — and whole-trajectory retrieval
> collapsed from 0.7027 to 0.0003. GYM-2 fails on two *new*, more general grounds, and both of
> them are quantitative and actionable.

I stopped hardening GYM-2 the moment the tripwire fired, as the task requires, and spent the rest
of the time characterising **why** it fired and what that costs wave 7.

## 1. The five gates, decided before adoption

| gate | statistic | value | threshold | |
|---|---|---|---|---|
| **D** path diversity | N3-rel median (translation-invariant) | **2.0** | >= 4, hard-fail <= 2 | ⛔ **FAIL** |
| **T** lookup tripwire | TW-A whole-trajectory | **0.0275** | fires above 0.50 | ✅ pass |
| | **TW-B coarse recombination** | **0.9825** | fires above 0.60 | ⛔⛔ **FIRES** |
| | TW-C fine recombination (not a gate) | 0.9975 | — | reported |
| | **TW-D lookup vs best baseline** | **0.9825 vs 0.1733 = 5.7x** | fires at >= 1.0x | ⛔⛔ **FIRES** |
| **O** rule-blind oracles | O1 obstacle-blind | 0.0387 | < 0.90 | ✅ pass |
| | O2 speed-blind | 0.4213 | < 0.90 | ✅ pass |
| | **O3 composition-blind** | **1.0000** | < 0.90 | ⛔ **FAIL** |
| **H** headroom | best baseline `valid_and_reached` | **0.1733** | in (0.05, 0.95) | ✅ pass |
| **V** exact validity | tolerance used anywhere | **none** | zero | ✅ pass |

Every threshold was written down in this log at 2026-09-20T00:52Z, before GYM-2 existed, with one
correction at 01:28Z — made **before any diversity, tripwire, oracle or baseline number was
computed** — because the N3 gate as first drafted was arithmetically impossible to pass. Both the
original and the correction are above.

## 2. ⭐ What two dimensions FIXED — reported first, because it is the part worth keeping

**The obstacle is now load-bearing.** Exhaustive sweep, 190 layouts x 4 speed limits x 40 goals,
**exhaustive in (start cell x start velocity)**: **109,440,000 configurations**, 87,959,520
solvable.

| | GYM-1 (1-D) | GYM-2 (2-D) |
|---|---|---|
| solvable configurations swept | 25,261,855 | **87,959,520** |
| ... made strictly more expensive by the obstacle | **0** | **36,810,755 = 41.85 %** |
| ... made infeasible by the obstacle | — | 21,480,480 |
| optimal ROUTE changes with the speed limit | not expressible | **1,882,092 / 16,377,426 = 11.49 %** |
| ... same, with cost ties removed | — | **1,518,313 / 16,405,110 = 9.26 %** |

The route mix moves monotonically as V rises — strict-tunnel 3,445,145 -> 1,514,206, strict-detour
4,140,994 -> 5,902,182 — and the flips concentrate at V1->V2 (1,154,277) as the
crawl-versus-cruise trade-off predicts. Flip rates are flat across tunnel-length buckets
(12.8 / 10.4 / 11.2 %), so this is the mechanism, not a corner of the generator.

**Whole-trajectory retrieval died.** T1 **0.0000**, T2 **0.0003**, T3 **0.0003**, against GYM-1's
T3 = 0.7027. In 2-D a stored trajectory replayed anywhere else simply collides.

**Both rules are genuinely read by the models.** In-distribution ablation (permute the rule across
episodes so every token stays a value the model has seen):

| | true | obstacle map shuffled | V shuffled |
|---|---|---|---|
| LSTM | 0.1127 | 0.0613 (**-0.0514**) | 0.0767 (**-0.0360**) |
| transformer | 0.1733 | 0.0987 (**-0.0746**) | 0.1193 (**-0.0540**) |

On GYM-1 the same ablation cost the LSTM 0.0140 and *helped* the transformer by 0.0020 — noise.

**The persistence protocol transfers.** Survival @48 / @96: LSTM 0.3900 / **0.3900**, transformer
0.3020 / **0.1627**. Parked-on-goal @24/36/48/96: LSTM 0.1053 / 0.1120 / 0.1127 / 0.1127;
transformer 0.1993 / 0.0607 / **0.0087** / 0.0113. The transformer reaches the goal twice as often
and holds it at one thirteenth the rate; its at-goal curve is non-monotonic and its survival
halves when the horizon merely doubles. ⭐ **Goal-reach ranks the transformer first, persistence
reverses it — on a completely different gym from the one where the parent found this.** This is
the piece of the parent's machinery that has now been validated twice.

## 3. ⛔ Failure 1 — the composition is real but STRUCTURALLY UNMEASURABLE (GATE O)

| oracle | blind to | `valid_and_reached` | faults caused |
|---|---|---|---|
| optimal-plan control | nothing | 1.0000 | 0 |
| O1 obstacle-blind | the obstacle field | **0.0387** | 5,673 collisions |
| O2 speed-blind | the speed limit | **0.4213** | 3,533 speed violations |
| **O3 composition-blind** | **only the interaction** | **1.0000** | **0** |

O3 keeps the route optimal at `V = V_MAX`, then re-plans **optimally under the true V, restricted
to that route**. Both rules fully respected; only the knowledge that *which route is cheaper
depends on how fast you may go* is removed. It differs from the true optimum on **8.69 %** of the
1,347 route decisions and costs **0.164 extra steps** on average — and **zero** faults.

> ⭐⭐ **The mechanism, and it is general to every dimension and every gridworld.** A speed limit
> is an **upper bound**, so raising it can never make a route infeasible. Composing a speed limit
> with an obstacle can therefore only ever change a route's **cost**, never its **feasibility**.
>
> ⛔ **"Compose the speed limit with the obstacle to find a different route" cannot be scored by
> `valid_and_reached`, `all_steps_valid`, `reached_goal` or `held_at_goal` — by any
> feasibility-or-goal-reaching metric at all.** Going to 2-D did not fix this and no number of
> dimensions will.

This is strictly stronger than the parent's finding. There the wall-blind oracle scored 1.0
because the obstacle did nothing. Here deleting the obstacle costs 96.1 points and relaxing the
speed limit costs 57.9 points, and **their interaction is still worth 0.0000 on the metric.**

## 4. ⛔ Failure 2 — the tripwire fires, and the byte budget is why (GATE T)

Budget **1,001,188 bytes** = 4 x 250,297 parameters, the parent's rule. `valid_and_reached` on
400 held-out episodes across 135 layouts:

| min_frag | fragments | bytes used | K=1 | K=2 | K=3 | K=4 | K=6 |
|---|---|---|---|---|---|---|---|
| 3 | 182,016 | 1,001,182 | 0.0025 | 0.7475 | 0.9650 | 0.9950 | 0.9975 |
| 4 | 178,599 | 1,001,182 | 0.0025 | 0.7500 | 0.9625 | 0.9950 | 0.9975 |
| **6** (`MIN_FRAG_GATE`) | 164,524 | 1,001,183 | 0.0050 | **0.8000** | **0.9825** | 0.9950 | 0.9950 |
| 8 | 145,079 | 1,001,186 | 0.0225 | **0.8450** | **0.9800** | 0.9900 | 0.9900 |
| 10 | 88,532 | 668,764 | 0.0275 | 0.7675 | **0.9775** | 0.9850 | 0.9850 |

**Two** stored macro-actions of >= 8 steps solve **84.50 %**; **three** of >= 6 steps solve
**98.25 %**. Tightening `min_frag` makes the table *better* at K=2, because the floor forces the
fixed budget to buy long macro-actions instead of a pile of short ones. There is no min_frag at
which this band is safe. At min_frag 10 the budget is not even binding — 66.8 % used.

`hit_rate == valid_and_reached` to four decimals at **all fifteen** settings, so every plan the
exact footprint test accepted also replayed cleanly. The tripwire is not scoring itself.

### ⭐⭐ Why it fires, as arithmetic rather than as a property of this gym

A `min_frag = 6` macro-action costs 5 bytes, so the budget buys `1,001,188 / 5 = 200,238` of
them. The environment has `G^2 (2V+1)^2 = 400 x 81 = 32,400` distinct (cell, velocity) states.
**The table can afford 5.08 macro-actions per state.** A table with a useful macro-action at every
state *is* a planner, whatever the gym looks like. Sweeping the budget confirms it exactly:

| budget | fragments | macro-actions **per state** | K<=3 |
|---|---|---|---|
| 1,001,188 (matched) | 164,524 | **5.078** | **0.9750** |
| 300,000 | 56,244 | 1.736 | 0.9150 |
| 100,000 | 20,000 | 0.617 | 0.8050 |
| 30,000 | 6,000 | 0.185 | **0.4900** |
| 10,000 | 2,000 | 0.062 | 0.1550 |
| 3,000 | 600 | 0.019 | 0.0000 |

> ⭐ **The coarse band crosses the 0.60 line at about 46,000 bytes — roughly 11,000 parameters —
> i.e. at about 0.28 macro-actions per state. At the matched budget the table has 18x more.**
>
> ⭐⭐ **Spec for wave 7: a matched-bytes stitching tripwire can only gate an environment whose
> state space is large compared with `4 * n_params / frag_bytes`.** At 250 k parameters and
> 5-byte macro-actions that is 200,238 entries, so the environment needs of order **7 x 10^5
> states** — a 100x100 grid at these velocities, not a 20x20 one. **GYM-2 has 32,400.**
> ⛔ Under-sizing the state space is not a flaw in *this* gym; it is a flaw every small gridworld
> shares, and it is why GYM-1 and GYM-2 failed the same gate for what looked like different
> reasons.

## 5. GATE D, and the frontier between "lookup-proof" and "learnable"

1,500 held-out episodes, exact substring tests against the whole 60,000-episode corpus.

| | GYM-1 | GYM-2 | required |
|---|---|---|---|
| N1 key novel | 1.000 | **1.000** | 1.000 ✅ |
| N2a position path novel | 0.509 | **0.988** | >= 0.95 ✅ |
| N2b action sequence novel | 0.033 | **0.964** | >= 0.90 ✅ |
| N2c behaviour string novel | 0.389 | **0.965** | >= 0.90 ✅ |
| **N3-rel median** | **1.0** | **2.0** | >= 4 ⛔ **FAIL** |
| N3-rel mean | 1.389 | 2.216 | |
| N3-rel uncoverable | 0.0007 | 0.026 | |

⭐ Surface novelty improved by a factor of 29 (action sequences 0.033 -> 0.964). **Fragment
novelty moved by one fragment.** And the result is not layout memorisation: on layouts appearing
**nowhere** in training, N3-rel median is still **2.0** (mean 2.266). What a stitching table
re-uses is the velocity-profile alphabet, which belongs to the dynamics, not to the map.

**The frontier, measured** (480 held-out episodes, 120 per length bucket):

| path length | N3-rel mean | T4 coarse K<=3 | LSTM | transformer |
|---|---|---|---|---|
| 10-12 | 2.07 | **1.000** | 0.092 | 0.125 |
| 13-15 | 2.29 | **0.983** | 0.067 | 0.150 |
| 16-19 | 2.34 | **1.000** | 0.117 | 0.192 |
| 20-36 | 2.44 | **0.800** | 0.167 | 0.183 |

and as a function of how much data the table may harvest (same budget):

| harvested from | fragments | bytes used | T4 coarse K<=3 |
|---|---|---|---|
| 2,500 episodes | 45,162 | 294,221 | **0.931** |
| 10,000 episodes | 132,095 | 853,397 | **0.981** |
| 40,000 episodes | 175,768 | 1,001,187 | **0.900** |

> ⛔ **There is no usable window on either axis.** Over a 3x range of path length the table moves
> 1.000 -> 0.800 while the best model moves 0.125 -> 0.183 — separated by 4-8x everywhere and not
> converging. And the table is **not data-bound**: 2,500 training episodes and 30 % of the budget
> already reach 0.931. Neither longer tasks nor better splits nor less training data is the lever.
> The lever is the state-space size in section 4.

## 6. Baselines — rules visible, parameters matched, counts printed

| | params | steps | seconds | final one-step loss |
|---|---|---|---|---|
| LSTM `d=64, h=200` | **250,289** | 6,000 | 680 | 0.3531 |
| transformer `d=64, 3L, 4H, d_ff=400` | **248,761** | 6,000 | 882 | 0.3280 |

Relative gap **0.61 %** against a pre-registered 2 % tolerance. Both see `V`, the **whole** 20x20
obstacle grid (learned linear projection) and a 7x7 egocentric window on it, at every step.
1,120,631 supervised one-step targets. Exact validity on 1,500 held-out episodes, horizon 48,
free-running and unmasked:

| | LSTM | transformer |
|---|---|---|
| **valid_and_reached** | 0.1127 | **0.1733** |
| all steps valid | **0.3900** | 0.3020 |
| goal reached | 0.1127 | **0.2347** |
| held at goal | **0.1127** | 0.0087 |
| speed violations / collisions / out-of-bounds | 83 / 8,772 / 83 | 306 / 9,605 / 505 |
| first fault: speed / obstacle / out | 12 / 877 / 26 | 41 / 868 / 138 |
| novel layouts, `valid_and_reached` | 0.0887 | 0.1293 |

**Refusal split** (1,000 episodes on sealed maps — goal unreachable, a fault-free behaviour
exists): LSTM `all_valid` **0.1350** with 10,467 collisions; transformer **0.0850** with 16,977.
Neither was trained on unsolvable episodes, so this is expected; it is reported because it
confirms the collision metric can fire.

## 7. Is validity checked exactly? YES (GATE V)

Every predicate is integer or exact-rational with **zero tolerance**.

- **Swept-cell collision**: a cell is hit iff the closed segment between the two cell centres
  meets its closed unit square, decided once at import with `fractions.Fraction` over 81
  displacements. Not a sampled line trace, not a bounding box. Consequence stated rather than
  hidden: a diagonal step grazes both orthogonal neighbours' corners, so it needs all four cells
  free — the standard no-corner-cutting rule, exact, not a safety margin.
- **Speed**: `|vx'| > V or |vy'| > V`, integers. **Out of bounds**: tested on **every** swept
  cell, so leaving the grid mid-step and returning is still recorded. **Goal**: `x == gx and
  y == gy and vx == vy == 0`, every preceding step valid.
- **No masking at rollout.** `verify_unmasked_rollout()` drives a deliberately illegal constant
  `(+1,+1)` policy and confirms the environment applies it verbatim: velocity reaches 2 against a
  limit of 1, the violation is recorded, and the recorded states equal raw integer arithmetic.
- **Batched vs scalar**: field-for-field on 300 episodes, **0 mismatches**.
- Two caveats, both measured rather than hedged. The velocity embedding clips model *input* at
  `|v| <= 8`; the largest `|v|` reached in any rollout was **6** (LSTM) and **7** (transformer),
  and **0.0000** of episodes ever exceeded it. After an out-of-bounds excursion the continuing
  rollout puts the agent back on the grid so the horizon can be counted out; episode-level
  validity is computed **before** any such repair.

> **No downstream refusal or deletion claim built on this gym would be approximate on account of
> the validity check.** That part of the instrument is sound and is re-usable as-is.

## 8. Where I was wrong, and corrections to the parent

- ⚠ **My pre-registered explanation of TW-C was wrong.** P3 predicted the min_frag=3 table would
  be behaviourally *complete*, making TW-C degenerate. I said I would verify rather than assert.
  Verified and **false**: coverage of the legal 3-step macro-action alphabet is 0.319 / 0.401 /
  0.337 / **0.218** for V = 1..4, and the table still reaches 0.9975. A fifth to two fifths of the
  alphabet is enough to re-plan. That is worse for the gym than what I predicted, and it removes
  the excuse I had pre-written for that row. The same is true on GYM-1 (coverage 0.411-0.592).
- ⚠ **My N3 gate as first drafted was impossible to pass** (`min_frag = half the path length` caps
  the fragment count at 2, so `>= 4` could never hold). Corrected at 01:28Z before any number was
  seen; both versions are above.
- ⚠ **A 32-byte budget error, in the table's favour.** The budget was computed as 4 x 250,297
  using a transformer with `max_len = 128`; the transformer actually built has `max_len = 104` and
  248,761 parameters, so the matched maximum is the LSTM's 250,289 and the budget should have been
  1,001,156. The tripwire therefore got 0.003 % more bytes than the rule allows — the safe
  direction, since it only strengthens a tripwire that fired.
- ⚠ **One correction to the parent.** Its claim that "T4's hit rate equals its `valid_and_reached`
  to four decimals at every K" does **not** hold in its own coarse band: re-run on its data,
  hit 0.789 vs vr 0.777 at K=2 and hit 0.985 vs vr 0.970 at K=3, so ~1.5 % of the plans its fast
  legality test accepted faulted on replay. Its reported score is the replay score, so its
  headline is unaffected and if anything conservative. Its **0.970** coarse-band number reproduces
  **exactly**.
- ⚠ **The parent's N3 over-states diversity**, as predicted before this run: on the same GYM-2
  episodes N3-abs = 2.74 and N3-rel = 2.22, and uncoverability collapses from 15.1 % to 2.6 %.
  Absolute-position tiling credits a gym with novelty a translation-invariant table does not pay
  for. Future gyms must report N3 the translation-invariant way.

## 9. What wave 7 should take from two failed gyms

1. ⭐⭐ **Do not score composition with a feasibility metric — but see section 12, where I tested
   this recommendation and it came out HALF RIGHT.** A speed limit is an upper bound, so its
   composition with an obstacle changes route *cost*, never route *feasibility*. Measured here:
   O3 = 1.0000 and 0.164 extra steps. Switching to an **optimality-gap** metric does fix the
   tripwire (TW-B falls 0.9825 -> 0.3900) but it does **not** fix GATE O (O3 only falls to
   0.9367, still above the 0.90 line) and it does **not** fix TW-D (the table still beats the
   best model 4.9x). What wave 7 actually needs is a rule whose composition can make a route
   **illegal** — a *minimum* speed, a fuel or time budget, a one-way constraint. A pure upper
   bound cannot be made measurable by changing the metric.
2. ⭐⭐ **Size the state space against the byte budget before building anything.** The matched-bytes
   stitching tripwire is informative only when `states >> 4 * n_params / frag_bytes`. At 250 k
   parameters that is of order 7 x 10^5 states. Measured collapse point: ~46,000 bytes, ~0.28
   macro-actions per state.
3. ⛔ **Exact-key lookup is not a tripwire.** T1 = 0.0000 on GYM-1 and 0.0000 on GYM-2 — two gyms
   that fail for different reasons, both of which it would have certified. Always run fragment
   stitching with a `min_frag` sweep, and report the whole (min_frag x K) grid.
4. ⭐ **Report N3 translation-invariantly, and report it before anything else.** Surface novelty
   is easy to buy (0.033 -> 0.964 just by going to 2-D) and means nothing; fragment novelty barely
   moved (1.39 -> 2.22) and is what binds.
5. ⭐ **Keep, unchanged:** the zero-tolerance exact validity predicates and swept-cell geometry;
   the unmasked rollout with its adversarial verification; the batched-vs-scalar equality check;
   the one-step persistence protocol with survival and at-goal-over-time curves, now validated on
   two unrelated gyms; the rule-blind oracle gate, which is nearly free and was decisive here;
   and pre-registration with a written falsification threshold, which is what made both of these
   results reportable rather than negotiable.

## 10. NOT-RUN declarations (declared, not null)

- **Scored run: NOT RUN.** Mandated — this spoke spends none.
- **Any claim about the CLU: NOT MADE.** Nothing here is evidence for it or against it.
- **GPU run: NOT RUN.** Not needed, so not justified. The whole study — a 109.4 M-configuration
  exhaustive sweep, 65,500 episodes with provably optimal ground truth, the full (min_frag x K)
  tripwire grid, a budget sweep and both baselines — runs in CPU-minutes to CPU-tens-of-minutes on
  3 threads. No GPU justification lines were written because no GPU run was contemplated.
- ⛔ **Further hardening of GYM-2: NOT RUN, deliberately**, from the moment TW-B fired at 01:34Z,
  under the task's stop rule. Everything after that timestamp is characterisation (why it fired,
  the frontier, the budget operating point) or a *mandatory* remaining gate (GATE O, and the
  baselines TW-D needs). The gym, the split and the metric were not touched again.
- **GYM-3: NOT BUILT.** A third gym would have needed a different metric (section 9.1) and a state
  space ~20x larger (section 9.2); designing one on the back of a fired tripwire is exactly what
  the stop rule forbids, and a second decisive negative with a spec attached is worth more.

## 11. ADDENDUM (02:21Z) — I tested my own recommendation, and it is only half right

Section 9.1 tells wave 7 to score composition with an **optimality gap** (steps against the
provably optimal cost) instead of a feasibility metric. A recommendation that has not been run
against the data is worth nothing, so I ran it. 300 held-out episodes, mean optimal cost 13.84;
`within +s` = fault-free, parked on the goal, and within `s` extra steps of provably optimal.

| policy | `valid_and_reached` | mean extra steps when solved | **+0** | +1 | +2 | +4 | +8 |
|---|---|---|---|---|---|---|---|
| **T4 coarse, K<=3** (the tripwire) | 0.9700 | 1.25 | **0.3900** | 0.6267 | 0.8467 | 0.9267 | 0.9700 |
| **O3 composition-blind** | 1.0000 | 0.22 | **0.9367** | 0.9567 | 0.9733 | 0.9867 | 0.9933 |
| LSTM | 0.1100 | 1.82 | **0.0767** | 0.0800 | 0.0900 | 0.0933 | 0.1033 |
| transformer | 0.1733 | 5.02 | **0.0800** | 0.1033 | 0.1200 | 0.1333 | 0.1433 |

Three conclusions, and they do not all point the same way:

1. ✅ **The cost metric rescues the TRIPWIRE.** TW-B falls from **0.9825 to 0.3900** — below the
   pre-registered 0.60 line. Stitched plans are valid and they reach, but they are *sloppy*:
   1.25 extra steps on average and only 39 % exactly optimal. A stitching table is a bad planner
   even when it is a sufficient one, and an optimality metric sees that where a feasibility
   metric cannot. ⭐ This is a real, re-usable improvement to the instrument.
2. ⛔ **It does NOT rescue GATE O.** O3 only falls from 1.0000 to **0.9367**, still above the
   0.90 pass line. Being blind to the interaction between the speed limit and the obstacle costs
   6.3 points of exact optimality and 0.22 steps. The composition is simply too cheap to measure,
   under *either* metric. **My section 9.1 recommendation, as originally written, would not have
   fixed the thing it was written to fix.**
3. ⛔ **It does NOT rescue TW-D.** Under the cost metric the table scores 0.3900 against the best
   model's 0.0800 — still **4.9x**. The gap between a trivial table and a trained model is
   metric-independent.

There is also a second architecture separation in this table, of the same shape as the
persistence result: when the transformer does solve an episode it takes **5.02** extra steps
against the LSTM's **1.82**. It reaches more often, holds worse, and wanders further.

> ⭐⭐ **Corrected recommendation.** Score with an optimality gap — it is strictly better and it
> is what makes the tripwire informative again. But do **not** expect it to make an upper-bound
> rule composable. To test "compose rule A with rule B and find a different route", at least one
> of the rules must be able to render a route **illegal**, not merely slower: a *minimum* speed,
> a fuel or step budget, a one-way constraint, a time window. That is a statement about the rule
> algebra, not about the metric and not about the number of dimensions, and it is the single
> most load-bearing thing these two spokes have established.

## 12. Artifacts

`START_UTC.txt` · `LOG.md` · `splits.pkl` · `splits_summary.json` · `diversity.json` ·
`structure.json` · `strictflip.json` · `tripwire.json` · `budget_sweep.json` · `frontier.json` ·
`param_match.json` · `train_report.json` · `eval.json` · `cost_metric.json` · `gym1_recheck.json` ·
`ckpt_lstm.eqx` · `ckpt_transformer.eqx` · and the stdout log of every run
(`split_ diversity_ structure_ tripwire_ train_ eval_ frontier_ budget_ strictflip_ gym1_recheck_`).

Code (3,950 lines): `chlu/gym2/{grid2d,layouts,episodes,tripwire,models,train,vecroll}.py` and
`chlu/experiments/exp_w51_gym2_{split,diversity,structure,strictflip,tripwire,train,eval,frontier,budget,costmetric}.py`
plus `chlu/experiments/exp_w51_gym1_recheck.py`.

---

## 2026-09-20T02:21Z — committed and bundled

Working tree clean. Two commits on top of the parent's tip `bfee159` (itself 3 commits on top of
`base/iter5 = 6675904`):

- `18cfa0d` GYM-2: 2-D gym, gated before adoption; tripwire FIRES, report and stop
- `f519c53` cost-metric addendum: test the recommendation instead of asserting it

`w51-gym2.bundle` — `refs/heads/w5.1/gym2 = f519c53`, `git bundle verify` reports
**"the bundle records a complete history"**.

Elapsed 1 h 37 m of an 8 h allowance. ⛔ Stopping here rather than spending the remaining time:
the tripwire fired at 01:34Z and the task's stop rule forbids hardening past it. Every mandatory
gate has been run and reported, the frontier the task asked for has been measured on both axes,
the cause has been reduced to arithmetic with a spec attached, the parent has been re-verified on
its own data, and the one recommendation I wanted to make has been tested and corrected. A third
gym would need a different rule algebra (section 11) and a state space ~20x larger (section 4);
designing one on the back of a fired tripwire is exactly what the stop rule forbids.

**Safety:** no box-wide kill was ever issued. The one process I terminated (PID 28818, my own
first `exp_w51_gym2_split` run, killed at 00:56Z when I found `sample_refusal` would spin against
a connectivity invariant) was located by its own start time and killed by PID. Five python
processes belonging to other agents were running throughout and were left untouched.
