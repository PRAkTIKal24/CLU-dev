# w5-gym1 — LOG

Spoke: `w5/gym1` · agent: experiment-engineer · **INSTRUMENT ONLY — no claim about the CLU, no scored run.**
Task file: `/c/chlu/tasks-w5/w5-spoke-gym1.md` · Worktree: `/c/chlu/worktrees/CHLU-w5-gym1` · Base `base/iter5 = 6675904`.

Dated bullets, appended throughout, never rewritten. Corrections are new bullets. Final report is the last section.

---

- **2026-09-19T23:33Z** — `START_UTC.txt` written to `/c/chlu/outputs/w5-gym1/START_UTC.txt` (`2026-09-19T23:33:03Z`).
  Checked first: no live `START_UTC.txt` existed in the output dir. Proceeding.

- **2026-09-19T23:34Z** — Environment verified before trusting any number:
  `chlu.__file__ = C:\chlu\worktrees\CHLU-w5-gym1\chlu\__init__.py` (inside my worktree — good).
  jax 0.9.0 / numpy 2.4.1 / equinox 0.13.4 / python 3.11.9 / devices `[CpuDevice(id=0)]`. System python, no venv created, nothing pip-installed.

- **2026-09-19T23:36Z** — **PRE-REGISTRATION.** Written *before* the tripwire is built or run, so that the
  headline tripwire number cannot be chosen after the fact. Everything below is fixed from here; deviations
  will appear as later dated bullets, marked DEVIATION.

  **GYM-1 v1 — the environment.** One agent on a line, with momentum, a speed limit, and a wall.
  - Line `L = 128` cells, `x in [0, 127]`. Velocity `v` is part of the state.
  - **Action** = acceleration `a in {-4..+4}` (9 actions). `v' = v + a`, `x' = x + v'`.
  - **Rule 1 (speed limit)** `|v'| <= V`, with `V in {1,2,3,4,5}` sampled per episode.
  - **Rule 2 (wall)** the closed swept segment `[min(x,x'), max(x,x')]` must not intersect the wall cells.
    Wall = contiguous cells `[w0, w0+ww-1]`, `ww in {1,2,3,4}`, sampled per episode.
    Swept-segment (not endpoint) intersection is what makes the wall uncrossable: jumping *over* it is a crossing too.
  - **Rule 3 (bounds)** `x' in [0, 127]`.
  - **Goal** reach `x = g` **at rest** (`v = 0`) and **stay** there.
  - The action alphabet is deliberately wider than the rules allow (`|a|` up to 4 vs `|v'| <= V`, and `a`
    can drive `x'` off the line), so every violation class is *expressible by the model*. A gym whose
    interface makes violations impossible cannot measure refusal.

  **Why momentum.** Without it, a greedy step-toward-goal is optimal and the wall never bends a single
  solvable path — the obstacle would be decorative. With momentum the agent can overshoot into the wall,
  so braking distance makes the wall load-bearing for *solvable* pairs, not just for solvability.

  **Ground truth.** The `(x, v)` transition graph has `128 * (2*5+1) = 1408` states; optimal paths and
  solvability come from exhaustive BFS on it. No heuristic teacher, no tolerance.

  **Pre-registered primary held-out split** (the number I will report as the headline):
  - Compositional split on the **rule combination** `(V, ww)`: all 5 `V` values and all 4 `ww` values
    appear in training, but 5 of the 20 `(V, ww)` combinations are held out entirely.
    Held-out combos, fixed now: `(1,4), (2,2), (3,1), (4,3), (5,2)`.
  - Held-out episodes are additionally required to be **wall-binding**: the optimal action sequence for
    the *same* episode with the wall removed is *invalid* in the walled environment. Exactly checkable.
  - Novelty filters, exactly checked and reported, not assumed:
    - **N1** held-out `(V, w0, ww, s, v0, g)` tuple absent from train.
    - **N2** held-out optimal **position sequence** is not a contiguous subsequence of any train position sequence.
    - **N3** minimum number of train-derived contiguous fragments (each length >= 3) needed to cover the
      held-out position sequence — exact DP. N3 is the anti-recombination measurement; it upper-bounds
      what any stitching lookup table can do.

  **Pre-registered tripwire.** Matched-**bytes** trajectory lookup. Budget `B = 4 * (#params of the
  parameter-matched neural baseline)` bytes (fp32). Four lookup strategies, all scored by *rolling the
  retrieved action sequence in the real environment* and measuring the same exact validity metrics as the
  neural models. Headline = **the best of the four** (a weak tripwire is a weak instrument):
  T1 exact-key, T2 nearest-neighbour absolute-key replay, T3 nearest-neighbour relative-key replay
  (`V, ww, v0, g-s, s-w0`), T4 fragment stitching.

  **Pre-registered falsification.** If the best tripwire's `valid_and_reached` on the held-out split is
  **>= 0.90**, the gym is not testing composition, it is not an instrument, and I report that and stop
  hardening it. `>= 0.50` is a serious warning and will be reported as such. This threshold is fixed now.

  **Pre-registered validity metrics** (all exact integer predicates, zero tolerance):
  speed-limit violation `|v'| > V`; wall crossing `swept segment meets wall`; teleport `x' outside [0,127]`;
  goal reach `x = g and v = 0 at some step with every preceding step valid`; plus `held_at_goal`.

  **Pre-registered baselines.** LSTM and transformer, **rules visible in the input** (`V, w0, ww` are
  embedded tokens), parameter-matched to within 2 %, exact counts printed. Trained with a **one-step**
  cross-entropy objective only — no multi-step or rollout term in the loss — then rolled out free-running.
  **No action masking, no clamping, no projection at rollout.** Validity is measured, never imposed.
  If either baseline reaches ~100 % `valid_and_reached` held-out, there is no headroom and I report that.

- **2026-09-19T23:52Z** — Environment built and smoke-tested. `chlu/gym1/line1d.py`: integer dynamics
  `v' = v + a`, `x' = x + v'`; exact fault predicates; ground truth by exhaustive BFS over the `(x,v)`
  graph. Confirmed the swept-segment rule really does make even a **1-cell** wall uncrossable at `V=5`
  (`wall_is_uncrossable(5, 60, 1) = True`), by flood-fill of the whole transition graph, not by sampling.

- **2026-09-19T23:58Z** — **FINDING 1 (structural, exact).** With the pre-registered `A_MAX = 4` and
  `V <= 5`, the agent can always halt in a single step (`|a| >= V` ⇒ braking distance 0). Sampling found
  **zero** wall-binding episodes for `V in {1,2}` out of thousands of draws. So the obstacle cannot
  constrain a path: it can only decide solvability. That is a defect in my pre-registered config, not a
  sampler bug.
  **DEVIATION from pre-registration:** `A_MAX` 4 → **1** (an acceleration limit, giving braking distance
  `V(V-1)/2`), and `V_CHOICES` `{1..5}` → **`{2,3,4,5,6}`** (`V=1` has braking distance 0 under `A_MAX=1`
  and so is structurally incapable of ever consulting the wall). `HELD_COMBOS` re-indexed to the same
  pattern: `(2,4),(3,2),(4,1),(5,3),(6,2)`. Violations stay expressible: repeated `+1` drives `|v|` past
  any `V`, and driving at speed into the end of the line puts `x'` off it.

- **2026-09-20T00:04Z** — **FINDING 2 (structural, exact — this one is the headline instrument property).**
  I asked, over the **whole configuration space** rather than a sample, whether the obstacle ever makes a
  *solvable* episode more expensive. For every `(V, w0, ww, g)` on a grid spanning the entire line
  (`V in {2..6}`, `ww in {1..4}`, `w0` every 3rd cell from 2 to 125, `g` every 3rd cell) I compared the
  exact BFS cost-to-goal field **with** the wall against the field **without** it, over all `(s, v0)`
  simultaneously.

  > **25,261,855 solvable configurations. Cost-increasing configurations: 0. Not one.**

  Broken out by free-side length, including pockets of 0–7 cells: zero everywhere.
  **On a line, an uncrossable obstacle can never bend a valid path — it can only decide feasibility.**
  The reason is that the minimum-time 1-D profile is also the minimum-excursion one, so whenever a
  wall-respecting path exists the wall-free optimum already is one. There is no "route around" in 1-D.

  Consequences, which the Head needs **before** wave 7 is scoped on this gym:
  1. On every **solvable** episode the wall carries **no information**: a policy that ignores the obstacle
     entirely can still be perfectly valid and reach the goal. Solvable episodes cannot test whether a
     model read the obstacle.
  2. The obstacle's information content lives **entirely** in the episodes it makes *unsolvable*. In 1-D,
     "obstacle" is a refusal/feasibility signal, not a routing signal.
  3. My pre-registered "wall-binding" filter for the held-out split is therefore near-empty, and the
     non-empty cases I found under `A_MAX=4` were **tie-break artifacts** (equal-cost wall-free optima
     that happened to clip a wall cell), not genuine constraint. I am not building a held-out split on an
     artifact.

- **2026-09-20T00:06Z** — **DEVIATION from pre-registration (held-out split), forced by Finding 2.**
  The primary held-out axis moves from "wall-binding + held-out `(V, ww)`" to a **rule × geometry
  compositional split**: hold out combinations of the speed limit `V` with the displacement magnitude
  bucket `|g-s|`. Every `V` and every bucket appears in training, never in these pairings:
  `(V=2, B4), (3, B2), (4, B0), (5, B3), (6, B1)` with
  `B0=[3,10] B1=[11,20] B2=[21,32] B3=[33,48] B4=[49,70]`.
  This is where the only real composition in a 1-D speed-limit gym lives: the length of the cruise phase
  at a speed that was never held for that long. The wall is still present in every episode (validity
  metrics need a crossable-by-a-bad-model wall), and I additionally report a **wall ablation** that
  measures directly how much the obstacle contributes — Finding 2 predicts "nothing, on solvable
  episodes", and a prediction of an instrument's own weakness should be checked, not assumed.
  A separate **REFUSAL split** (goal unreachable, but a fault-free behaviour exists) carries the
  obstacle's actual information and is reported separately; it is not the headline number, and the
  refusal *claim* belongs to another spoke, not this one.

- **2026-09-20T00:41Z** — Baselines instantiated and **parameter-matched before any training**, because the
  tripwire's byte budget is defined from them: **LSTM 249,474 params**, **transformer 249,397 params**,
  relative gap **0.031 %** (well inside the pre-registered 2 %). Budget = 4 bytes/param =
  **997,896 bytes**. Splits: train 120,000 / held-out 3,000 / in-distribution control 2,783 /
  refusal 1,500. Wall proved uncrossable on the transition graph for **all 160** rule settings
  (`wall_uncrossable_all_rule_settings: true`). `wall_raises_cost` = **0.0000** on all 123,000 sampled
  episodes — Finding 2 reconfirmed on the data actually used, not inherited from the sweep.

- **2026-09-20T00:48Z** — ⛔ **TRIPWIRE RESULT — REPORTED FIRST, AND IT FIRES.**
  Matched-bytes trajectory lookup on the 3,000 held-out pairs, every strategy scored by **replaying what
  it retrieves in the real environment** under the same exact metrics a neural model gets:

  | strategy | valid_and_reached | all_valid | hit | bytes used / 997,896 |
  |---|---|---|---|---|
  | T1 exact key | 0.0000 | 0.0000 | 0.000 | 997,895 |
  | T2 NN absolute key | 0.0043 | 0.8983 | 1.000 | 997,895 |
  | T3 NN **relative** key, replay | **0.7027** | 0.8167 | 1.000 | 997,890 |
  | T4 stitch K=1 | 0.5047 | 0.5047 | 0.505 | 41,657 |
  | T4 stitch K=2 | 0.8520 | — | 0.852 | 41,657 |
  | T4 stitch K=3..6 | 0.8573 | — | 0.857 | 41,657 |
  | **T4 stitch K=8** | **0.9797** | 0.9797 | 0.980 | 41,657 |

  **Best = 0.9797.** The pre-registered falsification threshold was **0.90**. It is exceeded.
  > **A table of stored trajectories at the matched byte budget solves the held-out pairs.
  > GYM-1 as specified is NOT testing composition, and it is NOT an instrument.
  > Reporting, and stopping: I am not hardening the gym further, and wave 7 must not be scoped on it.**

  Three details that make this worse, not better, for the gym:
  1. **T3 alone — plain nearest-neighbour replay of whole stored trajectories, no stitching at all —
     already reaches 0.7027**, far past the 0.50 "serious warning" line. This is not an artifact of my
     stitching search being too clever; unmodified trajectory recall gets most of the way there.
  2. **T4 used 41,657 of 997,896 bytes: 4.2 % of its budget.** The tripwire is not byte-limited. Matching
     bytes was supposed to be the binding constraint and it is slack by a factor of 24, so a *smaller*
     table would have done the same job.
  3. The in-distribution control is **0.8958** for T3 versus 0.7027 held out. The compositional split does
     cost a lookup table something (19 points) — it is simply nowhere near enough to make the task
     lookup-proof.

  Self-consistency check, which passed: T4's `hit_rate` equals its `valid_and_reached` to four decimals at
  every K. Every plan the fast O(1) fragment-legality test claimed was valid also replayed validly in the
  exact environment, so the search and the exact rollout agree; the tripwire is not scoring itself.

- **2026-09-20T01:02Z** — **Tripwire sweep: is T4 lookup, or is it secretly re-planning?**
  The fair objection to T4 is that stitching 3-step macro-actions is planning, not recall. So I swept the
  minimum fragment length the table is allowed to reuse. `valid_and_reached` on the 3,000 held-out pairs:

  | min_frag | K=1 | K=2 | K=3 | K=4 | K=6 | K=8 | K=10 | fragments |
  |---|---|---|---|---|---|---|---|---|
  | 3 | 0.505 | 0.852 | 0.857 | 0.857 | 0.857 | 0.980 | **1.000** | 8,359 |
  | 5 | 0.458 | 0.853 | 0.857 | 0.857 | 0.958 | **1.000** | 1.000 | 7,937 |
  | 8 | 0.293 | 0.847 | 0.912 | **0.999** | 0.999 | 0.999 | 0.999 | 6,755 |
  | 12 | 0.241 | 0.763 | **0.970** | 0.972 | 0.972 | 0.972 | 0.972 | 4,602 |

  The objection does not survive. At **min_frag = 12** — pieces of at least twelve steps, which are
  substantial stored trajectory chunks, not primitives — **three** of them concatenated solve **97.0 %**
  of the held-out pairs. At min_frag = 8, four pieces give 99.9 %. This is trajectory recombination in the
  plain sense, and the gym falls to it.

- **2026-09-20T01:06Z** — **Exact novelty checks (N1/N2/N3) on 1,500 held-out episodes vs all 120,000
  training paths.** These were supposed to certify "paths must be genuinely new". They instead explain
  the tripwire:
  - **N1 = 1.000** — every held-out episode key is novel. The split is clean at the key level, so the
    failure is not a leak.
  - **N2 (position sequence) = 0.509** — only **half** the held-out optimal position sequences are absent
    from training. The other **49 %** occur **verbatim** as a contiguous substring of a training path.
  - **N2 (action sequence) = 0.033** — **96.7 %** of held-out optimal *action* sequences occur verbatim in
    training. With an alphabet of {-1, 0, +1} and optimal plans that are ramp/cruise/ramp, a new action
    string is essentially unobtainable.
  - **N3 = 1.51 fragments on average** (737 of 1,500 need **one**, 761 need **two**, 1 needs three;
    0.07 % untileable). The typical held-out "new path" **is two stored fragments**.

  N3 and the tripwire are independent measurements that agree: a task whose held-out paths are two
  stored fragments long is a task a stitching table solves, and it does. (They are not the same number —
  N3 tiles position *strings* with no velocity-consistency requirement, so it is a lower bound on the
  recombination a real macro-action search needs; T4 at K=2–3 is the operational figure. The lower bound
  and the achieved number bracket each other, which is the consistency one wants.)

  **This settles deliverable 1 negatively and it should be said plainly: in GYM-1 the requirement "held-out
  pairs whose paths are genuinely new" is not merely unmet, it is close to unattainable.** A 1-D
  speed-limited agent has a 3-symbol action alphabet and bang-bang optimal plans; there are not enough
  distinct paths for a held-out path to be new. That is a property of the task specification, not of my
  split construction — a different split cannot fix an alphabet that small.

- **2026-09-20T02:05Z** — **Rollout machinery verified before any model number is trusted.** Two checks,
  both passing, both asserted in `exp_w5_gym1_eval.py` so a regression fails the run rather than being
  reported:
  - **No guardrails.** `verify_unmasked_rollout()` drives a deliberately illegal constant `+1` policy into
    a `V=2` episode and confirms the environment applies it verbatim: velocity reaches **3 against a limit
    of 2**, the speed violation is recorded, the first fault is flagged at step 3, and the recorded states
    equal raw integer arithmetic (`v += 1; x += v`) exactly. Nothing is clamped, masked or projected while
    a trajectory is still clean. Validity is measured, never imposed.
  - **Batched == scalar.** The vectorised environment used for model rollouts agrees with the scalar
    reference field-for-field (`all_steps_valid`, `reached_goal`, `held_at_goal`, and the raw speed / wall
    / teleport counts) on 300 episodes: **0 mismatches**.

- **2026-09-20T02:07Z** — **FINDING 3 — the wall-ablation oracle, and it is decisive.** On the 3,000
  held-out episodes I ran a policy that computes its plan with the **wall deleted** and replays it into the
  walled environment:

  > **wall-blind oracle: `valid_and_reached` = 1.0000. Optimal oracle: 1.0000.**

  A policy that never looks at the obstacle is **perfect** on the held-out split. This is Finding 2 turned
  from a statement about cost fields into a direct behavioural measurement: on the solvable split the
  obstacle is not merely non-binding, it is **entirely ignorable**. Any wave-7 claim of the form "the model
  respected the obstacle" that is evidenced on solvable 1-D episodes would be unfalsifiable — a model that
  cannot see the wall at all scores 1.0. The wall-crossing metric can still fire (a *bad* model crosses,
  as the partially-trained LSTM below does), so the metric is live; what is dead is the inference from
  "didn't cross" to "read the obstacle".

- **2026-09-20T02:18Z** — **CORRECTION to the 02:07 bullet's context (process, not result).** The first
  training launch was backgrounded with `nohup ... &` from inside a tool call; the child was killed when
  its launcher shell exited, and the run froze at step 1000 of 8000 while the log still looked alive.
  Caught by checking the checkpoint mtime rather than the log tail. Relaunched as a properly tracked
  background job, from scratch with the stale checkpoint deleted rather than resumed — the optimiser state
  is not checkpointed, so resuming would have restarted Adam mid-run and made the training curve a
  different object from the one I would be reporting. No result reported before this bullet is affected:
  the tripwire, the novelty checks, the oracles and the two rollout verifications use no trained weights.
  The `LSTM(partial)` rollout numbers seen at step 1000 were never reported as baseline results.

- **2026-09-20T02:31Z** — **Per-composition breakdown — this is the mechanism, and it is worth reading
  carefully.** `valid_and_reached` on each held-out (speed limit, displacement bucket) pairing, for plain
  nearest-neighbour replay (T3) and for the *conservative* stitcher (fragments of at least 12 steps, at
  most 3 of them):

  | held-out pairing | n | T3 replay | T4 min_frag=12, K=3 |
  |---|---|---|---|
  | V=2, \|d\| 49–70 | 428 | **0.000** | **1.000** |
  | V=3, \|d\| 21–32 | 682 | 0.743 | 0.993 |
  | V=4, \|d\| 3–10 | 730 | 0.810 | 0.932 |
  | V=5, \|d\| 33–48 | 428 | 0.839 | 0.974 |
  | V=6, \|d\| 11–20 | 732 | 0.889 | 0.969 |

  The `(V=2, |d| 49–70)` row is the whole story. It is the **only** composition that completely defeats
  retrieval — T3 scores **exactly zero**, because crawling 49–70 cells at a speed limit of 2 requires a
  cruise phase held for longer than any stored trajectory holds it. It is the hardest thing in the gym.
  And **three stored fragments solve it perfectly.**

  > The compositional axis I built — *hold a speed you know for a distance you have not held it for* — is,
  > in a 1-D speed-limited gym, **identical to repeating a stored constant-velocity segment**. Composition
  > and concatenation are the same operation here. That is not a flaw in my split; it is what "compose a
  > speed limit with a distance" *means* on a line, and it is why no split of this environment can be
  > lookup-proof.

  Note the stitcher is not required to find *optimal* paths, only valid goal-reaching ones inside the
  horizon, which is why long fragments can serve short displacements too.

- **2026-09-20T02:36Z** — **CORRECTION to the 02:18Z bullet — the cause was probably not mine.** The w5
  hub/delete spoke messaged to say it twice ran `taskkill /F /IM python.exe`, which kills **every** python
  process on this four-agent box, and that a process listing at the time showed `exp_w5_gym1_train` and
  `exp_w5_gym1_breakdown` running. So the frozen-at-step-1000 training run was most likely killed by that,
  not by the `nohup`/shell-exit mechanism I blamed at 02:18Z. I cannot separate the two causes after the
  fact — the process died without a traceback, which is consistent with both — so I am recording both and
  ranking the external kill first on the strength of their direct observation.
  **Impact on results: none.** The frozen run was discarded and retrained from scratch rather than
  resumed, and nothing from it was ever reported. Current status checked: training is alive and
  progressing (step 600, checkpoint written 01:51 local).
  Also checked, on their tip: no stray `C:\c\...` tree exists — every path this spoke hands to python is
  Windows-style (`C:/chlu/outputs/w5-gym1`), POSIX paths are used only in bash. Their other tip
  (`CHLU_GEN_FEATURES_ROOT` / cache generation) does not apply here: GYM-1 is self-contained integer
  dynamics and touches no generative bridge, feature cache or shapes3d data at all.

- **2026-09-20T02:12Z (UTC)** — LSTM finished: **8,000 steps, 815 s, final one-step loss 0.0980**
  (mean of last 200). Transformer training started. Eval pipeline validated end-to-end on the LSTM while
  the transformer trains; both pre-flight assertions passed again (no guardrails; batched == scalar).

- **2026-09-20T02:16Z (UTC)** — **Confound found in my own wall ablation, and fixed before reporting.**
  My first ablation hid the wall by setting `ww := 0`. The LSTM collapsed from 0.3760 to **0.0040**, which
  looks like overwhelming evidence that the model depends on the obstacle — and it would have contradicted
  Findings 2 and 3. It is an artifact: **`ww = 0` never occurs in training** (`ww ∈ {1,2,3,4}`), so that
  embedding row is untrained and the probe measures out-of-distribution input, not dependence on the wall.
  Replaced with an in-distribution probe — **permute `w0` across episodes**, so every rule token remains a
  value the model has seen and only the *correspondence* to the real wall is destroyed. Both are now
  reported, the `ww := 0` one explicitly labelled as an OOD-input probe rather than a wall probe.
  Worth stating plainly: the wall-blind **oracle** at 1.0000 is the clean measurement of the obstacle's
  contribution and it does not depend on any model, which is why Finding 3 rests on it and not on this.

- **2026-09-20T00:36Z** — Transformer finished: **8,000 steps, 1,919 s, final one-step loss 0.0967**;
  LSTM was 8,000 steps, 815 s, loss 0.0980. Same objective, same data, same step count, parameters matched
  to 0.031 % — the two baselines are matched in fit as well as in size. Full evaluation run and written to
  `eval.json`. Both pre-flight assertions passed again.

---

# FINAL REPORT — w5/gym1

**INSTRUMENT ONLY.** This spoke makes **no claim about the CLU**, and **spent no scored run**.
It built a measuring device and reports its properties. The properties are bad.

## 0. Verdict

> ⛔ **GYM-1 is not an instrument. The matched-bytes trajectory-lookup tripwire FIRES.
> Wave 7 must not be scoped on this gym.**

I stopped hardening the gym at the moment the tripwire fired, as the task requires, and spent the
remainder characterising *why* it fires and what survives for re-use. The reason is not a bad split —
it is that **a one-dimensional gym cannot carry the question being asked of it**, and that can be shown
exhaustively rather than argued.

## 1. The tripwire — reported first

Budget = 4 bytes × the parameter count of the matched baselines = **997,896 bytes**. Every strategy is
scored by **replaying what it retrieves in the real environment**, under exactly the metrics a neural
model gets. 3,000 held-out pairs.

| strategy | valid_and_reached | all_valid | hit | bytes used / 997,896 |
|---|---|---|---|---|
| T1 exact key | 0.0000 | 0.0000 | 0.000 | 997,895 |
| T2 nearest neighbour, absolute key | 0.0043 | 0.8983 | 1.000 | 997,895 |
| T3 nearest neighbour, **relative** key, replay | **0.7027** | 0.8167 | 1.000 | 997,890 |
| T4 fragment stitching, K=2 | 0.8520 | — | 0.852 | 41,657 |
| **T4 fragment stitching, K=8** | **0.9797** | 0.9797 | 0.980 | 41,657 |
| T4 fragment stitching, K=10 | **1.0000** | — | 1.000 | 41,657 |

**Best = 0.9797 against a pre-registered falsification threshold of 0.90.** The threshold was fixed in
writing before the tripwire was built (LOG 2026-09-19T23:36Z).

Three things make this worse rather than better for the gym:

1. **T3 needs no stitching at all.** Plain nearest-neighbour replay of whole stored trajectories reaches
   **0.7027**, past the 0.50 "serious warning" line on its own.
2. **The byte budget was never the binding constraint.** T4 used **41,657 of 997,896 bytes — 4.2 %**.
   Matching bytes was supposed to be what made this hard; it is slack by a factor of 24.
3. **It is not secretly a planner.** Sweeping the minimum fragment length the table may reuse:

   | min_frag | K=1 | K=2 | K=3 | K=4 | K=8 |
   |---|---|---|---|---|---|
   | 3 | 0.505 | 0.852 | 0.857 | 0.857 | 0.980 |
   | 8 | 0.293 | 0.847 | 0.912 | **0.999** | 0.999 |
   | 12 | 0.241 | 0.763 | **0.970** | 0.972 | 0.972 |

   **Three stored chunks of at least twelve steps each solve 97.0 % of the held-out pairs.** That is
   trajectory recombination in the plain sense of the word, not re-planning from primitives.

Self-consistency check that passed: T4's hit rate equals its `valid_and_reached` to four decimals at
every K, so every plan the fast legality test accepted also replayed cleanly in the exact environment.
The tripwire is not scoring itself.

## 2. Why it fires — three exhaustive results about one dimension

**(a) On a line, an obstacle can never bend a valid path.** Comparing exact BFS cost-to-goal fields with
and without the wall, over a grid spanning the whole line and all `(s, v0)` simultaneously:

> **25,261,855 solvable configurations. Configurations made more expensive by the wall: 0.**

The minimum-time 1-D profile is also the minimum-excursion one, so whenever a wall-respecting path exists
the wall-free optimum already is one. There is no "route around" in one dimension.

**(b) A policy that cannot see the obstacle is perfect.** The wall-blind oracle — plan with the wall
deleted, replay into the walled environment — scores **`valid_and_reached` = 1.0000** on the held-out
split. Confirmed on the trained models with an in-distribution ablation (permute `w0` so every rule token
stays a value the model has seen, but no longer corresponds to the real wall):

| | true wall | `w0` shuffled | delta |
|---|---|---|---|
| LSTM | 0.3760 | 0.3620 | **−0.0140** |
| transformer | 0.4080 | 0.4100 | **+0.0020** |

Destroying the wall information costs about one point, and for the transformer the change is positive,
i.e. noise. **Neither model uses the obstacle, and neither needs to.**

**(c) Held-out paths cannot be new here.** Exact checks, 1,500 held-out episodes against all 120,000
training paths:

| check | result |
|---|---|
| N1 — episode key novel | **1.000** (the split is clean; this is not a leak) |
| N2 — position sequence novel | **0.509** — 49 % occur verbatim in training |
| N2 — **action** sequence novel | **0.033** — **96.7 % occur verbatim** |
| N3 — fragments needed to tile the path | **1.51 mean** (737/1500 need one, 761 need two) |

**The typical held-out "new path" is two stored fragments.** With a 3-symbol action alphabet and
bang-bang optimal plans there are not enough distinct paths for a held-out path to be new.

**The mechanism, in one line.** The per-composition breakdown shows the only pairing that completely
defeats retrieval is `(V=2, |d| 49–70)` at **T3 = 0.000** — crawling a long way under a low speed limit
needs a cruise held longer than anything stored. Three fragments solve that same pairing at **1.000**.

> The compositional axis available in a 1-D speed-limited gym — *hold a speed you know for a distance you
> have not held it for* — **is** repetition of a stored constant-velocity segment. Composition and
> concatenation are the same operation here. No split of this environment can be lookup-proof.

## 3. Deliverables

### 3.1 Environment and held-out split — delivered; the "paths must be new" requirement measured and FAILED

One agent on a line (`L = 128`), integer dynamics `v' = v + a`, `x' = x + v'`, acceleration limit
`|a| ≤ 1`, speed limit `|v'| ≤ V` with `V ∈ {2..6}` sampled per episode. Obstacle: contiguous wall cells,
width 1–4, blocked by **swept-segment** intersection so jumping over it counts as a crossing — proved
uncrossable by flood-fill of the transition graph for **all 160 rule settings**, not by sampling. Goal:
arrive at rest and stay. Ground truth by exhaustive BFS over the `(x, v)` graph, so "optimal" is provable,
not heuristic. Split: compositional on (speed limit × displacement bucket); every `V` and every bucket
appears in training, never in the held-out pairings; 0 key overlap and 0 pairing overlap, verified.
**The novelty requirement is not merely unmet, it is close to unattainable in this task** (§2c).

### 3.2 Tripwire — delivered, and it fires

§1.

### 3.3 Persistence at one step — delivered, and it separates the two architectures

**How it is enforced.** (i) The loss is cross-entropy on the immediate next acceleration and nothing else:
no multi-step term, no rollout in the objective, no scheduled sampling. (ii) "Stay parked at the goal" is
supervised for **4 steps only**, while the evaluation horizon is 48 and the persistence probe is 96 — so
holding for a further 44 or 92 steps is never supervised. (iii) At rollout there is **no action mask, no
clamp of `v` to the speed limit, no projection of `x` back onto the line**. This is verified in code, not
asserted in prose: `verify_unmasked_rollout()` drives a deliberately illegal constant `+1` policy and
confirms the environment applies it verbatim — velocity reaches **3 against a limit of 2**, the violation
is recorded, and the recorded states equal raw integer arithmetic. Validity is measured, never imposed.

**How it is measured, and what it shows.** Survival = fraction of episodes still fault-free through step t.

| | LSTM | transformer |
|---|---|---|
| survival @ t=48 | 0.9700 | 0.8740 |
| survival @ t=96 | **0.9680** | **0.7360** |
| parked on goal @ t=24 | 0.2927 | 0.2993 |
| parked on goal @ t=36 | 0.3607 | **0.1413** |
| parked on goal @ t=48 | **0.3673** | **0.0600** |
| parked on goal @ t=96 | 0.3720 | 0.0620 |

The transformer **reaches** the goal more often than the LSTM (0.4473 vs 0.3700) and **holds** it at one
sixth the rate. Its at-goal curve is non-monotonic — it arrives, then drifts off — and its survival falls
14 points when the horizon merely doubles, entirely outside anything it was supervised on. The LSTM is
flat to t=96.

> **Persistence is architecture-dependent and is invisible to goal-reach.** A gym reporting only goal-reach
> would rank the transformer above the LSTM; the persistence metric reverses the ranking. This protocol is
> the part of this spoke most worth keeping.

### 3.4 LSTM and transformer, rules visible, matched parameters — delivered, counts printed

Both models see `V`, `w0` and `ww` as embedded tokens at every step, alongside position, velocity and goal.

| | params | steps | seconds | final one-step loss |
|---|---|---|---|---|
| LSTM | **249,474** | 8,000 | 815 | 0.0980 |
| transformer | **249,397** | 8,000 | 1,919 | 0.0967 |

Relative parameter gap **0.031 %** (pre-registered tolerance 2 %). Matched in fit as well as in size.

### 3.5 Exact validity metrics — delivered

1,500 held-out episodes at horizon 48 = **72,000 steps** per model. Free-running, unmasked.

| | LSTM | transformer |
|---|---|---|
| **valid_and_reached** | **0.3700** | **0.4173** |
| all steps valid | 0.9700 | 0.8740 |
| goal reached | 0.3700 | 0.4473 |
| held at goal | 0.3673 | 0.0600 |
| in-distribution control | 0.4387 | 0.4787 |
| **speed-limit violations** | **267** | **375** |
| **wall crossings** | **542** | **946** |
| **teleports** | **9** | **83** |
| episodes with any speed violation | 1.40 % | 3.20 % |
| episodes with any wall crossing | 1.33 % | 6.73 % |
| episodes with any teleport | 0.60 % | 3.27 % |

First-fault attribution (LSTM / transformer): speed 21/48, wall 16/96, teleport 8/45.

**Refusal split** (500 episodes: goal unreachable behind the wall, but a fault-free behaviour does exist —
the one place the obstacle carries information in 1-D): LSTM `all_valid` **0.0140** with **1,268** wall
crossings; transformer **0.0040** with **1,155**. Both drive into the wall on ~99 % of trapped episodes.
Neither has learned any refusal behaviour, which is expected — neither was trained on unsolvable episodes
— and it confirms the wall-crossing metric is live rather than structurally unable to fire. The refusal
*claim* belongs to another spoke; this is reported only as a property of the instrument.

### 3.6 Is validity checked exactly? YES

Every predicate is integer arithmetic with **zero tolerance**: speed `|v'| > V`; wall `swept segment
[min(x,x'), max(x,x')] meets the wall cells`; teleport `x' outside [0,127]`; goal `x == g and v == 0 with
every preceding step valid`. No rule needs a tolerance, so **no downstream refusal claim built on this gym
would be approximate on account of the validity check.** Two caveats, both measured rather than hedged:

- The velocity embedding clips the model's *input* at `|v| ≤ 16`. **Measured: the largest `|v|` reached in
  any rollout was 6, and 0.0000 of episodes ever exceeded the clip.** The caveat never binds in practice.
- After a teleport the continuing-mode rollout returns the agent to the line so the horizon can be counted
  out. Episode-level validity (`all_steps_valid`, `first_fault_step`) is computed **before** any such
  repair, so no reported rate is softened by it; only the raw post-fault counts are affected.

The batched environment used for model rollouts was checked field-for-field against the scalar reference
on 300 episodes: **0 mismatches**.

## 4. The second falsification criterion

"Baselines reach ~100 % ⇒ no headroom to measure" is **NOT triggered**: 0.3700 and 0.4173 leave ample
headroom. But the comparison that matters is this one:

> **The matched-bytes trajectory table (0.9797) beats both trained, rules-visible, 250k-parameter
> baselines (0.3700, 0.4173) by more than a factor of two.**

A gym on which a trivial lookup table outscores every trained model by 2.3× cannot be used to argue that a
trained model composed anything.

## 5. NOT-RUN declarations (declared, not null)

- **Scored run: NOT RUN.** Mandated — this spoke spends none.
- **GPU headline run: NOT RUN.** Not needed, so not justified. The entire study — the 25.3 M-configuration
  sweep, 123 k episodes, the full tripwire and both baselines — runs in CPU-minutes to CPU-tens-of-minutes
  on 3 threads. No GPU justification lines were written because no GPU run was ever contemplated.
- **Further hardening of GYM-1: NOT RUN, deliberately**, under the task's stop rule, from the moment the
  tripwire fired. §2 is the argument that no amount of hardening would have helped.
- **Any claim about the CLU: NOT MADE.** Nothing here is evidence for it or against it.

## 6. What wave 7 needs instead

1. **Dimension.** An obstacle cannot bend a path in 1-D — proved over 25.3 M configurations. If wave 7
   wants to test "compose the speed limit with the obstacle to find a different route", the gym must be at
   least 2-D. In one dimension an obstacle is a *feasibility* signal only.
2. **Enough path diversity that held-out paths can be new.** Measure N2/N3 *before* adopting a gym. If the
   median held-out path tiles from two stored fragments, no split will save it.
3. **Run the tripwire with stitching and a min_frag sweep, before the gym is adopted.** T1 alone scored
   0.0000 here and would have certified this gym as sound. Exact-key lookup is not a tripwire.
4. **Make the rule-blind oracle a mandatory gate.** It is nearly free and it is decisive: if a policy that
   cannot see a rule scores 1.0, that rule is untested and no claim may rest on it.
5. **Keep these — they are the reusable part of this spoke:** the exact zero-tolerance validity predicates;
   the unmasked rollout with its adversarial verification; the one-step persistence protocol with survival
   and at-goal-over-time curves (§3.3), which caught a failure goal-reach hides; and pre-registration with
   a written falsification threshold, which is what made this result reportable rather than negotiable.

## 7. Artifacts

`START_UTC.txt` · `LOG.md` · `splits.pkl` · `splits_summary.json` · `param_match.json` ·
`tripwire.json` · `tripwire_sweep.json` · `tripwire_breakdown.json` · `novelty.json` ·
`train_report.json` · `train_stdout.log` · `eval.json` · `ckpt_lstm.eqx` · `ckpt_transformer.eqx`

Code: `chlu/gym1/{line1d,episodes,models,tripwire,train,vecroll}.py` and
`chlu/experiments/exp_w5_gym1{,_train,_eval,_tripsweep,_breakdown}.py`.

Branch bundle: `w5-gym1.bundle` — `refs/heads/w5/gym1 = bfee159`, `git bundle verify` reports
**"the bundle records a complete history"**. Working tree clean; 3 commits on top of `base/iter5 = 6675904`.
