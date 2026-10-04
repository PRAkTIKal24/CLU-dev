# w15/cmaze-t-harness — LOG

Spoke: `cmaze-t-harness`.  Worktree `C:/chlu/worktrees/CHLU-w15-cmaze-t-harness`,
branch `w15/cmaze-t-harness`, base `base/iter8 = 148f3b7`.
Output dir `C:/chlu/outputs/w15-cmaze-t-harness/`.

## 2026-09-22T21:26Z — start, environment
- `START_UTC.txt` written.
- `source /c/chlu/env_w3.sh` then the import assert.  **FIRST ATTEMPT FAILED**:
  `env_w3.sh` builds `PYTHONPATH` with a POSIX `:` separator and an msys-style
  `/c/Users/...` second entry.  Windows CPython splits `PYTHONPATH` on `;`, so the
  whole string was ONE bogus path and `equinox` was not importable; `chlu` only
  resolved because `python -c` puts the cwd on `sys.path`.
- **Fix, recorded so every number is attributable:** after sourcing, I re-export
  `PYTHONPATH='C:/chlu/worktrees/CHLU-w15-cmaze-t-harness;C:/Users/PratikJawahar/chlu/chlu-bundle-v2/.venv/Lib/site-packages'`
  (Windows separator, Windows paths, worktree FIRST so the editable
  `_editable_impl_chlu.pth` pointing at the Head's checkout cannot win).
  Assert then passes:
  `OK C:\chlu\worktrees\CHLU-w15-cmaze-t-harness\chlu\__init__.py`,
  `jax 0.9.0 numpy 2.4.1 eqx 0.13.4`.  No venv built, no `uv sync`.
  The shim lives in `scripts/w15_cmaze_t_env.sh` inside the worktree.

## Survey of what already exists (imported, never edited)
| piece | module | role in CMAZE-T |
|---|---|---|
| engine + solver + exact judge | `w8_envscout.maze_colour` | **unmodified**; `generate`, `solve`, `judge` |
| stepper (transition function) | `w9_harness.dynamics.step` | the transition CMAZE-T asks an arm to reproduce |
| exactness audit (4 checks) | `w9_harness.judge_maze` | `judge_ref` is the **independent driver** for acceptance 1 |
| maze image | `w9_harness.render` | 15 planes at k=4 |
| shared frozen encoder | `w9_harness.encoder` | the **frozen state encoder** the neural rivals read |
| shared observation | `w9_harness.features` | patch ⊕ global-pool ⊕ dyn |
| matched-param trunks | `w9_harness.arms` | MLP / LSTM / transformer, one `ActionHead` shape |
| blind relaxations | `w9_harness.columns` | the route-level definition CMAZE-T specialises to a step |
| CLU single design | `w9_maze.arms.CluRead` + `chlu.core.gen_request_head` + `chlu.experiments.exp_w8_single_design.CovContentStore` | the CLU arm |

## DECISIONS WHERE THE TASK FILE LEFT A CHOICE (COMMON.md: pick the ICLR option, record it)

### D1 — what "the factors of the transition" are, concretely
Task file: *agent colour × key colour × door colour × tinted-floor colour × direction
(× carry state)*.  CMAZE-1 has no separate "agent colour"; it has a **pocket**, and an
agent carrying key c is the only sense in which an agent in this environment HAS a
colour.  So `agent colour ≡ the colour in the pocket before the step`, and
`carry state` (empty / carrying) is its coarsening — which is exactly why the task
file puts carry state in parentheses rather than listing it as a sixth factor.
**An inert factor would have been the alternative and it was rejected: a word the
engine never reads is a word every arm gets right for free and it would dilute the
instrument.**  The five words are therefore:

| word | alphabet (k = 4) | what it is |
|---|---|---|
| `w_agent` | 0..k (5) | pocket colour before the step, 0 = empty |
| `w_key`   | 0..k (5) | colour of an **untaken** key home on the CURRENT cell, 0 = none |
| `w_door`  | 0..k+2 (7) | target cell: 0 none, 1..k CLOSED door of colour c, k+1 OPEN door, k+2 WALL/off-grid |
| `w_tint`  | 0..k (5) | target cell tint colour, 0 = untinted |
| `w_dir`   | 5 | N E S W P |

**These five words determine the transition exactly** (verified as acceptance check 3b,
`words_determine_transition`), which is what makes hold-outs over word combinations a
statement about the transition rather than about a featurisation.

### D2 — why this is a composition instrument and not a lookup
Rule A is `pocket == door colour`; rule B is `pocket == tint colour` **evaluated after
the door has consumed the key**.  So `(w_door=c, w_agent=c, w_tint=c)` is LEGAL — the
key is spent opening the door, the pocket empties, and the tint no longer bites —
while `(w_door=0, w_agent=c, w_tint=c)` is ILLEGAL.  That is an irreducible
**3-way interaction** between door colour, tint colour and pocket colour, and it is
the thing the n-way hold-out holds out.

### D3 — BLIND-SEQUENTIAL at the transition level
The route-level definition (`key order taken from the rule-A-only plan`) has no
meaning for one step.  Its transition-level twin is the same idea — *the two rules
treated as independent stages instead of as an interaction*: **BLIND-SEQUENTIAL
evaluates rule B against the pocket BEFORE the door consumed the key.**  It is
optimal in its own factored world and is graded by the exact judge, so it cannot be
a straw man, and it differs from the truth on exactly the D2 configuration.

### D4 — what an arm EMITS
`(legal, displacement, key_event, pocket_next)` — one-hot blocks, 2 + 5 + (2k+1) + (k+1)
= 21 dims at k=4.  The harness APPLIES `(legal, displacement, key_event)` to the real
`s_t` to build the generated `s_{t+1} = (pos', ks')`, and the transition is correct iff
`(legal, pos', ks')` equals the engine's `step()` output **exactly** and `pocket_next`
agrees with `pocket(ks')`.  All integer comparisons; no tolerance.
`pocket_next` is a separate head precisely because "correct pocket" is one of the three
things the task file says the engine judges — an arm that gets legality right and the
pocket wrong must be able to fail.

### D5 — the frozen codec, and where the deletable object stops
Following the CMAZE-2 pattern: a **frozen next-state codec** (21-dim code → latent
D = 64 → logits → argmax) trained by reconstruction on TRAIN codes only and frozen
before any arm exists; and a **frozen state encoder** (`w9_harness.encoder`, conv AE
on maze images, reconstruction only) that the neural rivals read.  BOTH are declared
**outside the deletable object**: deletion is store-scoped, and nothing in the codec or
the encoder is downdated or refitted.

### D6 — scale of the VAL smoke
Task file: ZERO scored runs, val smoke is free.  3 seeds, all rows, all rivals.
Sized so the whole smoke completes on this CPU box inside one turn.


## 2026-09-22T21:5xZ — the structural finding that shaped the layer

First smoke of the core module on 12 mazes / 600 transitions:
- `words_determine_transition` PASS (101 distinct word tuples, 0 with two outcomes)
- `replay_audit` 600 replayed, **0 disagreements**
- `impossible_example_audit` 600 checked, 0 without an accepting emission
- `rule_b_bites` **0.0150** of all transitions, 0.0188 of move actions
- **`n_door_eq_pocket_eq_tint` = 0**

That last zero is the finding.  `w8_envscout.maze_colour.generate` sets tints on random
floor cells and then **zeroes the tint on start, goal, door and key cells**.  Measured
over 40 generator seeds: **160 door cells, 0 of them tinted.**  So in CMAZE-1 as shipped,
a door cell is NEVER a tinted cell, and therefore:

* the D2 three-way interaction `(door=c, pocket=c, tint=c)` **cannot occur in one step**;
* **BLIND-SEQUENTIAL is vacuous at the transition level** — evaluating rule B against the
  pre-door pocket versus the post-door pocket can never differ, so it would print 1.0000
  identically to ORACLE and measure nothing;
* rule B bites at 1.5 % of transitions, all of them plain `pocket == tint` refusals with
  no rule-A involvement.

### D7 — THE ONE DISTRIBUTIONAL CHANGE CMAZE-T MAKES: tinted doors
The **rules already specify the tinted-door case** — `maze_colour.judge`, `dynamics.step`,
`judge_maze.judge_ref` and `maze_colour.solve` all read `door` first and then `tint`
against the UPDATED pocket.  The generator simply never emits the case.  So CMAZE-T's
request layer tints a fraction of door cells and hands the result to the **unchanged**
engine, solver and judge.  ⛔ No engine code is touched; this is the same class of move
`w9_maze.task._poison_pairs` and `_seal` already make on the tint and grid arrays.

Why it is the right move rather than a convenience:
* a transition instrument with no within-step interaction between its two rules is not a
  composition instrument at all — it is two independent 2-way rules, and the task file's
  own two-gym law ("the rules' interaction must be worth enough that a composition-blind
  policy cannot score well") would be unsatisfiable by construction;
* the configuration it creates is exactly the interesting one: entering a **CLOSED** door
  `c` tinted `c` while carrying `c` is **LEGAL** (the key is spent, the pocket empties,
  the tint no longer bites), while entering the **SAME CELL once the door is OPEN** while
  carrying `c` is **ILLEGAL**.  Same three colours, opposite verdicts, and the third
  factor that flips it is the door's open/closed state;
* **self-tinting (tint colour = the door's own colour) cannot make an instance
  infeasible**: any route that could pass the closed door still can, because the key is
  consumed before rule B is evaluated.  Cross-tinting can, so every tinted instance is
  re-solved with the unmodified solver and the tint is reverted if the oracle dies.

Rejected alternatives, recorded: (i) print BLIND-SEQUENTIAL = 1.0000 and call the
instrument done — that ships an instrument that cannot fail its own falsifier;
(ii) redefine BLIND-SEQUENTIAL as something contrived — a relaxation that is not a named
flag on the rules is a straw man.

## 2026-09-23T00:1x–01:3xZ — dry runs (12–14 mazes), five findings, and PREREG AMENDMENT 1

All dry-run numbers below are from 12–14 train mazes / 8 val mazes and are **NOT** part
of the VAL smoke.  Logs: `logs/dry0.log` … `logs/dry6.log`, `logs/probe_latent.log`.

**D8 — `landscape_shuffled` as a row permutation is a no-op BY CONSTRUCTION.**
The address book is fitted AFTER the control is applied and `head_read` computes
`(Phi W_k) G_k`, so permuting the atom ROWS of `G_k` is absorbed exactly by the matching
column permutation of `W_k`.  Measured: row-permuted control **0.8195** on `unseen
n-way` against the CLU's **0.8195**, identical to every digit.  Redefined as a
permutation of each well's atom **CHANNELS**, which moves the deployed span and cannot
be absorbed.  Recorded, not quietly fixed.  (PREREG A1.1.)

**D9 — the BLIND-* PLANNER columns are not policies, so a second, LEARNED falsifier was
added.**  The `blind_outcome` columns are rule-KNOWING relaxed oracles: they read the
grid, they never train, and so they never have to generalise to a held-out combination.
On the dry run `BLIND-SEQUENTIAL (planner)` scored 0.9098 on `unseen n-way` against the
CLU's 0.8195 — which says nothing about composition, only that an untrained oracle does
not suffer a hold-out.  Two **LEARNED** composition-blind arms were added, same store,
same head machinery, same rows, same codec, degraded in exactly one way each:
`BLIND-COLOUR (learned)` = colour identity removed from the request;
`BLIND-SEQUENTIAL (learned)` = main effects only, no interaction terms.
Dry-run `unseen n-way`: CLU **0.8195**, learned blind-colour **0.0000**, learned
blind-sequential **0.2331**.  ⭐ The falsifier does NOT fire for the learned arms — the
instrument measures composition.  Both families are printed, each labelled.

**D10 — the rollout row must run the FULL optimal route.**  An 8-step prefix of a
~45-step route cannot reach the goal, so `validity` was 0 for every arm by construction
and measured the truncation.  Fixed; an `L = 8` block is kept as a declared diagnostic
whose readable number is `all_steps_exact_rate`.  The rollout is now LOCKSTEP-batched
(one forward pass per step for all live rollouts, w9's discipline), which took the
rollout row from ~700 s to 0.3 s; `_pad_to` pins the jitted leading dimension so a
shrinking live set does not retrace at every step.

**D11 — the within-maze lookup column was reading its own answer.**  Its table was built
from the very rows being scored and returned 1.0000 trivially.  It is now built from the
val mazes' NATURAL-distribution rows — a different sample of the same mazes — and stays
DECLARED NOT transfer, printed only as the ceiling a lookup reaches when the maze repeats.

**D12 — ⭐⭐ THE INSTRUMENT'S OWN LIMIT: the charter-C controls do not collapse at high
span, and the reason is Johnson–Lindenstrauss.**
The next-state code has **26 distinct values in train** (`next_state_code` in every seed
JSON).  A random projection of 26 well-separated points into a handful of dimensions
preserves their separation, so a RANDOM-ATOM store can stay decodable although its
content was never fitted.  Measured across the latent sweep (`logs/probe_latent.log`,
14 train mazes, `seen` row, worst of the four charter-C controls):

| latent | K | atoms | span | CLU seen | worst control | store claim? |
|---|---|---|---|---|---|---|
| 16 | 4 | 3 | 0.750 | 0.9813 | 0.9394 | no |
| 24 | 4 | 3 | 0.500 | 0.9989 | 0.9256 | no |
| 32 | 8 | 3 | 0.750 | 0.9901 | 0.9868 | no |
| 32 | 4 | 3 | 0.375 | 0.9829 | 0.8545 | no |
| 64 | 8 | 3 | 0.375 | 0.9752 | 0.8848 | no |
| **64** | **8** | **2** | **0.250** | **0.9741** | **0.6926** | **YES** |
| **64** | **4** | **3** | **0.188** | **0.9725** | **0.7697** | **YES** |
| **64** | **4** | **2** | **0.125** | **0.9047** | **0.5052** | **YES** |

So the saturation ratio is the knob, exactly as `w9_maze.arms` says, and a store claim
on CMAZE-T is available **only below about span 0.25**.  ⭐ Therefore the four
charter-C controls were made part of the Config-A selection bar and the config grid was
rewritten to span the saturation ratio.  (PREREG A1.2.)

**D13 — rival budget.** At 200 gradient steps the MLP and LSTM were still emitting the
majority class (per-step exactness 0.0000 on an all-legal rollout, i.e. "always illegal").
Raised to **8000** steps for every rival and for both deletion rivals, with the loss curve
printed.  (PREREG A1.4.)

All four changes were written into `PREREG.md` as **AMENDMENT 1** before any
`val_seed*.json` existed.

## 2026-09-22T23:00–23:30Z — the VAL smoke, and three more findings fixed before it shipped

**Smoke 1 (pre-amendment-2, preserved in `preamendment2/`).** 3 seeds, 90 train mazes /
45 val mazes each, ~37k train rows and ~18k val rows per seed.  All six acceptance
conditions passed, `status: ready`.  It surfaced two defects in its own instrument:

**D14 — ⭐ the frozen codec capped the value hold-out row at 0.4963 ± 0.0368.**
The codec's OWN encode→decode→judge ceiling, measured per row: `seen` 1.0000,
`unseen 2-way` 0.8722, `unseen n-way` 0.9495, **`value` 0.4963**.  The train rows realise
only 23–31 of the 45 distinct next-state codes the val rows need, because a code carrying
the held-out colour in its `key_event` or `pocket_next` block never occurs in train.  So
the value row was measuring the CODEC, not the arm.  Fixed by fitting the codec on the
train codes **plus the enumerated emission alphabet** — the full Cartesian product of the
four emission blocks (450 codes at k = 4), a statement about the FORMAT carrying no row,
no word, no rule and no outcome, so it leaks nothing.  The STORE is still built from the
train codes only.  After the fix the codec ceiling is **1.0000 on every row**.
(PREREG AMENDMENT 2.)

**D15 — the store-scoped exactness check was comparing two different objects.**
The "fitted without the rows" reference was built with
`TransitionClu(Ztr[keep], ...)`, which recomputes `ward_balanced_partition` on the
SURVIVING rows and therefore puts different CHANNELS in each well.  It reported
`max_relative_delta = 1.26` while the per-well counts matched exactly (11620 = 11620) —
counts matching while statistics could not is what gave it away.  Fixed with a
`well_channels=` override so the comparison is between two versions of the same object.
Re-measured: `S_k` max relative delta **7.8e-16 … 1.1e-15** and the deployed atoms
**exactly 0.0**, across both deletions and all three seeds.  (PREREG AMENDMENT 3.)

**D16 — an emission symbol that was not a displacement.**  The displacement block was
6 wide (`stay, N, E, S, W, P`).  The engine never produces "displaced by P", but once the
codec was trained on the full alphabet an arm could EMIT it, and `apply_outcome` raised
`KeyError: 'P'`.  The block is now 5 wide — `{stay, N, E, S, W}` — so every symbol the
format admits is a realisable displacement.  `code_dim` 22 → 21, alphabet 540 → 450.

**D17 — the capacity diagnostic.**  Config-A's read path is small (4,008–8,016 floats),
so the matched rivals are width-5 to width-11 trunks on a 636-dim observation.  The first
question a referee asks is whether the CLU only wins because the rival was starved, so a
**capacity diagnostic** was added: the same three trunks at **10× the read-path budget**,
same observation, same frozen encoder, same rows, same 8,000 gradient steps, printed
under `capacity_diagnostic` and labelled **NOT a matched rival and NOT part of the
naive-vs-naive table**.  ⛔ This is not COMMON.md's forbidden "inflated-rival row": that
prohibition is against matching a rival to the CLU's MEMORY LEDGER, which is never done
here — the float64 sufficient statistic is printed separately and matched by nothing.

Every earlier run is preserved: `preamendment2/` (smoke 1) and `amendment23/` (smoke 2).

## 2026-09-22T23:46Z — FINISHED

Branch committed whole as `c689c31` on base `148f3b7`; bundle created and
`git bundle verify`-ed (complete history, one ref, sha1).  `git status` clean.
Self-test `scripts/w15_cmaze_t_selftest.py` passes in 1.7 s on a 10-maze bank
(`logs/selftest.log`).

**Final artefact state, verified by directory listing** (git cannot see gitignored
outputs):

    FINDINGS.md 40 KB · HARNESS-READY.json 45 KB (status: ready) ·
    VAL-SMOKE.json 103 KB · val_seed{0,1,2}.json 65-66 KB each ·
    PREREG.md 15 KB · LOG.md · PIDS.txt · START_UTC.txt · END_UTC.txt ·
    w15-cmaze-t-harness.bundle 7.7 MB ·
    logs/ (dry0-6, probe_latent, probe_delete, seed0-2, selftest) ·
    preamendment2/ (smoke 1, 8 files) · amendment23/ (smoke 2, 8 files)

**Every PID this spoke launched is in `PIDS.txt` and every one is gone.**  ⛔ No
box-wide kill was ever run; the only processes killed were the ones listed there, by
PID.  One unrelated python process (181545, another agent's) was running on this box at
finish and was left alone.

Wall clock 21:26:35Z -> 23:46:29Z, 2 h 20 min.  Local CPU only, three concurrent seed
processes, OMP/MKL/OpenBLAS threads pinned to 4.  ZERO scored runs spent.
