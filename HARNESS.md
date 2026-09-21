# The w9 maze harness — what hub/maze scores through

One sentence: **`w9_harness.judge_maze.JUDGE` is env-scout's exact integer judge,
re-exported unwrapped, and it is the only thing in this harness that decides anything.**

```python
from w9_harness.judge_maze import JUDGE          # the one judge
ok, reason, steps = JUDGE(instance, "NNESWP...")  # exact; no tolerance anywhere
```

## The five pieces

| piece | module | what it is |
|---|---|---|
| exact judge | `judge_maze.py` | `w8_envscout.maze_colour.judge`, unwrapped, + a four-part exactness audit |
| stepper | `dynamics.py` | the transition function rollouts need, audited to agree with the judge on every route |
| shared encoder | `render.py`, `encoder.py` | conv autoencoder on maze images, **reconstruction only**, frozen before any arm exists |
| shared observation + head | `features.py`, `arms.py` | one 429-dim vector; one `ActionHead`, `d_model=32`, one PRNG key, identical across arms |
| arms and columns | `arms.py`, `tabular.py`, `columns.py` | MLP / LSTM / transformer at matched parameters, table floor, reported columns, refuser |

## The judge is exact — the four checks

Run `python test_w9_harness.py`, or read `judge_audit` in any seed report.

1. **Source.** AST scan of `judge` and `_pocket`: no float literal, no true division, no
   `isclose`/`allclose`/`atol`/`rtol`/`eps`/`tol`/`round`/`float`, no float attribute.
2. **Types.** Replay under an `int` subclass that raises on any comparison with, or
   coercion to, a float.
3. **Redundant.** `judge_ref` — a second replay written from the RULES, not from the
   judge's code — must agree on verdict, reason *including the colour blamed*, and step
   index, on every route.
4. **Oracle.** Optimum accepted at exactly its length; optimum minus one step rejected;
   one step over budget rejected *as over-budget*; every single-action mutation decided
   identically by both judges.

If any of these ever fails, `HARNESS-READY.json` reports `judge_exact: false` and the
spoke stops, because everything downstream inherits it.

## What "shared" means here, exactly

- **Encoder — fully shared.** One set of weights per seed, trained by a loss whose only
  arguments are `(model, images)`. No arm's objective can reach it; it is frozen before
  any arm is built. `encoder.independence_report()` prints the loss source as evidence.
  Every arm reads the *same* frozen feature arrays.
- **Observation — fully shared.** `features.obs` is one function. MLP, LSTM, transformer
  and refuser all consume its output and nothing else. They differ only in how they
  combine observations *across time*.
- **Action head — shared shape and shared initialisation, trained per arm.** `d_model` is
  pinned at 32 for every arm and the head is built from one module-level PRNG key, so all
  heads start byte-identical. They are then trained per arm, because a head frozen on one
  trunk would be a head tuned for that trunk. That last part is **not** shared and is
  stated rather than glossed.
- **Table floor — deliberately not on the encoder.** It is the memorisation floor over raw
  discrete state. `table-enc` is the same lookup on the shared encoder's bits and is
  reported beside it.

## Scored column vs. diagnostics

- **`validity`** — the scored column. Fraction of episodes whose emitted route `JUDGE`
  accepts: legal at every step, goal entered, length ≤ budget.
- **`score`** — env-scout's column, mean of `valid ? opt_len/len(route) : 0`.
- **`goal*`** — diagnostic: reached the goal legally with the budget lifted. This is the
  same judge applied to a *declared* budget-lifted variant of the instance, so it is still
  exact; it is never the scored number.
- **`legal*`** — diagnostic: mean legal prefix length / optimal length.

The diagnostics exist because zero tolerance means a 45-step route needs all 45 actions
right, so `validity` is a coarse instrument at these training budgets. They give the hub
dynamic range without softening the judge.

## Reported columns, never gates

`fragment-lookup` (env-scout's colour-canonicalised retrieval policy, `r=1` and `r=2`) and
the `blind-*` composition-blind planners are **printed columns**. Nothing aborts on them,
no instance is filtered because of them, and the generator is env-scout's adopted default
configuration unchanged.

## The refuser and the DISPLAY line

`data.make_impossible` builds instances the judge can never accept, certified exactly:

- `budget` — budget one below the exact BFS optimum;
- `legality` — BFS over (cell, keystate) finds no route at all.

Only the refuser sees them. `data.assert_unseen` checks the impossible seeds are disjoint
from every other arm's data, so **"no impossible example was ever seen" is a checked
property of the code, not a claim in prose.** The refuser is the only comparator for
DISPLAY rows; the other arms have no refuse token by construction and score 0 on the
impossible set by definition, which is reported rather than hidden.

## Reproducing

```
python test_w9_harness.py                                  # the properties above
python exp_w9_harness.py --seed 0 --scale std --out <dir>  # one seed of everything
python publish_ready.py --out <dir>                        # HARNESS-READY.json
```
