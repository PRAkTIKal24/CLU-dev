# `w14_request_layer` — CMAZE-2b, the repaired request layer

**Spoke:** `w14/request-layer-fix`. **Scored runs: 0.** Output dir mirrored in this repo at
`w14_request_layer_fix/`; the live copy is `/c/chlu/outputs/w14-request-layer-fix/`.

## What this package is

CMAZE-2 is a REQUEST layer over CMAZE-1 (`w9_maze/task.py`): named agents and named targets
inside one maze, with `w8_envscout.maze_colour`'s engine, solver and **exact integer judge**
imported unmodified. Wave 9 found that CMAZE-2's blind-sequential column read **0.9037–0.9130**
where CMAZE-1 reads **0.4980** published / **0.5317** at the harness tier, and diagnosed it:
**1242 of 1449 optimal routes pick up exactly one key**, so there is no key ORDER for RULE B to
trap.

This package is the repair, and it is **one acceptance rule**:

> a pair is kept iff its RULES A+B optimal route picks up **at least two keys**.

Nothing in the engine, the solver or the judge is touched. The key count is read off the
solver's own route by `w8_envscout.screen.key_order_of`. It is an integer. **`tolerance_used`
stays `null`.**

## Files

| file | what it is |
|---|---|
| `task2.py` | `build_site_v2` — `w9_maze.task.build_site` with the `>= MIN_KEYS` rule and the one structural consequence it forces |
| `doorgraph.py` | components of the floor with door cells removed, and the door graph over them |
| `budget.py` | the rebased ABSOLUTE budget grid, by a rule that mentions route lengths and nothing else |
| `build_bank.py` | build one BEFORE or AFTER bank at w9's own seeds and configuration |
| `measure.py` | `w9_maze.envcols`' columns, unmodified, under three budget conditions |
| `probes.py` | the three diagnostics the repair rests on (`breakdown`, `density`, `direction`) |
| `readiness.py` | assembles `REQUEST-LAYER-READY.json` and checks the four acceptance conditions |
| `verify.py` | **independent** re-derivation from the bank pickles; does not import `readiness.py` |

## Reproducing it

```
python -m w14_request_layer.probes breakdown            # the mechanism, CMAZE-1 proper
python -m w14_request_layer.probes density              # >= 2 keys is a property of the MAZE
python -m w14_request_layer.probes direction            # ... and it is DIRECTIONAL
python -m w14_request_layer.build_bank after 0 96 <banks>
python -m w14_request_layer.build_bank before 0 96 <banks>
python -m w14_request_layer.measure after 0 <pool.pkl> <banks> <out.json> <before.pkl>
python -m w14_request_layer.readiness <out_dir>
python -m w14_request_layer.verify   <out_dir>          # exit 0 only if every check passes
```

## The two things a reader should not confuse

1. **FRAGMENT-LOOKUP cross-maze and same-maze are different quantities.** The published
   CMAZE-1 figure is the **cross-maze** one (0.0006 / 0.0000): the fragment table is built
   from mazes the evaluation never sees. The **same-maze** figure (w9's 0.0741 / 0.5105) is
   built from routes through the very maze being evaluated and is a much stronger floor.
   Both are published here, labelled.
2. **CMAZE-2b's blind-sequential lands BELOW 0.4980, not at it.** 0.4980 is a MIXTURE over
   CMAZE-1's own key counts — 73.6 % of CMAZE-1's routes are single-key and score 0.6532,
   the rest score 0.1351 / 0.0000. CMAZE-2b is 100 % multi-key, so it sits at CMAZE-1's
   **multi-key band**, not at CMAZE-1's marginal. See `artefacts/anchor_breakdown.txt`.
