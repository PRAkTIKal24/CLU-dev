# w15/cmaze-o-harness — LOG

`START_UTC.txt` written. Output dir `/c/chlu/outputs/w15-cmaze-o-harness/`.

## 0. Environment

`source /c/chlu/env_w3.sh` sets `PYTHONPATH="$CHLU_ROOT:$_H/.venv/Lib/site-packages"` with a
POSIX `:` separator. **The interpreter on this box is Windows CPython 3.11**, which splits
`PYTHONPATH` on `;`, so the whole string was one bogus entry and only `python -c`'s implicit
cwd entry put `chlu` on the path — `equinox` was unimportable.

**Choice (recorded):** I do not edit `env_w3.sh` (not my file). Every run of mine sources it and
then re-exports the SAME two paths joined with `;` in Windows form:

```
export PYTHONPATH="C:/chlu/worktrees/CHLU-w15-cmaze-o-harness;C:/Users/PratikJawahar/chlu/chlu-bundle-v2/.venv/Lib/site-packages"
```

Assertion after the fix:
```
OK C:\chlu\worktrees\CHLU-w15-cmaze-o-harness\chlu\__init__.py
jax 0.9.0 numpy 2.4.1 eqx 0.13.4
eqx file C:\Users\PratikJawahar\chlu\chlu-bundle-v2\.venv\Lib\site-packages\equinox\__init__.py
```
`chlu.__file__` resolves INSIDE the worktree. Pinned versions are the wave-2 ones. No venv built,
no `uv sync`.

Box: 4 cores, 4 other `python.exe` processes live (other agents). Compute plan sized accordingly.

## 1. What CMAZE-O is, and the choices I made (all recorded, none waited on)

**The instrument.** Request = nine words (layout id, agent/key/door/tint colour, agent/key/door/goal
zone). Answer = the rendered grid state. Judge = exact placement of every object AND legality under
the engine's rules. `tolerance_used` is `null` everywhere.

| choice | what I picked | why (closest to an ICLR main-track result) |
|---|---|---|
| cell words | **ZONE words**: a fixed 3x3 of 5x5 zones; each zone designates 4 distinct cells, one per object slot, per layout | a raw `(r,c)` is not a word and does not compose; a zone word makes `layout x zone` a real interaction that the same word resolves differently in every layout — which is the one thing a blind arm cannot do |
| door placement | the **balanced articulation point** of the floor graph in that zone | "a door on a corridor" is the task's own legality clause, and only an articulation point can ever be load-bearing |
| braid | **0.08**, not CMAZE-1's adopted 0.40 | measured: at 0.40 a door cuts the agent from the goal in **2.1 %** of requests and RULES A/B are wallpaper; at 0.08 it is **29.1 %**. CMAZE-1 adopted 0.40 for ROUTE multiplicity, which CMAZE-O does not use. `_braid` is called with a different number; the engine is untouched. |
| legality | on-floor, door-on-corridor, distinct, tint-clear, colours-in-range, **feasible under A+B by the engine's own `solve`** | makes the legality clause bite on arm outputs; the reference is always legal, so `admissible` = feasible |
| door load-bearing | **reported, never required** | requiring it forces `door colour == key colour` and collapses two words of the vocabulary into one |
| `valid` | `exact_objects AND legal` | the nine words NAME the eight object facts; the layout is the backdrop and is reported separately as `exact_layout` |
| colour in the item tensor | a **shared colour field**, not a per-slot colour block | a per-slot block makes "a colour never seen in THIS slot" an output channel that was never once positive, so every slot-value hold-out would be structurally zero and say nothing about any arm |
| value hold-outs | **both**: strict (colour 5 never seen anywhere) and slot (colour 4 never on an agent) | the strict one is the task file's literal wording and is a judge-integrity check; the slot one is the version that can separate arms |
| n-way | **n = 4** over (layout, agent zone, key colour, tint colour), cardinality 1,728 | uniform sampling cannot supply an n-way row over 4 zone words (6,561 combos, ~60 % unseen by accident); this set is **97 %+ covered by train**, so the held-out tuples are genuinely special |
| AE object loss | **spatial softmax over the 225 cells** + masked colour softmax | measured: per-cell BCE gave an AE ceiling of `exact_objects = 0.0025`; the judge takes an argmax, so the loss must be the cross-entropy of that argmax |
| stratification | 25 % of every pool drawn with a load-bearing door | a load-bearing door is a ~5 % event under uniform draw and the breakout would have no n |

## 2. Environment note

`PYTHONPATH` fix recorded in section 0. **No scored run is spent, so no `/c/chlu/SCORED-LEDGER.md`
row is written** (COMMON.md scopes that discipline to spokes that spend one).

## 3. Runs

* Instrument probes (before PREREG, both instrument construction, neither an arm number):
  braid probe; AE loss probe (`artefacts/ae_probe_v3.log`).
* `PREREG.md` + `PREREG.timestamp.txt` written at 2026-09-22T21:53:56Z.
* Tiny end-to-end smoke (6 layouts, 500 train, 40 AE steps): whole chain green —
  disjointness pass, 0 impossible requests, **4,800 states replayed, 0 disagreements**,
  store-scoped gap 2e-15 relative. `tiny/`.
* **VAL smoke, 3 seeds (0,1,2), 12 layouts, 8,000 train, 6,000 AE steps, 3,000 rival steps**
  launched; PID in `PIDS.txt`. Output `valsmoke/`.

## 4. Results (all VAL, 3 seeds, zero scored runs)

**`HARNESS-READY.json` → `"status": "ready"`. All six acceptance criteria pass.**

* independent replay: **105,408 judged states, 0 disagreements**, layout-bank rebuild
  mismatches 0 (`valsmoke/VERIFY.json`).
* impossible requests shown to a non-refuser arm: **0 of 30,588 checked**.
* hold-out disjointness: pass on 3/3 seeds; n-way factor-set train coverage
  0.9740 / 0.9693 / 0.9595 of 1,728.
* `tolerance_used`: 0 non-null anywhere in the smoke document.

Headline `valid` (3-seed mean): AE ceiling 0.7522 / CLU **0.2728** / transformer 0.1817 /
MLP 0.1211 / LSTM 0.0094 / cont-MLP 0.0006 / LOOKUP floor 0.1506 on `seen`; on
`unseen n-way` CLU **0.1158** and the **LOOKUP floor 0.0000**.

**Bars (PREREG):** B1 pass (ceiling 0.7522 ≥ 0.50) · **B2 FAIL** (only
`landscape_removed` collapses) · B3 pass (lookup floor 0.1158 below the CLU on
`unseen n-way`) · B4 pass on `seen` · B5 pass · B6 pass (store-scoped gap 2.7e-15
against a 1e-6 bar).

### Two findings I did not expect, both recorded and shipped

1. **`landscape_shuffled` (atom ROW permutation) is a PROVED NULL for this read**, not a
   weak control. It agreed with the CLU to every printed digit on all 5 rows and all 3
   seeds; the span sweep measures the read gap at **1.9e-06 – 5.7e-06** against ~15–23 for
   every other control. The algebra is in
   `chlu/exp_w15_cmaze_o_arms.py`'s docstring: re-solving `W` against `P G` gives `W* Pᵀ`,
   so `Φ (W* Pᵀ)(P G) = Φ W* G`. **Choice made:** keep the control (a control that cannot
   move is worth stating once), add `landscape_shuffled_channels` to
   `CONTROLS_SHIPPED`, and measure it in the span sweep. The VAL smoke's control set is
   reported as run, and `HARNESS-READY.json → controls` names both sets explicitly. I did
   **not** re-run the 71-minute smoke for a renamed control, because the charter-C verdict
   (interface, not store) is unchanged by it.
2. **The charter-C controls do not collapse**, so by COMMON.md this is an **interface
   claim**, and FINDINGS says so. **Choice made:** rather than report that as a bare
   failure, I added a **span sweep** (`scripts/w15_cmaze_o_span.py`, 3 seeds, 26 min) to
   locate where the store starts to matter. It does not: the random-atom fraction falls
   from 0.55× the CLU at span 0.688 to 0.25× at 0.312, but the CLU falls with it
   (0.2728 → 0.0267), so no point on the grid has both a working CLU and collapsing
   controls. The atoms are **0.76 %** of the deployed read; the address book is 99.24 %.
   The pair-block ablation is named as the next measurement and DECLARED NOT-RUN.

## 5. Artefacts written (verified by directory listing, not by git status)

`FINDINGS.md` · `HARNESS-READY.json` · `PREREG.md` + `PREREG.timestamp.txt` · `TABLES.md` ·
`LOG.md` · `START_UTC.txt` / `END_UTC.txt` · `PIDS.txt` ·
`artefacts/{braid_probe.json, ae_loss_probe.json, ae_probe_v1.log, ae_probe_v2.log, ae_probe_v3.log}` ·
`valsmoke/{VAL-SMOKE.json, VERIFY.json, SPAN-SWEEP.json, seed0-2.json, states_seed0-2.npz, bank_seed0-2.npz, run.log, span.log}` ·
`tiny/` (the end-to-end chain smoke).

## 6. Background jobs

All four of my PIDs (`PIDS.txt`) ran to completion inside a turn and are finished. ⛔ No
box-wide kill was ever run; no process of mine was killed.
