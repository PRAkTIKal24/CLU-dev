# FIGURES.md

## ⭐ w21 `figures-redraw` (2026-09-24): the current state. Read this section first.

**No figure carries an in-image banner.** Placeholder status lives only in PROVENANCE.json and the
draft's placeholder index. `make_figures.py`'s `tag_placeholder` is a no-op now. The w17 text
below is kept as history; its banner rule and its placeholder list are **superseded**.

**Rebuild:** `bash build.sh` from this directory. It runs `make_figures.py`, then
`make_figures_w21.py` (with `--panels data/cmazet_items.json` when that file is present), twice
into scratch dirs. It diffs SHA-256 over every output and installs only if both builds match.
Proof files: `SHA256-w21-build1.txt` = `SHA256-w21-build2.txt`. The earlier `SHA256-build*.txt`
and `SHA256-maze-w18-*.txt` are kept as history. `PROVENANCE.json` is merged, never dropped: each
entry carries its sources and the **SHA-256 of every source file**.

**Rows, rival sets and panel sets are DATA** (`data/`). Adding one means editing a JSON file and
rebuilding. It needs no code change.
- `data/ladder_rival_sets.json`: one entry per rival column set, and one panel per landed set.
  Each set's pool size is asserted from its own artefact. Landed: the 32,256-row
  combination-split set and the data-matched MLP (32,768 rows). **Not landed on this machine:**
  the 131,072-row set (w20 `shapes-data-matched`) and `spoke/ladder-rivals-32k`. To add one, set
  `status: landed` and give `source`, `reader`, `arms`, `pool_source` and `pool_path`, then add a
  caption line to the draft.
- `data/maze_conditions.json`: the rows of the condition table, top to bottom. The
  `words_plus_observation_improved` row waits for `w21-maze-t-clu-improve`: set `status: landed`
  and give its BARS artefact and schema.
- `data/cmazet_panels.json`: one entry per Part-C panel set, and one column per arm. An improved
  CLU is added as an arm entry, plus its own per-item re-read made by `cmazet_extract.py`, which
  must pass the same reproduction gate.

| figure | shows | source artefacts | seeds |
|---|---|---|---|
| `fig_crossover` | (a) val composition vs K: mean ± 1 seed sd, each seed, and the highest control; (b) four-part ledger (seed mean, stacked, sd on the total) | `w15-crossover-seeds/CROSSOVER-3SEED.json` | 3 at **every** K |
| `fig_ladder` | store vs each rival set, one panel per set, with the table floor | `w12b-nr6-and-kstar/KSTAR-LADDER.json`, `w12-composition-ladder/LADDER.csv`, pool files | 3 |
| `fig_deletion` | (a) survivor composition before/after for the store and 4 rivals × 3 scopes; (b) seconds; (c) rows read | `w15-single-design-seeds/T7-3SEEDS.json` | 3 |
| `fig_maze_conditions` | CMAZE-T condition table: per-seed CLU and R\*, the Head's-rule label, no margin bar | BARS JSONs listed in `data/maze_conditions.json` | 3 per row |
| `fig_cmazet_families` / `_door_then_tint` / `_per_arm` / `_deletion` | every arm's generated s_{t+1} and the judge's mark | persisted w19 BOTH TEST objects → `data/cmazet_items.json` (re-read gate: `data/cmazet_items.REPRO.json`) | families: seed 0 (rule\_a from seed 2); others: seed 0 |
| `fig_encoder_adapt`, `fig_maze_readverb`, `fig_maze_requestlayer` | unchanged w17 figures, rebuilt without a banner (bytes equal the w18 installs) | as below | as below |

**Maze labels (the Head's rule, computed by one function from each BARS artefact):**
- Rivals on the observation: **ASYMMETRIC** (no claim label).
- Words, main effects (VAL): **LOSS**.
- Words, full design (TEST): **LOSS**. This equals the artefact's own `HEAD_LABEL` (asserted).
- Observation (VAL): **COMPARABLE**. |Δ̄| = 0.1053 ≤ σ = 0.1068. ⚠ The brief's table said LOSS;
  the rule applied to the artefact says COMPARABLE.
- Words + observation (TEST): **WIN** against R\* = LSTM at 80,000 steps, although w19's own
  pre-registered margin bar B1 failed.

**Part C gate:** the persisted objects were hash-verified (132/132, with MANIFEST, `.sha256` and
the recomputed `param_sha256`). They were then re-read on batmac. The BOTH TEST n-way reads are
0.800 / 0.566 / 0.808 (mean 0.7246), with controls 0.000 / 0.225 / 0.244 / 0.166. Every one of the
16 arms, and every deletion row, equals the scored JSON floats exactly on all 3 seeds.

**Palette:** these are the w17 arm colours, the dataviz reference slots. They were validated with
`validate_palette.py`, a Python twin of the skill's JS validator, because this box has no node.
The port reproduces the skill's documented 9.1 / 19.6. The 5-hue ladder set, checked over all
pairs, gives CVD WARN (6.1) and normal-vision PASS (15.6). Secondary encoding is therefore
applied: a distinct marker per family plus direct end labels, and the paper's tables serve as the
table view. Output is print PDF on white, light mode only.

---

