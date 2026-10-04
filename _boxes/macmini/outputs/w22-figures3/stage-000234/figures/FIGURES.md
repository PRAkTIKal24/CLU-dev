# FIGURES.md

## ⭐ w22 `figures-redraw` pass 2 (2026-09-24): the current state. Read this section first.

**Zero scored runs; nothing written to any ledger.** Build: `bash build.sh` = `make_figures.py` then
`make_figures_w22.py` (which imports `make_figures_w21.py` as a library; `maze_six.py` reads the
six-seed CMAZE-T artefacts), twice, SHA256-compared (`SHA256-w22-build{1,2}.txt`), installed
byte-identically into **both** `w17-paper/figures/` and `outputs/paper-visuals/`.

| stem | what it shows | source (every value read, never typed) |
|---|---|---|
| `fig_ladder` | store vs **three** rival sets, one panel each; **pool rows AND optimiser steps in every panel title**: (a) 32,256 rows · 5,040 steps (spoke-A `@ref`), (b) 32,768 rows · 2,560 steps (data-matched MLP), (c) **131,072 rows · 20,480/204,800 steps** (w20, P1-carried fit per family × seed; filled dot = seed carried at 204,800) | `data/ladder_rival_sets.json`; w20 `SHAPES-MATCHED.json ladder.*`; steps asserted per rung × seed × family |
| `fig_headline_steps` (NEW, `fig:headline-steps`) | rows 3–4 TEST: rival score against optimiser steps at 16,384 / 32,768 / 131,072 rows — the operative variable was steps, not data | `SHAPES-MATCHED.json rows34.arms`; each fit's `declared.json`/`tenx.json` `optimizer_steps`, `n_fit_rows` |
| `fig_maze_conditions` | (a) condition table + CLU-REL-A10 row (Head's rule, seeds 0–2); (b) A1 and (c) CLU-REL-A10, **six seeds**, per-seed **ceiling** bars, **hard seeds 1, 4 shaded** (A1 ceiling < cut 0.6407), best matched rival (open ▲), best **+rel** rival (filled ▲), highest control (×); every seed where a rival/control reaches the CLU is **marked by rule** (seed 4: control = CLU; seed 5: rival = CLU; A1 seeds 1, 5: rival > CLU) | per-seed run JSONs `ROWS.nway`, asserted float-equal to `REL-NEWSEEDS.json nway_summary`; `MAZE-T-SEED4.json classification` |
| `fig_maze_rel_tuples` (NEW, `fig:maze-rel-tuples`) | CLU-REL-A10, every held-out n-way tuple × 6 seeds: CLU, best matched rival, best +rel, highest control; "= CLU on k/n" per row (seed 4: control and +rel 16/16; seed 5: rival and +rel 16/16) | `w21-rel-newseeds/replay/BOTH_test_seed*_PERTUPLE.json` (stored `[n, correct]`); objects hash-identical to the staged ones |
| `fig_deletion` | the **four measurements** (w20, book refit): (a) store = refit-without, (b) book = refit-without, (c) refusal display, (d) before→after with **the random-1 % item control beside the conjunction**, (e) paired Δ + CI, (f) CLU after − each rival retrained without | `SHAPES-MATCHED.json deletion_matched.per_scope.*` |
| `fig_crossover`, `fig_cmazet_*`, others | unchanged from pass 1 | — |

**Colour roles (dataviz, validated palette unchanged):** CLU blue; per-seed "best matched rival" is a
*role* (its family varies by seed) → slot 2 orange; **open = without the relational request columns,
filled = with them**, for both the CLU (A1 vs REL) and the rival (matched vs +rel); controls grey ×;
ceiling ink bar. Families keep their pass-1 hues wherever a family is the entity.

**Pending slots (data, not code):**
- `data/ladder_rival_sets.json` → `ladder_rivals_32k` (32,768 rows; still running on forgis-14). Set
  `status: landed`, `source`, `reader`, `arms`, `steps`, `steps_check`; rebuild. Tested with a stand-in
  4th set: the figure becomes a 2 × 3 grid, no code change.
- `data/cmazet_panels.json` → `fig_cmazet_rel_per_arm` (CLU-REL-A10 per-row panels): gate 1 PASSED
  (192/192 objects), gate 2 cannot run — the REL pickles need module `w21_clu`, absent on batmac
  (`NEEDS-ARTEFACTS.json` in the w22 outputs).


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

## w17 `plots` (history; banner rule and placeholder list superseded by w21 above)

Owner: `w17/plots`. This directory (`figures/`) is owned by `plots`; ⛔ `figures/rendered/` belongs to
`rendered-compositions` and is never written by this spoke.

**Evidence base:** `~/clu/evidence/outputs/` (read-only). **No w15 or w16 artefact is read by any
figure** (AMENDMENT 1). Every plotted value is read from the files listed below by
`make_figures.py`; nothing is typed in, rounded or recalled. Machine-readable provenance:
`PROVENANCE.json` (written by the same build).

**Rebuild:** `bash build.sh` (from this directory, or from the `w17/plots` branch at
`w17_plots/build.sh`). It builds twice into scratch dirs, diffs the SHA-256 of every output and
refuses to install unless they match. Python 3.12.14 · matplotlib 3.10.8 · numpy 2.4.1 ·
backend Agg · font DejaVu Sans (matplotlib-bundled, TrueType embedded, `pdf.fonttype 42`).
Proof: `SHA256-build1.txt` = `SHA256-build2.txt` (see `~/clu/outputs/w17-plots/FINDINGS.md`).

## One arm, one name, one colour — in every figure

| arm | name on every figure | colour | marker |
|---|---|---|---|
| our object | **CLU** | blue `#2a78d6` | ● |
| 1× byte-matched learner | **Learned 1× (four-part total)** | orange `#eb6834` | ■ |
| MLP rival | **MLP** | aqua `#1baf7a` | ▲ |
| LSTM rival | **LSTM** | yellow `#eda100` | ▼ |
| Transformer rival | **Transformer** | magenta `#e87ba4` | ◆ |
| continuous-MLP rival | **Continuous MLP** | green `#008300` | ✚ |
| lookup table | **Lookup table (floor)** — a printed floor, never a rival | grey `#6b6a66`, dashed | × |
| charter-C controls | the control's own name | light grey `#b5b4ae` | × |
| CLU encoder configurations (fig 4 only) | **frozen encoder / adapter / anchor** | blue / violet `#4a3aa7` / red `#e34948` | ● ■ ▲ |

Colours are slots 1–8 of the dataviz reference palette in its documented (validated) order;
every series also carries a distinct marker or hatch, so identity is never colour-alone. The
CLU's own byte-ledger parts use one hue (the CLU's blue), light → dark. The maze request-layer
audit (fig 5b) shows **environments, not arms**, so it uses neutral greys and never an arm colour.

## ⛔ Byte convention (COMMON: "use ONE convention throughout, or print the bracket")

**ONE convention in every figure: the crossover hub's FOUR-PART honest ledger** — atoms +
covariance/count + address book + coupling/hole table, **summed, never netted** — with
524,288 B drawn only as a **reference line / reference-pin label**. The K = 5 full-square-priced
1× opponent (3,177,896–3,241,076 params, the row the 938,413–1,710,324 bracket was about) is
**not drawn anywhere**. The 524,288 B rivals are the read-path-matched naive rivals (~130 k
params). The w12 `pca_tensor` / `DowndatableStore` convention does not appear.

## The figures

### 1. `fig_crossover.{pdf,png}` — ⚠ PLACEHOLDER
(a) val composition vs K (wells); (b) four-part byte ledger vs K, stacked by part.
- **Source:** `w12-single-design-crossover/CROSSOVER-val.json` (`per_K.*.val_composition_per_seed`,
  `val_composition_mean`, `ledger_four_part.*`, `control_gate_at_K_star`).
- **Seeds:** n = 3 (0, 1, 2) at K = 5 and K = 32; **n = 1 (seed 0) at K = 8…28** — printed in
  the legend. The "3-seed band" the task asks for exists only at K = 5 and K = 32 in w≤14 evidence.
- ⚠ The artefact's first key is `SMOKE_NOT_A_CLAIM: true`. The plotted byte column is
  `honest_total_bytes` and its four parts (asserted to sum to it), **not** the parts-1-2-4
  column the w13 advisor report §5 flags as the Hub's error.
- **Awaiting:** w15 `crossover-seeds` (3 seeds at every K). ⛔ It may confirm, narrow, or
  overturn the no-crossover reading; the figure presumes nothing.

### 2. `fig_ladder.{pdf,png}` — ⚠ PLACEHOLDER
(a) composition vs n per arm at K\* = 32, mean line + min–max seed band, per-corner ceiling
dotted; (b) margin CLU − Learned 1×, per seed + mean + band, 0 and the declared +1.0 margin.
- **Sources:** `w12b-nr6-and-kstar/KSTAR-LADDER.json` (`by_rung.*.kstar_K32.{store,learned_1x,margin}.per_seed/mean`,
  `legs.*.kstar_K32.*.ceiling_per_corner`) and `w12-composition-ladder/LADDER.csv`
  (`rival_{mlp,lstm,transformer,cont_mlp,table}` rows, `mean_distinct_valid_novel_tuples`, seeds 0–2;
  these 524,288 B rivals do not depend on K).
- **Seeds:** 3 on every rung.
- ⛔ **The n-axis is labelled for what it is:** "n ≥ 3: the same 120 corners" — for n ≥ 3 it is
  a TRAINING-DATA axis; only n = 2 vs n = 6 is a depth contrast. The draft's caption must carry
  the Head's inference note; the figure carries it on the axis.
- ⛔ **Panel (b) is "margin vs Learned 1×", NOT "vs best rival".** The build checks whether
  Learned 1× is the strongest learned arm at every (rung, seed): **it is not** — at n = 3,
  seed 0 the 524,288 B LSTM and at n = 3, seed 2 the 524,288 B Transformer score above it.
  `\TODO{no artefact for the K* margin against the best of ALL learned arms at n = 3}` — the
  KSTAR-LADDER artefact prints margins against Learned 1× only, and this spoke does not
  re-derive a margin.
- **Awaiting:** wave-18 regeneration (Amendment 1 names the ladder). No w15/w16 ladder row is read.

### 3. `fig_deletion.{pdf,png}` — ⚠ PLACEHOLDER
(a) CLU composition before vs after deletion, per scope, TEST, K\* = 32; (b) adaptation seconds
(log); (c) data rows read — CLU downdate, CLU refit-without (the exact reference) and each rival's
retrain-without.
- **Sources:** `w12-single-design-crossover/deletion/test_seed0_K32/deletion.json`
  (`scopes.{value,conjunction_live,item}.{before_same_corners,after}`, `scopes.value.our_cost_line`,
  `scopes.value.exactness_store_scoped.refit_without_cost_line`, `scopes.value.receipt.n_before`);
  `w12-deletion-baselines/comparators/seed_0/{mlp,cont_mlp,transformer,lstm}__retrain_without__value-object_hue-0.rival.json`
  (`cost_line.{seconds,data_rows_read}`, `deletion_counts.n_fit_before`).
- **Seeds:** **1 (seed 0)** for the CLU; rival cost lines are seed 0 (rows read and bytes are
  seed-invariant per the spoke's own FINDINGS §4).
- ⛔ Pools differ and are printed on the axis: CLU 131,072-row pool, rivals 16,384-row pool.
- ⛔ Language carried into the notes: exactness is **store-scoped**; frozen encoder/decoder and the
  address book are **outside the deletable object**; adaptation is **light and data-free** — never
  "no changes". The vacuous red ∧ cube scope is **not plotted** (it deletes 0 rows).
- Rival post-deletion composition on the scored corners: **never measured** in w≤14 evidence
  (DECLARED NOT-RUN there), so (a) is CLU-only. `\TODO{no artefact for rival post-deletion composition on the scored corners}`
- **Awaiting:** w15 `single-design-seeds` (deletion rows at seeds 1–2). ⛔ Those may reproduce
  or break the seed-0 reading.

### 4. `fig_encoder_adapt.{pdf,png}` — FINAL (not a placeholder)
Per-seed margin (single design − best learned baseline, `mlp_180k`) for frozen → adapter and
frozen → anchor, TEST, K\* = 32.
- **Source:** `w13-encoder-adapt/CROSS-ARM.json` (`MARGIN.{adapter,anchor}.per_seed.*`).
- **Seeds:** 3 seeds × 3 configurations = 9 arm-runs (3 SEEDS, never "9 seeds").
- ⛔ Opponent is `mlp_180k`, **not** Learned 1× — never merged with fig 2.
- Needs no w15/w16 artefact.

### 5a. `fig_maze_readverb.{pdf,png}` — ⚠ PLACEHOLDER (maze)
Validity under the exact judge, AVG (latent mean) vs MERGE (plan-space verb), per arm, on the
`unseen_pair` and `n-way` rows; per-seed dots; pipeline-ceiling seed range shaded.
- **Sources:** `w14-read-verb/runs/scored1_seed0_test.json`, `scored2_seed1_test.json`,
  `scored3_seed2_test.json` (`table.{unseen_pair,nway}.<arm>.{AVG,MERGE_target_wins}.validity`,
  `pipeline_ceiling_rows.*.validity`). `MERGE_target_wins` is the arm per `w14-read-verb/TABLES.md`
  (verified by inspection this session: the JSON value 0.51256… matches the table's 0.5126 for CLU unseen_pair seed 0).
- **Environment:** CMAZE-2b only. **Seeds:** 3.
- ⛔ This is an **INTERFACE (verb)** result: the same merge lifts the lookup floor more than the
  CLU and the charter-C controls do not collapse ⇒ no store row. The lookup floor at ~0.97 is the
  **within-maze** floor that cannot transfer across mazes.
- **Awaiting:** w16 maze-wm rows (composition, rollout validity, deletion). ⛔ Each may land as a
  declared negative.

### 5b. `fig_maze_requestlayer.{pdf,png}` — ⚠ PLACEHOLDER (maze, APPENDIX)
(a) keys per optimal route, before vs after the request-layer fix; (b) scripted-planner validity
at the opt + 10 % budget.
- **Sources:** `w14-request-layer-fix/artefacts/measure_before_0.json`,
  `measure_after_{0,1,2}.json` (`key_count_histogram`, `all_pairs_slack10.blind.*.validity`).
- **Seeds:** before = seed 0 only (the only before-artefact on disk); after = 3.
- ⚠ **Judgement call, flagged for the Hub/referee:** this figure shows the CMAZE-2 and CMAZE-2b
  request layers side by side **because the figure is the fix itself** — an environment audit
  with no learned arm and no score compared across environments. If the Hub reads COMMON's
  "never in one table" rule as covering figures, drop panel (a)'s before bars and (b)'s before
  bars, or drop the figure (appendix only).
- **Awaiting:** w16 maze-wm (CMAZE-T) rows supersede the maze section; may be dropped or re-scoped.

## Placeholder list (for wave 18 `placeholder-fill`)

| figure | placeholder? | w15/w16 artefact it awaits | may land as a negative? |
|---|---|---|---|
| fig_crossover | **YES** | w15 `crossover-seeds` | yes — a crossover may appear or the flat reading may narrow |
| fig_ladder | **YES** | wave-18 regeneration (Amendment 1) | yes |
| fig_deletion | **YES** | w15 `single-design-seeds` (deletion rows, seeds 1–2) | yes |
| fig_encoder_adapt | no | — | — |
| fig_maze_readverb | **YES** | w16 maze-wm composition / rollout / deletion | yes — every w16 row may be a declared negative |
| fig_maze_requestlayer | **YES** (appendix) | w16 CMAZE-T harness | may be dropped |

Every placeholder figure carries the red tag **"PLACEHOLDER · w≤14 evidence · wave 18
regenerates"** above its panels. ⛔ The tag must be removed only by the regenerating build, never
by hand.

## LaTeX include block (paths relative to the draft; source comments travel with the figure)

```latex
% SOURCE: w12-single-design-crossover/CROSSOVER-val.json  (PLACEHOLDER: w15 crossover-seeds)
% BYTES: four-part honest ledger, summed never netted; 524,288 B is a reference line only
\includegraphics[width=\linewidth]{figures/fig_crossover.pdf}

% SOURCE: w12b-nr6-and-kstar/KSTAR-LADDER.json ; w12-composition-ladder/LADDER.csv  (PLACEHOLDER)
% BYTES: Learned 1x at the four-part K* total; rivals at the 524,288 B reference pin
% n >= 3 rungs share the identical 120 corners: a TRAINING-DATA axis
\includegraphics[width=\linewidth]{figures/fig_ladder.pdf}

% SOURCE: w12-single-design-crossover/deletion/test_seed0_K32/deletion.json ;
%         w12-deletion-baselines/comparators/seed_0/*__retrain_without__value-object_hue-0.rival.json
% (PLACEHOLDER: w15 single-design-seeds; CLU n = 1)
\includegraphics[width=\linewidth]{figures/fig_deletion.pdf}

% SOURCE: w13-encoder-adapt/CROSS-ARM.json  (3 seeds; opponent mlp_180k)
\includegraphics[width=0.55\linewidth]{figures/fig_encoder_adapt.pdf}

% SOURCE: w14-read-verb/runs/scored{1,2,3}_seed{0,1,2}_test.json  (PLACEHOLDER: w16 maze-wm)
\includegraphics[width=\linewidth]{figures/fig_maze_readverb.pdf}

% SOURCE: w14-request-layer-fix/artefacts/measure_{before_0,after_0,after_1,after_2}.json  (PLACEHOLDER; appendix)
\includegraphics[width=\linewidth]{figures/fig_maze_requestlayer.pdf}
```

Compiled against `iclr2027/iclr2027_conference.sty` with pdflatex (TeX Live 2026): all six
include at `\linewidth`, 0 errors, 0 overfull boxes.
