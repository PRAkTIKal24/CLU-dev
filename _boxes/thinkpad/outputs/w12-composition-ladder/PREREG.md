# PRE-REGISTRATION — w12 hub/composition-ladder

**Written 2026-09-21T11:38:41Z** (`PREREG.timestamp.txt`), **before any arm number existed
on disk**: `C:/chlu/outputs/w12-composition-ladder/runs/3_seed0/` was empty of `rows_*.json`
at that instant and no other `runs/` leg directory existed. The only things read beforehand
were source code, `LADDER-READY.json`, `CROSSOVER-val.json`, the two advisor reports, and
index arithmetic written to `instrument/INSTRUMENT.json` (corner counts and row sets — no
arm, no score).

---

## 1. What is NOT re-declared here

The **estimand and the bar are the harness's**, pre-declared in
`/c/chlu/outputs/w12-kway-ladder-harness/PREREG.md` and carried in
`LADDER-READY.json:estimand`:

- unit = **corner**; statistic = mean paired per-corner difference;
- metric = `distinct_valid_novel_tuples`; margin `DECORATION_MARGIN_NOVELTY = +1.0`;
- paired bootstrap CI95, 10 000 resamples, `default_rng(20_260_919)`; two-sided sign test;
- function = `chlu.experiments.exp_gen_factor_wells.paired_estimand`.

This hub introduces **no new metric and no new margin**, and does not restate them.

## 2. Seeds — 3, DECLARED

`/c/chlu/outputs/w12-cache-gen2-seeds-3-4/CACHE-READY.json` **did not exist** when this hub
started (11:31:33Z; that directory then held only `START_UTC.txt`, `LOG.md`, `PIDS.txt`,
`bridge/`, `bridge_seed_3.log`). Generation-2 **full** 480 000-row caches exist for seeds
**0, 1, 2** only
(`C:/chlu/outputs/spoke-a-rivals/cache_generation2/shapes3d_recon/seed_{0,1,2}/feature_cache.json`
→ `generation: 2`, `scope: "full"`, `n_rows: 480000`; seed_3 and seed_4 absent).

⇒ **n_seeds = 3 on every row of this hub.** No 3-seed and 5-seed row is mixed, and every
plot and table prints `n`.

## 3. K\* — NOT PUBLISHED, so K = 5, DECLARED

`/c/chlu/outputs/w12-single-design-crossover/CROSSOVER-val.json` reads
`"K_star": null`, `"K_star_reason": "K=5 reference row absent; rule not applicable"`, with all
nine swept K listed under `not_run`. ⇒ the single-design arm runs at **K = 5**, DECLARED, and
that declaration is carried on **every** single-design row.

## 4. The arms, fixed now

| arm | what it is | byte matching |
|---|---|---|
| `single_design_K5` | `chlu.core.downdate_store.DowndatableStore`, the w8 single design, K=5 DECLARED | its own deployed total (**1×**) |
| `pca_tensor` | the wave-5 subspace store (`exp_w5_compose_field.PCASpanStore` + `TensorAddress`) — the Head's *"re-run with the old w5 store if n-way fails"*, an arm from the start | its own deployed total (**1×**) |
| `rival_mlp`, `rival_lstm`, `rival_transformer`, `rival_cont_mlp`, `rival_table` | spoke A's fitted learners and the table FLOOR | the **524,288 B reference** pin |

**Controls on every store arm at every k** (charter C.2, non-negotiable after wave 9):

- `landscape_shuffled` — `perturb_modes('value_permute')` applied **after** the address book
  is fitted: the arm's own book, the arm's own bytes, the atom values destroyed.
- `random_atoms` — `perturb_modes('random_span')` applied **before** the book is fitted, and
  the book **refitted** against the random atoms. The strong form; the same object
  `exp_w5_compose_field` calls `pca_random_refit` and `w9_maze/arms.py` calls `random_atoms`.

**If a control does not collapse at a rung, the store is DECORATIVE at that rung and no row
is claimed for it.** `rank/width` is printed on every store row.

## 5. ⭐ THE BEST LEARNED BASELINE — the rule, fixed before any number

Panel 2 of the figure is the paired margin against the **best learned baseline**, and "best"
is defined **now**:

> For each (rung k, seed s) independently, the best learned baseline is the learned arm with
> the **largest mean `distinct_valid_novel_tuples`** over that rung's corners, among
> `{rival_mlp, rival_lstm, rival_transformer, rival_cont_mlp, rival_table}`.
> The paired margin is then `paired_estimand(arm, that baseline)` on the same corners in the
> same order.

This rule picks the **strongest** opponent available at that rung and seed. It is therefore
the hardest bar of the five and it runs against this hub's own interest; it is not "the mean"
and it is not a baseline chosen after seeing which one flatters an arm. The identity of the
winner at each (k, seed) is reported beside the margin.

## 6. The rungs, and the reproduction reference

Rungs **k = 2, 3, 4, 5, 6** come from `chlu.data.gen_kway_ladder.build_kway_split` — the
harness's families. ⛔ This hub builds none of its own; if the harness is wrong, this hub
reports that.

A sixth condition, **`shipped`**, is run with the identical arms and seeds on the
value-hold-out condition **as it exists on disk** (`C:/chlu/data/gen/shapes3d/value_holdout/`),
scoring `pair_subcorners` (row 3, 60 corners) and `nway_corners` (row 4, 49 corners). It is
the **reproduction reference** for the non-negotiable *"k=2 must reproduce row 3 and k=6 must
reproduce row 4"*: the ladder's k=2 corners are row 3's corners and k=6's first 49 are row 4's
(row set for row set — measured, `instrument/INSTRUMENT.json`), but the rungs sit on
**different splits**, so the reproduction is a claim about numbers and is tested as one.

**Declared before the comparison:** the ends reproduce if, per seed, the k=2-vs-shipped-row-3
and k=6(first 49)-vs-shipped-row-4 arm means agree **within that arm's own across-corner sd**
on that rung. If they do not, the ladder is reported as **off-instrument at that end and not
plotted**, exactly as the task requires.

## 7. Declared confounds, carried not corrected

- **Held-out fraction varies with k** and cannot be equalised: the harness measured
  `required_floor = 45 000 > available_ceiling = 8 640` ⇒ `feasible: false`
  (`LADDER-READY.json:row_matching_feasibility`). Every rung's held-out row count and
  fraction is printed on every row. This hub does not invent a correction.
- **The store is fitted on the same 131 072 rows at every rung** (every rung's train index
  clears `n_pool`; `LADDER-READY.json:families.*.fit_pool_feasibility`), so the confound acts
  through pool *coverage*, not pool *size*.
- **The learned rivals are fitted on the `montero_combination` pool**, not on any rung's own
  pool (`rivals/seed_*/mlp.rival.json:population.fit_population`). Spoke A's own
  `row_eligibility.json` rules rows 3–4 — full six-factor corner requests —
  `can_enter: true`; that ruling is **carried**, not re-derived, to k = 3, 4, 5, and the
  exact fit-row overlap per rung is a **DECLARED NOT-MEASURED** (the fit-row index is not
  published in the checkpoints, and reconstructing it requires `CHLU_CLAUDE_ROOT`, which this
  box's standing rules forbid this hub to read).

## 8. What would prove this hub wrong (restated from the task, so it cannot drift)

1. **Every arm collapses at the same k** ⇒ that is a property of **Shapes3D**, not of any
   arm, and must be reported in those words — a finding about the dataset's composition
   ceiling.
2. **`pca_tensor` matches or beats the single design at every k** ⇒ reported plainly. The
   single design's justification is deletion, not composition, and this would say so.
3. **No arm deteriorates at all** ⇒ the ladder does not discriminate on this dataset; say so
   rather than reporting a flat line as a result.

## 9. Not run, declared in advance

- **The 1× byte-matched learned column.** A rival matched to the single design's own
  four-part total (3 922 450 B at K = 5 = **7.481 ×** 524 288 B,
  `/c/chlu/outputs/w12-single-design-crossover/LEDGER-ACROSS-K.json`) does not exist on this
  box: every fitted rival is matched to **131 072 float32 parameters**. So the learned column
  is the **524 288 B reference** column and is labelled as such on every row; the 1× column
  is a DECLARED NOT-RUN unless a matched checkpoint appears, and a 524 288 B rival **flatters
  this hub** (it is several times smaller than the object it is compared with).
- **dSprites** — the ladder is defined on Shapes3D's `object_hue`/`shape`; no number is
  produced for any other dataset and none is implied.
