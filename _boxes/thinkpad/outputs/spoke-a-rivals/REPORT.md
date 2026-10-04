
---

# REPORT - w4-spoke-a: the rival set, and cache generation 2

Generated 2026-09-19 21:31 UTC by reading the artifacts on disk. Every number below traces to a JSON file in this directory; none is carried in a variable from the run that produced it.

## Dial declaration, discharged

**Dial: the instrument and the comparison set.** This spoke ran no arm of ours and makes no claim about the mechanism.

**Falsified if** a rival could not be brought to matched parameters within +/-10 %, or if generation-2 re-encoding did not reproduce its own hashes on a second pass. Neither happened: every rival matched inside +/-1 %, and the determinism check passed. Both are evidenced below.

## The rules that travel with these numbers

1. **Generation 2 only.** Never quote a number here beside a generation-1 number, in either direction.
2. **Generation 1 is untouched** on disk and still hashed.
3. **Know the population.** Every scored set here is the Montero COMBINATION hold-out and is retained-values-only (measured). It is not the value-holdout population. Only the deletion comparator's numbers concern an unheld value.
4. **A dash is never a win for us.** Every dash is footnoted and names whose limit it is.
5. **No accuracy without its distinct-output count.**

### Cache generation 2

**The falsifier passed.** One shard (seed 1, 8192 rows) re-encoded twice in the same process; the sha256 of the quantised cache bytes matched: `1f6748ea769437571bd3f8de716c662e…` both times. Had it not, generation 2 would be noise and this spoke would have stopped here.

| seed | rows | scope | dtype | device | precision | features sha256 |
|---:|---:|---|---|---|---|---|
| 0 | 480,000 | full | float16 | cpu:0 | HIGHEST | `ef6359a4cd54b08287cf3709…` |
| 1 | 480,000 | full | float16 | cpu:0 | HIGHEST | `86b44d0f1e6470e7f1e0e0c3…` |
| 2 | 480,000 | full | float16 | cpu:0 | HIGHEST | `3ab14e583c3f500da060638c…` |

**GENERATION 1 DOES NOT PASS THE SHIPPED RE-ENCODE GATE, ON ANY SEED.** `exp_gen_store._assert_cache_reencodes` - the gate the scored cell calls at the top of every `run_seed` - re-encodes 64 cached rows through the encoder stored beside that cache and requires agreement within ONE ULP of the cache dtype on under 1% of elements. Run against both generations:

| seed | generation 1 | generation 2 |
|---:|---|---|
| 0 | **FAIL** - 51 ULP, 1280/131072 (0.98%) | PASS - 1 ULP, 1/131072 (0.00%) |
| 1 | **FAIL** - 81 ULP, 640/131072 (0.49%) | **FAIL** - 3 ULP, 3/131072 (0.00%) |
| 2 | **FAIL** - 59 ULP, 1231/131072 (0.94%) | PASS - 1 ULP, 2/131072 (0.00%) |

The generation-1 caches are not reproduced by their own stored encoder here, by **51-81 ULP on 0.49-0.98 % of elements**. That is the CPU-versus-GPU signature this spoke's task documented in advance (~53 ULP on ~1 % of elements), measured independently by running the programme's own gate. Generation 2 sits at 1-3 ULP on ~1e-5 of elements - the floor that batch-shape reordering allows, since the gate encodes 64 rows standalone while the cache was written in 8192-row blocks.

⛔ **What this does not say.** It does not invalidate any published generation-1 number: those numbers were computed on those features, and the features are what they are. It says generation 1 cannot be reproduced, extended or audited on this box - so it cannot be re-scored here, and a generation-1 number and a generation-2 number are measurements on different feature sets.

⭐ **What bounds the blast radius**, and it should be read beside the table above rather than after it: in the generation-2 re-score, the three DETERMINISTIC arms (`clu`, `knn_proto`, `store_removed`) move by **exactly 0.0000**. The regeneration is invisible to anything that is a deterministic function of the features. On its own the ULP table reads as alarming; the 0.0000 is what makes it precise.

⭐ **A consequence for the programme, not just this spoke.** A re-score on generation 1 was never possible: that gate would have refused all three seeds. Historical runs proceeded because the gate only fires ENFORCED when a cache is EXTENDED, and a full-scope cache with no misses never triggers it. The check existed, was correct, and was structurally unable to see the thing it was built to catch.

**Generation 1 and generation 2 disagree, with the predicted signature.** Same encoder parameter hash in every case, so this is the same encoder computing different bytes:

| seed | elements compared | differing | share | max abs diff |
|---:|---:|---:|---:|---:|
| 0 | 983,040,000 | 5,004,082 | 0.509% | 0.00390625 |
| 1 | 19,021,824 | 90,684 | 0.477% | 0.001953125 |
| 2 | 19,021,824 | 116,435 | 0.612% | 0.00390625 |

The max deviations are one float16 ULP at those magnitudes, on about half a percent of elements. That is the few-ULP / ~1 %-of-elements signature of a cache that was not all produced on one device. It does not invalidate any generation-1 number; it says those numbers belong to a different cache and cannot be quoted beside a generation-2 one.

### The rival set on GENERATION 2 (3 seeds: 0, 1, 2)

Judge = the frozen external classifier. Probe = its companion, on the same renders. `in-corner` is balanced accuracy on the held-out corner against its matched-seen control; chance is 0.5. `distinct` is distinct factor tuples the judge reads on B, out of 72 - an accuracy is never printed without it.

| rival | params | judge in-corner | probe in-corner | judge comp-B | distinct/72 | seeds |
|---|---:|---:|---:|---:|---:|---:|
| prototype table (incumbent) | 118,784 | 0.9165 ± 0.0306 | 0.8817 | 0.8650 | 56.0 | 3 |
| LACE (selection repaired) | 12,464,319 (95.1x) | 0.6155 ± 0.0261 | 0.6340 | 0.3048 | 71.7 | 3 |
| product-of-experts EBM | 3,556,665 (27.1x) | 0.5500 ± 0.0094 | 0.5139 | 0.1968 | 72.0 | 3 |
| MLP (one-hots) | 117,878 | 0.8983 ± 0.0497 | 0.8919 | 0.8943 | 46.0 | 3 |
| continuous-input MLP | 119,525 | 0.6377 ± 0.0743 | 0.6030 | 0.7184 | 45.0 | 3 |
| transformer (6 tokens, 2 layers) | 119,858 | 0.8463 ± 0.0201 | 0.8464 | 0.8526 | 48.3 | 3 |
| LSTM (6 tokens) | 119,125 | 0.8946 ± 0.0406 | 0.8961 | 0.8565 | 48.7 | 3 |

### Rivals at a second budget, and the request-dropout variants

Judge = the frozen external classifier. Probe = its companion, on the same renders. `in-corner` is balanced accuracy on the held-out corner against its matched-seen control; chance is 0.5. `distinct` is distinct factor tuples the judge reads on B, out of 72 - an accuracy is never printed without it.

These are kept out of the matched table above so they cannot be mistaken for it.

`mlp_180k` is at **179,968** parameters, 1.51x the matched target, because no arm on the read hub can be parameter-matched to this rival set: the pinned store alone is 131,072 parameters (524,288 B / 4), 10.3 % above the table's 118,784. That is a property of the programme's byte pin, not a choice any hub made. It answers whether a GENERIC LEARNER improves with 1.51x the parameters; it does not license 'parameter count does not matter' in general, because a table and an address book are different objects.

The `_dropout` variants are the SAME architectures at the SAME matched counts, trained with half their requests partially specified. They exist for rows 1, 2, 5 and 6, where an unrequested factor is genuinely `nan`. Rows 3 and 4 ask full six-factor requests, so the primary (full-only) variants above are the right rivals there — and the stronger ones: dropout training costs the MLP about 32 % on full-request validation MSE.

| rival | params | judge in-corner | probe in-corner | judge comp-B | distinct/72 | seeds |
|---|---:|---:|---:|---:|---:|---:|
| MLP at 179,968 params (the read hub's arm budget) | 178,952 | 0.9298 ± 0.0475 | 0.9620 | 0.9614 | 58.7 | 3 |
| MLP, request-dropout variant | 117,878 | 0.8800 ± 0.0640 | 0.8800 | 0.8326 | 40.3 | 3 |
| MLP at 179,968 params, request-dropout variant | 178,952 | 0.9229 ± 0.0662 | 0.9286 | 0.9367 | 51.7 | 3 |
| continuous-input MLP, request-dropout variant | 119,525 | 0.7108 ± 0.0324 | 0.6983 | 0.6073 | 30.0 | 3 |
| transformer, request-dropout variant | 119,858 | 0.8558 ± 0.0802 | 0.9017 | 0.8434 | 53.3 | 3 |
| LSTM, request-dropout variant | 119,125 | 0.8770 ± 0.0475 | 0.9043 | 0.8279 | 41.7 | 3 |

### Per-factor judge accuracy on B (mean over seeds)

The aggregate is the last thing to read. On this corner the prototype table reads every factor except `scale` at or near 1.000, and `scale` near chance — which is the whole story about what a per-value mean can and cannot represent, and it is invisible in the aggregate.

| rival | `floor_hue` | `object_hue` | `orientation` | `scale` | `shape` | `wall_hue` | aggregate |
|---|---:|---:|---:|---:|---:|---:|---:|
| prototype table (incumbent) | 1.000 | 1.000 | 0.995 | 0.204 | 0.991 | 1.000 | 0.865 |
| LACE (selection repaired) | 0.310 | 0.273 | 0.231 | 0.324 | 0.352 | 0.338 | 0.305 |
| product-of-experts EBM | 0.171 | 0.208 | 0.167 | 0.306 | 0.199 | 0.130 | 0.197 |
| MLP (one-hots) | 1.000 | 1.000 | 0.644 | 0.741 | 0.981 | 1.000 | 0.894 |
| continuous-input MLP | 1.000 | 0.963 | 0.292 | 0.616 | 0.440 | 1.000 | 0.718 |
| transformer (6 tokens, 2 layers) | 1.000 | 1.000 | 0.755 | 0.454 | 0.907 | 1.000 | 0.853 |
| LSTM (6 tokens) | 1.000 | 1.000 | 0.801 | 0.421 | 0.917 | 1.000 | 0.856 |

Read a rival by the factors it carries, not by its mean. A rival that wins an aggregate while losing a factor another arm carries is not better at the task; it is better at a different mixture of it.

### LACE: the historical selection and the repaired one, both printed

`GenCellConfig.lace_selection_source` was NOT edited. The historical leg is still run from the pinned value; the repair re-runs the SAME 36-trial selection (same grid, same short schedule, same criterion) on the STRATIFIED rows the arm is actually scored on, in this spoke's own module.

| seed | published (gen 1) | historical selection | repaired selection | delta | historical regressed? | repaired regressed? | repaired config | repaired whitening-exact? |
|---:|---:|---:|---:|---:|:--:|:--:|---|:--:|
| 0 | 0.7077 | 0.6313 | 0.6231 | -0.0082 | True | True | `sigma=0.1 both ledoit_wolf` | False |
| 1 | 0.6732 | 0.6353 | 0.6278 | -0.0075 | False | False | `sigma=0.03 both 1e-3` | False |
| 2 | 0.6582 | 0.6511 | 0.6511 | 0.0000 | False | False | `sigma=0.03 both ledoit_wolf` | False |

The last column is not decoration. R27.3's exactness gate asks whether the SELECTED shrinkage leaves `p0(u) = N(0, I)` near-exact; only `none` clears the pinned 0.10 tolerance on this cache. A repaired selection that lands on `ledoit_wolf` is better on the validation criterion AND outside that licence, so "repaired" does not automatically mean "better licensed" and the flag must be read with the number.

Statistic: probe-companion in-corner balanced accuracy on the primary sampler, which is the statistic the harness's own regression check uses. The published column is the generation-1 reference constant from `GenCellConfig.lace_ode_probe_reference_per_seed` and is shown for orientation ONLY — it is a generation-1 number and must not be quoted beside a generation-2 one.

### The product-of-experts EBM, at its OWN budget, run twice

| seed | params | deployed bytes | byte multiple vs the 524,288 B store | run 1 judge in-corner | run 2 judge in-corner | spread |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 3,556,665 | 14,226,660 | 27.1x | 0.5417 | 0.5701 | 0.0201 |
| 1 | 3,556,665 | 14,226,660 | 27.1x | 0.5482 | 0.5595 | 0.0080 |
| 2 | 3,556,665 | 14,226,660 | 27.1x | 0.5603 | 0.5703 | 0.0071 |

Two independent runs, each with its own expert fit and its own sampler keys, because a 72-item corner read from a stochastic sampler cannot carry a confident claim from one draw. The byte multiple belongs beside every number this arm produces: it is not held to the store's budget and its numbers are not comparable in level to an arm that is.

### Did the PoE-EBM win outright?

The PoE-EBM does not win outright: the best judge in-corner mean is table at 0.9165, against the PoE-EBM's 0.5500. Reported either way.

### Which rival can enter which row

A dash below is a structural fact about the rival, decided in advance and footnoted. It is never a win for us.

| rival | rows 3-4 (FULL requests) | row 2 (off-grid value) | partial requests (query sets) | rows 5-7 (value hold-out) | callable |
|---|:--:|:--:|:--:|:--:|:--:|
| table | yes | yes | yes | yes | yes |
| cont_mlp | yes | yes | — | yes | yes |
| mlp | yes | — | — | yes | yes |
| transformer | yes | — | — | yes | yes |
| lstm | yes | — | — | yes | yes |
| mlp_180k | yes | — | — | yes | yes |
| *_dropout (every learner) | yes | — | yes | yes | yes |
| lace_repaired | yes | — | yes | yes | — |
| poe_ebm | yes | — | — | yes | — |

**Population caveat on rows 2 and 5-7.** Every rival here is fitted on the Montero combination pool, which contains every factor value. On a value-holdout row it saw the value the arm under test never did - the fitted table has 3,157 rows at `object_hue` 0.5 against a value-holdout arm's 0. Eligible, but NOT like-for-like: report with a population footnote, or refit on your own pool. Check with `load_rival(seed, name).like_for_like({factor: [values]})`.

Footnotes for every dash:

- **cont_mlp / partial** — PRIMARY variant, full requests only; load cont_mlp_dropout for a partial request
- **mlp / offgrid** — closed vocabulary: this rival's input is a class index over the realised grid, so a value between or outside the grid points cannot be expressed at all. Snapping it to the nearest grid value would score the rival on a request it never received, so it is refused instead. (one-hot over the grid values)
- **mlp / partial** — the PRIMARY variant is trained on fully specified requests only, which is what rows 3 and 4 ask. For a partial request load mlp_dropout, which is the same architecture at the same matched count.
- **transformer / offgrid** — closed vocabulary: this rival's input is a class index over the realised grid, so a value between or outside the grid points cannot be expressed at all. Snapping it to the nearest grid value would score the rival on a request it never received, so it is refused instead. (token over the grid values)
- **transformer / partial** — PRIMARY variant, full requests only; load transformer_dropout for a partial request
- **lstm / offgrid** — closed vocabulary: this rival's input is a class index over the realised grid, so a value between or outside the grid points cannot be expressed at all. Snapping it to the nearest grid value would score the rival on a request it never received, so it is refused instead. (token over the grid values)
- **lstm / partial** — PRIMARY variant, full requests only; load lstm_dropout for a partial request
- **mlp_180k / offgrid** — closed vocabulary: this rival's input is a class index over the realised grid, so a value between or outside the grid points cannot be expressed at all. Snapping it to the nearest grid value would score the rival on a request it never received, so it is refused instead. (one-hot over the grid values)
- **mlp_180k / partial** — PRIMARY variant, full requests only; load mlp_180k_dropout for a partial request
- ***_dropout (every learner) / offgrid** — same as the primary variant of that family: only cont_mlp_dropout accepts an off-grid value
- **lace_repaired / offgrid** — LACE's categorical head has a closed vocabulary over the realised grid and cannot express an off-grid value (chlu.core.gen_request.DenseRequests). THE METHOD'S limit, not this spoke's.
- **lace_repaired / callable** — this arm's output is a stochastic draw, not a function of the request: two calls on the same request return different features by construction. Ask this spoke to score your corners.
- **poe_ebm / offgrid** — the per-factor experts are categorical over the realised grid. THE METHOD'S limit, not this spoke's.
- **poe_ebm / partial** — the experts are composed by summing one energy per factor over a FULL assignment; an absent factor has no expert term and the product of experts is undefined for it
- **poe_ebm / callable** — this arm's output is a stochastic draw, not a function of the request: two calls on the same request return different features by construction. Ask this spoke to score your corners.

### The deletion comparator: retrain from scratch without the value

**Seed 0 cost line.** deleting object_hue=0.5 costs 3.08s and a refit of 29,099 rows by the gold standard, against an O(D) copy of one prototype row and 0 refitted rows by mean-substitution (below timer resolution, so no ratio is quoted); on the forget set the judge reads the deleted factor at 0.0 (gold) vs 0.0 (edit), and on the retain set 1.0 vs 1.0

**Seed 1 cost line.** deleting object_hue=0.5 costs 3.75s and a refit of 29,067 rows by the gold standard, against an O(D) copy of one prototype row and 0 refitted rows by mean-substitution (below timer resolution, so no ratio is quoted); on the forget set the judge reads the deleted factor at 0.0196 (gold) vs 0.0588 (edit), and on the retain set 1.0 vs 1.0

**Seed 2 cost line.** deleting object_hue=0.5 costs 3.20s and a refit of 29,011 rows by the gold standard, against an O(D) copy of one prototype row and 0 refitted rows by mean-substitution (below timer resolution, so no ratio is quoted); on the forget set the judge reads the deleted factor at 0.0 (gold) vs 0.0 (edit), and on the retain set 1.0 vs 1.0

SCOPE OF THE DELETION COMPARATOR. The value is held out of the ARM'S FIT POOL only, not from the encoder, the decoder or the judge, which are frozen and were fitted on a split containing every hue. 'Never trained on' means the arm never fitted a row carrying that value. The decoder can still render it and the judge can still recognise it, which is what makes the comparison scorable at all.

| seed | method | wall clock | rows refitted | judge on the deleted factor, FORGET set | same, RETAIN set |
|---:|---|---:|---:|---:|---:|
| 0 | cont_mlp_retrain_from_scratch | 75.572s | 29,099 | 0.0000 | 0.2294 |
| 0 | lstm_retrain_from_scratch | 91.949s | 29,099 | 0.0000 | 0.9957 |
| 0 | mlp_180k_retrain_from_scratch | 68.298s | 29,099 | 0.0000 | 1.0000 |
| 0 | mlp_retrain_from_scratch | 69.547s | 29,099 | 0.0000 | 1.0000 |
| 0 | table_edit_mean_substitution | 0.000s | 0 | 0.0000 | 1.0000 |
| 0 | table_full_pool | 18.744s | 32,256 | 1.0000 | 1.0000 |
| 0 | table_retrain_from_scratch | 3.081s | 29,099 | 0.0000 | 1.0000 |
| 0 | transformer_retrain_from_scratch | 178.227s | 29,099 | 0.0000 | 0.9978 |
| 1 | cont_mlp_retrain_from_scratch | 77.821s | 29,067 | 0.0000 | 0.1800 |
| 1 | lstm_retrain_from_scratch | 88.801s | 29,067 | 0.0000 | 0.9957 |
| 1 | mlp_180k_retrain_from_scratch | 75.195s | 29,067 | 0.0000 | 1.0000 |
| 1 | mlp_retrain_from_scratch | 69.634s | 29,067 | 0.0000 | 1.0000 |
| 1 | table_edit_mean_substitution | 0.000s | 0 | 0.0588 | 1.0000 |
| 1 | table_full_pool | 19.450s | 32,256 | 1.0000 | 1.0000 |
| 1 | table_retrain_from_scratch | 3.754s | 29,067 | 0.0196 | 1.0000 |
| 1 | transformer_retrain_from_scratch | 185.623s | 29,067 | 0.0392 | 1.0000 |
| 2 | cont_mlp_retrain_from_scratch | 54.109s | 29,011 | 0.0000 | 0.2343 |
| 2 | lstm_retrain_from_scratch | 76.248s | 29,011 | 0.0000 | 0.9935 |
| 2 | mlp_180k_retrain_from_scratch | 59.953s | 29,011 | 0.0000 | 1.0000 |
| 2 | mlp_retrain_from_scratch | 60.976s | 29,011 | 0.0000 | 1.0000 |
| 2 | table_edit_mean_substitution | 0.000s | 0 | 0.0000 | 1.0000 |
| 2 | table_full_pool | 16.781s | 32,256 | 1.0000 | 1.0000 |
| 2 | table_retrain_from_scratch | 3.198s | 29,011 | 0.0000 | 1.0000 |
| 2 | transformer_retrain_from_scratch | 140.611s | 29,011 | 0.0000 | 0.9935 |

Both sets are needed or neither number means anything: performance on the FORGET set should fall, and performance on the RETAIN set should not. A method that forgets the value by getting worse everywhere has not deleted anything, it has been damaged.

### Every existing arm, re-scored on generation 2

generation 1 and generation 2 are separate columns. The delta says whether the instrument change is visible; it is never a result and the two columns are never combined.

| arm | gen-1 mean | gen-1 seed sd | gen-2 mean | gen-2 seed sd | max abs delta | moved more than seed spread? |
|---|---:|---:|---:|---:|---:|:--:|
| clu | 0.6759 | 0.0132 | 0.6734 | 0.0176 | 0.0000 | False |
| dynread | 0.5577 | 0.0372 | NOT RUN | NOT RUN | NOT RUN | None |
| dynread/A_request_off | 0.5000 | 0.0000 | NOT RUN | NOT RUN | NOT RUN | None |
| dynread/B | 0.5742 | 0.1047 | NOT RUN | NOT RUN | NOT RUN | None |
| dynread/B_request_off | 0.5000 | 0.0000 | NOT RUN | NOT RUN | NOT RUN | None |
| ebm_compose | 0.5148 | 0.0171 | 0.5215 | 0.0206 | 0.0069 | False |
| knn_proto | 0.8817 | 0.0028 | 0.8803 | 0.0022 | 0.0000 | False |
| lace | 0.6399 | 0.0025 | 0.6412 | 0.0140 | 0.0139 | False |
| lace/ld_corrected | 0.6161 | 0.0151 | 0.6327 | 0.0037 | 0.0177 | True |
| lace/ld_published | 0.5547 | 0.0757 | 0.6436 | 0.1648 | 0.1181 | False |
| store_removed | 0.5000 | 0.0000 | 0.5000 | 0.0000 | 0.0000 | False |

**Verdict: AT LEAST ONE ARM MOVES BY MORE THAN ITS SEED SPREAD - report this to every hub before it scores against generation 2**

### External corroboration, and a caution about the table above

Everything in this section is **hub/read's measurement, on hub/read's population, with hub/read's judge**, reported here with attribution. None of it is mixed into the tables above and no number here is this spoke's.

**The rival ordering is population-dependent, severely.** Same two rivals, two corner sets:

| | prototype table | `mlp_180k` | gap |
|---|---:|---:|---:|
| this spoke's corner B, this spoke's judge | 0.9165 | 0.9298 | 0.013 |
| hub/read's 49 n-way corners, its judge | 0.1550 | 0.7426 | 0.588 |

The prototype table is nearly competitive here and collapses there. Corner B is a single Montero cell - 72 rows, one `shape` value, retained values throughout - and an additive per-value-mean composition handles it well. hub/read's corners are n-way conjunctions, which the additive rule cannot represent.

⛔ **So do not read the table above as 'the prototype table is a strong rival'.** It is a strong rival *on the easiest compositional population in the programme*. The levels across the two rows are not on a common scale (different populations, judges and request sets); it is the change in ORDERING that is the finding.

**The matched-budget result replicates independently, and is larger elsewhere.** The same 1.0x -> 1.51x parameter step on the same architecture:

- this spoke, corner B: +0.033 balanced accuracy, +12.7 distinct factor tuples
- hub/read, 49 n-way corners: 37.245 -> 53.469 novel tuples per corner, i.e. +16.2

Two populations, two instruments, same direction. On hub/read's scoreboard that step exceeds its dial's entire measured headroom (+9.08 tuples/corner from its deployed read to an oracle address), and it reports its hub falsified by `mlp_180k` on both of its target rows.

⛔ **`mlp_180k` carries TWO budget framings and both must be stated, or it misleads in one direction or the other.** It is **1.51x** the prototype table's 118,784 - the budget this spoke's own rival set is matched to - and simultaneously **0.994x** hub/read's arm at 179,968, i.e. matched to that arm within -0.57%. Against the table it is the over-budget rival; against that hub's arm it is THE MATCHED RIVAL. This spoke initially quoted the first while reasoning about the second and offered hub/read a softening it correctly declined.

⚠ The caveat this spoke attached to `mlp_180k` survives and is repeated: it tests whether a GENERIC LEARNER improves with 1.51x the parameters, measured against the TABLE's budget. It does not license 'parameter count beats mechanism', and 1.51x of an MLP is not 1.51x of a store. hub/read's own ablation against its incumbent read is +1.143 [-2.898, +5.061], i.e. nothing - so on those corners BOTH addressed reads sit below the matched generic learner, which is a statement about the addressed-read FAMILY rather than about any one variant.

⭐ **Why the second budget existed at all.** `mlp_180k` was built on a side request, not from this spoke's plan. Without it, hub/read would have compared against the parameter-matched `mlp` alone, measured +3.592 [-0.816, +8.103], and read that as a win - while the pinned store's 131,072 parameters put its arm permanently outside the matched budget. A rival set matched to ONE budget is a comparison at one budget; the second budget is what made the first interpretable.

## Readiness

`RIVALS-READY.json` status: **ready**, seeds [0, 1, 2], rivals present: ['cont_mlp', 'lace_repaired', 'lstm', 'mlp', 'poe_ebm', 'table', 'transformer'].
