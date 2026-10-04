# hub/composition — RESULTS LEDGER (charter §A.6: "a results ledger of every scored run")

## SCORED RUNS AGAINST THE HELD-OUT SETS A / B / C
| # | date | PREREG | config | result |
|---|---|---|---|---|
| — | — | — | — | ⭐ **NONE. 0 of 3 spent.** |

⛔ **This hub performed ZERO scored runs.** Two independent reasons, both standing:
1. ⛔ **LAUNCH-COMMON §1c:** *"Do not perform your first scored run until a human tells you how many of the
   three remain for your target."* ⭐ **QUESTION 1 was asked at boot and is still unanswered.** This box
   cannot see the Mac's count.
2. ⛔ Even with an answer, a scored run here would be **seed 0 only** — seeds 1 and 2 are cache-blocked
   (32,395 / 32,437 of 33,039 rows missing; producing them means re-encoding on hardware 51 ULP away from
   the Mac cache, which LAUNCH-COMMON §4 forbids). ⭐ `PREREG-hub-composition-1.md` §6 therefore
   **recommends the run be spent on the Mac at 3 seeds, not here.**
3. ⛔ And `exp_gen_cell` refuses to start here at all: the re-encode gate's `will_extend` scoping is not
   wired at `run_seed`'s call site, so even seed 0 — shortfall provably **0** — is refused. ⛔ I did not
   edit the frozen scoring harness to get past it (**QUESTION 2**, unanswered).

## PRE-REGISTRATIONS FILED
| file | status |
|---|---|
| `PREREG-hub-composition-1.md` | ⭐ filed before any run; **not executed**; recommends execution on the Mac at 3 seeds |

## NON-SCORED WORK — all on `val` / `MS` / frozen artefacts. ⛔ B NEVER ENTERED.
| # | what | artefact | touches B? |
|---|---|---|---|
| N1 | cache-coverage measurement, per seed | `probes/cache_coverage.json` | no |
| N2 | valid∧novel instrument + validation vs harness (24/24 rows) | `instrument/valid_novel.py`, `probes/frozen_novelty.json` | no (reads frozen artefacts) |
| N3 | frozen Mac cell table, re-derived from disk | `probes/frozen_table.json` | no (reads frozen artefacts) |
| N4 | novelty reference at three strictnesses | `probes/novelty_reference.json` | labels only |
| N5 | `x0` / `clu` / ceiling on `val` + `MS` | `mech/x0_probe.json` | no |
| N6 | oracle-selection control | `mech/oracle.json` | no |
| N7 | store round-trip bound | `mech/roundtrip.json` | no |
| N8 | partition sweep, untrained | `mech/partition_init/*.json` | no |
| N9 | dreaming temperature grid (pre-registered grid, β ∈ {0,1}) | `mech/temp_grid_A_val_s1000/*.json` | no |
| N10 | decoder noise-tolerance + PCA capacity curves | `mech/capacity.json` | no |
| N11 | whitening remedy (**predicted PASS, measured FAIL**) | `mech/whiten_init/*.json` | no |
| N12 | residual-structure control ladder | `mech/residual.json` | no |
| N13 | byte requirement + bytes↔rel-err power law | `mech/requirement.json` | no |
| N14 | bootstrap CIs + arm-seed spread | `mech/robustness.json` | no |
| N15 | partition sweep, trained (partial) | `mech/partition_trained/*.json` | no |
| N16 | independent replication (separate agent, own code) | `replication/REPLICATION.md` | no |

⭐ Every one of these is selection / mechanism work, which charter §A.1–2 place under *"iterate freely on
MECHANISM"* and *"selection on `val` only"*. ⛔ Not one of them is a scored run and ⛔ not one reads B.
