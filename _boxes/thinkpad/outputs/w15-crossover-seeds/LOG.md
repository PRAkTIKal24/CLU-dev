# w15 spoke/crossover-seeds — LOG

Started: see `START_UTC.txt`. Base `base/iter8 = 148f3b7`, branch `w15/crossover-seeds`,
worktree `/c/chlu/worktrees/CHLU-w15-crossover-seeds`, outputs
`/c/chlu/outputs/w15-crossover-seeds/`.

**Scored runs spent by this spoke: ZERO.** Everything below is `val`, the free
selection condition (charter A.2). Every number is `SMOKE_NOT_A_CLAIM`.

---

## 1. Environment

`source /c/chlu/env_w3.sh` alone does NOT give a working interpreter on this box, and
the reason is recorded rather than patched silently:

- `env_w3.sh` builds `PYTHONPATH="$CHLU_ROOT:$_H/.venv/Lib/site-packages"` with a POSIX
  `:` separator. Native Windows python splits `PYTHONPATH` on `;`, so MSYS hands it a
  single glued entry `C:\chlu\...\CHLU-w15-crossover-seeds:\c\Users\...\site-packages`
  and the wave-2 site-packages is not on the path at all —
  `ModuleNotFoundError: No module named 'equinox'`.
- Fix, applied in `/c/chlu/outputs/w15-crossover-seeds/env.sh` and in
  `scripts/w15_launch.sh`: re-export `PYTHONPATH` with `;` and Windows-form paths after
  sourcing. Nothing else about the environment is touched; no venv is built, no
  `uv sync` is run.

Assert passes:

```
OK C:\chlu\worktrees\CHLU-w15-crossover-seeds\chlu\__init__.py
jax 0.9.0   numpy 2.4.1   equinox 0.13.4    python 3.11.9
```

`CHLU_GEN_FEATURES_ROOT=C:/chlu/outputs/spoke-a-rivals/cache_generation2` is set in the
launcher and asserted in-process (`assert_generation2`) — without it the run silently
reads the frozen bridge, cache GENERATION 1, which w5-common 3.8 forbids for any number.

## 2. What w12 actually left behind (read-only, never edited)

From `/c/chlu/outputs/w12-single-design-crossover/CROSSOVER-val.json`:

| K | n_seeds in w12 | seeds |
|---|---|---|
| 5 | 3 | 0,1,2 |
| 8, 10, 12, 16, 20, 24, 28 | **1** | 0 |
| 32 | 3 | 0,1,2 |

So the seven interior K carried a single draw of (feature cache, partition, address-book
fit), and the four-part total — which depends on the realised well widths and on the
fitted address book, both seed-dependent — had no seed band at all. The 14 missing cells
are exactly this spoke's job.

Evidence that the total IS seed-dependent, already visible inside w12 at K=32:
seed 0 total 2,143,138 B (`logs/sweep_s0_A2.log`) vs seed 1 total 2,136,378 B
(`logs/sweep_s1_r2.log`) — the address book differs, 1,059,824 B vs 1,053,064 B.
A U-shape minimum located on one seed is therefore not yet a result.

## 3. Choices made (pre-authorised; recorded here, not escalated)

1. **The instrument is imported, not reimplemented.** `scripts/w15_crossover_seeds.py`
   imports `scripts/w12_val_sweep.py` and calls its `main()` verbatim, overriding only
   the module-level `OUT` path so rows land in this spoke's output dir. w12's files are
   not edited. This is the choice closest to an ICLR main-track result: a leg produced
   here is comparable to w12's seed-0 leg line by line, because it IS the same code.
2. **SWEEP stays (5, 8, 10, 12, 16, 20, 24, 28, 32).** The range is not extended.
3. **The figure is regenerated from scratch** (`scripts/w15_crossover_figure.py`) rather
   than w12's `w12_crossover_figure.py` being edited, and emits vector PDF **and** PNG
   with deterministic metadata so the rebuild is byte-identical.
4. **The four-part ledger is reported per K per seed**, mean and band. w12's report
   printed `per[ss[0]]["ledger_four_part"]` — seed 0's ledger only — which is exactly
   the thing that cannot stand for the U-shape claim.
5. **Lanes: 4, one BLAS thread each**, on a 4-core box. Oversubscribed BLAS is slower
   than serial BLAS and the address-book fit is the whole cost of a leg.

## 4. Launch

4 lanes launched 2026-09-22T21:31:05Z, PIDs in `PIDS.txt`:

| lane | seed | K | pid |
|---|---|---|---|
| laneA_s1_lowK | 1 | 8, 10, 12, 16 | 174637 |
| laneB_s1_hiK  | 1 | 20, 24, 28    | 174640 |
| laneC_s2_lowK | 2 | 8, 10, 12, 16 | 174643 |
| laneD_s2_hiK  | 2 | 20, 24, 28    | 174646 |

All four report `cache {'generation': 2, ...}` on their first line — the generation
assertion fired and passed.

## 5. THE U-SHAPE TEST, DECLARED BEFORE THE SEED-1 AND SEED-2 TOTALS EXIST

Written 2026-09-22T21:31:05Z. At this moment `val_sweep/` in this spoke's output dir contains no
`row.json` for any (K, seed) cell — the four lanes have been running for well under a
minute and the address-book fit alone takes 400–1200 s per leg. The seed-0 totals ARE
already known: they are w12's, read from
`/c/chlu/outputs/w12-single-design-crossover/CROSSOVER-val.json`, and they are what the
"minimum near K = 24" sentence was built on (seed 0: K=20 2,100,786 B, K=24 2,094,130 B,
K=28 2,126,890 B). The test below therefore decides the sentence on seeds 1 and 2, whose
numbers do not yet exist.

Let `T(K, s)` = the honest four-part total in bytes at well count K for seed s, and
`Kmin(s) = argmin_{K in SWEEP} T(K, s)`, SWEEP = (5, 8, 10, 12, 16, 20, 24, 28, 32).

**S1 — shape.** `T(., s)` is U-shaped for seed s iff it is strictly decreasing on
`K <= Kmin(s)` and strictly increasing on `K >= Kmin(s)`. Any interior violation is
reported as a violation, with the offending K printed.

**S2 — location.** "minimum near K = 24" holds for seed s iff
`Kmin(s) in {20, 24, 28}` — the swept grid point and its two neighbours on the ladder.
⛔ The ladder is not re-spaced to make a neighbourhood wider.

**S3 — the claim.** The sentence "the total is U-shaped in K with its minimum near
K = 24" may be claimed at n = 3 iff S1 and S2 hold for **all three** seeds.
If `{Kmin(0), Kmin(1), Kmin(2)}` is not contained in `{20, 24, 28}`, the minimum has
moved materially across seeds, the sentence is **REFUTED**, and the per-seed minima are
the result — not a failure.

**S4 — resolvability, reported beside S3 and never instead of it.** Even if S2 holds,
the minimum is only *resolvable* if the gap to each neighbour exceeds the seed band:
`mean_s T(20,s) - mean_s T(Kmin,s) > sd_s T(Kmin,s)` and likewise for K=28. If the
band swallows the gap, the claim becomes "the total is flat in K over 20–28" and that
is what gets said.

## 6. Running notes

- The address-book fit is the whole cost of a leg: 507–1,100 s against 12 s to build the
  store and ~250 s for the five 24-corner reads (arm + four controls). Legs land in
  `val_sweep/val_K<K>_seed<s>/row.json`.
- **The ledger is not always seed-dependent, and the reason is structural.** At K = 8
  `ward_balanced` gives nine equal 256-wide wells on both seeds, every well deploys
  rank 64, so atoms, covariance and address book are identical and both seeds print
  `2,959,602 B`. Where the widths differ by one or a well deploys a lower rank the two
  seeds separate — e.g. K = 20 seed 2 spans 63..64 per well (sum 1,279) and prints an
  address book of 694,264 B against seed 0's 694,784 B. So the seed band on the total is
  narrow at coarse K and opens at fine K, which is exactly the region the U-shape claim
  lives in.
- Composition, by contrast, is plainly seed-dependent: K = 8 gives 0.8837 (seed 0, w12),
  0.9230 (seed 1), 0.9045 (seed 2) — a spread of 0.039, about twice the 0.02 selection
  tolerance. w12's single-seed interior rows could not have shown this.
- Determinism harness `scripts/w15_rebuild_check.sh` written and passing on partial
  rows: report + figure rebuilt twice, all four artefacts byte-identical
  (`REBUILD-pass1.sha256` vs `REBUILD-pass2.sha256`). It is re-run on the complete set.

## 7. Result

All 14 legs landed. Every K in the sweep now carries n = 3 seeds {0, 1, 2}.
Last leg: `val_sweep/val_K28_seed1/row.json`, 2026-09-22T23:06:26Z (UTC; `ls --time-style` on this box prints LOCAL time, which is UTC+2 — the earlier reading of 01:06 was local). All four lanes exited
on their own; no kill of any kind was issued.

- **U-shape S3 verdict: CONFIRMED.** Per-seed minima of the honest four-part total are
  K = 24, 24, 24. S4 resolvability: seed band sd 7,060 B at K = 24 against neighbour gaps
  of 12,203 B (K = 20) and 30,506 B (K = 28) — both clear the band.
- **The composition story inverts against w12's seed-0-only interior rows.** Every
  `drop_from_K5` at n = 3 is <= 0; the largest drop anywhere in the sweep is 0.0000, at
  K = 5 itself. w12's "largest drop anywhere is 0.0093 at K = 10" was a seed-0 artefact.
- **K\* = 32 under the frozen rule, unchanged** — but the satisfying set is now every K in
  the sweep, and the seed sd on composition exceeds the 0.02 tolerance at 6 of the 9 K, so
  the rule cannot separate K = 32 from K = 24 or K = 28 at n = 3. Reported as a limit, not
  repaired: the rule was not moved and the tolerance was not widened.
- **One control does not collapse:** `landscape_shuffled_across_wells` at K = 20 seed 1,
  0.0139 on 1 of 24 corners (mean over corners 0.000579) against an arm of 0.9213. Under
  the strict gate K = 20 FAILS and no row may be claimed there. The shipped C.2 control
  `landscape_shuffled`, `landscape_removed` and `random_atoms` collapse to exactly 0 at
  every K and every seed. I checked the documented fixed-point weakness of the
  across-wells permutation and it does NOT explain this cell: K = 20 has a fixed point on
  seed 2 as well, and seed 2 collapsed to 0.0000.
- **All four controls collapse at K = 24**, the U-shape minimum, on all three seeds.

## 8. Choices made after the numbers landed (recorded, not escalated)

6. **The strict control gate was kept strict.** `collapses` means exactly zero on the
   worst of 24 corners on the worst of 3 seeds. A softer "collapses to within a numerical
   floor" gate would have passed K = 20, and picking it after seeing the number would be
   moving a bar. The strict verdict is reported, the residual is named with its K, seed,
   max and mean, and the shipped C.2 control's verdict is printed beside it.
8. **A second selection-resolvability table was added** (`CROSSOVER-3SEED.md` section 1b,
   `per_K.<K>.selection_resolvability` in the JSON). It puts the 0.02 tolerance beside the
   seed sd at every K. This is the ICLR-main-track choice: reporting K\* without it would
   let a reader believe three seeds separate K = 32 from K = 24, which they do not.
9. **The figure gained an inset** on K >= 16. The U lives in a 60 kB window against a
   4.1 MB first bar and is invisible at the stack's scale; the claim under test gets its
   own axes rather than a sentence asking the reader to trust an unreadable wiggle.

## 9. Determinism

`scripts/w15_rebuild_check.sh` builds all four acceptance artefacts twice from the same
`row.json` files and diffs the hashes: **BYTE-IDENTICAL ON REBUILD: PASS**. Hashes in
`ARTEFACTS.sha256`. The report writes no wall-clock field and sorts every mapping; the
figure pins PDF/PNG metadata (no creation date, no Software tag) and draws no clock.

