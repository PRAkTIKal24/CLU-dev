# harness-debt — the byte-law bug, monitor #6's missing half, and ONE published re-score diff

**Campaign 2, wave C2W4. Agent:** experiment-engineer. **Small — hours, not days.**
**Worktree MANDATORY** — you hold **worktree slot 3 of 3**. Base local `main` @ **`d4f56c8`**.
Branch `agent/experiment-engineer/harness-debt`.
Worktree: `git worktree add ../CHLU-debt -b agent/experiment-engineer/harness-debt`.
Charter **ADDENDUM 3 §A15 task 4**, implementing **§A14.4 verbatim** (*"Harness debt, one owner"*).

**Read first:** `.claude/AGENT_PROTOCOL.md`; `.claude/advisor-head-c2-charter.md` **ADDENDUM 3 (§A12 ·
§A14.4 · §A15)**; `.claude/outputs/bprime-theory.md` **§6 code requests C1, C2, C8 and T1 (the corrected
law and its verification)**; `.claude/outputs/bprime-fb4-gate.md` **PART D / R4 (why monitor #6's fix was
deliberately NOT landed)**; the **`2026-07-31 (later still)` `[C2W3]` §10 entry** in
`.claude/handover_context.md` — **reconciliations 1, 2, 4 are yours.**

⭐ **REGISTRY STATUS — CURRENT.** *"Quote outputs/§10 only"* is **LIFTED**; `claims_matrix.md` §0 has the
dated never-quote list. **Two of the three live errata are literally your deliverables** (the byte law
and monitor #6's provisional count); the third is **MUNKEY = ICLR-2026 workshop (oral), name
QUARANTINED**, not ICML 2026.

---

## ⭐ DIAL DECLARATION (protocol §7, C2 form) — echo before your first result
- **Dial / pillar:** **none — this is instrument repair.** You are not measuring a dividend and you have
  no claim. Two of your three items are **defects in numbers the audit paper prints**, and the third is
  **a published count that is currently PROVISIONAL**. This is the highest-value-per-hour task in the
  wave precisely because it is not a result.
- **Laundering control:** n/a — but ⛔ **every number you change is a number someone already published**,
  so the control that matters is the **before/after diff** (§3). No silent re-scores.
- **Falsifies:** §5. **Does NOT falsify:** the byte-floor theorem (it is an accounting identity, exact
  28/28 in rational arithmetic — only the *shipped formula* and one published *sentence* are wrong) ·
  monitor #6's C2W2 dead-band repair (that half is correct and stays).

---

## 0. Why this task exists, in three sentences
`chlu/experiments/memory_gym.py` **had no owner in C2W3** and carries a **live P0 formula bug in the
byte law that B′ prints**; monitor #6's C2W2 repair is **half-landed**, so its published "27
post-repair" count is **PROVISIONAL**; and the C2W3 engineer **correctly declined** to land the two-line
fix because doing so would silently re-score another agent's published number mid-wave. **All three are
one owner's job by Head ruling §A14.4, and that owner is you.** ⭐ The standard is the **C2W2 D4
pattern**: change the predicate, re-score the *recorded* readings offline, and **publish the diff** —
never re-run the store and never let a count move without a table saying which cells moved and why.

---

## 1. D1 — ⛔ Fix `byte_ratio_law` (theorist C1, **P0** — the audit paper prints this)

**The defect.** `chlu/experiments/memory_gym.py::byte_ratio_law` divides by the **store** dimension
`D = addr_dim + payload_dim + n_spectator` where the launder's row is **`(d + m)` floats**:

```python
dim = float(addr_dim + payload_dim + n_spectator)      # ⛔ D, but used as the launder row width
return float(atoms_per_item) * (dim + 2.0) / dim + float(addr_dim) / dim
```

**The corrected law** — exact in **all 28** C2W1 cells in integer/rational arithmetic, **0 ulp**:

> **`ratio = [A·(D + 2) + d] / (d + m)`**, with `A = atoms_per_item`, `D = d + m + n_spectator`,
> `d = addr_dim`, `m = payload_dim`.

**Consequences, all in the conservative direction (⭐ no published claim was inflated — the store costs
*more* relative to the table than we published):**
- **Wrong only when `n_spectator > 0`.** `n_spectator = 0` ⇒ `D = d + m` ⇒ the two forms coincide.
- The four `manifold` (`n_spectator = 1`) cells measure **52.00×** against a published **43.33×**
  (**+8.667, 20 %**).
- The **floor RISES**: the `floor_note` prints **2.00×** where the true floor at `n_spec = 1` is
  **2.40×** (gauss). The `n_spec = 0` floor **2.20×** is unchanged, and the **measured min 2.28×** is
  unchanged. Shell floors are **2.40 / 2.60×** with a `×9/8` surcharge on the atom term
  (`+1/(D+2) = 12.5 %`).
- ⭐ **`PREREG-Bprime.md` §7's reuse licence STANDS** and **`bprime-rivals` does NOT re-measure the
  theorem.** ⛔ **`PREREG-Bprime.md` is NOT edited** — a pre-registration whose text is revised after the
  fact stops being one. The correction is a dated **erratum** (`doc-curator-c2w3-sync` files it).

**Land:** the corrected formula, the corrected `floor_note`, and a docstring that states the corrected
law, the 24/28-vs-28/28 history, and the conservative direction.

### D2 — the missing regression test (theorist C2, **P0**)
⭐ **The bug is invisible in every test we have, and that is the actual lesson.**
`test_byte_ratio_law_matches_the_measured_ledger` passes `n_spectator = 0` **literally**, and
`test_cell_reports_all_three_harness_native_controls_and_a_byte_ledger` is parametrised over
`aggregate`/`recency` only — **no test exercises a spectator dim.** Add `manifold` (or an explicit
`n_spectator = 1`) to **both**, and add a direct assertion that the corrected law reproduces the
**52.00×** manifold cell and the **2.40×** floor.
⚠ **Also assert the identity structurally, as integers, not as a float ratio**: `full == 4·[N_at(D+2) +
K·d]` and `launder == 4·K·(d+m)`. ⛔ **The `byte_account` half of that (theorist C3) belongs to
`bprime-rivals` — `chlu/eval/dividend.py` is THEIR file this wave.** You assert it on the gym side only;
if you need the `dividend.py` change, **STOP and report** so the Hub routes it.

---

## 2. D3 — ⛔ Monitor #6's missing `+eps_acq` half (reconciliation 4, **P0**)

**What landed in C2W2** (`chlu/core/monitors.py`, `ObjectiveDivergenceMonitor` /
`objective_divergence_predicate`): a **dead-band on the loss slope** —
`slope_loss < -eps` with `eps = eps_rel · scale`, `scale = max|loss|` over the window,
`eps_rel = 1e-9`, and `eps_rel = 0.0` restoring the pre-repair predicate **exactly**.
That half is **correct and stays.**

**What did NOT land: the same dead-band on the acquisition leg.** The predicate still reads
`slope_acq <= 0.0`, so a converged run whose acquisition slope is **flat to round-off on the positive
side** (e.g. `+5.9e-17`) **fails the test and does not trip** — a **false negative**. The theorist
predicted **two recovered false negatives** from this half, and they **never materialised** because the
half was never landed.

**Land:** `slope_acq <= +eps_acq`, with `eps_acq` built the same relative way as `eps`
(`eps_acq_rel · scale_acq`, `scale_acq = max|acq|` over the window), **defaulting to the same
`1e-9` relative band**, and **`eps_acq_rel = 0.0` restoring the current predicate exactly** — the same
offline-re-score affordance the loss half already has. Keep `objective_divergence_predicate` a free
function so recorded readings stay re-scorable offline.
⚠ **Symmetry is the point:** one leg with a dead-band and one without is not a repair, it is a
half-repair that changes the trip count in one direction only.

### D4 — ⭐ ONE published re-score diff, and it is the deliverable's real content
**The C2W2 D4 pattern, verbatim: re-score the RECORDED readings offline at the new `eps_acq`. Do NOT
re-run the store.** Publish a **trip-state diff table**:
- **every cell**, with pre-repair (`eps_rel = eps_acq_rel = 0`) · loss-half-only (the shipped C2W3
  state, which produced the published "27") · **both halves** (your state);
- **which cells changed, in which direction, and why** — a `no-trip → TRIP` from the acq half is a
  **recovered false negative** and is the outcome the theorist predicted; a `TRIP → no-trip` from this
  half would be a **surprise and must be explained, not reported**;
- ⭐ **the corrected post-repair count, stated once, with its diff beside it.** Until this lands, *"27
  post-repair"* carries a **PROVISIONAL** qualifier and is a never-quote-adjacent hazard for the paper.
  ⛔ **If the two predicted recoveries do not materialise, say so plainly** — the prediction failing is
  a finding about the recorded readings, not a reason to tune `eps_acq` until it fires. **Do not tune
  the band to produce the predicted result.** That is the whole rule.
- ⛔ **Every other monitor must be bit-identical across the diff.** The C2W3 precedent for this is
  `bprime-fb4-gate`'s monitor-#3 diff: *15 cells, every `full`/`launder`/`dividend` bit-identical,
  exactly two monitors changing, ZERO trips added.* Match that standard.

### D5 — monitor #2's domain guard (theorist C8, **P1** — land only if D1–D4 are clean and time remains)
`r_i := 0` and **INAPPLICABLE** wherever `λ_min,i ≤ 0`; use the corrected inradius **inside its domain
only**; add the **capture-radius** leg. ⚠ Note the **silent `max(λ, 1e-9)` clamp** at
`chlu/core/clu_system.py:906` — ⛔ **`clu_system.py` is NOT yours; if the guard requires touching it,
STOP and report.** ⚠ `bprime-c6`'s **D3 (OQ-A)** is re-locating `B = 0.33` with the corrected inradius
**as a measurement** this wave — **coordinate through the Hub, do not land a `B` change**, and do not
consume their number before it is reported.
⚠ **`sep/2` is never a certified inradius** (never-quote list). **`λ_min > 0` does NOT certify a
nonempty basin** — measured **0.000** at `λ_min = +0.910`.

---

## 3. PREREG (`.claude/outputs/harness-debt/PREREG.md`) — **before you re-score anything**
Short but mandatory — D4's acceptance criterion is a **changed published count**. Commit to:
- the **corrected manifold ratio (52.00×)** and the **corrected floor (2.40×)**, and that `n_spec = 0`
  cells are **bit-identical** (24/28 unchanged);
- **how many cells you expect monitor #6's acq half to flip, and in which direction** — the theorist
  predicted **two recovered false negatives**; register that, then measure it;
- that **no other monitor moves**;
- **`eps_acq_rel = 0` reproduces the current predicate bit-for-bit** (this is a blocking check, not a
  hope).

## 4. Verification bar (blocking — do not report `done` without these)
1. `uv run pytest -q` **green on the full suite** (C2W3 closed at **1061 passed**; report your number and
   **account for every delta** — new tests added, none regressed). ⚠ JAX cold-start ~20 min; budget it.
2. `uv run ruff check chlu/ tests/` — **All checks passed.**
3. ⭐ **Bit-identity gates, both directions:** `n_spectator = 0` byte cells **unchanged**; `eps_acq_rel =
   0` monitor readings **unchanged**. A repair that cannot be turned off is not auditable.
4. The **trip-state diff table** exists and every changed cell has a stated reason.

## 5. Falsifiers
- ⛔ **"The corrected law does not reproduce the measured ledger."** If `[A(D+2)+d]/(d+m)` does not match
  the measured `manifold` ratio to 0 ulp in rational arithmetic, then the theorem's *statement* (not
  just its published verification) is wrong, **B′'s reuse licence is in doubt, and `bprime-rivals` may
  have to re-measure.** ⛔ **Report in your first 10 lines, same day** — it is the single highest-impact
  thing you could find.
- ⛔ **"The acq dead-band changes more than the predicted cells, or changes them the wrong way."** A
  `TRIP → no-trip` flip from a dead-band that only *widens* the trip condition is a contradiction ⇒
  something else moved ⇒ **stop, do not publish the count, report.**
- **Does NOT falsify:** the two predicted recoveries failing to materialise (report it) · the corrected
  floor differing from a registry sentence (that is the erratum, and the curator is filing it).

## 6. Compute & environment
Hours. ⚠ **JAX cold-start ~20 min** — keep the session warm, `--quick` for smoke runs, reuse the main
venv (`PYTHONPATH=<worktree> /Users/user/Desktop/CHLU/.venv/bin/python …`, cwd in the worktree); `uv sync
--frozen` only if you must, and report the resolved JAX version in your flag-provenance table.
⚠ You are **worktree 3 of 3**, concurrent with `bprime-rivals` (the wave's spine) and `bprime-c6`. The
thermal cap is real (w26: load 575/8 cores). **No long parallel sweeps.** ⚠ `git -C <worktree>` always.

## 7. File ownership (11 branches / 4 waves / 0 conflicts — keep it)
**Yours, exclusively this wave:** `chlu/experiments/memory_gym.py` · `chlu/core/monitors.py` ·
`tests/test_memory_gym.py` · `tests/test_monitors.py`.

**Read-only to you:** `chlu/eval/dividend.py`, `chlu/eval/rivals/`,
`chlu/experiments/exp_bprime_rivals.py` (**`bprime-rivals`** — theorist C3 is theirs) ·
`chlu/eval/attribution.py`, `chlu/experiments/exp_route3_attribution.py`,
`chlu/core/soft_certificate.py`, `chlu/cli/experiment_cmd.py` (**`bprime-c6`**) ·
`chlu/core/blocks.py`, `chlu/data/` (**`cluformer-pilot`, if it spawns**) ·
`chlu/config.py` (**standing read-only to all C2 engineers**) · `chlu/eval/race.py` (**frozen schema**) ·
`chlu/core/{clu_system,admission,memory_potentials,placement,integrators,controller,psi_readout}.py` ·
`chlu/eval/fb4_gate.py`.
⛔ **If your work requires editing a file you do not own, STOP and report to the Hub.**

## 8. ⛔ Never-quote (the ones that will bite you; full dated list in `claims_matrix.md` §0)
**"verified to 1e-9 in all 28 cells"** (it is **24/28**; use the corrected law) · **monitor #6's "58
trips"** without *"pre-repair"* (**the artefact count is 31 of 58, not 29**) and **"27 post-repair"**
without **PROVISIONAL** until *your* diff lands · **`sep/2`** as a certified inradius · **`λ_min > 0`**
as certifying a nonempty basin (**0.000** at `λ_min = +0.910`) · **MUNKEY as "ICML 2026"** or with a
named workshop · **"Prop D1 is violated"** (retired) · any C2W3-or-later cell as a **byte-matched**
dividend (min ratio **17.11×**) · any **`AttentionPsi`** trajectory number (it **raises** now) ·
⭐ **any tier-ii/tier-iii claim** (§A13; "CLU-former" is a placeholder that must never reach a draft).

## 9. Output — `.claude/outputs/harness-debt.md`, protocol §5 format
- ⭐ **The two diff tables in the first screen**: (a) the byte-ledger before/after across all 28 cells,
  with the **24 unchanged** and the **4 changed** marked and the +8.667 shown; (b) the monitor-#6
  **trip-state diff** at three predicate settings, with every changed cell reasoned;
- ⭐ **the corrected post-repair monitor-#6 count, stated once, with its diff beside it** — this is the
  sentence that lifts the PROVISIONAL qualifier;
- the **PREREG scorecard** (registered · measured · verdict), including the predicted-two-recoveries
  outcome **whichever way it went**;
- the **flag-provenance table** (commit, seeds, every non-default flag — mandatory);
- the full-suite pass count with **every delta accounted for**, and the ruff result;
- ⭐ **an explicit list of every DOCUMENT that now needs updating** because of what you landed —
  `doc-curator-c2w3-sync` consumes this directly, so name files and sections;
- **your reconciliation list in the FIRST 10 LINES** if you produce one;
- ⛔ **declared NOT-RUNs, never reported as nulls** (D5 if you skip it, and say why).
- **Git footprint**: branch, commits, files, tests. Before removing the worktree, **verify from the MAIN
  repo that your branch ref shows your commits** (`git -C /Users/user/Desktop/CHLU log --oneline
  main..agent/experiment-engineer/harness-debt`). ⛔ **Never push `origin`**; do not push at all.
