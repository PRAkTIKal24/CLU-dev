# c3-trackb-tripwire — the criterion-4 tripwires for CAMELS-US and N-CMAPSS (+ N-CMAPSS's missing crit-2 baselines)

**Campaign 3, wave 1. Agent:** results-analyst. **ZERO worktrees. ZERO branches. ⛔ ZERO tracked-code edits.**
Everything you write lives under `.claude/scratch/c3-trackb-tripwire/` and `.claude/outputs/c3-trackb-tripwire/`.
Writes `.claude/outputs/c3-trackb-tripwire.md`.
**Budget:** ≈ 1.5 days. **Compute: CPU-only, hours.** ⛔ No GPU, no CSF3 job, no training run.

> ⭐ **WHY YOU TAKE NO WORKTREE AND WRITE NO PRODUCTION CODE.** You are the **kill-condition for two
> venues we have not adopted.** Productionising a loader for a venue that may be rejected is exactly
> the waste the standing doctrine forbids (*build the kill-condition before the thing it can kill*).
> If a venue is adopted, its loader becomes a **separate engineer task**; if it dies here, we spent
> hours of CPU and no engineer-days. ⚠ **`c3-csf3-harness` (wt1) owns `chlu/data/**` and its
> registry — you must not touch it.** Your file ownership is `.claude/**` and nothing else.

**Binding documents, read first, in this order:**
1. `.claude/advisor-head-c3-charter.md` **§3 IN FULL** — the venue table and ⛔ **the admissibility
   rule: criterion 4 is MEASURED, not argued, before any venue is adopted.** You are that measurement.
2. `.claude/advisor-head-intervention.md` **§6** (the five criteria) and **§8** (prohibitions).
3. ⭐ `.claude/outputs/c3-benchmark-scout/trackB-scorecard.json` — rows 1 (CAMELS-US) and 2
   (N-CMAPSS). **Their `crit4_tripwire` blocks are your specification**; take `query_repr`,
   `key_repr`, `exemplar_baseline`, `state_byte_budget` and `dies_if` **from the JSON, never
   re-derived.** The sibling report `.claude/outputs/c3-benchmark-scout.md` §2 carries the prose.
4. ⭐⭐ `.claude/outputs/c2w10-benchmark-gate.md` **IN FULL** — the instrument you are re-using, its
   port, its unit-tested shims, and the two lessons in §5 below. Its SAM-kNN port lives under
   `.claude/outputs/c2w10-benchmark-gate/`.

---

## ⭐ DIAL DECLARATION (protocol §7) — echo before your first result

- **Dial / pillar:** **none — admissibility instrument.** ⛔ You produce **no** CLU number, **no**
  performance claim, and **no** venue adoption. You produce, per venue, a **measured criterion-4
  verdict** and (for N-CMAPSS) the **missing criterion-2 pairing**.
- **Laundering control:** ⭐ **you ARE the laundering control** (C2W10's own framing). The exemplar
  store at matched bytes is the launder, and this task measures how strong it is.
- **Falsifies venue adoption:** the exemplar store at the matched byte budget **meets or beats** the
  best strong learned reference under the venue's `dies_if` rule ⇒ the venue is **metric-native** and
  inadmissible as a primary claim venue.
- **Does NOT falsify anything about the CLU.** ⛔ **The exemplar store winning does not falsify the
  CLU — it falsifies the BENCHMARK's admissibility** (C2W10 §1 verbatim). Write that sentence in your
  report before your first number, so no reader mis-reads a fired tripwire as a CLU result.

## ⛔⛔ THE PRE-REGISTRATION IS MANDATORY AND COMES FIRST

Your acceptance criterion is a **measured comparison against a registered threshold** ⇒ protocol §5's
prereg rule binds. **Write `.claude/outputs/c3-trackb-tripwire/PREREG.md` BEFORE running any
harness**, containing: the `dies_if` rule **quoted from the scorecard**, your **point predictions**
for each arm (commit to numbers — a prediction that survives is evidence, one that fails is a
*finding*), the store-size ladder, and every protocol choice you had freedom over. ⛔ If a number is
computed before the prereg is written (e.g. a deterministic baseline with no free parameter),
**disclose it in the prereg** — C2W10 did exactly this and it is the standard.

## ⚠⚠ THE TWO C2W10 LESSONS — either one, missed, inverts your verdict

1. ⭐⭐ **ANTI-HOBBLING DECIDED THE LAST GATE.** Every exemplar arm runs **RAW *and* CAUSALLY
   STANDARDISED**, and you **consume the MAX**. On INSECTS, standardisation was worth **+4.0 to +6.3
   points** to the kNN arms, and **had C2W10 run raw-only, `criterion4_cleared` would have come out
   `true`** — i.e. the wrong answer. The classical store gets its **strongest admissible form**, or
   the tripwire is worthless. Causal means statistics from instances `< t` only.
2. ⭐⭐ **A SINGLE STORE SIZE CANNOT REPRESENT THE FAMILY.** On INSECTS, `kNN_S` scored **76.03 % at
   L = 500** and **59.75 % at L = 14,782** — *the CLU's own byte budget was the WORST arm*, 19 points
   below a store 30× smaller. **Storing more was actively harmful.** ⇒ Run the **full L ladder** from
   the scorecard (`{250, 500, 1000, 2000, 5000, at-budget}`) and **report the curve, then consume the
   ladder MAX** — ⛔ never a single L, and ⛔ never only the at-budget point.

## 1. Venue A — CAMELS-US (the registered PRIMARY candidate)

Run the scorecard's row-1 spec. Non-negotiable points from it:
- **Both query variants are mandatory** — the 365-day window (1,852 dims) **and** the 30-day
  (177 dims). Lesson 2 is why.
- **Both store arms** — **REGIONAL** (exemplars pooled across all 531 basins; the registered
  **primary**, because it matches how the LSTM baseline is actually trained: one model for all
  basins) and **LOCAL** (same-basin only). Report both.
- **Causal standardisation from training-period statistics only.** A per-basin z-score computed over
  the full record is a test-set leak and would silently strengthen the store.
- **The strong learned reference** must be the published LSTM rainfall-runoff result at the standard
  531-basin protocol and metric (NSE), cited to its primary source with the table named. ⛔ Do not
  hand-roll a weak "learned baseline" and declare the store beaten by it.
- ⭐ **State the structural asymmetry in your report, as the hypothesis under test:** the target is a
  function of an **unobserved accumulated state** (soil moisture, groundwater, snowpack), not of the
  forcing window — two identical 365-day windows with different antecedent storage produce different
  discharge, and the 27 static attributes only partially disambiguate. **If that non-identifiability
  is material, no metric over the observable window can be the ceiling.** ⛔ That is the reason the
  venue is worth running, **not** a prediction that it clears; measure it.

**⛔ HEAD-FACING BLOCKER, CHECK IT FIRST (10 minutes):** the scout could **not confirm CAMELS-US's
explicit licence string** — this is the single open blocker on the primary recommendation. Check it
before any download; if it is not unambiguously open for our use, **report BLOCKED on the mirroring
question and hand it up** — do not mirror data on an unconfirmed licence. ⚠ The scout's "15 GB /
130 GB" size figure is **secondary-sourced (hyper.ai) — ⛔ do not quote it**; its DERIVED estimate for
forcing+discharge alone is ≈206 MB (≈0.6 GB with all three forcing products). Verify by measurement.

## 2. Venue B — N-CMAPSS DS02 (the ranked FALLBACK) + its missing criterion-2 pairing

Run the scorecard's row-2 spec, **plus one thing CAMELS does not need:**

⭐ **N-CMAPSS's criterion-2 pairing is DECLARED NOT PUBLISHED — you supply it.** The venue cannot be
scored on criterion 2 without the trivial-baseline numbers, so compute the **mandatory companion
rows**: **mean-RUL** and **affine-in-cycle-index**. ⚠ **The affine-in-cycle-index baseline is
expected to be dangerously strong by construction** — RUL is *defined* piecewise-linear in cycle
index — and that is precisely the measurement: if trivial baselines sit at the frontier, the venue
**fails criterion 2** and you say so plainly. Minutes of CPU; it is the cheapest decisive number on
the board.

⚠ **This venue carries the board's highest criterion-4 prior, and the reason is published:**
**similarity-based RUL estimation** — matching a query trajectory against a library of stored
run-to-failure trajectories — is a classical, competitive method on the CMAPSS family. That is a
nearest-neighbour store by another name. Run it as an exemplar arm (DTW **or** Euclidean over
health-index curves at the same byte budget), alongside the k-NN arm.

**⛔⛔ THE CAFE EMBARGO IS ABSOLUTE.** No CAFE-derived C-MAPSS number is externally comparable (the
banked label-bug report: test labels under-estimate by exactly `RUL_unit`). ⛔ Nothing from our CAFE
prognostics work may be quoted for or against this venue **in either direction**, and any number of
ours flowing through CAFE-derived preprocessing **inherits the embargo**. Go back to the original
NASA PCoE files. State in your report which files you used and their checksums.

**⛔ Classic C-MAPSS is NOT a claim venue** (Advisor ruling, 2026-08-13) and is out of scope here. Do
not run it, do not price it, do not compare to it.

## 3. The byte budget

Use the scorecard's **1,966,080 B** for the store ladder's at-budget point, and state the
exemplar-count arithmetic as it does (N-CMAPSS: 24-dim float32 frame = 96 B; 20-frame window = 1,920 B
+ 4 B target = 1,924 B/exemplar ⇒ 1,021 exemplars at budget; a 1-frame representation buys 19,660).
⚠ **A ≈2 MB Track-A budget was ruled on 2026-08-13 and its last digit is being confirmed** — it does
**not** retroactively change this ladder. If the confirmed figure differs, the at-budget point moves;
**report the ladder so that re-pricing is a lookup, not a re-run.** That is the point of the ladder.

## 4. Kill / stop conditions

- CAMELS licence not confirmable → **BLOCKED on mirroring**, hand up, proceed with N-CMAPSS.
- A venue's strong learned reference cannot be sourced to a primary table → ⛔ **do not substitute a
  weaker one**; report the gap and score what you can.
- ⛔ If a tripwire **fires**, STOP at the measurement. **Do not** propose a replacement venue, do not
  re-spec the task to make it clear, and do not soften the reading. A fired tripwire is a **finding**
  and it returns to the Head + Advisor, ⛔ **not to a spoke** — it would be the **SEVENTH** criterion-4
  confirmation, and the standing rule is that we do not shop for one.

## 5. Acceptance criterion (one line)

`PREREG.md` filed before any harness ran; for **each** venue a measured criterion-4 verdict against
the scorecard's own `dies_if` rule, computed from the **ladder max** over **raw-and-standardised**
arms with the strong reference cited to a primary table; **N-CMAPSS's mean-RUL and
affine-in-cycle-index criterion-2 rows supplied**; the CAMELS licence string resolved or escalated;
every number carrying its flag-provenance table; and a one-line-per-venue **ADOPT-RECOMMENDED /
DIES** statement that never conflates a fired tripwire with a CLU result.

## 6. Report format

Protocol §5, dial declaration first, the *"a fired tripwire falsifies the benchmark, not the CLU"*
sentence before the first number, and the prereg scored honestly at the end (predictions that
survived vs failed — a failed prediction is a finding, not an embarrassment).
