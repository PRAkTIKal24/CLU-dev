# bprime-c6 — third-party store attribution: what a table structurally CANNOT do, measured

**Campaign 2, wave C2W4. Agent:** experiment-engineer (or results-analyst — this is measurement on an
existing rig, not a build). **Small.**
**Worktree MANDATORY** — you hold **worktree slot 2 of 3**. Base local `main` @ **`d4f56c8`**.
Branch `agent/experiment-engineer/bprime-c6`.
Worktree: `git worktree add ../CHLU-c6 -b agent/experiment-engineer/bprime-c6`.
Charter **ADDENDUM 3 §A15 task 2**, implementing **§A14.1** (the SCOPED Route-3 kill) and the theorist's
code request **C6** (`bprime-theory.md` §6, priority **P0**).

**Read first:** `.claude/AGENT_PROTOCOL.md`; `.claude/advisor-head-c2-charter.md` **ADDENDUM 3 in full
(§A12 · §A13 the claim architecture v3 · §A14.1 the scoped kill · §A15)** and **ADDENDUM 2 §A9.4/§A9.5
(the bar and the kill you are working under)**; `.claude/outputs/bprime-theory.md` **§T5.4 (the
proposition), §T5.5 (the exchange-rate table that tells you where to run), §6 C6/C8, §9 items 2/4/5
(what is NOT derived)**; `.claude/outputs/route3-stage1-plus-2x2.md` (**the rig you are extending and
the attribution machinery that already exists**); the **`2026-07-31 (later still)` `[C2W3]` §10 entry**
in `.claude/handover_context.md` (**Decision Point 2 and the "adjudication I will not make alone"
block**).

⭐ **REGISTRY STATUS — CURRENT.** The *"quote outputs/§10 only"* restriction is **LIFTED**;
`claims_matrix.md` §0 holds the dated never-quote list. **Three live errata override the registries**:
the byte law is **24/28** (corrected law `[A(D+2)+d]/(d+m)`, floor rises to **2.40×** at `n_spec=1`) ·
**MUNKEY is an ICLR-2026 workshop (oral), workshop name QUARANTINED**, not ICML 2026 · **monitor #6's
"27 post-repair" is PROVISIONAL** until `harness-debt` lands the `+eps_acq` half this wave.

---

## ⭐ DIAL DECLARATION (protocol §7, C2 form) — echo before your first result
- **Dial / pillar:** **none — this is protocol evidence for TIER i (the audit paper).** You are not
  reviving Route 3 and you are not claiming a dividend. §A9.5's kill **stands** and is **scoped by
  §A14.1 to inference-read claims**; you are measuring the one coupling that a per-slot table gives
  **exactly 0** for by construction, so that the audit paper can say *"here is what a table
  structurally cannot do, measured"* — which is an audit column, not a revival.
- ⛔ **You do not reopen §A9.5.** A large third-party coupling does **not** unlock `route3-stage2`.
  `route3-stage2.md` stays parked. The **only** pre-registered revival trigger for inference-read claims
  is **OQ-2**, and it is **wave-boundary only** (see §4 — you declare it NOT-RUN).
- **Laundering control:** the launder is native to the rig you inherit (per-slot matched-bytes table
  launder, settle-deleted launder, launch-noise floor, blank-store control). **A table's third-party
  Δ is exactly 0 by construction — report it as the measured 0 it is, and say "by construction" every
  time**, because otherwise it reads as a win rather than as the definition of the contrast.
- **Falsifies:** §4. **Does NOT falsify:** a small coupling at the designed gate (that is *predicted*,
  `7.0e-4`) · the CLU still losing on the answer channel (already measured; not your question).

---

## 0. Why this task exists, in five sentences
C2W3's §A9.5 kill fired on the *answer channel*: a per-slot matched-bytes table reproduced the slotted
read at **37/38 slots** and **18/18 of the slots that cleared the §A9.4 unlock bar**, with **zero
read-beats**. The theorist then **named and proved** the one coupling that table cannot express — **a
change in slot content caused by a stored item the query did *not* select** (Prop T5.4: for fixed `x`,
`∂ŷ/∂(any non-selected row) = 0` **exactly**, because the table's store-dependence factors through
finitely many *selected* rows; the CLU's `q̈ = −M⁻¹∇V` has `∇V` summing over **all** wells). **That
measurement was available all wave, at zero build cost, and was never made.** Its magnitude decides
whether the contrast is real or cosmetic, and the magnitude scales as **`exp(−½(d/s)²)`** — a **1089×
span** between the designed admission gate and the rig the gym actually runs. **You make the
measurement.**

## 1. Deliverables

### D1 — ⭐ The third-party attribution probe (the task's spine)
On the **existing** rig (`chlu/eval/attribution.py` + `exp_route3_attribution.py`, both landed in C2W3):
delete a stored item **the query did not select**, and measure the change in each slot's content.
- **Per-slot Δ vs `t`** — the curve, not the endpoint. **Report `p_t` as well as `q_t`** (theorist C7:
  momentum peaks at **1.88 σ_q at `t ≈ 10`**, ~4× the position channel).
- **Slot grid in `t ∈ [1, 240]`** (C7: beyond `t ≈ 240` the within-basin variance is `<1e-3 σ_q`, and by
  `t = 1200` it is `4.7e-9 σ_q` — slots out there measure nothing).
- **≥3 seeds. Multi-seed before any paper number, no exceptions.**
- **The table's Δ is computed, not assumed** — run it, report the **exact 0**, and caption it *"0 by
  construction (Prop T5.4)"*.

### D2 — ⭐ The curve vs `d/s`, across the three declared regimes
This is what makes D1 a *result* rather than a data point. Measure the coupling at, at minimum:

| regime | `d/s` | predicted coupling (theorist T5.5) | why it matters |
|---|---|---|---|
| **the rig C2W3 actually runs** (`d_safe_override = 0.58`, `s ≈ 0.30`) | **1.9–2.0** | **0.69–0.80** | O(1); the merger band, `λ_min` collapsing |
| **the soft-certificate region** (deficit ≈33 % of `sep`) | **2.9** | **0.111** | the principled middle — SC-1…SC-7 landed in C2W3, default-off |
| **the designed admission gate** (`d_safe = 4.4 s`) | **4.4** | **7.0e-4** | below every noise floor the harness has |

⭐ **The design-deciding sentence you are testing, and it is the audit paper's:** the only coupling a
per-slot table cannot express is **exponentially suppressed by our own admission gate**, so *"a store
that is organised well enough to be safe is organised well enough to be a table"* — measure whether that
is true, quantitatively, with the curve.
⚠ **Two bounds the theorist declared and you must carry, not silently discharge:**
- **T5.5's prediction is a two-well toy at `p₀ = 0`** (`bprime-theory.md` §9 item 4). Your rig is the
  learned multi-atom store — **agreement is a finding, disagreement is also a finding, and neither is a
  bug until you show it is.**
- ⚠ **`s` for a learned multi-atom well is an UNSOLVED modelling question** (§9 item 2) — T5.5 uses
  `atom_init_width` as a proxy and calls it *"a bracket, not a measurement"*. ⭐ **Declare which `s` you
  used, and report the curve against BOTH your chosen `s` and the raw geometric separation**, so the
  x-axis convention cannot silently move the conclusion. If you can measure a fitted well width on the
  shipped learned `V_θ`, that is a bonus finding worth more than the curve itself (it gates the transfer
  of T4.1–T4.3 and is carried from `doctrine-repairs` OQ-C).
- ⚠ The theorist's own ballistic prediction **missed its registered 20 % bar** (`|Δq(t)|` matched
  `(t²/2M)‖∇V_j(q₀)‖` at only **0.61–0.73× at `t=10`**, 0.17–0.34× at `t=20`, the residual being
  free-fall toward the item's own well) while the **scaling** held (**2.60 decades** over `r/s = 2→4`
  against a registered 2–4). **Pre-register your own prefactor expectation with the free-fall correction
  included** and score it honestly.

### D3 — OQ-A: re-locate `B = 0.33` with the corrected inradius (owed since C2W2, ≈10 min)
The soft certificate's violation budget **`B = 0.33`'s outer edge was located with the broken `sep/2`
ruler** (`bprime-theory.md` T4.3, §9 item 5). **Re-locate it with R1's corrected inradius and report the
corrected `B`.**
- ⛔ **This is a MEASUREMENT, not a re-parametrisation.** You report what `B` should be. ⛔ **Do NOT
  change any shipped default** and do NOT edit `chlu/core/monitors.py` (owned by `harness-debt` this
  wave). If the corrected `B` differs materially, that is a Hub item and a `doc-curator` erratum, and
  you say so — you do not land it.
- ⚠ It matters to D2 because the soft-certificate row's `d/s ≈ 2.9` is *defined* by `B = 0.33`.
- ⚠ **`sep/2` as a certified inradius is on the never-quote list.** Do not reproduce it, even to
  contrast.

### D4 — the CLI hook (reconciliation 6, assigned to you)
`chlu/cli/experiment_cmd.py` had **no declared owner in C2W3**, so `exp-route3-attribution` has no CLI
hook and runs only via `python -m`. The engineer correctly stopped and reported. **You own that file
this wave: land the hook**, matching the surrounding registration idiom exactly, with a test.
⛔ **`chlu/config.py` is standing read-only to all C2 engineers.** If the hook needs a config change,
STOP and report.

## 2. What you must NOT do
- ⛔ **No stage 2.** No slotted write objective, no `allocate`, no γ-as-selector. `route3-stage2.md` is
  parked and stays parked. §A9.4's `unlock = true` is on the record so the met condition is not lost;
  §A9.5's **override** is the reason it did not proceed, and that reason is **not** a dividend verdict.
- ⛔ **No inference-read claim.** §A14.1: the per-item slotted read **is table-expressible and that axis
  is closed.** Route 3's *training-time* machinery stays live as **tier-ii organizer tooling** — that is
  C2W5's, not yours, and you do not build toward it.
- ⛔ **Do not re-measure** the stage-1 attribution curve, the §A9.5 table launder result, the 2×2, the
  byte-floor theorem, or anything in `PREREG-Bprime.md` §7.
- ⛔ **Do not run OQ-2** (a learned `p₀` steering the path toward non-selected wells). It is the **only**
  pre-registered revival trigger for inference-read claims and it is **wave-boundary only** (§A14.1) —
  running it inside a wave would launder a revival past its own gate. **Declare it NOT-RUN, with this
  reason.** ⚠ You may *note* in Open Questions what your measured curve implies about it; that is
  evidence for a future Head decision, not a result.

## 3. PREREG (`.claude/outputs/bprime-c6/PREREG.md`) — **before ANY measured run**
Mandatory: your acceptance criterion is a measured **curve and exponent**. Commit to:
- the predicted coupling at each of the three `d/s` regimes (**start from T5.5's 0.69–0.80 / 0.111 /
  7.0e-4 and state whether you expect the learned multi-atom store to agree, and why**);
- the predicted **`exp(−½(d/s)²)` scaling** — how many decades over your measured `d/s` span;
- the predicted **prefactor with the free-fall correction**, since the theorist's bare-ballistic version
  missed by 0.61–0.73×;
- the predicted **q-vs-p ratio** at the slot grid's peak (C7 says momentum ≈4× position at `t ≈ 10`);
- the table's Δ (**exactly 0**) — trivially, but register it, because it is the contrast's definition;
- **how each was derived.** Guessed numbers are not pre-registrations.

## 4. Falsifiers (declare them, then adjudicate them)
- ⛔ **"The coupling is not measurable on the shipped rig."** If the third-party Δ at `d/s ≈ 1.9–2.0`
  does not clear the launch-noise floor beyond 2 SE on ≥3 seeds, then the one coupling a table cannot
  express is **below our instrument even where theory says it is O(1)** ⇒ ⭐ **that is a finding for the
  audit paper, not a failure of the task**, and it makes the §A14.1 scoped-kill *stronger*: the escape
  exists in principle and is unmeasurable in practice at this weight class. **Write it plainly.**
- ⛔ **"The `exp(−½(d/s)²)` law does not hold on a learned store."** If the measured decades over your
  `d/s` span disagree with the prediction by more than your registered tolerance, the **exchange rate
  that prices the whole Route-3/soft-certificate trade is wrong**, and every statement built on the
  1089× span (including charter §A14.1's own pricing) needs a reconciliation entry. **Report it in your
  first 10 lines.**
- ⛔ **"The `s` convention decides the answer."** If your two x-axis conventions (chosen `s` vs raw
  separation) give qualitatively different conclusions, then `d/s` is not a well-defined ruler on a
  learned store and **T5.5's table is a bracket only** — say so, and flag every downstream site.
- **Does NOT falsify:** a coupling smaller than predicted at the designed gate (that is the prediction)
  · a noisy curve at large `t` (C7 predicts the signal dies past ~240 steps) · disagreement with a
  two-well toy at `p₀ = 0` (declared out of domain).

## 5. Compute & environment
Hours, not days — this is measurement on a rig that exists. ⚠ **JAX cold-start ~20 min**; keep the
session warm, `--quick` for smoke runs, reuse the main venv (`PYTHONPATH=<worktree>
/Users/user/Desktop/CHLU/.venv/bin/python …`, cwd in the worktree); `uv sync --frozen` only if you must,
and report the resolved JAX version. ⚠ **N94 epoch discipline.** ⚠ Use `git -C <worktree>` always.
⚠ You are **worktree 2 of 3** and `bprime-rivals` (the wave's spine, worktree 1) is running concurrently
— the thermal cap is real (w26 incident, load 575/8 cores). **Do not launch long parallel sweeps.**

## 6. File ownership (11 branches / 4 waves / 0 conflicts — keep it)
**Yours, exclusively this wave:** `chlu/eval/attribution.py` ·
`chlu/experiments/exp_route3_attribution.py` · `chlu/core/soft_certificate.py` (**read/measure freely;
⛔ change no shipped default — D3 is a measurement**) · `chlu/cli/experiment_cmd.py` ·
`tests/test_attribution.py`, `tests/test_route3_attribution.py`, `tests/test_soft_certificate.py`.

**Read-only to you:** `chlu/core/monitors.py`, `chlu/experiments/memory_gym.py`,
`tests/test_{monitors,memory_gym}.py` (**`harness-debt`**) · `chlu/eval/rivals/`,
`chlu/eval/dividend.py`, `chlu/experiments/exp_bprime_rivals.py` (**`bprime-rivals`**) ·
`chlu/core/blocks.py`, `chlu/data/` (**`cluformer-pilot`, if it spawns**) ·
`chlu/config.py` (**standing read-only to all C2 engineers**) · `chlu/eval/race.py` (**frozen schema**) ·
`chlu/core/{clu_system,admission,memory_potentials,placement,integrators,controller,psi_readout}.py`.
⛔ **If your work requires editing a file you do not own, STOP and report to the Hub.**

## 7. ⛔ Never-quote (the ones that will bite you; full dated list in `claims_matrix.md` §0)
**`sep/2`** as a certified inradius · **`λ_min > 0`** as certifying a nonempty basin (measured **0.000**
at `λ_min = +0.910`) · the ridge saddle **`λ_min = −0.5946`** as multi-seed (seed 0; 3-seed mean
**+0.177 ± 0.469**) · any **`AttentionPsi`** trajectory number (it **raises** now; `q0_only` 0.35–0.45 vs
a bar of 0.19) · **monitor #6's counts** ("58 trips" without *"pre-repair"*; "27 post-repair" is
**PROVISIONAL**) · **"verified to 1e-9 in all 28 cells"** (24/28) · **MUNKEY as "ICML 2026"** ·
**`ε` as the manifold-payload lifetime dial ∝ 1/ε** without the `2α` ceiling (`τ_max = Γ/2α`) ·
**`k*`** without *"of `∂q_N/∂θ`, and only where the fixed-point sensitivity dominates the transient"* ·
**"Prop D1 is violated"** (retired) · any C2W3-or-later cell as a **byte-matched** dividend (min ratio
**17.11×**) · ⭐ **any tier-ii/tier-iii claim** (§A13; "CLU-former" is a placeholder that must never
reach a draft).

## 8. Output — `.claude/outputs/bprime-c6.md`, protocol §5 format
- ⭐ **The `d/s` curve in the first screen**, with the three declared regimes marked, both x-axis
  conventions, q **and** p channels, ≥3 seeds, error bars, and the table's **0 by construction** on it;
- the **PREREG scorecard** (registered · measured · verdict, including refutations — *a refutation that
  produces a corrected law is the best outcome this task has*);
- **D3's corrected `B`**, stated as a measurement with its ruler named, plus whether it moves the
  soft-certificate row of D2;
- the **flag-provenance table** (commit, seeds, every non-default flag — mandatory);
- **admissible-cell coverage**, first-class;
- ⭐ **one paragraph written for the audit paper**: *what a per-slot table structurally cannot do, its
  measured magnitude, and where our own admission gate puts it.* `bprime-draft` will lift this.
- **your reconciliation list in the FIRST 10 LINES** if you produce one;
- ⛔ **declared NOT-RUNs, never reported as nulls** — including **OQ-2**, with its reason.
- **Git footprint**: branch, commits, files, tests. Before removing the worktree, **verify from the MAIN
  repo that your branch ref shows your commits** (`git -C /Users/user/Desktop/CHLU log --oneline
  main..agent/experiment-engineer/bprime-c6`). ⛔ **Never push `origin`**; do not push at all.
