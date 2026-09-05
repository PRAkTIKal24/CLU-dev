# V1 — the two floors §4.1 has never met (X-1 matched-bytes · X-2 matched-compute)

---
## ⛔⛔ STATUS: BANKED — NOT SCOPED, NEVER LAUNCHED. ⛔ DO NOT RUN THIS.
**Head ruling, 2026-08-26:** V1 ships **without** these runs. §4.1's escalatability result is framed as an **empirical proof of concept**, with the matched-compute / matched-bytes comparison stated in-paper as **future work** — *"maybe for the ICLR long."*

⇒ This file is a **pre-scoped bank**, kept because the scoping is done and the ICLR long can lift it whole. It is **not** an instruction to anyone. ⛔ A future Advisor or spoke reading `.claude/tasks/` must not treat it as in-flight.

⚠ **If it is ever revived, re-verify §3's and §4's code pointers first** — they were checked on disk 2026-08-26 and *a provenance fact has a shelf life* (Add.82: a stale generator pointer in a task file would have silently reverted a completed fix while reporting success).
---

**Scoped by the V1 Shorts Advisor at the Head's direction, 2026-08-26; banked the same day by Head ruling.** Basis: `.claude/outputs/v1-claims-currency-audit.md` §4-Q2 and §10b (X-1, X-2), Advisor-verified.

**Agent:** `experiment-engineer` · **one declared worktree** · **Output:** `.claude/outputs/v1-escalatability-floors.md`
⛔ **This spoke edits NO paper file.** Not `papers/v1-short/**`, not `NIPSsubmission/v1-ttcl/**`. It runs an experiment and writes one report.

---

## 1. The question, and the honest prior

V1's §4.1 claims the CLU-specific asset is being an **escalatable** memory — *"extra compute buys accuracy; one-shot memories get no allocation payoff"* (**CM-2**). **That claim has never met a floor.** Its entire comparison set is **the authors' own full-budget CLU** and **a modern Hopfield**. There has never been a matched-bytes exemplar store or a matched-compute feedforward line anywhere in the paper.

Charter §4.1, the genuine-win bar: *win-by-construction results are **supplementary only**; a short's primary claim must survive competitive baselines doing fairly well.*

⛔⛔ **THE PRIOR IS THAT THE TRIPWIRE FIRES, AND YOU SHOULD EXPECT TO LOSE.** Criterion 4 (intervention §6) has **six confirmations** and exactly one measured survivor (CAMELS-US, C3 charter Add.1). The mechanism to expect here: **an exemplar store at matched bytes on MQAR vocab-256 at kv ≤ 32 answers in one pass at or near 1.000** — so it has **no graded-compute dial at all**, which makes the CLU's escalatability *trivially* unique and *simultaneously* worthless, because there is nothing to ration when the baseline is already at ceiling in one step. That is the same geometry that produced **N90** (*"flat at 0.99–1.00 — it cannot rise **because it is already saturated**"*) and **N95** (*"no headroom versus the correct ML floor"*).

⭐ **A fired tripwire is a REPORTABLE OUTCOME, not a failed pass.** Both branches are usable and the Head has pre-committed to both:
- **Fires** ⇒ CM-2's escalatability is supplementary-only, and §4.1 re-pitches to the **certified rationing apparatus** (LTT 30/30 valid, measured risk 0.030 at ε=0.05) — which is what the paper already calls it.
- **Does not fire** ⇒ §4.1 is V1's first genuine win.

⛔ **Do not tune toward a non-firing result.** A pass that reports "no fire" after adjusting the budget, the arms or the metric is worth less than nothing.

---

## 2. ⛔ PREREG FIRST — the gate that makes either outcome usable

**C3 charter §5 invariant:** *prereg with numeric falsifiers BEFORE each job ladder · kill-conditions built first · ≥3 seeds before any paper number.*

⛔ **Commit `PREREG.md` BEFORE any implementation commit, so prereg-first is git-provable** (the Add.24 precedent: `bprime-dividend-family-2`'s registration commit provably preceded its implementation commit and every artifact). A result whose prereg does not precede it **cannot enter the paper** under our own rules — that is the whole reason this section is first.

`PREREG.md` states, before you write experiment code:
1. **The exact byte budget** and how it was derived (§4), with `dtype_bytes` **declared, never inferred**.
2. **Numeric falsifiers** — the `margin_rel` threshold is **−0.02** (the C2W10 convention). State what value of `margin_rel` you would call a fire, and what you would call a survival, *before* you see one.
3. **Kill conditions** — what result would make you stop and report rather than continue.
4. **Point predictions** for each arm's accuracy at matched compute and at matched bytes. ⭐ Registering a prediction you then miss is a *calibration record*, not an embarrassment (N314's precedent scored 4 resolved / 8 refuted and filed all of them).

---

## 3. The arms (four), the cells, the seeds

**Cells:** MQAR vocab-256, **kv ∈ {16, 24, 32}** — §4.1's own three levels, no others.
**Seeds:** **5** (base 42), matching §4.1's own n. ⛔ Never fewer; ≥3 is the floor and 5 is what the paper already quotes.

| arm | what it is | status |
|---|---|---|
| **(i) calibrated CLU gate** | §4.1 as it stands — staged relaxation, Platt head, LTT exit threshold | ✅ **exists**: `chlu/experiments/exp_v1_calibration.py` (its docstring: *"kv in {16, 24, 32} x >= 5 seeds"*) |
| **(ii) matched-bytes exemplar store** ⭐ **X-1** | a 1-shot exemplar / kNN store over the **same key–value pairs**, sized to the **same total state bytes** as the CLU store | ⚠ **byte accounting exists but needs adaptation** — see §4 |
| **(iii) matched-compute feedforward floor** ⭐ **X-2** | the N90/N95 baseline transplanted onto these cells: a feedforward net given the **same compute** as the gate's ladder | ⚠ a floor of this class lives in `chlu/experiments/exp_retry_compute.py` (N90's rig). ⛔ **Verify it at the moment of use — do not trust this pointer** (a provenance fact has a shelf life; that lesson cost this program a near-silent revert at Add.82) |
| **(iv) modern Hopfield** | already the paper's baseline | ✅ **exists in the same file**: `_hopfield_confidences()` |

---

## 4. ⛔ The byte budget is the claims-critical part, and it is where a referee will attack

`chlu/experiments/cl_baselines.py` **already carries pre-registered matched-bytes accounting** — `# ⛔ THE BYTE ACCOUNTING (w26 matched-bytes-frontier, PREREG §1)`, with `floats_per_stored_item()`, `fixed_state_floats()` and `items_for_budget()`.

⚠ **But it is written for the CL entry** — Split-MNIST/CIFAR class-IL, keyed on `n_classes` and raw exemplars. **MQAR key–value recall is a different accounting problem** and the adaptation is yours to derive and to justify in `PREREG.md`:
- what the CLU store actually costs per stored item on these cells (wells, addresses, payloads — **φ included**);
- what an exemplar store costs per key–value pair at the same dtype;
- how many items each gets at the shared budget.

**Binding conventions (C3 charter Add.1 §2, and they are not optional):**
- ⛔ **TOTAL state bytes, AS-DEPLOYED dtype.** No dtype normalisation — an fp32 store pays its real cost.
- ⛔ **`dtype_bytes` is declared per row, never inferred.**
- ⛔ **Never quote a state-byte figure without its convention.**
- ⛔ **φ is ledgered on every arm** (C3 §5: *byte ledgers on every arm incl. φ*).

⭐ **The scorer is ten lines and already exists**: `margin_rel = (E - S) / S` at `.claude/scratch/c3-trackb-tripwire/analyse.py:62`. Reuse the statistic; ⛔ do not re-derive it differently.

---

## 5. What to measure and report

For every arm × cell × seed, report **both axes**:
1. **accuracy at matched COMPUTE** — the axis §4.1 currently reports, extended to arms (ii) and (iii);
2. **accuracy at matched BYTES** — the axis that has never been run, with `margin_rel` against the **−0.02** threshold.

⭐ **And report the shape, not just the endpoints** (**CM-22(bb)**: *quote the curve, never the endpoint*): the gate's accuracy-vs-compute ladder per arm, so a reader sees whether the exemplar baseline has a dial at all. **The absence of a dial is the finding, if it is the finding.**

Also required:
- **The LTT coverage certificate re-checked** under the new arms (30/30 valid, risk 0.030 at ε=0.05). ⭐ This is a property of the *apparatus*, not of the comparison, and criterion 4 does not touch it — so it survives either branch and is worth measuring precisely.
- **Per-arm byte ledger table**, dtype declared.
- **Your prereg scorecard**: every point prediction, hit or missed, scored honestly.

---

## 6. Acceptance criteria

- `.claude/outputs/v1-escalatability-floors.md` exists, carrying: the prereg (quoted), the byte-ledger table, both accuracy axes, `margin_rel` per cell, the ladder curves, the LTT re-check, and the prereg scorecard.
- `PREREG.md` is committed **before** the first implementation commit, and the report prints both commit hashes so the ordering is checkable.
- ⛔ **`papers/v1-short/**` and `NIPSsubmission/v1-ttcl/**` byte-untouched** — md5 manifest before and after, printed. *(Advisor note: no other spoke is in flight against those paths; this criterion is satisfiable — cf. Add.43, where a byte-untouched criterion named a directory another spoke was commissioned to write to.)*
- **One declared worktree**, named in the report. ≤3 engineer worktrees is the standing cap.
- Every negative positive-controlled (§8).

---

## 7. ⛔ Prohibitions

1. **No paper edits.** Findings go in the report.
2. ⛔ **No tuning toward a non-firing result** (§1). If an arm needs a choice you did not preregister, **stop and report the choice**, do not make it silently.
3. ⛔ **Never quote a state-byte figure without its convention**; never infer `dtype_bytes`.
4. ⛔ **This produces a NEW measurement, which means a new registry entry — and that is the Hub's to file, routed via the Head.** ⛔ You do not write to `claims_matrix.md` or `negative_results.md`, and you do not self-file an N-number. Propose; never register.
5. ⛔ **Do not touch the C3 CSF3 artifacts** (`outputs/cluformer-pilot/**`). This is a laptop job on V1's own cells and has nothing to do with the scale runs.
6. **Declare NOT-RUNs.** A declared not-run is never a null (C3 §5).

---

## 8. ⚠ Two grep hazards on this machine — both will silently lie to you

1. ⛔ **`grep` here is a shell function resolving to `ugrep 7.5.0`.** On bounded-context patterns (`.\{0,70\}word.\{0,70\}`) over long lines it either **errors *"exceeds complexity limits"* and exits 0** — a silent false negative that looks like success — or **hangs**. Both reproduced 2026-08-26. ⇒ **use `/usr/bin/grep` for context patterns**; count with `grep -o … | wc -l`, ⚠ **never `grep -c`** (lines, not occurrences).
2. ⛔ **Directory-level grep over `.claude/` returns nothing** — it is gitignored. **Sweep per-file.**
3. ⚠ **zsh does not word-split**: a `--include=*.py` style glob is expanded by the shell before grep sees it. Quote it, or omit it.

⭐ **Positive-control every negative before reporting it.**

---

## DIAL DECLARATION
**Dials touched: the arm set only** — two baseline arms (matched-bytes exemplar store; matched-compute feedforward floor) are ADDED to §4.1's existing MQAR protocol. ⛔ **No CLU-side dial is changed**: kv levels, seeds, epochs, relaxation ladder, calibration head and LTT targets all stay exactly as `exp_v1_calibration.py` runs them today, so arm (i) reproduces §4.1's published numbers. **Laundering control:** the matched-bytes exemplar store *is* the laundering control for the escalatability claim. **Falsifies:** CM-2's *"one-shot memories get no allocation payoff"* if the exemplar store reaches ceiling at matched bytes.
