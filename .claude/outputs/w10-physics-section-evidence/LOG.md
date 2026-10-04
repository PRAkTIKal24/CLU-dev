# w10 spoke/physics-section-evidence — LOG

**Agent:** `physics-theorist`. **Branch:** `w10/physics-section-evidence`.
**Output dir:** `/c/chlu/outputs/w10-physics-section-evidence/`.
⚠ **Path note, declared:** the task file names the output dir `/c/chlu/outputs/w10-physics-section/`;
the invocation names `/c/chlu/outputs/w10-physics-section-evidence/`. I write **only** to the
invocation's dir (the one I was told I own). `w10-physics-section/` does not exist and I did not create it.

⛔ `$CHLU_CLAUDE_ROOT` NOT read. Evidence base used: `/c/chlu/evidence/outputs/`
(+ `/c/chlu/evidence/claims_matrix.md`, `/c/chlu/evidence/negative_results.md`).

---

## Timeline

- **2026-09-21T08:26:07Z** — START_UTC.txt written. Task file read:
  `/c/chlu/tasks/w10-spoke-physics-section-evidence.md`.
- **08:27Z** — Evidence survey. Located the CRR spine in
  `evidence/outputs/w6-advisor-report.md` §1 and in the wave-6 record
  `evidence/outputs/_thinkpad/w6/` (`SCORED-LEDGER.md`, `w6-potentials/{LOG.md,RESULTS-LEDGER.md}`,
  `w6-shorts-inventory/shorts-experiments-inventory.md`).
  ⚠ A box-wide `grep -rIl` over `/c/chlu` was launched and exceeded 120 s; it was superseded by
  scoped greps and is polled to completion before this spoke ends (see §Background jobs).
- **08:32Z** — ⛔ **Scope check on the two registry files I was pointed at.**
  `evidence/negative_results.md` (3,794 lines, N1–N313) ends at the **C2W11 fold** and
  `evidence/claims_matrix.md` (v2.17) is **Campaign-2/Campaign-3 accounting**. ⛔ **Neither contains a
  single wave-5 or wave-6 ICLR-long entry** — `grep -in "gym-1|gym-2|illegal"` over
  `negative_results.md` returns **0 hits**. The w5/w6 negatives therefore live only in the wave
  records (`evidence/outputs/w5-advisor-report.md`, `w6-advisor-report.md`,
  `evidence/outputs/_thinkpad/w5|w6/**`). **DECLARED:** the benchmark statement is drafted from those,
  and the two registry files are cited only where they genuinely bear (they do not, for w5/w6).

- **08:45Z** — Located and read the spine's primary sources (all under `/c/chlu/evidence/outputs/`):
  * `w6-advisor-report.md` §1 §7 — the CRR headline and its limits.
  * `_thinkpad/w6/SCORED-LEDGER.md` — the w6 outcome rows (potentials, retention, compose-physics).
  * `_thinkpad/w6/w6-potentials/{LOG.md,RESULTS-LEDGER.md,PREREG-potentials-1.md}` + the raw
    artefacts `test/s{0,1,2}/results.json`, `replication/s0/`, `dev/s0/`.
  * `_thinkpad/w6/w6-shorts-inventory/shorts-experiments-inventory.md` — D1–D50, A1–A8,
    E1.1–E1.21 / E2.1–E2.19 / E5.1–E5.23.
  * `_thinkpad/w6/w6-retention/LOG.md` — the SECOND, independent reproduction of the retention law.
  * `_thinkpad/w6/{w6-compose-physics,w61-compose-physics}/LOG.md` — the refutation half.
  * `_thinkpad/w5/{w5-gym1,w51-gym2}/LOG.md` + `w5-advisor-report.md` — the benchmark negatives.
- **09:05Z** — ⭐ **Numeric reproduction written and run:** `checks/derivation_checks.py`
  (pure stdlib, no `chlu` import, no production code touched) → `checks/derivation_checks.out`.
  **34 / 34 checks reproduce, 0 FAIL.** Three real defects in my own first pass, fixed and logged:
  (i) D12 summed `(1-gamma)^n` from n=1 not n=0 (my loop order) — the law was fine, my rollout was not;
  (ii) D17's numeric derivative was taken at fixed `h/h*` rather than fixed `h`, which returns
  **-1** instead of **+1.14** because `h* ∝ gamma` — a sign error that would have read as a REFUTATION
  of D17; (iii) D10's collision tolerance was set below `sqrt(machine eps)`, which is the floor for a
  discriminant that cancels to zero. ⛔ None of the three was a defect in the evidence base.
- **09:20Z** — Two findings that are mine, not the record's, both written up in `EVIDENCE.md` §4:
  **(a)** the D17 constant **+1.1385** is `1+gamma/(2-gamma)` at **gamma = 0.243303**, which is NOT a
  point of the recorded 9-point gamma grid, and the per-cell fit windows are **not in any
  machine-readable artefact** ⇒ the headline constant is reproducible as a FORMULA but not as a
  MEASUREMENT-SIDE number. **(b)** D18's quoted derived value **-1.0050** sits at `gamma = 0.009917`
  while its quoted measured value **-1.1090** sits at `gamma = 0.184104` on the same derived curve
  — an evaluation-point gap that would fully explain the "thin margin", and that CANNOT be
  adjudicated from the artefacts. Both labelled, neither asserted.
- **09:35Z** — `EVIDENCE.md` + `BENCHMARK-STATEMENT.md` written.

- **09:50Z** — **Branch committed.** `cc84c8e` on `w10/physics-section-evidence`, one commit on top
  of `base/iter8 = fa60b15`, working tree clean.
  ⚠ **Only the code artefact is in the repo, and here is why.** I first staged the prose
  deliverables under `projects/w10-physics-section-evidence/` and git refused them: this repo's own
  `.gitignore` carries `projects/*/` and `docs/**`. ⛔ **I did not `-f` past the repo's own
  convention.** The commit therefore carries `w10_physics_section_evidence/{derivation_checks.py,
  derivation_checks.out,README.md}` at top level (the `w8_envscout/` precedent, and ⛔ nothing under
  `chlu/`), and `EVIDENCE.md` / `BENCHMARK-STATEMENT.md` / `LOG.md` stay in this output directory.
  The README in the committed dir says so, so a reader of the bundle alone is not misled.
- **09:55Z** — **Bundle created and verified, that branch only.**
  `git bundle create .../w10-physics-section-evidence.bundle w10/physics-section-evidence`
  (⛔ **not** `--all`). `git bundle verify` → *"is okay … The bundle records a complete history."*
  `git bundle list-heads` → **exactly one ref**,
  `cc84c8eb0bda6d8fd2a7b77b746e1110b1dbb156 refs/heads/w10/physics-section-evidence`. 5,104,134 B.
  ⭐ **Replication check:** a clean `git clone -b w10/physics-section-evidence <bundle>` into the
  scratchpad re-ran `derivation_checks.py` and the output **diffed IDENTICAL** to the committed
  `.out`. The scratch clone was removed afterwards.

---

## Background jobs — all accounted for

| job | what | end state |
|---|---|---|
| `bg074vviy` | the over-broad `grep -rIl` over `/c/chlu` launched at 08:27Z | ⭐ **completed inside this turn, exit 0**, output file empty; superseded by scoped greps and not used. ⛔ No detached work is running at the end of this spoke. |

⛔ **No box-wide kill was ever run.** No process was killed at all — nothing needed it.

---

## Deliverables

```
START_UTC.txt                       2026-09-21T08:26:07Z
LOG.md                              this file
EVIDENCE.md                         ⭐ the evidence base (the deliverable)
BENCHMARK-STATEMENT.md              ⭐ the "a benchmark is needed" requirement list, standalone
checks/derivation_checks.py         34 checks, pure stdlib, ~4 s
checks/derivation_checks.out        34 / 34 reproduce, 0 FAIL
w10-physics-section-evidence.bundle branch w10/physics-section-evidence @ cc84c8e, verified
```

## Scored runs

⛔ **NONE.** Mandated by the task file (*"Scored runs: ⛔ none. Small numerical sanity checks
only."*). Nothing in this spoke's output touches a scored budget.

## What would have proved this spoke wrong, and what actually happened

| falsifier from the task file | outcome |
|---|---|
| *"A derivation you cannot reproduce numerically ⇒ mark it REFUTED and say which step fails."* | ⭐ **34 / 34 reproduce.** Three apparent failures arose and **all three were defects in my own first pass** (D12 summation order, D17 differentiated at fixed `h/h*` instead of fixed `h`, D10 tolerance below `sqrt(machine eps)`); each is logged above and in `EVIDENCE.md` §4.1. ⛔ **Two derivations ARE marked REFUTED, and their failing steps are named:** **D16** as a rollout prediction on the field store (the pure-exponential envelope is invalid at the defective exceptional point, D10) and **D32b** byte-identity-to-never-inserted on this object (V5's canonical-placement precondition; measured as **only 16.5 %** of the obstruction — **83.5 %** is that the interface is fitted on the pool). |
| *"The half-life law failing to reproduce from the recorded artefacts ⇒ that is the finding, and it outranks the section."* | ⭐ **It reproduces.** B1 = **0.043256 / 0.043516 / 0.043102**, B2 mean Δ = **−0.596831 / −0.579639 / −0.598172**, **49/49** corners better on every seed, recomputed by me from `test/s{0,1,2}/results.json` and matching `RESULTS-LEDGER.md` to every printed digit; byte pin, field sha, integrator check, instrument floor and B6 likewise. ⚠ **One bounded exception, reported rather than buried:** the D17 pair **+1.1385 / +1.1368** is reproducible from the hub's report but **not** from any recorded machine-readable artefact, because the per-cell fit windows were never written to one (`EVIDENCE.md` §4.2). That is a reporting gap in the programme's sharpest number and it should be closed; nothing I ran contradicts the number itself. |
