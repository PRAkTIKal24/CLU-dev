# WAVE 10 — REPORT FOR THE MAIN ADVISOR
**Compiled by the ICLR-long Hub, 2026-09-21. Five spokes, FORGIS_14, `base/iter8 = fa60b15`.**
Launch 10:25Z, all five finished by 11:16Z. ⭐ **ZERO scored runs, zero GPU, zero new measurements** —
this was a consolidation wave by design. Record: `.claude/outputs/w10-*/`.
⚠ Every number is quoted from a spoke's own report. ⛔ **The Hub has re-derived no statistic.**

---

## 0. The scoreboard

| spoke | verdict | scored |
|---|---|---|
| ⭐ `seeds-and-tables` | **240 cells, 27 sources, all machine-verified** — regenerates from its own bundle | 0 |
| ⛔⛔ `registry-fold` | **the lead claim's phrasing is wrong as written** — "9/9 seeds" is not 9 seeds | 0 |
| ⭐ `physics-section-evidence` | evidence base + benchmark statement assembled; ⚠ registry gaps exposed | 0 |
| ⛔ `draft-v0` | skeleton built — and it **DELETED** the unsupported sentences rather than styling them | 0 |
| ⛔⛔ `referee` | **REJECT as it stands.** 16 must-fix. ⭐ Accept-with-major-revision at **0 scored runs** | 0 |

**Budgets unchanged by this wave:** `single-design` 3/3 · `generation-table` run 3 unspent ·
`composition` 2/3 · `deletion` 2/3 · `encoder-loop` **2** · `maze` **0 (exhausted in w9)**.

---

## 1. ⛔⛔ THE HEADLINE: the programme's lead claim is real, its PHRASING is not

**Three spokes converged on this independently, from three directions.**

⛔ **"1e-16…1e-14 on Shapes3D, 9/9 seeds, two independent implementations" is wrong as written.**
1. **"9/9 seeds" = 3 CONFIGURATIONS × 3 seeds inside ONE implementation** (K=40/44/48).
2. **The band is not shared.** The peer implementation runs **2.842e-14 · 4.547e-13 · 1.164e-10 ·
   1.126e-09** — five orders coarser; ⚠ its worst leg sits **below its own reproducibility floor of
   4.282e-09**, so there the deviation is not resolvable at all.
3. ⛔ **It was not unconditional — it FAILED first**, on test seed 2 at **32.90 float32 ulp**, fixed
   by a variance-share admission cut with **no free parameter** (`tau = 4.144858e-09`), ⭐ declared
   before the re-run and **not moved**.
4. ⭐⭐ **The strongest form of the two-implementation claim is the FAILURE AGREEMENT:** the second
   implementation reproduced the **same seed-2 failure** with different code and its own sign
   convention — 14.377 ulp against 32.90.

⭐ **What IS unambiguously in the 1e-16 band, and what the paper should lead with:** wave 9's own
CMAZE-2 number — **7.5e-16…8.8e-16, subspace projector exactly 0.000e+00, all three scored seeds,
replicated from a clean checkout with ZERO numeric differences.**

⭐ **The `draft-v0` spoke deleted "9/9 seeds" on its own**, before reading the registry's finding.
Two spokes, different inputs, same conclusion.

---

## 2. ⛔⛔ `spoke/referee` — REJECT, and the fix costs no compute

> **REJECT as it stands. Accept-with-major-revision is reachable, and the revision costs 0 scored runs.**

| reviewer | score | the sentence that decides it |
|---|---|---|
| **R1 — machine unlearning** | **3 → 6** after MF1–MF7 | *"Exact" is never defined, the frozen encoder/decoder is never refit, and the sufficient statistic that buys the exactness is **24.5× the byte budget** and appears in no table* |
| **R2 — compositional generation** | **5 → 6** | two main-text tables report the same arm in two metrics on two splits |
| **R3 — benchmarks / rigour** | **6** | ⭐ *"the negatives discipline and `verify_tables.py` are genuinely above the field"* |
| **AC** | **reject, invite resubmission** | ⛔⛔ two artefact-level contradictions **inside the file the paper cites for its lead result**: `a1_subspace.passed_every_seed = false` and `note: "No scored run and no claim"` |

**The four that carry the rejection:**
- **MF3** ⛔ Table 1 — the lead — **prints no byte row**, and the object is **24.5× the pin** the paper
  holds everything else to. ⛔ Without it *"R1 rejects on sight — it is the standard unlearning objection."*
- **MF4** ⛔ **"Exact" is never defined**, and the only definition the measurements support is
  **store-scoped**: the frozen encoder and decoder are never refit.
- **MF7** ⛔ A row labelled **RESULT · 3 / TEST** whose own artefact says *"No scored run and no claim"*.
- **MF10** ⛔⛔ **THE ENVIRONMENT IS MISLABELLED — see §3.**

⭐ **The referee also VERIFIED rather than only attacking**, and says not to re-litigate these: it
re-ran `verify_tables.py` clean (27 sources / 240 cells / 720 per-seed values / 5 figure hashes),
re-checked **every** w9 deletion cell and **every** w5 paired delta, and independently recomputed K15
(max 0.013761, equal on 7 of 12). ⭐ It names **§6.1 plan-vs-latent "the best thing in the paper, and
exact."** It declared **six things NOT checked, with reasons.**

---

## 3. ⛔⛔ THE ENVIRONMENT WAS MISLABELLED — w9 scored CMAZE-**2**, not CMAZE-1

`hub/maze`'s own final report: *"**CMAZE-2** — a REQUEST layer over CMAZE-1. The engine, the solver
and the exact integer judge are `w8_envscout.maze_colour`, imported unmodified."*

⚠ **The Hub wrote "CMAZE-1" throughout the w9 report. That was wrong** and is corrected in
**w9 Amendment 2**. The three numbers only become readable once separated:

| number | what it is |
|---|---|
| **0.4980** | published CMAZE-1 (env-scout adoption) |
| **0.5317** | **CMAZE-1 proper**, harness tier — ⭐ the environment reproduces as adopted |
| **0.9037** | **CMAZE-2**, the request layer — ⛔ a REQUEST-LAYER weakness, not a CMAZE-1 defect |

⛔ **The maze section and the harness tier are a DIFFERENT ENVIRONMENT, not merely a different
budget, and must never share a table.**
⚠ **Also corrected (MF11):** *"FRAGMENT-LOOKUP 0.0000 everywhere"* is **false on the hub's own
split** — same-maze reads **0.0741 / 0.5105**; the 0.0000 figures are the **cross-maze** quantity.

---

## 4. ⭐ What the wave built that is worth keeping
- **`seeds-and-tables`:** 27 sources, **240 cells**, **720 per-seed values**, 5 figures, **5 declared
  NOT-RUN rows**, 4 corrections — **PASS** under a verifier that does not import the builder.
  ⭐ **Figures are byte-identical across rebuilds**, and ⭐⭐ **the whole table set regenerates from the
  BUNDLE alone** into byte-identical output. ⭐ The verifier **caught two real defects of the spoke's
  own** (an unregistered source, and a path shorthand expanding to 9 files where 3 exist).
- **`registry-fold`:** claims matrix + negatives registry updated; ⭐ prints the **three
  implementations as three separate rows** so no draft can quote one's band as another's.
- **`physics-section-evidence`:** EVIDENCE.md + BENCHMARK-STATEMENT.md; ⭐ re-ran its checks from a
  clean clone of its own bundle and diffed **identical**.
- **`draft-v0`:** 74 KB skeleton; ⭐ **deleted** every sentence with no artefact, and recorded the
  Campaign-2 skeleton as a **different object from a different campaign** so no one merges two.

## 5. ⚠ Disclosures the spokes made against themselves
- ⚠ **The negatives registry (N1–N313) contains ZERO w5/w6 entries** — `grep -in "gym-1|gym-2|illegal"`
  returns 0 hits. The physics spoke drew those negatives from the wave records instead.
- ⛔ **N89/N90 are allocated TWICE** (a 2026-09-11 gen-pivot addendum re-used them). ⭐ The spoke
  **allocated no number in either series rather than compounding it.** ⛔ The gen-pivot negatives
  still have **no matrix row at all.**
- ⚠ **C6: the paper's sharpest number (+1.1385) has its WEAKEST artefact trail.** ⛔ Do not write D18
  up as a confirmation; `d14_confinement` is tautological — quote the Hessian check.
- ⚠ **4 of 9 figure slots have no artefact**; F6 cannot be run from the artefacts at all.
- ⭐ `physics-section-evidence` **refused to force its prose past `.gitignore`** rather than `-f` past
  the repo's own convention.

## 6. Limits
No new measurement exists in this wave · the Hub re-derived nothing · the draft is a **skeleton, not
prose** (the Head writes prose) · **16 must-fix are open and the draft is REJECTED until they are
closed** · `w10/draft-v0` and `w10/referee` are ⛔ **deliberately NOT merged** into `base/iter8` so a
rejected draft does not become the baseline.

## 7. What the Advisor is asked to rule on
1. ⭐⭐ **The lead claim survives but must be re-stated.** ⛔ Is *"exact deletion, store-scoped,
   at 24.5× the byte pin, with the cost printed"* a claim the paper can lead with — or does the byte
   cost move it out of the headline?
2. ⛔ **MF4: "exact" must be DEFINED.** The measurements support a **store-scoped** definition only.
   **Is store-scoped exactness the contribution, stated plainly?**
3. ⚠ **The registry has structural gaps** (no w5/w6 entries, N89/N90 double-allocated, gen-pivot
   unrecorded). **Who owns repairing the numbering?** ⛔ The Hub did not renumber.
4. ⚠ **C6:** the sharpest physics number has the thinnest provenance. **Strengthen the trail, or drop
   the number from the main text?**
5. ⭐ **The referee says accept-with-major-revision costs ZERO scored runs.** ⛔ That makes the
   revision a WRITING pass, not an experimental one — **is that the next wave, ahead of w12?**
