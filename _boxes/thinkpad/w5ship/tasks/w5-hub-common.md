# WAVE 5 — COMMON RULES. Boot: `autonomous-hub-charter.md` → this file → your target file. Nothing else.
**Base `base/iter5 = 6675904`** (pin + 36 additive files; 0 deletions; frozen harness byte-identical).
⛔ Do not read wave-2/3/4 reports. §3 carries what you need.

## 1. The comparison
**Every result row vs the FULL RIVAL SET** (built by w4 spoke-a, on disk): prototype table · repaired
LACE · PoE-EBM · **MLP · `mlp_180k` · transformer · LSTM · continuous-input MLP**, 3 seeds.
⛔⛔ **A win over an earlier CLU configuration is an ABLATION, labelled so. A win over a RIVAL is a
RESULT.** Landscape-removed + landscape-shuffled on every arm, every row. Rows a rival cannot enter
get a **dash and a footnote** — ⛔ never a win for us.

⭐ **Two budgets, both declared, and you print both:** the **byte** pin (524,288 B store) is PRIMARY;
**parameter** matching is secondary and cannot hold simultaneously — the pinned store alone is 131,072
parameters, 10.3 % above the prototype table. `mlp_180k` is the parameter-matched rival at the second
budget. ⛔ Never quote one matching without naming it.

## 2. Estimand, seeds, budget
⭐ **≥ 49 paired corners, margin declared in the PREREG before the first held-out number.** ⛔ A single
cell never decides a target. ⭐ **Rivals are scored on the SAME corners** — if a rival artefact exists
only as a single 72-request cell, re-run it per corner; that is a property of how it was run, not a
licence to decide on one cell. *(Hub default, logged: raised by w4 hub/placement, unruled by the Advisor.)*
**Generation 2 only (seeds 0,1,2).** ⛔ No generation-1 number shares a table with a generation-2 number.
**3 scored runs per named target; replications of the same PREREG are FREE.** Write your row to
`/c/chlu/SCORED-LEDGER.md` **before** the run. All scored runs reported; the best never alone.

## 3. ⛔ KNOWN — do not rediscover
1. ⛔⛔ **No read of the WAVE-3 store beats a parameter-matched MLP — including a perfect one.**
   `oracle_span` (the true feature projected onto span of the wells' modes — the best answer *any*
   read of those atoms can give) is **−6.966 below `mlp_180k`** on all three seeds. Read headroom is
   **+4.354** and closing all of it still leaves the arm 6.97 short.
   ⭐⭐ **BUT that ceiling is a property of the wave-3 store's 64 FITTED modes. The field store CHOOSES
   its atoms and spans a different subspace at the same 524,288 B. Whether some other 524,288 B of
   content clears the learner is OPEN and is wave 5's most valuable question.**
2. ⛔ **Wave 3's composition win (+11.612) was over `basis_head` — an ABLATION.** Against the full
   rival set the same object **loses**. ⛔ Do not cite it as a result or as something to "keep".
3. ⭐ The tensor **product** address is the mechanism (+32.408 over the summed variant); the store
   does earn its bytes against the per-value-mean specialist (+29.054) and the table (+34.238).
4. ⛔ **Row 2 belongs to a generic learner:** a size-matched continuous MLP beats the CLU read on every
   seed under two independent implementations. The kernel-settling mechanism is a **structural null** —
   the pinned read already *is* the chord midpoint. ⚠ The second judge has a broken-clock defect a
   constant render exploits; row 2 needs **both** judges.
5. ⛔ **Placement is exhausted:** re-deriving how labels become a partition is worth ~1 tuple/corner
   with the wrong sign on another seed. ⭐ But *where* channels go matters enormously (+23.653 over
   equal wells with no labels) and **using labels is most of it**. Inherit the label-MI partition.
6. ⛔ **Refusal is winnable at 8 bytes** (per-factor presence bitmap + bracket check: rows 1,2,5,6 all
   1.0000). ⭐⭐ **The one exception is exact: a per-factor gate cannot express a CONJUNCTION deletion
   at all (zero bits change), and the joint table that can costs every row-3/4 request.**
7. ⭐⭐ **The field store's central property holds EXACTLY:** eviction 26,275 B changed / **0 outside**;
   the **hole = 1 byte** / 0 outside; `assert_no_refit` passes on 14 structures; the negative control
   fires at 5,762 B. The hole refuses one conjunction while **96 unseen conjunctions stay answered**.
8. ⛔ **Generation 1 fails its own encoder's re-encode check on all three seeds** (51/81/59 ULP on
   ~1 % of elements). Generation-1 caches cannot be re-scored on this box. ⭐ Deterministic arms moved
   **exactly 0.0000** between generations; only sampler arms moved.
9. ⚠ **LACE is 95.1× the store budget; PoE-EBM 27.1×.** The prototype table at **0.91×** leads the
   rival set at 0.9165. ⛔ Print the byte multiple beside any row either one appears in.
10. ⛔ Dreaming BANKED · admission RETIRED · earlier-layer content RETIRED · placement EXHAUSTED.

## 4. Labels
Labels are used **once at fit time** to align wells, and at the address book. ⭐ **Declare it in every
table caption.** ⛔ Zero-supervision structure is BANKED, not claimed.

## 5. Process
One launcher per hub; `START_UTC.txt` first, refuse to start if a live one exists. `LOG.md` in dated
bullets throughout, never rewritten; corrections are new bullets; final report is its last section.
⛔ A declared NOT-RUN is never a null. Local CPU, 3 threads. GPU cloud-only, headline runs only, three
justification lines in `LOG.md` first.
⭐⭐ **COMMIT YOUR CODE AND BUNDLE THE BRANCH WHOLE BEFORE YOU FINISH.** ⚠ In wave 4 a spoke ended with
12 files untracked and no bundle; the Hub had to rescue it. ⛔ `git status` does not show gitignored
outputs — verify the outputs tree with a **directory listing**, and verify the code with
`git status` + `git bundle verify`. **Both halves, every time.**
