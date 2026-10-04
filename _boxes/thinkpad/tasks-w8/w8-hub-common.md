# WAVE 8 — COMMON RULES. Boot: `autonomous-hub-charter.md` → this file → your target file. ⛔ Nothing else.
**Base `base/iter7 = ec2ef7c`** (pin + 69 additive files, 0 deletions, frozen harness byte-identical).

## 1. ⭐⭐ FOUR BINDING RULES, NEW IN WAVE 8 (Head) — these change how you run
1. ⭐ **SAME ENCODER FOR EVERY ARM.** Every arm maps the request into the frozen feature space and
   renders through the **shared frozen decoder and judge**. ⭐ **When the encoder is unfrozen, every
   arm trains its OWN encoder at EQUAL COMPUTE** — no arm gets a better representation for free.
2. ⛔⛔ **NO RUN-KILLING TRIPWIRES.** Instrument checks — fragment-lookup score, composition-blind
   score, table floor — are **REPORTED COLUMNS, never abort conditions.** ⭐ **Every commissioned
   experiment runs the CLU and the learned baselines to completion.** ⚠ This reverses waves 5–6,
   where a fired tripwire stopped the run. It does not any more. **Print it and keep going.**
3. ⭐⭐ **TWO ROW CLASSES, and you label every row as one of them.**
   - **RESULT rows** — scored vs **learned baselines** (MLP, LSTM, transformer, continuous MLP) at
     **matched parameters**. ⭐ The prototype table is a **printed floor**, not the bar.
   - **DISPLAY rows** — post-deletion refusal, post-deletion composition, refusal of impossible
     requests. These **show inductive bias**. ⭐ **The ONLY comparator is a transformer with a refuse
     token trained on impossible examples**, and our line is: ⭐ **"no impossible example ever seen."**
     ⛔ A DISPLAY row is never quoted as a RESULT.
4. ⭐ **Paper object = the Naive-CLU**, composition-led. Physics is an extended section. ⛔ Do not
   argue the physics into a composition claim — wave 6 measured it out of that story.

## 2. Estimand and budget
⭐ **≥ 49 paired corners; margin declared in the PREREG before the first held-out number.**
**Generation 2 only; 3 seeds on every number.** Rivals scored on the **same** corners.
**Budget: `single-design` 3 · `encoder-loop` 3 · `generation-table` 1 left.** Replications free.
Write your ledger row to `/c/chlu/SCORED-LEDGER.md` **before** the run. All scored runs reported.
⛔ Spent and closed: `refusal` 3/3 · `placement` exhausted. `composition` 2/3 and `deletion` 2/3 are
**BANKED** — ⛔ not yours to spend in wave 8.

## 3. ⛔ KNOWN — the state of the programme. Do not rediscover.
1. ⭐⭐ **THE RESULT:** `pca_tensor` — per-well leading principal subspace, **unsupervised**, 524,288 B
   exact — beats `mlp_180k` on rows 3 AND 4, all three seeds (row 4 +10.143/+8.673/+7.796).
   ⚠ Row 3 seed 1 clears by 0.033. **That object is the incumbent and the bar.**
2. ⛔⛔ **THE OPEN PROBLEM, and wave 8 exists to close it:** the content that wins composition
   **cannot do exact per-item deletion**; the object that forgets exactly (**the coupling hole**:
   1 byte, 0 outside, no refit, **0.0000 composition cost**) **loses composition when it evicts
   atoms, whichever atoms.** ⭐ The single design is the attempt to have both.
3. ⛔ **The dynamics COST composition** at every strength that moves the state (−39.96 at the arm) and
   are bit-identical to the incumbent only at zero displacement. ⭐ The mechanism is a **resolution
   law**: the read's answers form a dense lattice ~5.5 % of the state norm apart and the dynamics step
   across the judge's decision cells. ⛔ `span(G_k)` is an **invariant subspace** — atom-grounded
   dynamics cannot leave it, so "motion off the subspace" is not an available explanation.
4. ⛔ **A lifetime is strictly worse than instant eviction** — same composition cost, paid early,
   refusal delivered late. ⛔ **Symbolic refusal is closed**: an 8-byte presence+bracket rule ties the
   field store at (1.0000, 1.0000) on rows 5 and 6.
5. ⛔ **Row 2 belongs to a generic learner.** ⛔ Placement exhausted. ⛔ Earlier-layer content retired.
   ⛔ Dreaming banked. ⛔ Admission retired.
6. ⭐ **The physics is validated as a description of dynamics**: the shorts' retention law predicts
   settle half-life **parameter-free**, beating a rival fitted **with sight of the held-out data**,
   49/49 × 3 seeds. ⛔ It predicts dynamics, not a scoreboard row.
7. ⛔ **"Mexican-hat width → interpolation floor" does NOT exist in the shorts** — established twice.
   ⛔ Never pre-register against it. ⭐ Generation-1 caches fail their own encoder's re-encode check;
   ⛔ gen-1 numbers never share a table with gen-2 numbers.

## 4. Labels
Used **once at fit time** to align wells and build the address book, and at the interface. ⭐ Declare it
in every caption. ⭐ `pca_tensor`'s **content is unsupervised** — if your object keeps that, say so.

## 5. Process — the three rules that have cost this programme real time
1. ⛔⛔ **NEVER `taskkill /F /IM python.exe` or any box-wide kill.** Kill only your own PIDs, captured
   at launch. In wave 5 a box-wide kill destroyed two hubs' work.
2. ⛔⛔ **Do NOT end a turn while detached (`nohup`) work is running** — nothing re-invokes you and the
   run finishes into silence. ⭐ **Poll your jobs to completion INSIDE a turn.**
3. ⭐⭐ **Commit and bundle the branch WHOLE before you finish.** ⛔ `git status` cannot see gitignored
   outputs — verify outputs by **directory listing**, code by `git status` + `git bundle verify`.
One launcher per hub; `START_UTC.txt` first. `LOG.md` in dated bullets; final report is its last
section. ⛔ A declared NOT-RUN is never a null. Local CPU, 3 threads.
