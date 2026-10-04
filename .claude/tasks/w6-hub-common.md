# WAVE 6 — THE PHYSICS WAVE. COMMON RULES.
Boot: `autonomous-hub-charter.md` → this file → your target file. ⛔ Nothing else.
**Base `base/iter6 = df025685`** (pin + 44 additive files, 0 deletions, frozen harness byte-identical).

## 1. The comparison
**Every result row vs the FULL RIVAL SET** (on disk from w4 spoke-a): table · repaired LACE · PoE-EBM ·
MLP · **`mlp_180k`** · transformer · LSTM · continuous-input MLP, 3 seeds, cache **generation 2**.
⛔⛔ **A win over an earlier CLU configuration is an ABLATION, labelled so. A win over a RIVAL is a
RESULT.** Landscape-removed + landscape-shuffled on every arm, every row. Rivals that cannot enter a
row get a **dash and a footnote** — ⛔ never a win for us.
⭐ **Byte pin 524,288 B is PRIMARY** (tolerance 0); parameter matching is secondary and cannot hold
simultaneously. Print both. ⛔ Never quote one matching without naming it.

## 2. Estimand and budget
⭐ **≥ 49 paired corners; margin declared in the PREREG before the first held-out number.** ⛔ A single
cell never decides a target. Rivals scored on the **same** corners. **Generation 2 only.**
**Budget: `potentials` 3 · `retention` 2 (deletion target) · `compose-physics` 2 (composition target).**
⛔ **`refusal` is 0/3 — SPENT. No hub may score a refusal row.** Write your ledger row to
`/c/chlu/SCORED-LEDGER.md` **before** the run. All scored runs reported; the best never alone.

## 3. ⛔ KNOWN — the state of the programme. Do not rediscover.
1. ⭐⭐ **THE RESULT (w5):** `pca_tensor` — per-well **leading principal subspace** of the fit pool,
   64 atoms × 2048 f32, **unsupervised, no label in the content** — **beats `mlp_180k` on rows 3 AND 4,
   all three seeds**, past a pre-declared bar (row 4 +10.143/+8.673/+7.796; row 3 +4.033/+2.483/+2.417).
   It beats every rival that can enter those rows. ⚠ Row 3 seed 1 clears by only 0.033.
   ⭐⭐ **THIS IS THE BAR NOW.** ⛔ The per-value-mean read is superseded as a comparator.
2. ⛔⛔ **THE CENTRAL PROBLEM: compose and forget are on DIFFERENT OBJECTS.** A principal direction is
   not a written item, so **charter §B.4 exact per-item deletion does NOT hold of the winning content**
   (keeping items instead costs ≈18.6 tuples/corner). Meanwhile the item/field store forgets exactly —
   but **loses composition when it evicts atoms, whichever atoms they are.**
3. ⭐ **The coupling hole is exact and free:** refuses `red ∧ cylinder` at 1.0000 for **1 byte**,
   0 bytes outside, **no refit**, **0.0000 composition cost**, 101 other never-held conjunctions
   untouched, zero seed spread. ⛔ **Eviction is NOT free** (−2.09/−0.91/−6.74 on composition).
4. ⛔ **Refusal rows 5/6 are closed:** the 8-byte presence+bracket rule ties the field store at
   (1.0000, 1.0000) on both. `table_absence` **is** `bitmap8`. ⛔ No refusal claim is available.
5. ⛔ **`knn_density` fails at 0.0203** — a length scale from the spacing of stored items does not
   separate held from never-held. ⛔ **Do not derive any length scale that way.**
6. ⛔ **No read of the wave-3 *fitted-mode* store beats a matched MLP, including a perfect read**
   (`oracle_span` −6.966 below `mlp_180k`). ⭐ w5 cleared that by changing **content**, not the read.
7. ⛔ **Placement exhausted** · **row 2 belongs to a generic learner** (size-matched continuous MLP
   wins on every seed, two implementations) · **dreaming banked** · **admission retired** ·
   **earlier-layer content retired**.
8. ⭐ **Generation 1 fails its own encoder's re-encode check** on all three seeds. ⛔ Generation-1
   numbers never share a table with generation-2 numbers.

## 4. Labels
Used **once at fit time** to align wells and build the address book, and at the interface. ⭐ Declare it
in every table caption. ⛔ Zero-supervision structure is BANKED, not claimed.
⭐ Note `pca_tensor`'s content is unsupervised — if your object keeps that, say so; it is worth stating.

## 5. Process — three rules that cost this programme real time
1. ⛔⛔ **NEVER run `taskkill /F /IM python.exe` or any box-wide kill.** You share this machine with
   three other agents; in wave 5 that command destroyed two hubs' work. **Kill only your own PIDs,
   captured at launch.**
2. ⛔⛔ **Do NOT end a turn while detached (`nohup`) work is running.** Nothing will re-invoke you and
   the run finishes into silence. ⭐ **Poll your jobs to completion INSIDE a turn** (loop on their
   output files with `sleep`).
3. ⭐⭐ **Commit your code and bundle the branch WHOLE before you finish.** ⚠ In wave 4 a spoke ended
   with 12 files untracked and no bundle and the Hub had to rescue it. ⛔ `git status` does not show
   gitignored outputs — verify outputs by **directory listing**, code by `git status` + `git bundle verify`.
One launcher per hub; `START_UTC.txt` first, refuse to start if a live one exists. `LOG.md` in dated
bullets throughout, never rewritten; final report is its last section. ⛔ A declared NOT-RUN is never a
null. Local CPU, 3 threads. GPU cloud-only for a headline run, three justification lines first.
