# TABLE-FIXES.md — w15 `spoke/table-fixes`

**Acceptance artefact for `/c/chlu/tasks/w15/table-fixes.md`.**
Branch `w15/table-fixes` · base `base/iter8 = 148f3b7` · worktree `/c/chlu/worktrees/CHLU-w15-table-fixes`.
**Scored runs spent: ZERO. Compute: ZERO.** No model fitted, no arm run, no artefact regenerated,
no background job launched. ⛔ `verify_tables.py` **NOT touched.** ⛔ `w10/draft-v0` and `w10/referee`
**NOT merged.**

**File changed:** `w13_draft_revise/DRAFT-v1.md` (one file; 137 insertions, 40 deletions).
**How:** `w15_table_fixes/apply_table_fixes.py` — 22 guarded exact-string replacements, idempotent
(a second run reports `already applied` on all 22 and writes nothing). Line numbers below are
**post-edit** line numbers in `DRAFT-v1.md`.
**Pre-edit copy of the file:** `/c/chlu/outputs/w15-table-fixes/DRAFT-v1.BEFORE.md`.
**Unified diff:** `/c/chlu/outputs/w15-table-fixes/DRAFT-v1.diff`.

**Source of the six items:** `/c/chlu/outputs/w13-referee-2/REFEREE-v1.md` §2 (MF2, MF14) and §3
(N5–N8).

⭐ **Every number written into the draft by this pass was re-derived from its primary artefact first**
— not copied from the referee. Two consequences are recorded below (one referee rounding slip
corrected; one citation target disambiguated).

---

## SUMMARY

| item | status | where |
|---|---|---|
| **MF2** — statistic's byte cost, convention named, in the lead sentence | ⭐ **DONE** | `DRAFT-v1.md:1340-1345` (§I lead sentence), `:232` (§3.3 table), `:1037` (§A MF2 row) |
| **MF14 part (i)** — BLIND-SEQUENTIAL and BLIND-FACTOR into the maze main-text table | ⭐ **DONE** | `DRAFT-v1.md:665-671` (§6.1 table), `:673-680`, `:708-721`, `:782-791` (§6.3), `:1049` (§A MF14 row) |
| **N5** — one ratio against three totals; two ratios for one object | ⭐ **DONE** | `DRAFT-v1.md:233-234`, `:242-250` (§3.3), `:64-69` (§0.2), `:592`, `:596-601` (§5.2), `:1039` (§A MF3), `:1104` (§B K19), `:1276` (§G C16) |
| **N6** — two different objects both called "K = 5" | ⭐ **DONE** | `DRAFT-v1.md:486-496` (§4.4), `:576` (§5.1), `:78` (§0.2), `:1097` (§B K12) |
| **N7** — the four-part honest total called "a statistic" | ⭐ **DONE** (it is the same clause as MF2) | `DRAFT-v1.md:1340-1345` |
| **N8** — three different statements about where the K = 5 loss is printed | ⭐ **DONE** | `DRAFT-v1.md:1005-1016` (§11 item 19) |
| provenance banner naming this amendment | ⭐ **DONE** (not asked for; added so the Head cannot read an amended file as w13's) | `DRAFT-v1.md:14-25` |

⛔ **Nothing is DECLARED NOT-DONE.** All six landed. What did **not** run at all is listed in
`FINDINGS.md` §"WHAT DID NOT RUN".

---

## MF2 — the statistic's own cost, in the lead sentence, with its convention named

**Referee's charge** (REFEREE-v1 §2): *"MF2's second half — 'state the statistic's byte cost in the
same sentence' — is not done: §I's sentence prints `24.55×`, which is the FOUR-PART HONEST TOTAL,
not the statistic."*

**BEFORE** — `DRAFT-v1.md` §I, contribution paragraph (pre-edit line 1249):

> … at the cost of a statistic that is **2.19 %** of the deployed read in one environment and
> **24.55× the byte pin** in the other **[K4]**;

**AFTER** — `DRAFT-v1.md:1340-1345`, ⭐ **the exact lead sentence, quoted, with its convention**:

> … at the cost of a retained sufficient statistic whose own bytes are **2.19 %** of the deployed
> read in one environment (**35,904 B of 1,642,752 B deployed**; convention: statistic ÷ deployed
> read) and **23.5121× the 524,288 B store pin** in the other (**12,327,096 B ÷ 524,288 B**;
> convention: statistic ÷ pin, the statistic alone), which takes the never-netted four-part honest
> total — atoms + statistic + request-count table + coupling/holes — to **24.5466× the pin** **[K4]**;

⭐ **Both conventions are now named in the clause that carries the figure**, and the four-part total
is kept as a separate, separately-labelled quantity instead of standing in for the statistic.

**Two supporting edits so the table agrees with the sentence:**

- `DRAFT-v1.md:232` — §3.3's w8 row. BEFORE the `statistic` cell read `12,327,096 / 11,957,784 /
  11,867,336` with no ratio and the row's last column read `24.5466× / 23.8422× / 23.6697×` with no
  convention. AFTER it reads `… — ⛔ **the statistic ALONE is 23.5121× / 22.8077× / 22.6351× the
  pin** (statistic ÷ pin)` and the last column gains `— four-part honest total ÷ pin`.
- `DRAFT-v1.md:1037` — §A's MF2 disposition row, which claimed MF2 CLOSED. AFTER it records that the
  second half was **not** closed in v1, names REFEREE-v1 §2 / N7 as the catch, and states the new
  clause.

**Arithmetic, re-derived here, not quoted:**

| quantity | seed 0 / 1 / 2 | ÷ 524,288 | source |
|---|---|---|---|
| statistic (part 2 TOTAL: covariance + running means + per-well counts) | 12,327,096 / 11,957,784 / 11,867,336 B | **23.5121× / 22.8077× / 22.6351×** | `/c/chlu/outputs/w13-seeds-and-tables/TABLES.md` T4 byte ledger, row `2. covariance / count ledger, TOTAL` |
| four-part honest total | 12,869,482 / 12,500,170 / 12,409,722 B | **24.5466× / 23.8422× / 23.6697×** | same table, rows `honest total` / `x THE 524,288 B PIN` (stored field — reproduced exactly by this spoke's own division) |
| w9 statistic share | 35,904 B of 1,642,752 B deployed | **2.19 %** | `/c/chlu/outputs/w13-seeds-and-tables/TABLES.md` T1 §1c-quater; `w9` ledger `/ledger/statistic_bytes` |

⚠ **CORRECTION TO THE REFEREE, recorded rather than passed on.** REFEREE-v1 §2 and N7 give the
statistic's ratio as **23.5117×**. Recomputed, `12,327,096 / 524,288 = 23.512069702148438` →
**23.5121×**. The referee's figure is a rounding slip in the 4th decimal. The draft now carries
**23.5121×**. ⭐ At the two decimals the task file names (**23.51×**) nothing changes.

---

## MF14 part (i) — the BLIND columns move INTO the maze main-text table

**Referee's charge** (REFEREE-v1 §2): *"§6.1's printed table has five columns … and none of them is
a BLIND column. The BLIND-SEQUENTIAL numbers appear in §6.3, a separate sub-section, and
⛔ BLIND-FACTOR's values appear NOWHERE in the draft."* Confirmed independently: `grep -c
"0.4762" DRAFT-v1.BEFORE.md` = **0**.

**BEFORE** — `DRAFT-v1.md` §6.1, five columns:

| row | CLU | best learned baseline | lookup (matched bytes) | `random_atoms` (C.2) | pipeline ceiling |
|---|---|---|---|---|---|
| 2 unseen pair | 0.1279 (0.1590 / 0.0597 / 0.1651) | **0.1840** (transformer) | 0.1565 | 0.1414 | 0.6319 |

**AFTER** — `DRAFT-v1.md:665-671`, seven columns; the two new ones and their measured cells:

| row | **BLIND-SEQUENTIAL** (ref.) | **BLIND-FACTOR** (ref.) |
|---|---|---|
| 1 seen | ⛔ **n/a — not a composition row; no blind block** | ⛔ **n/a — not a composition row; no blind block** |
| 2 unseen pair | **0.9037** (0.9077 / 0.8905 / 0.9128) | **0.4762** (0.4564 / 0.4677 / 0.5046) |
| 3 n-way, CONJUNCTION | **0.9001** (0.9229 / 0.8933 / 0.8841) | **0.4720** (0.4447 / 0.4665 / 0.5048) |
| 3 n-way, PER ROUTE | **0.9001** (0.9229 / 0.8933 / 0.8841) | **0.4720** (0.4447 / 0.4665 / 0.5048) |
| 4 budget interpolation | ⛔ **DECLARED NOT MEASURED — not a blank, not a zero** | ⛔ **DECLARED NOT MEASURED — not a blank, not a zero** |

**Provenance — read by this spoke straight out of the three scored JSONs, not out of TABLES.md:**
`/c/chlu/outputs/w9-maze/runs/scored1_seed0_test.json`,
`/c/chlu/outputs/w9-maze/runs/scored2_seed1_test.json`,
`/c/chlu/outputs/w9-maze/runs/scored3_seed2_test.json`, at
`/env_columns/{unseen_pair,nway}/blind/BLIND-SEQUENTIAL(key order from A)/validity` and
`/env_columns/{unseen_pair,nway}/blind/BLIND-FACTOR    (A x B local only)/validity`.
Values reproduce `/c/chlu/outputs/w13-seeds-and-tables/TABLES.md` T1 §1c cell for cell.
`/env_columns` carries exactly **four** keys — `unseen_pair`, `nway`, `fragment`,
`fragment_cross_maze` — on every scored seed, which is MF14 part (ii)'s evidence, re-verified.

**Supporting edits:**
- `DRAFT-v1.md:673-680` — the caption that reads the two new columns (below).
- `DRAFT-v1.md:708-721` — the old *"MF14 — the caveat columns come into the MAIN table"* instruction
  becomes a statement that they **are** in the table, and the DECLARED GAP is widened from row 4 to
  **rows 1 and 4**, each with its own reason (unsatisfiable vs inapplicable).
- `DRAFT-v1.md:781-790` — §6.3 retitled *"WHY the caveat columns read as they do"*; it no longer
  introduces the numbers (they are upstairs) and gains BLIND-FACTOR's CMAZE-1 contrast,
  **0.2350 (0.1850 / 0.2500 / 0.2700)** (`/c/chlu/outputs/w13-seeds-and-tables/TABLES.md` T3),
  so both blind references are shown to roughly double under the request layer.
- `DRAFT-v1.md:1049` — §A's MF14 row records that part (i) was **not** closed in v1.

### ⛔ THE THING THE TASK FILE ASKED ME TO REPORT AND NOT SOFTEN

**Yes — with the BLIND columns beside the claim, the maze section reads worse.** Measured, on the
same held-out TEST rows, under the same exact integer judge, on the same three seeds:

| row | CLU | best learned baseline | BLIND-SEQUENTIAL | BLIND-FACTOR | SEQ ÷ CLU | FACTOR ÷ CLU |
|---|---|---|---|---|---|---|
| 2 unseen pair | 0.1279 | 0.1840 | **0.9037** | **0.4762** | **7.07×** | **3.72×** |
| 3 n-way (conj.) | 0.0284 | 0.0493 | **0.9001** | **0.4720** | **31.7×** | **16.6×** |

⛔ **Both request-ignoring references beat the CLU and beat the best learned baseline on both
composition rows.** ⛔ I did not move them back, did not relegate them to a footnote and did not
soften the caption.

⭐ **What it actually costs the paper, stated exactly.** It does **not** cost a result, because §6.1
already claims none: BAR 3 fails and *"under the hub's own pre-registration NO RESULT ROW IS CLAIMED
for the CLU arm here"* was already in the draft. What it costs is the *reading*: a policy that never
reads the request scores 0.90, so **this split does not test composition at all** — which is the
mechanism §6.3 already measured (1,242 of 1,449 optimal routes pick up exactly one key, so rule B
has no key ORDER to trap). ⭐ The statement the paper can make is therefore **narrower and harder to
attack** than *"the CLU fails on a route task"*: **the environment's own tripwire fires before the
arm is reached.** ⛔ Neither column may be quoted as a baseline the CLU lost a fair comparison to —
they are class `reference` in the artefact, not arms — and neither may be dropped.

---

## N5 — one ratio against three totals, and two ratios for one object

**BEFORE** — `DRAFT-v1.md` §3.3, K\* row (pre-edit line 216), last column: **`4.0897×`**, a single
value beside a three-seed total. §0.2 bullet 1 repeated it.

**AFTER** — `DRAFT-v1.md:233-234`, the row splits in two and the ratio becomes per-seed:

| leg | address book | honest total | × the 524,288 B pin |
|---|---|---|---|
| **the K\* = 32 object (T7/T9), TEST, 3 seeds** | 1,060,864 / 1,054,104 / 1,058,264 | 2,144,178 / 2,137,418 / 2,141,578 | **4.0897× / 4.0768× / 4.0847×** |
| **the K\* = 32 object, ⚠ the VAL pin leg (T12's K = 32 row)** | 1,059,824 | 2,143,138 | **4.0877× — ⚠ val, and labelled val wherever it appears** |

Plus, at `DRAFT-v1.md:242-250`, the N5 note: T7 and T9 are **one-seed** tables (`seeds (n) | 1`,
seed id 0) whose 4.0897× is seed 0 alone; `w12b .../FINDINGS.md` §4 prints **three totals and no
ratio**; parts 1, 2 and 4 are constant across seeds and only part 3 varies.

**Every other locus of the same defect, closed in the same pass:**

| line | BEFORE | AFTER |
|---|---|---|
| `:64-69` (§0.2 bullet 1) | *"**4.0897×** on the K\* object (`TABLES.md` T7 byte ledger)"* | *"**4.0897× / 4.0768× / 4.0847×** per TEST seed … T7 byte ledger is **one seed**, id 0 … T12's **4.0877×** is the **val** pin leg, ⚠ a fourth number and not a fourth seed"*; and the w8 row gains *"of which the **statistic alone** is **23.5121× / 22.8077× / 22.6351×**"* |
| `:592` (§5.2 table) | `**4.0877×**` | `**4.0877× ⚠ val**` |
| `:596-601` (§5.2 caption) | — | new N5 note: every cell of T12 is the **val** sweep leg; §3.3's three TEST seeds are 4.0897× / 4.0768× / 4.0847×; ⛔ the four numbers are one object on two conditions and never a seed spread |
| `:1039` (§A MF3 row) | *"**4.0897×** (K\*)"* | per-seed TEST triple, T7's one-seed status flagged, T12's val leg flagged, both conventions named |
| `:1104` (§B K19) | *"7.8545× → 3.9942× at K = 24 → 4.0877×"* | same, `— ⚠ **the VAL sweep leg throughout; K\*'s three TEST seeds are 4.0897× / 4.0768× / 4.0847×**` |
| `:1276` (§G C16) | *"the honest total is 7.8545× → 4.0877×, U-shaped"* | same, `— ⚠ **all VAL** (N5); the K\* **TEST** totals are **4.0897× / 4.0768× / 4.0847×**` |

**Arithmetic, re-derived here:** 2,144,178 / 524,288 = **4.0897**; 2,137,418 / 524,288 = **4.0768**;
2,141,578 / 524,288 = **4.0847**; 2,143,138 / 524,288 = **4.0877**.
**Sources:** three-seed totals `/c/chlu/outputs/w12b-nr6-and-kstar/FINDINGS.md` §4 (table
*"The K\* = 32 four-part ledger across seeds"*, rows `0`, `1`, `2` and `(val pin)`);
one-seed ledger and its stored 4.0897 `/c/chlu/outputs/w13-seeds-and-tables/TABLES.md` T7
(*"Byte ledger — … Seed 0, K\* = 32, TEST"*, and the seed-count block
*"PRINTED AS A COLUMN … n = 1 (seed 0). Every row of T7 is a 1-seed row."*);
val leg `/c/chlu/outputs/w13-seeds-and-tables/TABLES.md` T12 row `K = 32` (`4.0877`).

---

## N6 — two different objects both called "K = 5"

**BEFORE** — `DRAFT-v1.md` §4.4 (pre-edit line 458): *"At **K = 5**, over 18 fitted legs: the single
design costs **12,523,008 – 13,041,864 B = 23.886× – 24.875×** the 524,288 B reference"*; and §5.1
(pre-edit line 536): *"at K = 5 composition holds … at an honest four-part ledger of **4,118,034 B =
7.8545×** the pin"*. Two byte totals **3.1× apart**, 80 lines apart, no cross-reference.

**AFTER** — both objects are **named**, and each points at the other:

- `DRAFT-v1.md:486-496` (§4.4) now opens: ⛔⛔ *"**NAME THE OBJECT: this is K = 5 (MI partition,
  full-square)** — the w12 ladder's `DowndatableStore` on the label-MI `factor` partition, well
  widths 777/936/216/60/59, statistic priced as the FULL SQUARE. ⛔ It is **NOT** the K = 5 of
  §5.1/§5.2, which is **K = 5 (ward, triangle)** …"* — with the 3.1× gap attributed to **both**
  convention and partition, and the note that **C20 moves both at once and so reads as a 3.6×
  convention effect when the convention alone is ≈2×**.
- `DRAFT-v1.md:576` (§5.1) carries the mirror clause naming **K = 5 (ward, triangle)** and pointing
  back at §4.4 and C20.
- `DRAFT-v1.md:78` (§0.2 bullet 3) and `:1097` (§B K12) gain the same two names, so no claim
  register entry can be lifted without them.

**Sources:** `/c/chlu/outputs/w12-composition-ladder/BYTES.json` `/summary/single_design_K5` →
`{n_legs: 18, total_bytes_min: 12523008, total_bytes_max: 13041864, times_reference_pin_min:
23.8857, times_reference_pin_max: 24.8754}` (read directly by this spoke);
`/c/chlu/outputs/w13-seeds-and-tables/TABLES.md` T12 K = 5 row (4,118,034 B = 7.8545×);
`/c/chlu/outputs/w12b-nr6-and-kstar/KSTAR-LADDER.json` `/legs/*/bridge_K5/0/ledger_four_part`;
correction **C20** at `DRAFT-v1.md:1280`.

---

## N7 — the four-part honest total called "a statistic"

⭐ **Same clause as MF2, and it closes with it** — see the MF2 section above for the before → after
and the quoted lead sentence. `DRAFT-v1.md:1340-1345`. The referee's own smallest fix was eleven
words; the landed clause is longer because it names **both** conventions explicitly rather than
leaving *"23.5× the store"* to be read as a third convention.

---

## N8 — three different statements about where the K = 5 loss is printed

**BEFORE** — `DRAFT-v1.md` §11 item 19 (pre-edit line 930):

> 19. ⭐ **NEW — the composition verdict is K-conditional**, and the K = 5 reading the programme
>     published first is **withdrawn from main text** and carried in the appendix **with its sign
>     stated as a property of K = 5, not of the store** (§4.2). *Source:*
>     `/c/chlu/outputs/w12b-nr6-and-kstar/FINDINGS.md` §13 item 1.

**AFTER** — `DRAFT-v1.md:1005-1016`: the withdrawal clause is **deleted** and replaced with the
governing reading —

> ⛔⛔ **N8 — CORRECTED HERE, AND THE CORRECTION IS THE POINT.** v1 of this draft said the K = 5
> reading is *"withdrawn from main text"* and carried in the appendix. It is **not**, and no ruling
> says it is: that sentence was lifted from `…/w12b-nr6-and-kstar/FINDINGS.md` §13 item 1, whose own
> heading is **"WHAT THE HUB IS ASKED TO RULE ON"** — ⛔ a recommendation, converted into a
> statement, contradicting this draft's own §C. ⭐ **The governing reading:** the Head's one-K ruling
> governs **which K the composition CLAIM is stated at**, not **which K's NUMBERS are printed**. The
> **K = 5 margins are printed in MAIN TEXT beside K\*'s** — §4.2's ladder table and §C **Table 5
> (main)** — and T8's *"Placement: APPENDIX"* applies to **the full 13-arm table**, not to the
> verdict. ⛔ **Appendixing the programme's largest negative would be overclaiming by omission; this
> draft does not authorise it.**

**Verified, not assumed:** `/c/chlu/outputs/w12b-nr6-and-kstar/FINDINGS.md` §13's heading is indeed
*"WHAT THE HUB IS ASKED TO RULE ON"*; `DRAFT-v1.md` §C (`:1129`) does slot *"**Table 5 (main)** |
the **K = 5 vs K\*** ladder verdict with the bridge column"*; §4.2 (`:424-430`) does print the K = 5
per-seed margins in a main-text table. ⭐ After the edit the draft says **one** thing in all three
places.

---

## §A — the disposition header

`DRAFT-v1.md:1026` BEFORE: *"⭐ **All sixteen CLOSED.**"* — which REFEREE-v1's **SF-a** correctly
called false against v1 as written.
AFTER: *"All sixteen CLOSED — fourteen by w13/draft-revise, and MF2 and MF14 by w15/table-fixes
after `w13/referee-2` returned both as ⚠ PARTIAL"*, with SF-a explicitly discharged and pointed at
this file. ⛔ The claim is now true and carries the history that makes it checkable.

---

## HOW TO REPRODUCE OR REVERT

```bash
cd /c/chlu/worktrees/CHLU-w15-table-fixes
git diff 148f3b7 -- w13_draft_revise/DRAFT-v1.md      # the whole change
python w15_table_fixes/apply_table_fixes.py            # idempotent; "already applied" x22
git checkout 148f3b7 -- w13_draft_revise/DRAFT-v1.md   # revert
```
