#!/usr/bin/env python3
"""w15/table-fixes -- the zero-compute must-fix pass on DRAFT-v1.md.

Lands MF2, MF14(i), N5, N6, N7, N8 as exact string replacements so the change is
auditable line by line.  ZERO scored runs, ZERO compute, no artefact regenerated,
`verify_tables.py` NOT touched.

Every number written by this script was re-derived from its primary artefact
before the edit; see /c/chlu/outputs/w15-table-fixes/LOG.md.

Idempotent: re-running on an already-patched file reports "already applied".
"""
import sys
import pathlib

DRAFT = pathlib.Path(__file__).resolve().parent.parent / "w13_draft_revise" / "DRAFT-v1.md"

EDITS = []  # (item, old, new, mark)

# `mark` is a string present ONLY in `new`.  It is how the script recognises an
# edit it has already made.  It is mandatory because several `new` blocks keep
# their `old` text verbatim and merely prepend to it, so "does `old` still
# match?" is NOT a safe applied/not-applied test -- it would double-insert.


def E(item, old, new, mark=None):
    if mark is None:
        # Safe only when `new` does not itself contain `old`; otherwise a second
        # run would match `old` again inside the text this script just wrote and
        # double-insert.  Those edits must pass an explicit `mark`.
        assert old not in new, item + ": new contains old, an explicit mark is required"
        mark = new
    assert mark in new, item
    assert mark not in old, item
    EDITS.append((item, old, new, mark))


# ---------------------------------------------------------------- MF14 part (i)
# The BLIND-SEQUENTIAL and BLIND-FACTOR reference columns move INTO the 6.1
# main-text table.  Source: /c/chlu/outputs/w9-maze/runs/scored{1,2,3}_seed{0,1,2}_test.json
# /env_columns/{unseen_pair,nway}/blind/*/validity ; tabulated at
# /c/chlu/outputs/w13-seeds-and-tables/TABLES.md T1 section 1c.
E("MF14(i)",
"""| row | CLU | best learned baseline | lookup (matched bytes) | `random_atoms` (C.2) | pipeline ceiling |
|---|---|---|---|---|---|
| 1 seen | **1.0000** | 0.9865 (transformer) | **1.0000** | 0.9601 | 1.0000 |
| 2 unseen pair | 0.1279 (0.1590 / 0.0597 / 0.1651) | **0.1840** (transformer) | 0.1565 | 0.1414 | 0.6319 |
| 3 n-way, CONJUNCTION | 0.0284 (0.0383 / 0.0469 / **0.0000**) | **0.0493** (lstm) | 0.0159 | 0.0195 | 0.5255 (per route) |
| 3 n-way, PER ROUTE | 0.0852 | **0.1246** (transformer) | 0.0946 | 0.0693 | 0.5255 |
| 4 budget interpolation | 0.0996 | **0.1430** (transformer) | 0.1122 | 0.0905 | 0.5573 |
""",
"""| row | CLU | best learned baseline | lookup (matched bytes) | `random_atoms` (C.2) | **BLIND-SEQUENTIAL** (ref.) | **BLIND-FACTOR** (ref.) | pipeline ceiling |
|---|---|---|---|---|---|---|---|
| 1 seen | **1.0000** | 0.9865 (transformer) | **1.0000** | 0.9601 | ⛔ **n/a — not a composition row; no blind block** | ⛔ **n/a — not a composition row; no blind block** | 1.0000 |
| 2 unseen pair | 0.1279 (0.1590 / 0.0597 / 0.1651) | **0.1840** (transformer) | 0.1565 | 0.1414 | **0.9037** (0.9077 / 0.8905 / 0.9128) | **0.4762** (0.4564 / 0.4677 / 0.5046) | 0.6319 |
| 3 n-way, CONJUNCTION | 0.0284 (0.0383 / 0.0469 / **0.0000**) | **0.0493** (lstm) | 0.0159 | 0.0195 | **0.9001** (0.9229 / 0.8933 / 0.8841) | **0.4720** (0.4447 / 0.4665 / 0.5048) | 0.5255 (per route) |
| 3 n-way, PER ROUTE | 0.0852 | **0.1246** (transformer) | 0.0946 | 0.0693 | **0.9001** (0.9229 / 0.8933 / 0.8841) | **0.4720** (0.4447 / 0.4665 / 0.5048) | 0.5255 |
| 4 budget interpolation | 0.0996 | **0.1430** (transformer) | 0.1122 | 0.0905 | ⛔ **DECLARED NOT MEASURED — not a blank, not a zero** | ⛔ **DECLARED NOT MEASURED — not a blank, not a zero** | 0.5573 |

⛔⛔ **MF14(i) — READ THE TWO BLIND COLUMNS BESIDE THE CLAIM, WHICH IS WHY THEY ARE HERE.** They are
**REFERENCE** columns in the artefact's own class field, not arms: **BLIND-SEQUENTIAL** = *key order
from A*, **BLIND-FACTOR** = *A × B local only* — request-ignoring policies scored by the **same exact
integer judge, on the same held-out TEST rows, on the same three seeds**. On rows 2 and 3 **both of
them beat the CLU and both beat the best learned baseline**: BLIND-SEQUENTIAL by **7.1×** and
**31.7×** the CLU's own score, BLIND-FACTOR by **3.7×** and **16.6×**.
⛔ **This makes the maze section's negative HARDER, not softer, and it is printed that way on
purpose:** on this split a policy that never reads the request scores 0.90, so ⛔ **the split does
not test composition at all** — which is exactly what §6.3's measured cause says (1,242 of 1,449
optimal routes pick up exactly one key, so rule B has no key ORDER to trap). ⭐ The statement the
paper can make is therefore **stronger and narrower** than *"the CLU fails here"*: **the
environment's own tripwire fires before the arm is reached.** ⛔ Neither column may be quoted as a
baseline the CLU lost a fair comparison to, and ⛔ neither may be dropped to make the row read
better. *Per-seed triples printed on every measured cell (SF12).*
""")

E("MF14(i)",
"""⛔⛔ **MF14 — the caveat columns come into the MAIN table**, per the charter: BLIND-SEQUENTIAL and
BLIND-FACTOR for rows 2 and 3. And **row 4 carries a DECLARED GAP, not a blank:**

> ⛔ **Row 4 has no BLIND column on any seed.** `/env_columns` carries exactly four keys —
> `unseen_pair`, `nway`, `fragment`, `fragment_cross_maze`. The charter rule is **unsatisfiable** for
> that row. **DECLARED NOT MEASURED, never a blank and never a zero.**

*Source:* `TABLES.md` T1 §1c and §1c-ter; correction **C13**.""",
"""⭐⭐ **MF14 is now CLOSED IN BOTH PARTS.** Part (i): BLIND-SEQUENTIAL and BLIND-FACTOR for rows 2
and 3 are **columns of the table above**, not a later sub-section — a reader meets the caveat and the
claim in one glance, which is the whole point of a blind column. Part (ii): rows 1 and 4 carry a
**DECLARED GAP, not a blank:**

> ⛔ **Rows 1 and 4 have no BLIND column on any seed.** `/env_columns` carries exactly four keys —
> `unseen_pair`, `nway`, `fragment`, `fragment_cross_maze` — verified directly on all three scored
> JSONs by w15/table-fixes. The charter rule is **unsatisfiable** for row 4 (a composition row) and
> **inapplicable** to row 1 (a seen row, not a composition row). Row 4 is therefore **DECLARED NOT
> MEASURED, never a blank and never a zero**; row 1 is marked n/a with its reason.

*Sources:* `/c/chlu/outputs/w13-seeds-and-tables/TABLES.md` T1 §1c and §1c-ter (and the in-tree
`w12r_artefact_mustfix/TABLES.md` §1c, identical cells); primaries
`/c/chlu/outputs/w9-maze/runs/scored1_seed0_test.json`, `.../scored2_seed1_test.json`,
`.../scored3_seed2_test.json` at
`/env_columns/{unseen_pair,nway}/blind/BLIND-SEQUENTIAL(key order from A)/validity` and
`.../BLIND-FACTOR    (A x B local only)/validity`; correction **C13**.""")

# 6.3 no longer INTRODUCES the column -- it now reads the column already printed above.
E("MF14(i)",
"""### 6.3 ⛔ The caveat column, measured on THIS split — and MF11

BLIND-SEQUENTIAL on CMAZE-2 is **0.9077 / 0.8905 / 0.9128 (mean 0.9037)** on row 2 and **0.9229 /
0.8933 / 0.8841 (mean 0.9001)** on row 3 — ⛔ **not** the published CMAZE-1 **0.4980**.""",
"""### 6.3 ⛔ WHY the caveat columns read as they do — and MF11

⭐ **The numbers themselves now live in §6.1's main table (MF14(i)); this section carries only their
mechanism and the CMAZE-1 contrast.** BLIND-SEQUENTIAL on CMAZE-2 is **0.9077 / 0.8905 / 0.9128
(mean 0.9037)** on row 2 and **0.9229 / 0.8933 / 0.8841 (mean 0.9001)** on row 3 — ⛔ **not** the
published CMAZE-1 **0.4980**. ⭐ **And BLIND-FACTOR moves the same way and must be quoted with it:**
**0.4762** (row 2) / **0.4720** (row 3) on CMAZE-2 against **0.2350 (0.1850 / 0.2500 / 0.2700)** at
the CMAZE-1 harness tier (`/c/chlu/outputs/w13-seeds-and-tables/TABLES.md` T3) — ⛔ **both blind
references roughly double under the request layer, so the rise is a property of the LAYER and not of
one column's rule.**""")

# ------------------------------------------------------------------- MF2 / N7
# The statistic's own cost, with its convention named, in the LEAD sentence.
E("MF2/N7",
"""retained data and items never kept **[K1, K2]** — at the cost of a statistic that is **2.19 %** of
> the deployed read in one environment and **24.55× the byte pin** in the other **[K4]**; that the two""",
"""retained data and items never kept **[K1, K2]** — at the cost of a retained sufficient statistic
> whose own bytes are **2.19 %** of the deployed read in one environment (**35,904 B of 1,642,752 B
> deployed**; convention: statistic ÷ deployed read) and **23.5121× the 524,288 B store pin** in the
> other (**12,327,096 B ÷ 524,288 B**; convention: statistic ÷ pin, the statistic alone), which takes
> the never-netted four-part honest total — atoms + statistic + request-count table + coupling/holes
> — to **24.5466× the pin** **[K4]**; that the two""")

# ------------------------------------------------------------------------- N5
E("N5",
"""| **the K\\* = 32 object (T7/T9)** | 516,096 | 560,608 | 1,060,864 / 1,054,104 / 1,058,264 | 6,610 | 2,144,178 / 2,137,418 / 2,141,578 | **4.0897×** |""",
"""| **the K\\* = 32 object (T7/T9), TEST, 3 seeds** | 516,096 | 560,608 | 1,060,864 / 1,054,104 / 1,058,264 | 6,610 | 2,144,178 / 2,137,418 / 2,141,578 | **4.0897× / 4.0768× / 4.0847×** |
| **the K\\* = 32 object, ⚠ the VAL pin leg (T12's K = 32 row)** | 516,096 | 560,608 | 1,059,824 | 6,610 | 2,143,138 | **4.0877× — ⚠ val, and labelled val wherever it appears** |""")

E("N5",
"""`/c/chlu/outputs/w12b-nr6-and-kstar/FINDINGS.md` §4 (the three-seed K\\* ledger — ⭐ the task file's
535,784 1× param pin holds across all three seeds to within **0.32 %**, and only part 3 varies with
the seed).""",
"""`/c/chlu/outputs/w13-seeds-and-tables/TABLES.md` T7 byte ledger and T12 K = 32 row;
`/c/chlu/outputs/w12b-nr6-and-kstar/FINDINGS.md` §4 (the three-seed K\\* ledger — ⭐ the task file's
535,784 1× param pin holds across all three seeds to within **0.32 %**, and only part 3 varies with
the seed).
⛔⛔ **N5 — ONE RATIO MAY NOT STAND FOR THREE TOTALS, AND ONE OBJECT MAY NOT CARRY TWO RATIOS
UNLABELLED.** `TABLES.md` **T7 and T9 are ONE-SEED tables** (`seeds (n) | 1`, seed id 0) and their
**4.0897×** is `2,144,178 / 524,288` for **seed 0 alone**; the three-seed totals come from
`w12b .../FINDINGS.md` §4, which prints **three totals and no ratio**. This spoke divided them:
**4.0897× / 4.0768× / 4.0847×**, printed above per seed. ⭐ **Parts 1, 2 and 4 are constant across
seeds — they are arithmetic on the partition shape — and only part 3, the address book, varies**
(the source's own sentence). ⚠ **And the 4.0877× that §5.2 and T12 print for the same object is the
VAL pin leg, a fourth number and not a fourth seed**; it now carries the word **val** in both
places.""",
  mark="ONE RATIO MAY NOT STAND FOR THREE TOTALS")

E("N5",
"""   printed on the same page (MF3). The object that forgets exactly is **24.5466× / 23.8422× /
   23.6697×** the 524,288 B pin on the w8 leg (`TABLES.md` T4 byte ledger) and **4.0897×** on the K\\*
   object (`TABLES.md` T7 byte ledger). ⛔ Quoted with its overrun attached or not at all.""",
"""   printed on the same page (MF3). The object that forgets exactly is **24.5466× / 23.8422× /
   23.6697×** the 524,288 B pin on the w8 leg (`TABLES.md` T4 byte ledger — ⛔ of which the
   **statistic alone** is **23.5121× / 22.8077× / 22.6351×**; the two conventions are named wherever
   either is quoted) and **4.0897× / 4.0768× / 4.0847×** per TEST seed on the K\\* object
   (`/c/chlu/outputs/w13-seeds-and-tables/TABLES.md` T7 byte ledger is **one seed**, id 0 — the other
   two totals are `w12b .../FINDINGS.md` §4; T12's **4.0877×** is the **val** pin leg, ⚠ a fourth
   number and not a fourth seed). ⛔ Quoted with its overrun attached, its convention named and its
   seed count printed, or not at all.""")

E("N5",
"""| ⭐ **the HONEST four-part total** | **7.8545×** | **3.9942× ← minimum** | **4.0877×** | ⛔ **U-SHAPED** |""",
"""| ⭐ **the HONEST four-part total** | **7.8545×** | **3.9942× ← minimum** | **4.0877× ⚠ val** | ⛔ **U-SHAPED** |""")

E("N5",
"""⛔ **Any sentence quoting "7.48× → 2.07×" is quoting the three-part curve and has dropped part 3** —""",
"""⚠⚠ **N5 — every cell of this table is the VAL sweep leg, and its K\\* column is NOT the TEST row of
§3.3.** T12's K = 32 honest total is **2,143,138 B = 4.0877×** (address book 1,059,824 B, the val
pin); §3.3's three **TEST** seeds are 2,144,178 / 2,137,418 / 2,141,578 B = **4.0897× / 4.0768× /
4.0847×**. ⛔ The four numbers are one object measured on two conditions and must never be quoted as
a seed spread. *Source:* `/c/chlu/outputs/w13-seeds-and-tables/TABLES.md` T12 row `K = 32`;
`/c/chlu/outputs/w12b-nr6-and-kstar/FINDINGS.md` §4 row `(val pin)`.

⛔ **Any sentence quoting "7.48× → 2.07×" is quoting the three-part curve and has dropped part 3** —""",
  mark="every cell of this table is the VAL sweep leg")

# ------------------------------------------------------------------------- N6
E("N6",
"""At **K = 5**, over 18 fitted legs: the single design costs **12,523,008 – 13,041,864 B = 23.886× –
24.875×** the 524,288 B reference;""",
"""⛔⛔ **NAME THE OBJECT: this is K = 5 (MI partition, full-square)** — the w12 ladder's
`DowndatableStore` on the label-MI `factor` partition, well widths 777/936/216/60/59, statistic
priced as the FULL SQUARE. ⛔ It is **NOT** the K = 5 of §5.1/§5.2, which is **K = 5 (ward,
triangle)** — the crossover/bridge `CovContentStore` on the label-free `ward_balanced` cut, five
≈410-wide wells, statistic priced as the SYMMETRIC TRIANGLE. The two four-part totals differ by
**3.1×** (12,523,008–13,041,864 B here vs 4,118,034 B there) because they differ in **both**
convention **and** partition — ⛔ roughly **2× from the convention alone**, the rest from the
partition. ⚠ Correction **C20** in §G compares 12,284,920 B against 3,391,552 B and, because it moves
both at once, **reads as a 3.6× convention effect when the convention alone is ≈2×**. ⭐ **Use the two
names everywhere; a referee who meets two "K = 5" byte totals 80 lines apart with no cross-reference
reads a contradiction, and on the page that is what it is.** See §5.1 for the other object.

At **K = 5 (MI partition, full-square)**, over 18 fitted legs: the single design costs
**12,523,008 – 13,041,864 B = 23.886× – 24.875×** the 524,288 B reference;""")

E("N6",
"""| at K = 5 composition holds | **0.8918** vs the incumbent's **0.8941** — a tie — at an honest four-part ledger of **4,118,034 B = 7.8545×** the pin | `TABLES.md` T12 K = 5 row |""",
"""| at K = 5 composition holds | **0.8918** vs the incumbent's **0.8941** — a tie — at an honest four-part ledger of **4,118,034 B = 7.8545×** the pin. ⛔⛔ **NAME THE OBJECT: this is K = 5 (ward, triangle)** — the crossover/bridge `CovContentStore` on the label-free `ward_balanced` cut, five ≈410-wide wells, SYMMETRIC-TRIANGLE statistic. ⛔ It is **NOT** §4.4's **K = 5 (MI partition, full-square)**, whose four-part total is **12,523,008 – 13,041,864 B = 23.886× – 24.875× — 3.1× larger**, on a different partition **and** a different convention. See §4.4 and correction **C20** (⚠ which moves both at once and so reads as a 3.6× convention effect; the convention alone is ≈2×). | `/c/chlu/outputs/w13-seeds-and-tables/TABLES.md` T12 K = 5 row; `/c/chlu/outputs/w12b-nr6-and-kstar/KSTAR-LADDER.json` `/legs/*/bridge_K5/0/ledger_four_part` |""")

# ------------------------------------------------------------------------- N8
E("N8",
"""19. ⭐ **NEW — the composition verdict is K-conditional**, and the K = 5 reading the programme published
    first is **withdrawn from main text** and carried in the appendix **with its sign stated as a
    property of K = 5, not of the store** (§4.2). *Source:*
    `/c/chlu/outputs/w12b-nr6-and-kstar/FINDINGS.md` §13 item 1.""",
"""19. ⭐ **NEW — the composition verdict is K-conditional**, with its sign stated as a property of
    **K = 5, not of the store** (§4.2). ⛔⛔ **N8 — CORRECTED HERE, AND THE CORRECTION IS THE POINT.**
    v1 of this draft said the K = 5 reading is *"withdrawn from main text"* and carried in the
    appendix. It is **not**, and no ruling says it is: that sentence was lifted from
    `/c/chlu/outputs/w12b-nr6-and-kstar/FINDINGS.md` §13 item 1, whose own heading is **"WHAT THE HUB
    IS ASKED TO RULE ON"** — ⛔ a recommendation, converted into a statement, contradicting this
    draft's own §C. ⭐ **The governing reading:** the Head's one-K ruling governs **which K the
    composition CLAIM is stated at**, not **which K's NUMBERS are printed**. The **K = 5 margins are
    printed in MAIN TEXT beside K\\*'s** — §4.2's ladder table and §C **Table 5 (main)** — and T8's
    *"Placement: APPENDIX"* applies to **the full 13-arm table**, not to the verdict.
    ⛔ **Appendixing the programme's largest negative would be overclaiming by omission; this draft
    does not authorise it.** *Sources:* `/c/chlu/outputs/w12b-nr6-and-kstar/FINDINGS.md` §13 item 1
    (read as the open question it is); `/c/chlu/outputs/w13-seeds-and-tables/TABLES.md` T8 placement
    header; §4.2 and §C of this file.""")

# ------------------------------------------------------------------- Appendix A bookkeeping
E("MF2/MF14",
"""⭐ **All sixteen CLOSED. ⛔ Zero scored runs, zero fitted models, zero new measurements.**
The *how* column names the change and the artefact that licenses it.""",
"""⭐ **All sixteen CLOSED — fourteen by w13/draft-revise, and MF2 and MF14 by w15/table-fixes after
`w13/referee-2` returned both as ⚠ PARTIAL (REFEREE-v1 §2).** ⛔ **Zero scored runs, zero fitted
models, zero new measurements** in either pass. The *how* column names the change and the artefact
that licenses it; the two rows re-closed in w15 say so, and say what was still open.
⚠ **REFEREE-v1's SF-a is discharged with them:** its *"fourteen CLOSED, two PARTIAL"* was correct
against v1 as written, and is superseded by the edits recorded in
`/c/chlu/outputs/w15-table-fixes/TABLE-FIXES.md`.""")

E("MF2",
"""with the statistic's byte cost **in the same sentence** (§I). Evidence: statistic/content = **2.19 %** (w9, `/ledger/statistic_bytes` 35,904 of 1,642,752 deployed) and **2,261–2,348 %** (w8, `/cov_bytes_per_seed` 12,310,672 / 11,941,360 / 11,850,912 against a 524,288 B store). The w9 ledger's own note says it: the statistic is *"declared beside them, never netted in"*. |""",
"""with the statistic's byte cost **in the same sentence** (§I). ⚠ **RE-CLOSED by w15/table-fixes: as written in v1 the lead sentence printed `24.55×`, which is the FOUR-PART HONEST TOTAL and not the statistic** — the second half of MF2, caught by REFEREE-v1 §2 and N7. §I's lead sentence now prints the statistic's own cost **with its convention named in the same clause**: **23.5121× the 524,288 B store pin = 12,327,096 B ÷ 524,288 B, statistic ÷ pin**, and carries the four-part honest total to **24.5466×** separately. Evidence: statistic/content = **2.19 %** (w9, `/ledger/statistic_bytes` 35,904 of 1,642,752 deployed) and **2,261–2,348 %** (w8, `/cov_bytes_per_seed` 12,310,672 / 11,941,360 / 11,850,912 against a 524,288 B store). ⛔ The statistic's **full** ledger line (covariance + running means + per-well counts) is 12,327,096 / 11,957,784 / 11,867,336 B = **23.5121× / 22.8077× / 22.6351×** the pin — `/c/chlu/outputs/w13-seeds-and-tables/TABLES.md` T4 byte ledger, part 2 TOTAL. The w9 ledger's own note says it: the statistic is *"declared beside them, never netted in"*. |""")

E("MF14",
"""| **MF14** | row 4 is a main-text composition row with **no BLIND column in existence** | ⭐ **CLOSED** | §6.1: BLIND-SEQUENTIAL and BLIND-FACTOR for rows 2 and 3 move **into the main table** per the charter; row 4 carries **DECLARED NOT MEASURED — not a blank, not a zero**, with `/env_columns`' four keys named. `TABLES.md` §1c, §1c-ter; correction **C13**. |""",
"""| **MF14** | row 4 is a main-text composition row with **no BLIND column in existence**, and rows 2–3's BLIND columns sat outside the main table | ⭐ **CLOSED — both parts** | ⚠ **Part (ii) was closed in v1; part (i) was NOT, and is closed by w15/table-fixes.** In v1, §6.1's table had five columns and no BLIND column; BLIND-SEQUENTIAL lived in §6.3 and **BLIND-FACTOR appeared nowhere as a number** (REFEREE-v1 §2). §6.1's table now carries **BLIND-SEQUENTIAL** and **BLIND-FACTOR** as columns with per-seed triples, rows 1 and 4 marked n/a and **DECLARED NOT MEASURED** respectively. ⛔ **The move makes the maze negative read HARDER — both blind references outscore the CLU and the best learned baseline on rows 2 and 3 — and it is reported, not softened.** `/c/chlu/outputs/w13-seeds-and-tables/TABLES.md` T1 §1c, §1c-ter; primaries `/c/chlu/outputs/w9-maze/runs/scored{1,2,3}_seed{0,1,2}_test.json` `/env_columns/*/blind/*/validity`; correction **C13**. |""")


# ------------------------------------------------- the amendment banner (provenance)
E("banner",
"""> ⛔ **THE HEAD WRITES THE PROSE.** This file is structure + claim placement + evidence pointers.
> A claim with no pointer was **deleted, not flagged** (§D).""",
"""> ⛔ **THE HEAD WRITES THE PROSE.** This file is structure + claim placement + evidence pointers.
> A claim with no pointer was **deleted, not flagged** (§D).

> ⭐⭐ **AMENDED 2026-09-22 by w15 `spoke/table-fixes`, branch `w15/table-fixes`** — six
> bookkeeping items returned by `w13/referee-2` (`/c/chlu/outputs/w13-referee-2/REFEREE-v1.md`):
> **MF2** (the statistic's own byte cost, with its convention named, in the lead sentence — §I),
> **MF14 part (i)** (the BLIND-SEQUENTIAL and BLIND-FACTOR columns moved INTO the maze main-text
> table — §6.1), and **N5 / N6 / N7 / N8** (§0.2, §3.3, §4.4, §5.1, §5.2, §11.19, §A).
> ⛔ **Zero scored runs, zero compute, no artefact regenerated, `verify_tables.py` NOT touched, and
> `w10/draft-v0` and `w10/referee` NOT merged.** The edit is reproducible from
> `w15_table_fixes/apply_table_fixes.py` and itemised before→after in
> `/c/chlu/outputs/w15-table-fixes/TABLE-FIXES.md`. ⚠ One substantive consequence, reported and not
> softened: with the BLIND columns in §6.1's table, **both request-ignoring references outscore both
> the CLU and the best learned baseline on rows 2 and 3** — see §6.1.""",
  mark="AMENDED 2026-09-22 by w15")

# ------------------------------------------------- MF2, in the byte table as well
E("MF2",
"""| **w8/downdate (Shapes3D)** | 524,288 | 12,327,096 / 11,957,784 / 11,867,336 | 11,488 (request-count table) | 5,288 + 1,322 | **12,869,482 / 12,500,170 / 12,409,722** | ⛔⛔ **24.5466× / 23.8422× / 23.6697×** |""",
"""| **w8/downdate (Shapes3D)** | 524,288 | 12,327,096 / 11,957,784 / 11,867,336 — ⛔ **the statistic ALONE is 23.5121× / 22.8077× / 22.6351× the pin** (statistic ÷ pin) | 11,488 (request-count table) | 5,288 + 1,322 | **12,869,482 / 12,500,170 / 12,409,722** | ⛔⛔ **24.5466× / 23.8422× / 23.6697×** — four-part honest total ÷ pin |""",
  mark="the statistic ALONE is 23.5121")



# --------------------------- N5 / N6, the same defect at its remaining loci
# The referee named the worst instances.  The charter rule is general: every byte
# figure travels with its convention named and its seed count printed.  These
# four are the remaining places in the file where it did not.
E("N5",
"""never netted: **24.5466× / 23.8422× / 23.6697×** (w8), **2.19 %** statistic share (w9), **4.0897×** (K\*), plus the **16,384 B declared residual**.""",
"""never netted: **24.5466× / 23.8422× / 23.6697×** four-part ÷ pin (w8, of which the statistic alone is **23.5121× / 22.8077× / 22.6351×**), **2.19 %** statistic ÷ deployed read (w9), **4.0897× / 4.0768× / 4.0847×** four-part ÷ pin on the three TEST seeds of K\* (⚠ T7 itself is **one seed**; T12's **4.0877×** is the **val** pin leg), plus the **16,384 B declared residual**.""",
  mark="four-part ÷ pin on the three TEST seeds of K")

E("N5",
"""| **K19** | ⛔ The honest four-part byte curve is **U-SHAPED** (7.8545× → **3.9942× at K = 24** → 4.0877×);""",
"""| **K19** | ⛔ The honest four-part byte curve is **U-SHAPED** (7.8545× → **3.9942× at K = 24** → 4.0877× — ⚠ **the VAL sweep leg throughout; K\*'s three TEST seeds are 4.0897× / 4.0768× / 4.0847×**);""",
  mark="the VAL sweep leg throughout; K")

E("N5",
"""| **C16** | "7.48× → 2.07×, monotonically" | that is parts 1+2+4; the honest total is **7.8545× → 4.0877×, U-shaped, minimum 3.9942× at K = 24** |""",
"""| **C16** | "7.48× → 2.07×, monotonically" | that is parts 1+2+4; the honest total is **7.8545× → 4.0877×, U-shaped, minimum 3.9942× at K = 24** — ⚠ **all VAL** (N5); the K\* **TEST** totals are **4.0897× / 4.0768× / 4.0847×** |""",
  mark="all VAL** (N5); the K")

E("N6",
"""| **K12** | ⭐⭐ At K = 5 the single design at **23.886×–24.875×** and `pca_tensor` at **1.358×–1.372×** score **identically to four decimals at every rung and every seed** ⇒ the four-part ledger buys **deletion, not composition** ⚠ **at K = 5** |""",
"""| **K12** | ⭐⭐ At **K = 5 (MI partition, full-square)** the single design at **23.886×–24.875×** and `pca_tensor` at **1.358×–1.372×** score **identically to four decimals at every rung and every seed** ⇒ the four-part ledger buys **deletion, not composition** ⚠ **at K = 5, and ⛔ NOT the K = 5 (ward, triangle) object of §5.1, whose four-part total is 7.8545×** |""",
  mark="NOT the K = 5 (ward, triangle) object of")

E("N6",
"""3. ⭐⭐ **At K = 5, `pca_tensor` at 1.358×–1.372× the reference scores IDENTICALLY — to four decimals
   at every rung and every seed — to the single design at 23.886×–24.875×**""",
"""3. ⭐⭐ **At K = 5 (MI partition, full-square), `pca_tensor` at 1.358×–1.372× the reference scores
   IDENTICALLY — to four decimals at every rung and every seed — to the single design at
   23.886×–24.875×** (⛔ **a different object from the K = 5 (ward, triangle) of §5.1**, N6)""",
  mark="a different object from the K = 5 (ward, triangle) of")



def main():
    text = DRAFT.read_text(encoding="utf-8")
    original = text
    applied, missing = [], []
    for item, old, new, mark in EDITS:
        if mark in text:
            applied.append(item + " (already applied)")
            continue
        n = text.count(old)
        if n == 1:
            text = text.replace(old, new, 1)
            applied.append(item)
        else:
            missing.append((item, n, old.splitlines()[0][:90]))
    if missing:
        for item, n, head in missing:
            print("NO-MATCH (%d hits) [%s] %s" % (n, item, head), file=sys.stderr)
        sys.exit(2)
    if text != original:
        DRAFT.write_text(text, encoding="utf-8")
    print("applied %d edits to %s" % (len(applied), DRAFT))
    for a in applied:
        print("  -", a)


if __name__ == "__main__":
    main()
