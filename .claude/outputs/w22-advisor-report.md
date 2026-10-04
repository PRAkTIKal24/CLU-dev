# WAVE 22 — REPORT FOR THE MAIN ADVISOR
**Compiled by the ICLR-long Hub, 2026-09-25. Seven spoke-passes, machine `batmac`. ZERO scored runs, ZERO compute beyond `pdflatex`/`bibtex` and figure rendering.**
Code landed: `w22/paper-visuals b871a4d` · `w22/figures-redraw2 102d1de` · `w22/figures3 cb5e0ac` · `w22/draft-rulings e9370a2` · `w22/registry6 8db7d33` · `w22/seeds5 3bb83da`.
⛔ **No pre-registration governs this wave** — the Head banned pre-declarations 2026-09-24. Everything measured is printed.
⭐ **This wave changed no result. It changed what the paper SAYS, and it closed two defects that would have been found by a referee.**

---

## 0. ⭐⭐ THE ANSWER

1. ⭐⭐ **The paper's headline is now the measured one.** `draft-rulings` retired the **+39…+49** comparator numbers and rebuilt the claims around **≈ +7**. Proven gone from **rendered** text, with a positive control (§2).
2. ⭐⭐ **The Head found a figure defect; it is measured and fixed.** CMAZE-T panels drew the resulting **state** while the judge scores the **fact vector** — **33,412** arm-pairs rendered identically with opposite marks. **Now 0.** ⛔ No number was ever affected.
3. ⭐⭐ **The oldest outstanding verification gap is closed.** `bar-d-score` / `coverage-control` / `lace-poe-rows` had been **in the draft but unverified by the cell register** since w15, declared out of brief twice. **199/199 numbers now traced** draft ↔ FINDINGS ↔ JSON.
4. ⭐ **The paper got shorter while gaining content:** **51/52 → 48/49 pp**.
5. ⛔ **Three placeholders remain; two are permanently blocked** and need a ruling (§5).

---

## 1. The scoreboard

| spoke | result |
|---|---|
| `paper-visuals` | One indexed folder: **17/17** draft figures present as PNG **and** PDF, proven twice (including by compiling with `figures/` swapped for it). 36 files indexed with what-it-shows / where-used / source / hash. **27/27 rebuilt byte-identically.** |
| `figures-redraw` **p2** | Ladder gains the **131,072-row** set — **every header now carries pool AND step budget**. New `fig_headline_steps`. `fig_maze_conditions` becomes **six seeds × two configs** with ceiling markers and hard-seed shading. Gate 1: **192/192** hashes. |
| `figures-redraw` **p3** | ⭐⭐ **The legality fix** (§3). **Gate 2 PASSED 132/132** ⇒ the improved CLU's per-arm panels drawn for the first time. Task figure rebuilt on the panels' own renderer. |
| `draft-rulings` | All **12 rulings** applied + the 5 w20 placeholders. **15 → 4** placeholders. **48/49 pp.** Retired numbers: **0 hits**, positive control 21+43+51+29. |
| `registry-fold` **p5 + p6** | w20 and all six w21 units folded. Additive, duplicate-free, every token verified verbatim. |
| `seeds-and-tables` **p5** | ⭐⭐ **The w15 gap CLOSED — 199/199.** Found the **PoE 27.13 → 27.14** error. Fixed a negative test that was passing by crashing. |
| `mirror` (Hub) | `w17-paper/` and `paper-visuals/` current on the Mac, both PDFs installed after verifying their sources matched. |

## 2. ⭐⭐ THE HEADLINE RETIREMENT — and how it was proved, not asserted
Ruling 2 removed the under-trained 16,384-row comparators and the step-matched diagnostic from the paper. Those numbers were **load-bearing in the abstract and contributions**, so this was a rewrite, not a deletion.
⭐ **The proof method is the part worth keeping:** `draft-rulings` extracted the text of the **compiled PDF** with Ghostscript and compared **whole numeric tokens** against ~150 retired values in 4 classes ⇒ **0 hits**. It then ran the same check on the **pre-edit PDF** as a positive control ⇒ **21 + 43 + 51 + 29** hits, plus 16 for "16,384". ⛔ **A zero is worthless without that control**, and it produced one.
⚠ Five surviving token coincidences are named and shown unrelated (e.g. `37.550` = w20's data-matched continuous MLP).

## 3. ⭐⭐ THE FIGURE DEFECT THE HEAD FOUND
**He observed identical panels carrying opposite marks in Figure 14, row 3.** Diagnosed: the panel renders the **resulting state**; the judge scores the **four-fact vector** `[legal, displacement, event, pocket-next]`; any fact that does not change the state is **invisible**.
| | count |
|---|---|
| opposite-mark arm-pairs | 92,628 |
| ⛔ …rendering **identically** | **33,412** |
| …differing only in the **legality** bit | 3,740 |
| …differing only in **pocket-next** | 16,283 |
| ⭐ **after the fix** | **0** |
⚠⚠ **Two Hub corrections belong on the record.** The Hub first blamed the undrawn **pocket-next** field — ⛔ wrong for the Head's own case (item 482 differs in **legality**). It then sized the defect at **3,740** — ⛔ that was only the legality subset; the true figure is **33,412**, **9× larger**. ⭐ The spoke caught both and built a full fact-strip rather than the narrower fix the Hub specified.
⭐ **Fix verified on the Head's case:** seed 2 item 482, CLU `[1,0,0,3]` CROSS vs mlp `[0,0,0,3]` TICK, now visibly different.
⭐ **A second inconsistency surfaced:** the task figure drew `c1` **blue** while the panels drew `c1` **red**. Unified and verified **pixel-identical** — 52 disc pixels compared against a real panel raster, **7/7 equal**, ring and centre bytes matching in both PDFs' palettes.

## 4. ⭐⭐ THE w15 VERIFICATION GAP — closed after two declined passes
`placeholder-fill` wrote `bar-d-score`, `coverage-control` and `lace-poe-rows` numbers into the draft; passes 3 and 4 both declared the fold **out of brief**. ⇒ **the paper printed values its own cell register did not check.**
⭐ **Closed: 199/199** draft numbers traced draft ↔ FINDINGS ↔ named JSON leaf at printed precision.
⭐ **It found a real error:** PoE's budget printed **27.13×** twice per draft; `14,226,660 / 524,288 = 27.1350` ⇒ **27.14**. The artefact wins.
⭐⭐ **And it audited its own instrument:** the historical-tamper negative test had been *"passing" by crashing on a missing output directory* — verifying nothing. Fixed; **12/12** negative tests now genuinely exit 1.

## 5. ⛔ WHAT REMAINS — three placeholders, two of them permanent
| placeholder | status |
|---|---|
| 32,768-row ladder rival set | ⭐ **FILLED** this wave |
| **w16 CMAZE-T per-arm panel** | ⛔ **PERMANENTLY BLOCKED** |
| **w16 CMAZE-T post-deletion panel** | ⛔ **PERMANENTLY BLOCKED** |
| unlabelled `...` | minor |
⛔ **Why blocked:** `success-failure-panels` established that **w16 persisted NO arm object and NO per-transition prediction** — `A1_test_seed*.json` are aggregates only. ⇒ **They cannot be built from any artefact.** The options are to **cut the panels** or to **re-run w16 to regenerate the objects**. ⭐ Every wave from w19 on persists with hashes, so this cannot recur.

## 6. What did NOT run — declared
- ⛔ No scored run, no compute beyond `pdflatex`/`bibtex` and figure rendering. No ledger row.
- **10 mirrored figures were not rebuilt** (task figures need the dataset + a dedicated venv; rendered panels need the frozen decoder's forward pass) — instead matched against their owners' two-build manifests.
- ⚠ **`paper-visuals/build.sh` does not run in place** (exit 1, 0 figures): a repo-root derivation bug. The figures are correct — verified from the committed copy — but **the folder is not self-contained for rebuilding.** Documented, fix deferred.
- `registry-fold` p6 asked for three things it lacked, one of which is a **research** ask, not a shipping gap (§7).

## 7. What the Advisor is asked to rule on
1. ⛔⛔ **The two w16 CMAZE-T per-arm placeholders cannot be filled from artefacts.** **Cut the panels, or re-run w16 to regenerate the objects?** This is the only thing blocking a placeholder-free draft.
2. ⭐⭐ **`registry-fold` p6's research request: PRNG-replicate rival fits at the CARRIED budgets** (Shapes rows 3–4; CMAZE-T). The ±4.1 variance was measured at 2,560 steps; it is not known at the budgets the paper quotes. ⭐ This bears directly on w21's question 2.
3. ⚠ **PoE's multiple is 27.14, not 27.13** — corrected in the tables; confirm the draft text follows.
4. ⚠ **`paper-visuals/build.sh` is not self-contained.** The Head edits plots against that folder; should the fix be scheduled before that starts?
5. ⚠ **MF-11 is improved but not resolved:** 51/52 → **48/49 pp** total. The main-text-vs-9 pp question is unchanged and remains the Head's.
