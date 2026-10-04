# pj-restore-v5 — paper-writer report

**Task + acceptance criterion:** BOUNDED RESTORATION of `NIPSsubmission/v5-palm/pj_sub.tex` — all 11 figures + 8 tables shipped with no truncation, Part-B additive restorations, Part-C enumerated corrections, clean rebuild, two-way numcheck, positive-controlled sweeps, `BUILD-NOTE-R3.md`.
**Status: done.**

**DIAL DECLARATION (echoed):** **none — editorial restoration.** Laundering control: n/a (zero performance numbers produced). Falsifies: n/a. **⛔ Zero new measurements, zero new numbers** — every added token traced to a named ancestor line.

> ⚠ **DOWNSTREAM RECONCILIATION LIST — READ FIRST (protocol §5 corollary; each needs an owner):**
> 1. ⛔ **Correction C(d) has NO ancestor in the named base.** `submission.tex` L104 carries the *same* redacted J&P entry as `pj_sub`. I sourced the full third-person entry **verbatim** from `NIPSsubmission/v2-neurreps/submission.tex` L179 (the sibling clean base, Add.52-accepted) and `papers/v5-short/draft.tex` L141 (V5's own draft lineage). **This is the single edit whose ancestor lies outside the folder — Advisor/Head to ratify by name.** Owner: Advisor.
> 2. ⚠ **The restored lifecycle negatives row has no host claim.** B3 mandates all missing rows and the App-E completeness sentence is only true with it, but the Head's rewrite cut contribution (4): `lifecycle` was 0 and is now 1, *only* in that App-E row. Keep (completeness true) or cut (soften "every"). Owner: Head.
> 3. ⭐ **Fidelity-report correction: the base's negatives estate is 20 rows, not 21.** `pj-fidelity-v5-r2` Tier-1 #2 says "5 of 21"; measured, the base has 20 (its second table's *"A tilt sets the soft scale"* row is the row `pj_sub` already carried). **5 present + 15 restored = 20 = complete.** Owner: doc-curator (registry note).
> 4. ⚠ **Four live findings were out of the worklist and are untouched** (listed §7 of the build note): A2-4 *"holds universally"* vs 10/10 · A2-12's *"5 validation seeds"* · the residual MF-8 intensifier layer · App C's *"Emergent Arm Translation"* heading. Owner: Head/Advisor at the next pass.

---

## What I did

- **Edited exactly one content file**, `pj_sub.tex`, via **50 scripted exact-match single-occurrence replacements** (`.claude/scratch/pj-restore-v5/restore{,2,3,4,5}.py`; each asserts `count(old) == 1` and aborts otherwise). Every other file in the folder is byte-untouched (hashes + mtimes in build note §0).
- **Part A** — wired all 11 PNGs (3 placeholders replaced, 8 unwired figures given homes), restored the 4 missing tables, and restructured the truncated ones.
- **Part B** — B1…B6 restored additively; approved wordings/riders diff-identical to the base.
- **Part C** — (a)–(h) applied exactly as enumerated, each printed before→after in the build note.
- **Deliverables:** `v5-palm/BUILD-NOTE-R3.md` (deliverable #1, 307 lines) + this report. No writes outside the folder except this report.

## How I verified (commands + observed output)

```
md5 pj_sub.tex                 6c1902f74ee9611d718cc65b9fd1a031  ->  d83447ef623345084529e1e4810c3e5c
/Library/TeX/texbin/pdflatex -interaction=nonstopmode pj_sub.tex   (x2, inside the folder)
  Output written on pj_sub.pdf (19 pages, 1759289 bytes)
grep -c "^!" pj_sub.log            -> 0        (errors)
grep -ci undefined pj_sub.log      -> 0        (undefined refs/citations)
grep "Overfull \hbox" pj_sub.log   -> no match (NOT merely "none > 10 pt on tables": zero, anywhere)
grep -c "Overfull \vbox" pj_sub.log-> 0
grep -c includegraphics pj_sub.tex -> 11       (was 0);  framebox -> 0 (was 3)
distinct figs used 11 / 11 available;  comm -13 (unused figs) -> EMPTY
begin{tabular} -> 8 (was 4)
```
- **Page split (from `pj_sub.aux` `\@abspage@last{19}` + `pdftotext`, not estimated):** main text **pp. 1–6**, references pp. 7–8, appendices A–E pp. 8–19 (A p.8 · B p.9 · C p.14 · D p.15 · E p.17), **total 19 pp**. (Base `submission.pdf` = 19 pp; pre-pass `pj_sub.pdf` = 11 pp with no figures and 5 negatives. Page limits deferred per Add.59 — reported, not fought.)
- **Rendered-output spot check:** 20/20 load-bearing restored strings verified present in `pdftotext pj_sub.pdf` (abstract conditions · encoder scope · certified denial · score sentence · trivial substitute · probe-floor rider · Δ+ℓ_θ/Δ at 3.77× · 86.97× · "never the vault" · direction-not-number · confinement controls · R₅₀ · seam · Titans · no-priority · corrected Guo · J&P entry · anonymization note · T\* · compute-adaptive-read dial). **0 misses.**
- ⚠ **Instrument incident, logged:** my first `pdftotext` wrote to `/tmp/pj.txt`, which the **parallel `pj-restore-v2` spoke had already written** — I read V2's page split as V5's for one command. Caught by the title line, re-run to a unique path. *Standing lesson for parallel spokes: never use a generic `/tmp` filename.*

## Findings / results

**Acceptance criteria, measured:**

| criterion | result |
|---|---|
| 11/11 figures shipped, zero banked | ✅ 11 `\includegraphics`, 0 placeholders, 0 unused PNGs. `fig2_vault.png` promoted App C → main text per task A1/MF-1; every other figure at its base home |
| 8/8 tables present, no truncation | ✅ 4 → 8. Overfull 177.5 pt → 0, 152.6 pt → 0, 49.2 pt → 0, and the newly-restored instrument-gap table's inherited 49.4 pt → 0. **No `\small` added; zero content change** (the four-instrument table was transposed to quantity×instrument rows / seed columns, all 39 values preserved + the base's 12-value mean±sd row restored) |
| every Part-B item present or honestly reported blocked | ✅ B1–B6 complete; nothing blocked. Detail table per item in build note §5 |
| Part-C applied exactly as enumerated | ✅ (a)–(h), each before→after printed. **(d) carries the ancestor deviation above** |
| zero edits outside the enumerated set + additive insertions | ✅ 50 edits, all classified; **3 additive insertions beyond the worklist declared by name** (the C-2 status-labelling rule, base L45; Fig-3's multi-seed disambiguation, base L144/A2-14; the base's Collapse/Scope sentences, base L207 — the in-file ancestor correction (c) and limitation (iv) need) |
| verbatim riders diff-identical to base | ✅ |
| zero new numbers without ancestors | ⭐ **248 numeric tokens added, ORPHAN LIST EMPTY.** 247 ancestored in `submission.tex`; 1 (`2603.01768`) in `v2-neurreps/submission.tex`. Only 2 tokens removed (`1.0068`, `0.0219`, −1 each) — the deleted placeholder box *duplicated* them; post-edit counts 3 and 3. **No result number dropped, rounded, moved or re-scaled** |
| sweeps clean | ✅ 38-pattern zero list all 0 (incl. `13.9`, `right-to-be-forgotten`, `memory provenance`, `Placeholder`, `redacted`); positive controls fire (`encoder` **0 → 4**, `ZERO` ×2, `107.77` ×7, `Blelloch` ×8); `\bMo\b` = 1 occurrence = the bibliography entry, prose/captions/labels/filenames 0; honest-scope sentence exactly 1 |
| App E completeness sentence true after restoration | ⭐ ✅ **20 rows counted by script = the base's full estate.** *"every negative result observed during evaluation is documented below"* is now a true statement |
| `submission.tex` + all other folder files byte-untouched | ✅ `1d0906fe45dc78436880c938ad227332` unchanged; sty unchanged; figs manifest unchanged; only `pj_sub.*` mtimes moved |

**`certified` per-occurrence (n = 3):** l.82 literature-description form (**now correctly stated** — the ε-vs-(ε,δ) inversion is fixed); **l.128 = the explicit denial, restored after two rounds absent**; l.161 = the Guo reference title. **Zero affirmative forms.**

**The four round-2 regressions are closed:** `encoder` 0 → 4 (the store-level/encoder-excluded scope is back at all three base sites) · App E's false completeness claim is now true · `(Mo, 2026)` gone from prose · the deletion store's scale (dim 3 / capacity 8–64 / no learning) restored in the abstract, §3.3, §4 and App D. **All eight referee MUST-FIXes are discharged** (MF-1 figures · MF-2 truncation · MF-3 the J&P entry · MF-4 the two false "strictly" claims · MF-5 the thrice-promised Δ + ℓ_θ/Δ rule, now delivered · MF-6a/b the probe-floor and 86.97× riders · MF-7 the deletion scale · MF-8 the four App-D garbles + "remarkable"), and SF-1/2/3/4/5/7/8 plus N-1/N-3-adjacent items.

## Git footprint

**None.** No tracked file touched, no branch, no commit — all work is inside gitignored `.claude/`. Files written: `.claude/NIPSsubmission/v5-palm/pj_sub.tex` (edited), `…/BUILD-NOTE-R3.md` (new), `…/pj_sub.{aux,log,out,pdf}` (build products), `.claude/scratch/pj-restore-v5/*` (scripts + the pre-edit backup `pj_sub.PRE.tex`), `.claude/outputs/pj-restore-v5.md` (this).

## Open questions / follow-ups / risks

1. **C(d)'s out-of-folder ancestor** — the one item needing a named ratification (reconciliation list #1).
2. **The lifecycle negatives row** — keep or cut (reconciliation list #2).
3. **19 pp with 11 figures.** The Head deferred page limits; when the page pass comes, the honest lever is figure placement (9 of 11 are appendix figures) and the appendix-maximalism policy (C-10) says pruning happens in a dedicated later pass, never now.
4. **A round-3 blind referee is now worth it** — the artifact changed substantially (11 pp → 19 pp, 5 → 20 negatives, all riders back). Add.59 lists it as a Head option after landing; the two blind instruments have caught something each round.
5. **Not a defect, but the Head should know:** the restored base content re-introduces the base's own posture on de-anonymization (`Anonymous (2026)` theory note + the CHLU continuity sentence), and C(d) now prints a full self-citation. That is the sanctioned C-1(d)/MF-3 posture, but it is a *change of direction* from the pre-pass file, so it should be a conscious Head choice.
6. **No missing-experiment note arose** — this pass produced no measurement and needed none; every number it wanted already existed.

## Proposed handover updates (for the Hub)

1. **`pj-restore-v5` DONE.** `pj_sub.tex` md5 `6c1902f74ee9611d718cc65b9fd1a031` → **`d83447ef623345084529e1e4810c3e5c`**; `BUILD-NOTE-R3.md` md5 `67e23b8939e652dac5009f8159552d40`. Build clean (0 errors / 0 undefined / **0 overfull boxes**), 19 pp, 11/11 figures, 8/8 tables, 20/20 negatives.
2. ⭐ **All four V5 round-2 regressions and all 8 referee MUSTs are closed**; the round-2 verdict "content accept-class / artifact not submittable as shipped" should now be re-scored on the artifact axis.
3. ⛔ **Two items need a Head/Advisor word before the next pass:** C(d)'s ancestor ratification, and the lifecycle negatives row.
4. **Registry candidates:** (i) *negatives registry* — nothing newly tried-and-failed; (ii) *`future_work.md`* — no new scientific boundary; (iii) ⭐ **process negative, worth recording:** *the round-2 hand-rewrite's four regressions were all "rider deleted", never "number changed" — three consecutive rounds now show numeric fidelity surviving hand-editing while claims scope does not. A scripted, assertion-guarded restoration (50 single-occurrence replacements with an ancestor per token) closed them in one pass with an empty orphan list; recommend it as the standing mechanism for rider restoration rather than a further hand-rewrite.* (iv) ⚠ **parallel-spoke instrument note:** generic `/tmp` filenames are unsafe when spokes run concurrently (§How I verified).
5. **`pj-restore-v2` is unaffected** — separate folder, separate report; no shared file was touched.
