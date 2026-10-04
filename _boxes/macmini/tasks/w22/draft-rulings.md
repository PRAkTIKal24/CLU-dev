# w22 `spoke/draft-rulings` — BRIEF (batmac, paper-writer)
**Wave-plan §7. ZERO compute, ZERO scored runs. ⛔ No ledger write. You hold the `.tex` lock.**

## THE ONE JOB
**Apply the Head's w20/w21 rulings to BOTH drafts, and replace every maze placeholder with the six-seed condition table.**

## ⛔⛔ THE STANDING RULE
**Pre-registrations and pre-declarations are BANNED** (Head, 2026-09-24). ⛔ Write no PREREG; declare no bar before a number. ⭐ **Print everything measured.** ⚠ Earlier waves' `PREREG-*.md` are an **anti-pattern**.

## THE RULINGS — apply each, verbatim in force
1. ⭐⭐ **Shapes headline = ≈ +7** novel tuples/corner vs the **strongest converged data-matched rival** (**+10…+22** vs the transformer), **3 seeds**.
2. ⛔⛔ **RETIRE from the paper entirely:** the **under-trained 16,384-row comparators** (the +39…+49 numbers) and the **step-matched diagnostic**. ⭐ They remain artefacts — keep an artefact pointer in a **LaTeX comment**, ⛔ never in rendered text.
3. Every Shapes table shows the **w20 data-matched set**. The ladder shows the **32,256-row** and **131,072-row** sets (⭐ and a **32,768-row** set when it lands — leave the column slot).
4. ⭐⭐ **Deletion:** **exact on the store AND the address book**; **value/item survivors unchanged**; ⛔ **conjunction deletion CHANGES WHAT THE BOOK KNOWS ABOUT THE SHARED VALUE** — print as a **measured limitation of the unit**, not as a footnote. The numbers: −6.13 / −6.43 / −6.30 on 3/3 seeds; the 20 red-with-other-shape corners fall −32.1; a **random-1 % control moves −0.02 / −0.11 / −0.01**, which is what shows it is **content, not refit instability**.
5. ⚠ **"Reads no training rows"** may be said of the store **only with the book's statistic size (0.97 GB) and its refit time (2,367–5,913 s) printed beside it.** ⛔ It is **not** a size claim.
6. ⛔ **`learned_1x` is not in the paper.** 7. The **w20 rebuilt K=5 arm replaces the old w12 ladder arm**.
8. ⭐⭐ **All six CMAZE-T seeds print, with a per-seed ceiling row.** Counts: **3 clear wins, 2 ties at ceiling, 1 loss**; **six-seed mean +0.046** ⛔ (**NOT** +0.128 — corrected by the Head 2026-09-24; verify against the run JSONs yourself).
9. **words-full TEST prints as measured: behind by 0.025 / 0.001 / 0.026.**
10. ⭐⭐ **The improved CLU (`CLU-REL-A10`) prints WITH its `+rel` diagnostic beside it, and the mechanism named: relational request coding carried through the closed-form head.** ⛔ Handing rivals the same columns recovers **97–100 %** of the margin; a **control ties** it on seed 4; a **rival ties** it on seed 5. ⛔ **It must not read as a store/memory result.**
11. ⭐ **The six-seed condition table replaces EVERY maze placeholder.**
12. ⛔ **The abstract's maze sentence stays `\TODO`** for the Head.

## TWO NEW PASSAGES TO WRITE
**(i) In the CMAZE-T task section — the MEASURED PER-SEED CEILING.** The best score **any arm** reaches on a seed. How it arises: on **seeds 1 and 4** the random 2-way hold-out removes **all** training rows with `pocket = c` and `tint = c` for some colour, leaving ~**40 %** of held-out n-way rows with a **zero-support colour pair**. **Every colour-identity reader caps there — the CLU within 0.003 of the cap.** ⭐ **A colour-equality code passes straight through it**; only the **`(c,c,c)`** interaction rows are unreachable by **any** arm.
⛔⛔ **State it as a ceiling FOR IDENTITY-CODED READERS, not as a constant of the data.** ⭐ Add that reporting it **per seed** is useful practice for anyone using the task.
**(ii) A few lines presenting CMAZE-T as a MINOR contribution:** a small world-model task with **exact engine judging**, **random test mazes so lookup cannot transfer**, five request words, **value / 2-way / n-way** hold-out families, **accept/refuse validity**, printed **floors**, **blind arms**, and the **per-seed ceiling** — offered to researchers working on composition. ⛔ **Explicitly NOT overclaimed as a benchmark.**

## ACCEPTANCE
- Both drafts compile: **exit 0, 0 undefined citations, 0 undefined references, 0 overfull**.
- **All four style diffs EMPTY**; the two drafts stay content-identical.
- ⛔ **Zero maze `\PLACEHOLDER{}` left**; the abstract's maze `\TODO` **still present**; report the new placeholder/TODO counts.
- ⛔ **No retired number survives in rendered text** — grep for the +39…+49 figures and the 16k comparator values and prove they are gone (comments may keep the artefact pointer).
- Every number you write is **verified against its artefact** and carries a `% SRC` pointer.
- Staged → compiled → checked → installed **byte-identically**; drafts `cmp`-equal to snapshot before install.
- `FINDINGS.md`; branch committed WHOLE and bundled; `git bundle verify` passes.

## OUTPUT
`~/clu/outputs/w22-draft-rulings/`; edits into `~/clu/outputs/w17-paper/`.
