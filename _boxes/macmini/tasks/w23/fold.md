# w23 `spoke/fold` — BRIEF (batmac, doc-curator)
**Wave-plan §9 bookkeeping. ⛔ ZERO compute, ZERO scored runs, no ledger row, no PREREG.**
⛔⛔ **YOU MAY NOT WRITE TO `w17-paper/`, `figures/` OR `paper-visuals/`.** A separate Hub owns the paper from now on. ⛔ **Do not take the `.tex` lock.** ⭐ You may **read** the drafts to check a number; ⛔ you may not edit one.

## THE ONE JOB
**Fold whatever W22 left unfolded — into `TABLES.md` and the registry ONLY.**

## ⭐⭐ THE KNOWN GAP (the Hub verified the timestamps)
`registry-fold` **pass 6 ended 06:30Z**. `seeds-and-tables` **pass 5 ended 06:39:59Z** — ⛔ **after it.** ⇒ **pass 5's findings are NOT in the registry.** Fold them:
- ⭐⭐ **The w15 fold is CLOSED**: `bar-d-score` / `coverage-control` / `lace-poe-rows` — **199/199** draft numbers traced draft ↔ FINDINGS ↔ named JSON leaf at printed precision. ⚠ Two earlier passes had declared it out of brief, so **record that it is now closed**, and that the paper no longer prints values its register does not check.
- ⭐ **A real error found:** PoE's budget printed **27.13×** twice per draft; `14,226,660 / 524,288 = 27.1350` ⇒ **27.14**. The artefact wins.
- ⭐⭐ **An instrument defect it found in ITSELF:** the historical-tamper negative test had been *"passing" by crashing on a missing output directory* — ⛔ verifying nothing. Fixed; **12/12** negative tests now genuinely exit 1. ⭐ **Fold this — a test that passes by accident is a defect class worth a registry entry.**
- The **ladder 32,768-row placeholder filled**; the **two w16 CMAZE-T per-arm placeholders declared PERMANENTLY BLOCKED** (w16 persisted no arm object and no per-transition prediction — they cannot be built from any artefact).

## ⭐ ALSO CHECK, then fold or declare
Sweep `~/clu/evidence/outputs/w22-*` and the w22 branches for anything else neither pass folded. ⛔ **If you find nothing further, SAY SO explicitly** — "nothing else outstanding" is a useful result; a vague sweep is not.

## ⛔ NOT RUN — by the Head's ruling
The **PRNG-replicate rival fits** (registry pass 6's ask) are **NOT run** — *"not cheap enough"*. ⭐ Record them as **declared not-run with that reason**, so the request is not lost.

## ACCEPTANCE
- Additive, proven with difflib (**inserted only; 0 replaced, 0 deleted**); duplicate-free `### N` headings; every decimal and 4+-digit token verified verbatim against its source.
- `⟲ PASS 7` annotations on superseded entries; ⛔ never edit an original.
- `TABLES.md` updated **only** where a cell is genuinely missing — ⛔ do not restate what pass 5 already verified.
- `FINDINGS.md`; branch committed WHOLE and bundled; `git bundle verify` passing.
- ⛔ **Nothing written to `w17-paper/`, `figures/` or `paper-visuals/`** — state this in your findings.

## OUTPUT
`~/clu/outputs/w23-fold/`. Extend `~/clu/outputs/w21-registry-fold/{negative_results.md,claims_matrix.md}` and the `w18-seeds-and-tables` register.
