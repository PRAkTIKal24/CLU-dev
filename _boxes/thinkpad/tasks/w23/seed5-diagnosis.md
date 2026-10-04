# w23 `spoke/seed5-diagnosis` — BRIEF
**Wave-plan §9. ⛔ ZERO scored runs. ⛔ NO re-fit. ⛔ NO new seed. ⛔ NO ledger row. ⛔ NO PREREG.**
⛔⛔ **You may NOT write to `w17-paper/`, `figures/` or `paper-visuals/`** — another Hub owns those now.
⭐ **Everything you do is a ZERO-FIT read of objects that already exist.**

## ⭐⭐ QUESTION 1 — the one the Head wants answered
**Seed 5, base design (`n-CLU` = A1). Two held-out "free"-family tuples score 0.000 where EIGHT trained baselines score 1.000:**
- **`(2,0,1,2)` n = 90** · **`(2,0,3,2)` n = 80**
⭐ **Both are class A — fully supported in train.** ⛔ So this is **not** a hold-out-coverage story; it is a CLU-specific failure on reachable data, and it is a large part of why seed 5 reads **CLU 0.682 vs best rival 0.867**.

**TRACE THE CLOSED-FORM READ for those two tuples, in this order, and print each step:**
1. **The design row** the request produces (all 318 / 396 columns as applicable) — and which are non-zero.
2. **Its projection on the fitted head** — the numbers, not a description.
3. **Which well and which atoms are addressed**, and with what weights.
4. **The emitted four facts** `[legal, displacement, event, pocket-next]` **vs the truth**, and **which fact is wrong**.
5. ⭐⭐ **LOCATE THE FAILURE.** Exactly one of:
   - **the DESIGN** — rank deficiency or collinearity with the pair block (⭐ check the rank of the design restricted to the rows that share this tuple's support, and whether its column is a linear combination of others);
   - **the ADDRESS BOOK** — the request addresses the wrong well/atoms;
   - **the CONTENT** — the right atoms are addressed but hold the wrong value.
⭐ **Compare against a tuple of the SAME family that the n-CLU gets RIGHT**, so the difference is isolated rather than merely described.

## ⭐ QUESTION 2 — the `n-CLU-REL` value hold-out
**Confirm mechanically that the `eq`-columns are ALL-ZERO for the never-seen colour**, and that this is why the value row is lost.
⚠ **The Hub verified these from the run JSONs — use them, and ⛔ flag any disagreement:**
- REL `value` row, per seed 0–5: CLU **0.4888 / 0.5008 / 0.5118 / 0.5252 / 0.6917 / 0.6038** ⇒ **6-seed mean 0.5537**; best arm per seed mean **0.8676**.
- ⚠⚠ **§9 quotes "0.500 vs 0.792". The Hub could NOT reproduce that pair** — the nearest is seed 1's CLU **0.5008**, whose best matched rival is **0.8612** (transformer @10× steps), not 0.792. ⭐ **Locate what 0.500 / 0.792 refers to and say so; if it cannot be reproduced, REPORT THAT — the artefact wins.**

## OBJECTS — all already on this box, ⛔ load them, never rebuild them
- **n-CLU seed 5:** `/c/chlu/outputs/w21-maze-t-seed4/test/persist/BOTH_seed5/test/`
- **n-CLU-REL seeds 3–5:** `/c/chlu/outputs/w21-rel-newseeds/test/persist/BOTH_seed*/test/` · **seeds 0–2:** `/c/chlu/outputs/w21-maze-t-clu-improve/test/persist/`
- Per-item verdicts and emissions already extracted: `w21-maze-t-seed4/replay/*PERTUPLE.json`, `w21-rel-newseeds/replay/`.
⭐ **Assert every object's `param_sha256` and `file_sha256` against its `MANIFEST.json` before reading it.** ⛔ One mismatch ⇒ stop and report.

## ACCEPTANCE
- **Per-tuple tables** for both questions — ⛔ not prose summaries.
- Question 1 answered with the failure **located** in design / address book / content, with the numbers that prove it, plus the right-answer comparison tuple.
- Question 2: the eq-column zero-ness **shown**, not asserted.
- ⭐ The §9 number discrepancy resolved or explicitly reported.
- `FINDINGS.md`, machine JSON, branch committed WHOLE and bundled, `git bundle verify` passing.
- ⛔ **Nothing written to `w17-paper/`, `figures/` or `paper-visuals/`.**

## OUTPUT
`/c/chlu/outputs/w23-seed5-diagnosis/`.
