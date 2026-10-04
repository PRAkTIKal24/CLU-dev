# Verbatim re-check — final paper source (2026-09-26)

**Source checked:** `src/DRAFT-v2.tex` from `~/Downloads/naive_CLU_final_v2fina_subF.zip` (zip sha256 `52144fdc…7ebe`, tex sha256 `e5380803…15c8`, 740 lines). It builds the 21-page PDF `naive_CLU_final_v2fina_subF (1).pdf` (sha256 `5cd1106f…3a29`).
**Rules:** pass 7's rules, copied verbatim from `w23/fold:w23_fold/gen_cells_pass7.py` (see `gen_cells_pass7.ref.py`): the same token regex, `% SRC` / `% SRCBLOCK` scopes, mechanical variants and boundary rule. There are two changes, both logged in the script:
1. SRC paths resolve against both the paper dir and the evidence root. The first run resolved only the paper dir, left 19 cited files unresolved, and "found" 4 of 736.
2. The `a..b` range form in records is accepted. Without it, 7 range upper bounds in Tab. kcurve3 were false misses. A guard confirms `10.1565` still does not match `0.1565`.

Nothing is computed or rounded. The check proves **presence** in the cited record, not meaning.

## Result
| | count |
|---|---:|
| number tokens in the source | 736 |
| layout widths, not results (`0.85`, `0.9` ×2; lines 35, 36, 295) | 3 |
| **printed values** | **733** |
| found verbatim in the record their `% SRC` cites | **693** |
| found verbatim after recovering the scope from the previous 42-page source (its SRC comment was cut) | 15 |
| **total found in a cited record** | **708 / 733** |
| no source comment in the final **or** the previous source | 25 |

**Negative control:** each found token's last digit was changed and the result searched in the same scope. **697 / 708 mutants were caught**; 11 escaped because the mutated value also appears in the same record (`negative_control_escapes.json`).

**Weak attribution (6 of the 15 recovered):** generic values whose recovered line has low text similarity (0.17–0.39): l.121 `524,288`, `131,072`; l.133 `1.0`; l.686 `0.5000`, `0.0000`; l.687 `1.0000`. They count as found but prove little.

## The 25 unsourced values — traced (`unsourced_trace.json`, `cooccur.json`)
| lines | what | values | record holding all of them |
|---|---|---:|---|
| 664–677 | zero-dynamics check | 17 | ⭐ `w12-clue-theorist/CLUE.md`: every value is present verbatim (primary record) |
| 574–575 | deletion, degraded arms (−0.3291, −0.3078, +0.0074) | 3 | ⭐ `w15-single-design-seeds/FINDINGS.md` (primary run record) |
| 686, 696–697 | retention law (0.902, −1.0020, +1.116) and capacity range (0.29–1.71) | 5 | ⚠ **no run record found** under `.claude/outputs`; only registries, claims matrices and old drafts |

## Not done
- **The source is not edited.** Adding `% SRC` comments for the 20 traceable values belongs to whoever owns the source.
- **Not searched:** the 5 retention-law values outside `.claude/outputs` (e.g. the repo's `projects/` runs from the earlier campaign).
- **No JSON-leaf / meaning check:** the check proves presence only, as pass 7 did.

Files: `verbatim_recheck.py`, `negative_control.py`, `trace_unsourced.py`, `cooccur.py`, `recheck.json`, `negative_control_escapes.json`, `unsourced_trace.json`, `cooccur.json`.
