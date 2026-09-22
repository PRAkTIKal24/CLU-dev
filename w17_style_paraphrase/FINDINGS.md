# w17 spoke/style-paraphrase — FINDINGS

## Delivered
- `~/clu/outputs/w17-paper/DRAFT-v2-style.tex` (sha256 `ec84c2d1…`): DRAFT-v2 paraphrased sentence by sentence under the rules of `PJ_Writing_Style_Context.pdf`. Built from `DRAFT-v2.tex` sha256 `edd81375…`, which is byte-identical to draft-v2's own copy and was unchanged from gate to install.
- `~/clu/outputs/w17-paper/STYLE-DIFF-CHECK.md`: the four required diffs, each **empty**, plus extra verbatim checks and a hand-written note on how each style rule was applied.
- `DRAFT-v2-style.pdf` (this dir only, not in the shared dir): pdflatex on the ICLR 2027 template, 31 pp, 0 errors, 0 undefined refs, 0 overfull boxes. Main text (§1–§6) ends on p. 12, as in DRAFT-v2.

## Acceptance: the four diffs (all empty)
1. **Numbers:** 2,525 number tokens (1,296 distinct) in the code of both files. Set diff and every-occurrence multiset diff are both empty. This counts every digit run, with no small-integer or year filter.
2. **`\PLACEHOLDER{}`:** 29 in both files, byte-identical and in the same order. `\TODO{}`: 4 in both, byte-identical.
3. **Artefact-path comments:** 278 path mentions (27 distinct paths) in both; 272 provenance comment strings in both, compared whole; and the 265 SRC/ENDSRC comments sit in the same document order.
4. **Section order:** 96 headings (section, subsection, subsubsection, paragraph) in an identical sequence; 74 `\label{}`s in an identical sequence.

Extra checks: all 28 table floats and figure slots, captions included, are byte-identical and in the same order; the preamble is byte-identical; the file has 0 non-ASCII characters. The draft-v2 provenance checker gives identical findings on both files (1,824 tokens, the same single tokenizer artefact, the same 10 unsourced template or wave-label lines). The diff checker was tested for sensitivity: it fails on a changed number (32.350 → 32.35), a changed placeholder, and two swapped subsections.

## Decisions (pre-authorised; recorded in LOG.md)
- **D1 Method:** I did not retype the file. `DRAFT-v2-style.tex` is DRAFT-v2 with 92 declared prose ranges replaced (`patches/`, `apply_patches.py`), and each range is conservation-checked on its own. Everything outside those ranges is byte-identical.
- **D2 Floats verbatim:** table and figure captions are not paraphrased. They carry numbers densely, and the no-content-change rule outranks restyling them.
- **D3 ABT openers** only reconnect content already in the draft (connectives and signposts). None adds a claim, a number or a direction. No placeholder text was touched, so nothing presumes a w15 or w16 outcome.
- **D4 The page bound (≤ 12 pp main text) outranks some scoping bullets:** four candidate lists went back to inline prose, and four succinctness passes (66 phrase-level edits plus one antecedent fix) recovered the ~35 lines the first paraphrase had added.
- **D5 Bold → `\emph`** for run-in list heads. Bold stays only on newly defined core concepts and the environment names.

## DECLARED NOT-DONE
- **Table and figure captions: not paraphrased** (D2). They are verbatim by design.
- **Some data-list paragraphs are verbatim** (App. F K=5-ward deletion list, App. K pins, App. M "The leg.", App. R ten requirements, App. W placeholder index, App. X not-run items, App. Y corrections). They have no sentence structure to paraphrase, or they are records.
- **`2e$-$7` in App. P was not restyled** as `\sci{2}{-7}`: doing so would change the number token that the diff checks.
- **`\cite{}` insertion:** not this spoke's job (cite-tagger inserts after `END_UTC.txt`).
- **Not re-synced with any later DRAFT-v2.tex.** If DRAFT-v2.tex changes after sha `edd81375…`, for example when cite-tagger inserts `\cite{}`, the style file does not follow automatically. The patches are keyed to line numbers of this exact version, so for later cite insertion cite-tagger should patch `DRAFT-v2-style.tex` directly.
- **No compute, no scored runs,** and no w15 or w16 number anywhere (none exists in the source; the numbers diff is empty).
