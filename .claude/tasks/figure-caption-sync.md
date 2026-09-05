# figure-caption-sync — paper-writer (captions only: make every caption describe the figure that actually ships) — RE-ISSUED

**Commissioned by the Shorts Advisor (charter: `.claude/advisor-head-shorts-charter.md`, Addendum 44, 2026-08-21). This re-issue supersedes the version that correctly BLOCKED on its precondition** (it ran before `figure-layout-fix` landed; zero files were edited, and its three process findings are folded in below). Read `.claude/AGENT_PROTOCOL.md`, then this file.

**Precondition — now MET:** `.claude/outputs/figure-layout-fix.md` exists. **Its §8 consolidated list is your worklist: 5 live items carried from the render pass (four of the original nine were made moot by the restored legends) + 3 new items created by the layout pass** (a new appendix-figure caption to write, and the V5 figure renumbering 2→3, 3→4).

**DIAL DECLARATION: none — caption-only editorial pass; zero number changes, zero prose changes outside caption bodies.**

## Three corrections folded in from the blocked run (all three were right)
1. **Paths are `.claude/papers/…`**, not `papers/…` — the earlier task file was wrong by one directory level.
2. **`pdflatex` is not on `PATH` here — it is `/Library/TeX/texbin/pdflatex`.** Use the absolute path for every build.
3. **`.claude/papers/v2-neurreps-descoped/` is IN SCOPE** — it was missing from the render pass's owed-edit list and carries the same defects. Check it against its own shipped figures and add any edits it needs to the worklist.

## Scope — strictly captions
Files: `.claude/papers/palm-variant/v5/submission.tex` · `.claude/papers/neurreps-variants/v2/submission.tex` · `.claude/papers/v2-neurreps-descoped/submission.tex`.
For each item: rewrite the caption so it describes the shipped figure — panel references match what is drawn, neutral seed labels where seeds are named, no reference to a stripped label, and the caption's **scientific content unchanged** (same quantities, same scope qualifiers, same fine print). Write the one new appendix-figure caption in the same register as its neighbours.
**Figure renumbering (V5 2→3, 3→4):** update the captions, the `\label`/`\ref` pairs and every in-text reference so the numbering is consistent end-to-end.
⛔ **Nothing outside `\caption{}` bodies, their labels, and figure cross-references.** Approved wordings and mandatory riders inside captions stay **verbatim**.

## Then
Rebuild each edited variant (`/Library/TeX/texbin/pdflatex` ×3) and report the page split against its current value: **V5 PALM variant 4.00 pp main / 10 pp total · V2 reframe 5.67 / 13 · V2 de-scoped 6.14 / 14.** ⛔ **V5's main text must stay 4.00 pp** — if a caption's length threatens it, shorten the caption, never the paper, and say so.

## Acceptance criteria
1. Every worklist item closed or explicitly deferred with a reason; **no caption names a label absent from its figure** — state the check you ran.
2. Figure numbering consistent across captions, labels and in-text references (state the check).
3. Zero edits outside caption bodies / labels / figure references; any in-text reference edits listed.
4. Page splits reported; V5 main text still 4.00 pp.
