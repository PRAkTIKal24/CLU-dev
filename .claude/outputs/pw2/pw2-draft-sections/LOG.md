# LOG pw2-draft-sections
- 2026-09-25T15:36:43Z start; bib sha ok b2cf715f… both; v2 tex 42cd4a48…, style f7ee4846…; tmux: only own pw2 session
- 2026-09-25T15:42:03Z stage 20260925T154128Z: restructure.py both drafts (62/64 chunks all emitted once); scratch builds before+after all exit 0, ucit=0 uref=0 err=0 overfull=0; pages v2 42->43, style 44->44
- 2026-09-25T15:45:16Z checks on stage: style diff-check ALL EMPTY (numbers, placeholders, artefact paths, section order 101 headings identical, 67 labels identical); provenance findings identical to before (line numbers masked) both drafts; MOVE-MAP proof PASS (62/62 v2, 64/64 style, multisets equal, 0 connective deletions, outside-span bytes identical, only heading lines changed); fig/table numbering unchanged (13 fig, 21 tab); HEAD-HELD written
- 2026-09-25T15:45:16Z positive controls: CTRL1 one-char change in a moved paragraph -> proof FAIL; CTRL2 dropped chunk -> assert; CTRL3 duplicated chunk -> assert; CTRL4 head-held one-space change -> FAIL; CTRL5 two subsection headings swapped in style -> style diff-check see below
CTRL5 -> style diff-check exit 1, headings sequence False (detected)
- 2026-09-25T15:45:47Z lock guard clean; live==snapshot; installed .tex+.pdf byte-identically into ~/clu/pw2/live/{v2,style} (INSTALLED-SHA256.txt); bib hash unchanged
- 2026-09-25T15:47:04Z FINDINGS.md written
- 2026-09-25T15:47:24Z committed 6b431af, bundle verified; bib hash b2cf715f… at end; live files still == installed; END
