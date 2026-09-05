# Task: v3-revision-2 — splice the scout bib + wire figures into V3 (w9, small)

- **Agent:** `paper-writer` · **Output:** report to `.claude/outputs/v3-revision-2.md`; edit `.claude/papers/v3-short/` in place (CHANGELOG v0.3).
- **Read first:** protocol · `.claude/outputs/scout-modular-interference.md` (the bib + the lift-ready paragraph + the novelty verdict: **CLEAR at specific-claim level, CROWDED at neighbourhood** — cite the neighbours) · `.claude/outputs/v3-revision.md` open items (Q3 figure, Q5 bib).

## Items
1. **§4 bib splice (closes editorial Q5):** replace the `[·]` anchors with the scout's verified citations (McCloskey & Cohen 1989; French 1999; Jacobs et al. 1991; Shazeer et al. 2017; Kirkpatrick 2017 EWC; Mallya & Lazebnik 2018 PackNet; Doan et al. 2021 NTK-overlap; Riemer 2019; Yu 2020; Boopathy 2025 — exactly as the scout's entries give them). Use the lift-ready paragraph (guard-railed): prior work *prevents* interference by construction or *measures* it as a diffuse monolithic property; **none prices it as a coupling law** — our claim is the measured kernel, ∝κ², exact zero beyond the graph, O(1)-vs-O(N).
2. **Figures (closes editorial Q3):** `v3-scaling-figure` (analyst, running concurrently) generates the S-vs-N scaling-curve PNG → embed as the headline Fig 1 when it lands; wire `\includegraphics` for the existing assets meanwhile (`fig1_interference_bars.png`, `fig3_pricing_parity.png`, the reversible `mem_grad_summary.png` for §3.5/App). If the scaling PNG hasn't landed by your run, embed bars as Fig 1 and leave a one-line swap note.
3. Citation-string slot (Q4) stays "(Anonymous, 2026)" until the F5 arXiv id exists — no action. Rebuild PDF.

**Acceptance:** zero `[·]` anchors remain; figures embedded; PDF builds. → `v3-referee` (w10).
