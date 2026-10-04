# M5 audit (w20 referee3-fixes): maze environments per table float and figure slot, DRAFT-v2.tex, code only

Tables: the verdict is taken from the tabular BODY (the data rows). Caption mentions are listed separately and were read by hand: every one is an exclusion ("contains no X row", "X is audited in its own table") or a reference to a rule (e.g. "CMAZE-1's own budget rule"), never a row. Figure slots: the verdict is taken from the whole caption.

| kind | label | envs in data rows / figure | envs mentioned in caption only | verdict |
|---|---|---|---|---|
| figslot | fig:task-shapes | (none: Shapes3D or no maze) | - | single |
| figslot | fig:task-cmazet | CMAZE-T | - | single |
| table | tab:t7 | (none: Shapes3D or no maze) | - | single |
| figslot | fig:t7-per-arm | (none: Shapes3D or no maze) | - | single |
| figslot | fig:t7-rival-wins | (none: Shapes3D or no maze) | - | single |
| figslot | fig:t7-rival-wins-s1 | (none: Shapes3D or no maze) | - | single |
| table | tab:ledger-kstar | (none: Shapes3D or no maze) | - | single |
| table | tab:ladder | (none: Shapes3D or no maze) | - | single |
| figslot | fig:ladder | (none: Shapes3D or no maze) | - | single |
| table | tab:del | (none: Shapes3D or no maze) | - | single |
| table | tab:hole | (none: Shapes3D or no maze) | - | single |
| table | tab:maze-del | (none: Shapes3D or no maze) | CMAZE-2 | single |
| table | tab:verb | (none: Shapes3D or no maze) | CMAZE-2b | single |
| table | tab:gentable-kstar | (none: Shapes3D or no maze) | - | single |
| table | tab:k5 | (none: Shapes3D or no maze) | - | single |
| table | tab:1x | (none: Shapes3D or no maze) | - | single |
| table | tab:kcurve | (none: Shapes3D or no maze) | - | single |
| table | tab:kcurve3 | (none: Shapes3D or no maze) | - | single |
| figslot | fig:crossover | (none: Shapes3D or no maze) | - | single |
| table | tab:impl-maze | CMAZE-2 | - | single |
| table | tab:impl | (none: Shapes3D or no maze) | - | single |
| table | tab:ablations | (none: Shapes3D or no maze) | - | single |
| figslot | fig:deletion | (none: Shapes3D or no maze) | - | single |
| table | tab:rivals-del | (none: Shapes3D or no maze) | - | single |
| table | tab:unlearn | (none: Shapes3D or no maze) | - | single |
| table | tab:rivals-del-seeds | (none: Shapes3D or no maze) | - | single |
| table | tab:adapt | (none: Shapes3D or no maze) | - | single |
| figslot | fig:adapt | (none: Shapes3D or no maze) | - | single |
| table | tab:encloop | (none: Shapes3D or no maze) | - | single |
| table | tab:bard-gpu | (none: Shapes3D or no maze) | - | single |
| table | tab:cmaze2 | (none: Shapes3D or no maze) | CMAZE-2 | single |
| table | tab:cmaze1 | (none: Shapes3D or no maze) | CMAZE-1 | single |
| figslot | fig:verb | CMAZE-2b | - | single |
| figslot | fig:maze-wm | CMAZE-T | - | single |
| table | tab:req | CMAZE-2 | CMAZE-1, CMAZE-2b | single |
| table | tab:req2b | CMAZE-2b | CMAZE-1, CMAZE-2 | single |
| table | tab:maze-del-2b | (none: Shapes3D or no maze) | CMAZE-1, CMAZE-2b, CMAZE-2 | single |
| table | tab:crr | (none: Shapes3D or no maze) | - | single |
| table | tab:cmazet,tab:cmaze-del | CMAZE-T | CMAZE-1, CMAZE-2b, CMAZE-2, CMAZE-O | single |
| figslot | fig:cmazet-families | CMAZE-T | - | single |
| figslot | fig:cmazet-door-tint | CMAZE-T | - | single |
| figslot | fig:cmazet-deletion | CMAZE-T | - | single |
| table | tab:cmazet-sym | CMAZE-T, CMAZE-O | CMAZE-1, CMAZE-2b, CMAZE-2 | SHARED |
| table | tab:cmazeo | CMAZE-O | CMAZE-1, CMAZE-2b, CMAZE-2, CMAZE-T | single |

**Floats whose data rows span more than one maze environment: 1.**

**Hand reading of the one flag.** `tab:cmazet-sym`: its only CMAZE-O token in the body is inside a wave-19
`\PLACEHOLDER{... may withdraw a margin, as on CMAZE-O}`, which is a cross-reference, not a CMAZE-O row. Every
row of that table is CMAZE-T with symmetric inputs, so its verdict is **single**. The placeholder itself is
wave 19's and was not touched, by instruction.
**M5 verdict: after this pass, no table or figure slot holds rows from two maze environments.**
CMAZE-1 / CMAZE-2 / CMAZE-2b / CMAZE-T / CMAZE-O each sit in their own floats. The splits themselves (`tab:req` /
`tab:req2b`, `tab:cmazet` / `tab:cmazeo`, `tab:impl-maze` / `tab:impl`, removal of `fig_maze_requestlayer`)
were already in the draft at this pass's start (base 148f3b7 drafts, SHA in before-*/). This pass made no M5
change: it verified the split.
