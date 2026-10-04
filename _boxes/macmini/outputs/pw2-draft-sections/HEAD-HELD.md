# HEAD-HELD — the paragraphs no later PW-2 spoke may change (except the item-3 and item-1 borrowings, whose spokes update this list)

Written by `pw2-draft-sections` after item 2 (re-sectioning). sha256 over the exact source text (lines joined by `\n`, no trailing newline). "First paragraph of a section" = the first prose chunk after the `\section` line (a float-only chunk, e.g. the two task figures at the top of Experiments, is not a paragraph; it is listed where it precedes). Re-check with `python3 pw2_draft_sections/headheld.py PATH/DRAFT-v2.tex ~/clu/outputs/pw2-draft-sections/HEAD-HELD.md` (and the same for DRAFT-v2-style.tex); exit 1 on any difference.

## DRAFT-v2.tex

| paragraph | lines | first 8 words | sha256 |
|---|---|---|---|
| abstract, opening sentence | 88–91 | `A world model that is to be trusted` | c5cb27b93dff256a438fce84b6541cab5421dc1350fc9e50e6ec4f6d1c4cf79d |
| abstract, maze `\TODO` (whole lines) | 128–130 | `\TODO{CMAZE-T sentence reserved for the Head: the six-seed` | f72b7276558b4a1deb43a0b5e9e8d139b7424b12c5434f486ec6633ed06c506b |
| Introduction, first paragraph | 144–150 | `The naive-CLU (n-CLU) is a \emph{naive AI unit}:` | 24ffe2c29c1044fa316e94df05c87f56c8d9cae3564cecbfd1dbb6f848ef3eb4 |
| Methods, first paragraph = the item-3 slot (to be replaced by the Head's opening) | 273 | `% PW2-ITEM3-SLOT` | 23fb15bd329b461240278dca1f9ea6df95ef098321386379f83a7172b480a845 |
| Experiments, first paragraph (preceded by the float chunk at lines 356–390) | 395–410 | `The deletable object is the \emph{single design}: the` | aa8259a746f3323c1771825267aded2d3a53436e536dd5f5b082f238e0101a02 |
| Results, first paragraph | 492–513 | `Tab.~\ref{tab:t7} is the scored result. Row 3 holds` | 03e6b622aea3eeb6de1c032504d7603a720baba3b5eb65b841001d7079dae240 |
| Discussion, first paragraph | 1111–1153 | `\begin{enumerate}` | 7db7fe74790424a7cf70075f3aaefa38039623364f1fcd90ccba8df212e9e6b5 |

## DRAFT-v2-style.tex

| paragraph | lines | first 8 words | sha256 |
|---|---|---|---|
| abstract, opening sentence | 94–96 | `A world model trusted with requests it has` | 95017d4bdb3ce805a20cb869748e671320c5b11fc43f84e472a44ab4535e3a77 |
| abstract, maze `\TODO` (whole lines) | 131–133 | `\TODO{CMAZE-T sentence reserved for the Head: the six-seed` | f72b7276558b4a1deb43a0b5e9e8d139b7424b12c5434f486ec6633ed06c506b |
| Introduction, first paragraph | 146–152 | `The naive-CLU (n-CLU) is a \emph{naive AI unit}:` | 0fff1056c10c184f4ecdbc2d05ec9da7e7afda14bca99da78c1787af29bdeea1 |
| Methods, first paragraph = the item-3 slot (to be replaced by the Head's opening) | 273 | `% PW2-ITEM3-SLOT` | 23fb15bd329b461240278dca1f9ea6df95ef098321386379f83a7172b480a845 |
| Experiments, first paragraph (preceded by the float chunk at lines 356–390) | 395–410 | `This section measures the three properties on Shapes3D.` | a21840ba9f72f07f573fa58a6894489cb1842ef4faaf62f632c0c99e2f25c483 |
| Results, first paragraph | 492–513 | `Tab.~\ref{tab:t7} is the scored result. Row 3 holds` | 03e6b622aea3eeb6de1c032504d7603a720baba3b5eb65b841001d7079dae240 |
| Discussion, first paragraph | 1103–1103 | `We now state what the naive unit does` | 2eb7da66068d7e875e5c0b4d38e60fe9060e300dcaa22f7f45b0c553f518d4ac |

