# HEAD-HELD — the paragraphs no later PW-2 spoke may change (updated by pw2-draft-tells after item 1 and the item-4 borrowings)

Updated from `~/clu/outputs/pw2-draft-methods/HEAD-HELD.md` (unchanged through pw2-draft-numbers). The style twin is unchanged: all 7 rows byte-identical to the previous list. DRAFT-v2 changes on exactly the three rows the borrowings touch (plan item 4, last bullet; brief: "the abstract's first sentence counts as part of the Head-held abstract opening"):

- **abstract, opening sentence**: v2 now carries the style twin's first sentence verbatim (B1), with the style twin's line layout, so its sha256 equals the style row (`95017d4b…`).
- **Introduction, first paragraph**: the only change is the sentence "We do not claim the best compositional-generation system or the best unlearning system." → the style twin's "We claim neither the best compositional-generation system nor the best unlearning system." (B4); every other byte of the paragraph is unchanged.
- **Discussion, first paragraph**: v2 had no Discussion signpost, so its first paragraph was the limitations list. B3 inserts the style twin's signpost at the same place, so v2's Discussion first paragraph is now that sentence (sha256 equal to the style row, `2eb7da66…`). The limitations list is no longer Head-held in either draft.
- Methods, Experiments, Results, the abstract maze `\TODO`: unchanged in both drafts.

Check: `python3 pw2_draft_sections/headheld.py DRAFT.tex HEAD-HELD.md` (exit 1 on any difference).

## DRAFT-v2.tex

| paragraph | lines | first 8 words | sha256 |
|---|---|---|---|
| abstract, opening sentence | 88–90 | `A world model trusted with requests it has` | 95017d4bdb3ce805a20cb869748e671320c5b11fc43f84e472a44ab4535e3a77 |
| abstract, maze `\TODO` (whole lines) | 126–128 | `\TODO{CMAZE-T sentence reserved for the Head: the six-seed` | f72b7276558b4a1deb43a0b5e9e8d139b7424b12c5434f486ec6633ed06c506b |
| Introduction, first paragraph | 142–148 | `The naive-CLU (n-CLU) is a \emph{naive AI unit}:` | 0560c09a3a62695e872ace1d3f7dd83ff639225bcd4d02d1a76cb45bd0923ea2 |
| Methods, first paragraph | 274–283 | `\citet{jawahar2026chlu} introduced the CHLU as a latent dynamics` | c100fed55e462eb5fa87694eeb063198605539b44cb28a2eeab53e2ab81e2988 |
| Experiments, first paragraph (preceded by the float chunk at lines 369–403) | 408–423 | `The deletable object is the \emph{single design}: the` | aa8259a746f3323c1771825267aded2d3a53436e536dd5f5b082f238e0101a02 |
| Results, first paragraph | 508–529 | `Tab.~\ref{tab:t7} is the scored result. Row 3 holds` | 03e6b622aea3eeb6de1c032504d7603a720baba3b5eb65b841001d7079dae240 |
| Discussion, first paragraph | 1172–1172 | `We now state what the naive unit does` | 2eb7da66068d7e875e5c0b4d38e60fe9060e300dcaa22f7f45b0c553f518d4ac |

## DRAFT-v2-style.tex

| paragraph | lines | first 8 words | sha256 |
|---|---|---|---|
| abstract, opening sentence | 94–96 | `A world model trusted with requests it has` | 95017d4bdb3ce805a20cb869748e671320c5b11fc43f84e472a44ab4535e3a77 |
| abstract, maze `\TODO` (whole lines) | 130–132 | `\TODO{CMAZE-T sentence reserved for the Head: the six-seed` | f72b7276558b4a1deb43a0b5e9e8d139b7424b12c5434f486ec6633ed06c506b |
| Introduction, first paragraph | 145–151 | `The naive-CLU (n-CLU) is a \emph{naive AI unit}:` | 0fff1056c10c184f4ecdbc2d05ec9da7e7afda14bca99da78c1787af29bdeea1 |
| Methods, first paragraph | 275–284 | `\citet{jawahar2026chlu} introduced the CHLU as a latent dynamics` | c100fed55e462eb5fa87694eeb063198605539b44cb28a2eeab53e2ab81e2988 |
| Experiments, first paragraph (preceded by the float chunk at lines 367–401) | 406–421 | `This section measures the three properties on Shapes3D.` | a21840ba9f72f07f573fa58a6894489cb1842ef4faaf62f632c0c99e2f25c483 |
| Results, first paragraph | 506–527 | `Tab.~\ref{tab:t7} is the scored result. Row 3 holds` | 03e6b622aea3eeb6de1c032504d7603a720baba3b5eb65b841001d7079dae240 |
| Discussion, first paragraph | 1166–1166 | `We now state what the naive unit does` | 2eb7da66068d7e875e5c0b4d38e60fe9060e300dcaa22f7f45b0c553f518d4ac |
