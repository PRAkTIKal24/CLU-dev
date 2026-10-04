# FINDINGS — pw1-draft-wording2 (PW-1 item 7 completion: the two seed-5 numbers)

Session `pw1-draft-wording2` · branch `pw1/draft-wording2` at `fbdcafb` (from `pw1/draft-wording` 29be0f8) · batmac · tooling `pw1_draft_wording2/` (verify_record.py, edit.py, check_derived.py). Stage `stage-123104/` (edited drafts, build/, STYLE-DIFF-CHECK.md, PROV-*, PROV-DERIVED.txt, SCAN-*, VOCAB-DELTA-*, PH-*, INDEX-*, PDF text). Snapshot `snap-before/`. Record check `VERIFY-RECORD.txt`.

## Status
| item | status |
|---|---|
| 7, first bullet: add the 0.04 margin and "106 of the 317 fit rows" to the seed-5 sentence in both drafts | **done**, installed byte-identically |
| replace the "% plan's authority alone" comment by a citation of the record | **done** |
| anything else | untouched (items 8 and 13 not re-opened; one anchored operation per draft) |

## The five numbers, verified verbatim (record `~/clu/evidence/outputs/w23-seed5-diagnosis/`; full printout VERIFY-RECORD.txt, ALL OK)
| value | file | path (JSON leaf preferred) |
|---|---|---|
| 0.04 | Q1-CONTENT.json | `codes[7]` (code [1,3,0,2]) `["clu.pkl"].oracle_disp_margin` = 0.0393 (leaf; rounds to 0.04) |
| 0.04 | W23-SEED5-DIAGNOSIS.json | `Q1.failing["2,0,1,2"].l2_oracle_margins[1]` = 0.0393; same for `"2,0,3,2"` |
| 0.04 | FINDINGS.md | printed verbatim l.27 ("The S-vs-E logit margin there is **0.04**"), l.70, l.77, l.120 |
| 317 | Q1-CONTENT.json | `codes[7].n_train` = 317; also W23-SEED5-DIAGNOSIS.json `Q1.failing[both].truth_code_fit_rows` = 317 |
| 211 | Q1-CONTENT.json | `codes[7].clu_in_sample_emits_top3` "(1, 3, 0, 2)" (S, the truth) → 211; FINDINGS.md l.30 "211 / 317", l.71 |
| 63 | Q1-CONTENT.json | same list, "(1, 2, 0, 2)" (E) → 63; FINDINGS.md l.71 "E 63" |
| 43 | Q1-CONTENT.json | same list, "(1, 1, 0, 2)" (N) → 43; FINDINGS.md l.71 "N 43" |
| 106 | arithmetic | 317 − 211 = 106 = 63 + 43; the three emits sum to 317 = n_train; `clu_in_sample_exact` 0.6656151419558359 = 211/317 |

The record agrees with the item-7 wording: CONTENT locus (FINDINGS l.22), one outcome code (legal, S, no event, pocket 2), "not a hold-out effect at all" (l.31), 10-atom relational design scores both tuples (already cited). Nothing contradicts; the STOP condition did not fire.

## The one sentence, before → after (identical in both drafts; DRAFT-v2 l.1008–1009 → l.1008–1012; style twin l.1003–1004 → l.1003–1007)
- **Before:** "… and the miss is a content-capacity failure of the 6-atom design on that code; the 10-atom relational design (n-CLU-REL, below) scores every row of both tuples valid." Comment: "mechanism as ruled in PW-1 PLAN item 7 …; the code's 0.04 decoding margin and its 106 of 317 wrong fit rows are in no record on this machine and are not printed."
- **After:** "… and the miss is a content-capacity failure of the 6-atom design on that code: the code decodes at the wrong side of a 0.04 margin in the deployed span and is already wrong on 106 of the 317 fit rows that carry it; the 10-atom relational design (n-CLU-REL, below) scores every row of both tuples valid."
  - `% SRC: w23-seed5-diagnosis/FINDINGS.md; w23-seed5-diagnosis/Q1-CONTENT.json` on the line carrying the numbers.
  - A comment line below gives "106 is 317 - 211 (equally 63 + 43), arithmetic, not printed in the record", along with every leaf path above.
  - The trailing comment now says the mechanism rests on the W23 record (FINDINGS l.22–33, l.66–78; W23-SEED5-DIAGNOSIS.json Q1), not on the plan alone. The earlier "Checked: …" list is kept.
- Wording choice: the plan's words are kept verbatim ("at the wrong side", "deployed span", "fit rows that carry it"). "Deployed span" is not defined in the paper's text; the comment defines it (the mean + 5 directions per well, FINDINGS l.25–26). **For the Head:** this may need a gloss when the prose is edited.
- W23-SEED5-DIAGNOSIS.json is deliberately left out of the SRC list. It contains the substring "106" by coincidence (in floats such as 0.10612554814735241), and citing it would make the provenance checker falsely "find" 106.

## Checks (stage stage-123104, built from the w17-paper layout)
| check | DRAFT-v2 before → after | DRAFT-v2-style before → after |
|---|---|---|
| exits pdflatex/bibtex/pdflatex | 0 → 0 | 0 → 0 |
| undefined citations / references | 0/0 → 0/0 | 0/0 → 0/0 |
| errors / overfull | 0/0 → 0/0 | 0/0 → 0/0 |
| pages | 42 → 42 | 42 → 42 |
| fig/sec/app numbers and pages (INDEX vs pw1-draft-wording's installed build) | identical | identical |
| checked number tokens | 2,594 → 2,597 (0.04, 106, 317) | 2,594 → 2,597 |
| provenance, raw checker: unsourced / not-found | 7/8 → 7/**9** | 7/9 → 7/**10** |
| the one added not-found token | "106 not in FINDINGS.md; Q1-CONTENT.json" (arithmetic, expected) | same |
| provenance, derivation-aware (check_derived.py) | 8 → 8, **identical**; 106 = 317 − 211 DERIVED-VERIFIED (both operands found in the cited files) | 9 → 9, identical |
| 0.04 and 317 | found in the cited files | found |
| `\PLACEHOLDER` / `\TODO` | 2/4 → 2/4, byte-identical, same order | 2/4 → 2/4 |
| style diff-check | ALL DIFFS EMPTY (numbers, placeholders, TODOs, artefact paths, provenance comments 440 = 440 in order, headings 94, labels 65, tables/figslots 31, preamble) | |
| rendered captions: banned words / "Figure pending" | 0 / 0 | 0 / 0 |
| vocabulary delta (scan.py) | none | none |

The raw provenance count does **not** stay identical: it gains exactly one token, 106. The record never prints 106, so no file lookup can find it. It is verified as arithmetic over two numbers that are in the cited file. **The Head decides whether that counts as "identical".**

## Installed
Before the install: the lock guard was clean (live sessions authtest, hubhost, pw1-draft-wording2, pw1-visuals-tooling), and all 4 live files were cmp-equal to the snapshot. After copying, all four were cmp-equal at the destination. Re-running edit.py on snap-before reproduces both installed .tex byte-for-byte (repro-*/).
| file | before | after |
|---|---|---|
| DRAFT-v2-style.pdf | `f9cf0ff15d96f008387689961851be2e8e94d1e9a2e665437f01dc6207b5a471` | `d908315a1f98a976eccf3cde6392268ade1bb1b04bfa0841b7fac96fcec5a702` |
| DRAFT-v2-style.tex | `3fa390b551853148699c05de3995974f1dd733d7dad2343440a8c0318cba0ca6` | `46fbd275c382608fa21b3e0d995f794c07201e9941a9ed259024d988ab38c8c7` |
| DRAFT-v2.pdf | `588d02f7c9708fbfbcdb6090499bd5ee27ef676b1fb618d5f0eaa17aef940069` | `4e3102e7504ad1ce9c303c7f50ef4a880a5cef4242fc525d82b90f86946b3803` |
| DRAFT-v2.tex | `2f720bae5a950f52816ee15871779980ecf882df661b3cf756370ff55d0ea568` | `8cc405af189adcab3c2974c779a7d0e50bee44686c7ac096fd8078b3f89fa64a` |

## What did NOT run
- No measurement, fit, GPU or model load. The record was only read.
- No figure, `figures/`, `paper-visuals/` or `references.bib` touched. Stale .aux/.bbl/.log in w17-paper/ were not rebuilt in place (only the two .tex and two PDFs installed, as in earlier passes).
- The shared checker `pw1_draft_sweeps/check_provenance.py` was not modified; the derivation check is a separate reader of its output.
- No prose beyond the one sentence and its comments. "Deployed span" is left unglossed (plan wording verbatim).
- Items 8 and 13, the abstract `\TODO`, MF-11 and MF-12 were not touched.

## Branch and bundle
`pw1/draft-wording2` at fbdcafb (tooling only). Bundle `pw1-draft-wording2.bundle` sha256 `dc6938ae20bf86c1b2352303c33f3d73daf588fd63458d1ef0689c24c8512e97`; `git bundle verify`: okay.
