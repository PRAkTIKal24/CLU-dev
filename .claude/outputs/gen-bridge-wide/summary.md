seeds: [0, 1, 2]

## pooled (macro over the 13 cells), mean ± std over seeds
| quantity | held-out | seen normaliser | REAL images |
|---|---|---|---|
| balanced_accuracy | 0.9798 ± 0.0088 | 0.9774 ± 0.0032 | 1.0000 ± 0.0000 |
| auroc | 0.9994 ± 0.0004 | 0.9989 ± 0.0003 | 1.0000 ± 0.0000 |
| tpr | 0.9694 ± 0.0177 | 0.9645 ± 0.0063 | 1.0000 ± 0.0000 |
| tnr | 0.9902 ± 0.0005 | 0.9902 ± 0.0005 | 1.0000 ± 0.0000 |
| **ratio (balacc held/seen)** | 1.0025 ± 0.0064 | per-seed 1.0059 / 0.9936 / 1.0079 | |

verdicts: ['PASS', 'PASS', 'PASS']  (floor=0.8, abs_floor=0.75)

## 4-way shape (⛔ a DIFFERENT quantity)
| pos | balacc 0.9686 ± 0.0185 | acc 0.9694 ± 0.0177 | n=936 |
| seen_pos | balacc 0.9668 ± 0.0065 | acc 0.9696 ± 0.0057 | n=2565 |

## per cell (mean ± std over seeds)
| cell | contrast | shape | n_pos/n_neg/n_seen | balacc held | balacc seen | ratio | AUROC | detection TPR | balacc on REAL | probe-on-real(shape) |
|---|---|---|---|---|---|---|---|---|---|---|
| montero | 0.050 | 1 | 72/141/141 | 1.0000 ± 0.0000 | 0.9846 ± 0.0089 | 1.0157 ± 0.0091 | 1.0000 ± 0.0000 | 1.0000 ± 0.0000 | 1.0000 ± 0.0000 | 1.0000 ± 0.0000 |
| wide-000 | 0.050 | 0 | 72/141/141 | 0.8542 ± 0.0967 | 0.8743 ± 0.0237 | 0.9748 ± 0.0870 | 1.0000 ± 0.0000 | 0.7083 ± 0.1934 | 1.0000 ± 0.0000 | 1.0000 ± 0.0000 |
| wide-001 | 0.100 | 1 | 72/215/215 | 0.9470 ± 0.0097 | 0.9377 ± 0.0068 | 1.0099 ± 0.0035 | 0.9920 ± 0.0056 | 1.0000 ± 0.0000 | 1.0000 ± 0.0000 | 1.0000 ± 0.0000 |
| wide-002 | 0.200 | 2 | 72/214/214 | 0.9992 ± 0.0011 | 0.9992 ± 0.0011 | 1.0000 ± 0.0000 | 1.0000 ± 0.0000 | 1.0000 ± 0.0000 | 1.0000 ± 0.0000 | 1.0000 ± 0.0000 |
| wide-003 | 0.300 | 3 | 72/214/214 | 1.0000 ± 0.0000 | 1.0000 ± 0.0000 | 1.0000 ± 0.0000 | 1.0000 ± 0.0000 | 1.0000 ± 0.0000 | 1.0000 ± 0.0000 | 1.0000 ± 0.0000 |
| wide-004 | 0.400 | 0 | 72/214/214 | 0.9938 ± 0.0088 | 0.9751 ± 0.0079 | 1.0192 ± 0.0105 | 1.0000 ± 0.0000 | 1.0000 ± 0.0000 | 1.0000 ± 0.0000 | 1.0000 ± 0.0000 |
| wide-005 | 0.450 | 1 | 72/215/215 | 0.9583 ± 0.0300 | 0.9992 ± 0.0011 | 0.9591 ± 0.0293 | 1.0000 ± 0.0000 | 0.9167 ± 0.0600 | 1.0000 ± 0.0000 | 1.0000 ± 0.0000 |
| wide-006 | 0.050 | 2 | 72/179/179 | 1.0000 ± 0.0000 | 1.0000 ± 0.0000 | 1.0000 ± 0.0000 | 1.0000 ± 0.0000 | 1.0000 ± 0.0000 | 1.0000 ± 0.0000 | 1.0000 ± 0.0000 |
| wide-007 | 0.100 | 3 | 72/179/179 | 0.9912 ± 0.0088 | 0.9953 ± 0.0013 | 0.9958 ± 0.0080 | 1.0000 ± 0.0000 | 0.9861 ± 0.0196 | 1.0000 ± 0.0000 | 1.0000 ± 0.0000 |
| wide-008 | 0.200 | 0 | 72/211/211 | 0.9938 ± 0.0072 | 0.9403 ± 0.0192 | 1.0572 ± 0.0149 | 0.9999 ± 0.0001 | 0.9907 ± 0.0131 | 1.0000 ± 0.0000 | 1.0000 ± 0.0000 |
| wide-009 | 0.300 | 1 | 72/216/216 | 1.0000 ± 0.0000 | 1.0000 ± 0.0000 | 1.0000 ± 0.0000 | 1.0000 ± 0.0000 | 1.0000 ± 0.0000 | 1.0000 ± 0.0000 | 1.0000 ± 0.0000 |
| wide-010 | 0.400 | 2 | 72/215/215 | 1.0000 ± 0.0000 | 1.0000 ± 0.0000 | 1.0000 ± 0.0000 | 1.0000 ± 0.0000 | 1.0000 ± 0.0000 | 1.0000 ± 0.0000 | 1.0000 ± 0.0000 |
| wide-011 | 0.450 | 3 | 72/211/211 | 1.0000 ± 0.0000 | 1.0000 ± 0.0000 | 1.0000 ± 0.0000 | 1.0000 ± 0.0000 | 1.0000 ± 0.0000 | 1.0000 ± 0.0000 | 1.0000 ± 0.0000 |

## contrast
OLS balacc = 0.9597 + +0.0858 * contrast   (n=39 cell-seeds, pearson r=0.265)
  contrast 0.050: balacc 0.9514 ± 0.0886  (cells ['montero', 'wide-000', 'wide-006'])
  contrast 0.100: balacc 0.9691 ± 0.0239  (cells ['wide-001', 'wide-007'])
  contrast 0.200: balacc 0.9965 ± 0.0058  (cells ['wide-002', 'wide-008'])
  contrast 0.300: balacc 1.0000 ± 0.0000  (cells ['wide-003', 'wide-009'])
  contrast 0.400: balacc 0.9969 ± 0.0070  (cells ['wide-004', 'wide-010'])
  contrast 0.450: balacc 0.9792 ± 0.0297  (cells ['wide-005', 'wide-011'])
bottom 3 cells by balanced accuracy: ['wide-000', 'wide-001', 'wide-005']
/Users/user/Desktop/CHLU/.claude/scratch/gen-bridge-wide/summarize.py:124: UserWarning: Glyph 11088 (\N{WHITE MEDIUM STAR}) missing from font(s) DejaVu Sans.
  fig.tight_layout()
/Users/user/Desktop/CHLU/.claude/scratch/gen-bridge-wide/summarize.py:124: UserWarning: Glyph 9940 (\N{NO ENTRY}) missing from font(s) DejaVu Sans.
  fig.tight_layout()
/Users/user/Desktop/CHLU/.claude/scratch/gen-bridge-wide/summarize.py:125: UserWarning: Glyph 11088 (\N{WHITE MEDIUM STAR}) missing from font(s) DejaVu Sans.
  fig.savefig(OUT / "ceiling_vs_contrast.png", dpi=140)
/Users/user/Desktop/CHLU/.claude/scratch/gen-bridge-wide/summarize.py:125: UserWarning: Glyph 9940 (\N{NO ENTRY}) missing from font(s) DejaVu Sans.
  fig.savefig(OUT / "ceiling_vs_contrast.png", dpi=140)

wrote /Users/user/Desktop/CHLU/.claude/outputs/gen-bridge-wide/ceiling_vs_contrast.png and summary.json
