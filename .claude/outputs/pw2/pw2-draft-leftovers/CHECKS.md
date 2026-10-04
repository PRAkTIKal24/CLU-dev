# CHECKS — pw2-draft-leftovers (PW-2 item 4 + item-5 follow-up), before -> after

## DRAFT-v2

### \TODO: 4 -> 3  PASS (expected 4 -> 3)
- before line 126: `CMAZE-T sentence reserved for the Head: the six-seed symmetric-input c…` — stays
- before line 1022: `no record prints the byte ratio of the composition-only n-CLU to the s…` — **removed**
- before line 1303: `AI use statement --- the authors to confirm which of the venue's requi…` — stays
- before line 1325: `anonymised code archive not yet built.}…` — stays
- after line 126: `CMAZE-T sentence reserved for the Head: the six-seed symmetric-input c…`
- after line 1307: `AI use statement --- the authors to confirm which of the venue's requi…`
- after line 1329: `anonymised code archive not yet built.}…`

### \PLACEHOLDER: 2 -> 0  PASS (expected 2 -> 0)
- before line 2291: `per-model $s_{t+1}$ for the observation-only input of Tab.~\ref{tab:cm…` — **removed**
- before line 2328: `per-model $s_{t+1}$ on the door-then-tint transitions for the…` — **removed**

### Absolute paths (anywhere, comments included): 2 -> 0  PASS expected 0 after
- before line 12: `%     (called EV below; on the drafting machine it is ~/clu/evidence/outputs/).`
- before line 35: `% COMPILE:  pdflatex DRAFT-v2.tex (from ~/clu/outputs/w17-paper/, template in ./iclr2027/)`

### Figs. 5–8: `\ref` uses (code only)

| label | number | page before -> after | refs before | refs after | citing line (after) |
|---|---|---|---|---|---|
| `fig:ladder` | 5 | 12 -> 12 | 0 | 1 | l.757: `…printed score (Fig.~\ref{fig:ladder})` |
| `fig:crossover` | 6 | 27 -> 27 | 0 | 1 | l.1586: `…at three seeds distinguish from $K=24$ or $K=28$ (Fig.~\ref{fig:crossover})` |
| `fig:deletion` | 7 | 29 -> 29 | 0 | 2 | l.897: `…bination, on all three seeds (Tab.~\ref{tab:del}, Fig.~\ref{fig:deletion})`; l.937: `… at seeds 1 and 2 and against the LSTM at seed 2 (Fig.~\ref{fig:deletion})` |
| `fig:adapt` | 8 | 31 -> 31 | 0 | 1 | l.1049: `…e under a low-rank adapter (Tab.~\ref{tab:adapt}, Fig.~\ref{fig:adapt})` |

### Quote block

PASS `\begin{quote}\small` 1 -> 0; no `\small` inside the quote after

### Labels: 77 -> 77; missing []; new []; renumbered []  PASS none missing, new or renumbered
Pages moved (3): app:laws p.39→38, app:notrun p.42→41, sec:limits p.17→16

## DRAFT-v2-style

### \TODO: 4 -> 3  PASS (expected 4 -> 3)
- before line 130: `CMAZE-T sentence reserved for the Head: the six-seed symmetric-input c…` — stays
- before line 1017: `no record prints the byte ratio of the composition-only n-CLU to the s…` — **removed**
- before line 1291: `AI use statement --- the authors to confirm which of the venue's requi…` — stays
- before line 1313: `anonymised code archive not yet built.}…` — stays
- after line 130: `CMAZE-T sentence reserved for the Head: the six-seed symmetric-input c…`
- after line 1295: `AI use statement --- the authors to confirm which of the venue's requi…`
- after line 1317: `anonymised code archive not yet built.}…`

### \PLACEHOLDER: 2 -> 0  PASS (expected 2 -> 0)
- before line 2279: `per-model $s_{t+1}$ for the observation-only input of Tab.~\ref{tab:cm…` — **removed**
- before line 2316: `per-model $s_{t+1}$ on the door-then-tint transitions for the…` — **removed**

### Absolute paths (anywhere, comments included): 2 -> 0  PASS expected 0 after
- before line 18: `%     (called EV below; on the drafting machine it is ~/clu/evidence/outputs/).`
- before line 41: `% COMPILE:  pdflatex DRAFT-v2-style.tex (from ~/clu/outputs/w17-paper/, template in ./iclr2027/)`

### Figs. 5–8: `\ref` uses (code only)

| label | number | page before -> after | refs before | refs after | citing line (after) |
|---|---|---|---|---|---|
| `fig:ladder` | 5 | 13 -> 13 | 0 | 1 | l.755: `…printed score (Fig.~\ref{fig:ladder})` |
| `fig:crossover` | 6 | 28 -> 28 | 0 | 1 | l.1574: `…at three seeds distinguish from $K=24$ or $K=28$ (Fig.~\ref{fig:crossover})` |
| `fig:deletion` | 7 | 30 -> 30 | 0 | 2 | l.895: `…bination, on all three seeds (Tab.~\ref{tab:del}, Fig.~\ref{fig:deletion})`; l.935: `… at seeds 1 and 2 and against the LSTM at seed 2 (Fig.~\ref{fig:deletion})` |
| `fig:adapt` | 8 | 32 -> 32 | 0 | 1 | l.1043: `…e under a low-rank adapter (Tab.~\ref{tab:adapt}, Fig.~\ref{fig:adapt})` |

### Quote block

PASS `\begin{quote}\small` 1 -> 0; no `\small` inside the quote after

### Labels: 77 -> 77; missing []; new []; renumbered []  PASS none missing, new or renumbered
Pages moved (3): app:clue p.32→33, app:encloop p.31→32, app:notrun p.43→42


**0 FAIL**
