# PREREG — hepa-lh-sweep (results-analyst, written BEFORE any training run)
Written 2026-09-07, before `run_hepa.py` was executed even once. Governing predictions for the
*datasets* are the Hub's frozen `.claude/outputs/hepa-longhorizon-triage/PREREG.md` §5 — I do not
restate or amend them. This file registers only the two things that frozen file does not cover:
(a) my point predictions for the **reproduction**, and (b) the curve-shape call.

## A · Reproduction (task §3) — my point predictions, blind
| arm | HEPA's stated value | my predicted reproduction (3 seeds, our harness/cut points) | interval |
|---|---|---|---|
| FD001 @512 h-AUROC | 0.918 ± 0.008 | **0.92** | [0.86, 0.96] |
| GECCO @512 h-AUROC | 0.888 ± 0.024 | **0.86** | [0.75, 0.94] |

Reasoning: `REPRODUCIBILITY.md` says these are "the values a clone of this repo will produce", the
C-MAPSS finetune is fixed-epoch (deterministic given seed), and I am running the clone's own code
through its own loaders. The declared risks that could move it: (i) 3 seeds not 5; (ii) MPS/CPU
float kernels, not CUDA; (iii) GECCO uses **val-loss early stopping**, so its seed spread should be
much wider than FD001's.
**The decision this feeds (task §3):** near 0.918 ⇒ the wave-1 tripwire's +0.060 on FD001 stands;
near 0.978 ⇒ it evaporates. I predict **it stands**.

## B · Curve shape (task §6)
I predict **FLAT curves on all four datasets** — h-AUROC at 2048/8192 inside the 512 seed spread —
because the audit measured the most recent analogous regime at 46–193 native steps on every one of
them. Specifically I predict `|h-AUROC(longest) − h-AUROC(512)| < seed_spread(512)` for ETTm1,
Beijing-AQ, PSM and Weather.
Registered counter-possibility (what would surprise me): a **monotone rise** on PSM, whose shipped
context is 100 — going 100 → 512 is the one rung step in this sweep that is *inside* the measured
46–61-step similar-regime window and could plausibly buy real signal. I therefore predict PSM
**100 → 512 rises** and **512 → 2048 → 8192 is flat**. That is a rise from *undersizing*, not from
long history, and I will label it as such.

## C · Confound registered in advance
P=16 at every rung ⇒ n_tokens = 7 / 32 / 128 / 512 at rungs 100 / 512 / 2048 / 8192. Any rise is
therefore confounded with capacity (more tokens), and wave 1 already showed the tripwire gaining
+0.004 h-AUROC from *bytes alone* with zero extra history. A rise is not evidence for history until
the token count is controlled.
