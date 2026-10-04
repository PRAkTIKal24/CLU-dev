# figures/rendered/ — w17 spoke `rendered-compositions` (sole owner of this directory)

## STATUS: DECLARED PLACEHOLDER — NOTHING HERE IS A RENDER YET
The frozen decoder, the external classifier (the judge), the CLU arm objects (`arm.pkl`) and the
rival weights (`*.rival.eqx`) are **not on batmac**. The evidence base holds only JSON/MD (0
checkpoint/array files). Exact paths: `NEEDS-ARTEFACTS.json`. **No decoder was substituted.**
Every figure carries a red "DECLARED PLACEHOLDER — NOT RENDERED" banner and hatched panels.

To finish (once the Hub stages the files under `$W17_RC_STAGE`, mirroring `C:/chlu/outputs`):
```
cd <worktree w17/rendered-compositions>; source ~/clu/env_bm.sh
export W17_RC_STAGE=<staged root> CHLU_GEN_DATA_ROOT=<dir with shapes3d/value_holdout>
python -m w17_rendered_compositions.render --render
```
It refuses (with the missing-file list) until every file is staged; it overwrites these figures,
`REQUESTS.json` (exact 6-factor tuples) and `STATUS.json`, and writes `SAME-DECODER.json`.

## What is real already
* **The corner selection** — by rule, from the T7 scored per-corner verdicts (K\* = 32, seed 0),
  `python -m w17_rendered_compositions.selection`; recorded in `REQUESTS.json.selection`.
* **The verdict marks on sets 1 and 3** — fixed by the selection rule from those counts.
* **The same-decoder guard** — `w17_rendered_compositions/same_decoder.py`, tested in
  `tests/test_w17_rendered_same_decoder.py` (5 passed on batmac). At render time it also asserts
  that re-derived `n_valid` per arm per corner equals the scored.json, and that its
  render→classify path equals `Judge.read_back` element for element.

## The four figures
| file | set | corners (seed 0) | verdict pattern |
|---|---|---|---|
| `rc_set1_clu_composes_rivals_fail` | 1 | `pair-001/f4-hh` (2-way, large + tilted), `nway-034` (six-factor box) | CLU ✓, every rival + floor ✗ |
| `rc_set2_all_arms_succeed` | 2 | candidate `nway-029` only | every arm ✓ — **may be EMPTY at request level**; then SEEN fallback, labelled, or DECLARED |
| `rc_set3_all_arms_fail` | 3 | `pair-002/f1-lh`, `pair-002/f3-lh` (+ an n-way request if a scan finds one) | every arm ✗ incl. the CLU |
| `rc_deletion_red_sphere` | deletion | one red-sphere fit row (largest scale, then orientation) | CLU before/after downdate vs rival as-trained / retrained-without |

Set 3 has **no n-way corner** on the counts: every seed-0 n-way corner has at least one rival or
floor success. Set 2 exists on the counts only in the necessary sense: the continuous MLP is valid
on at most one request of `nway-029` at seed 0.

## LaTeX (drop-in; compiles against the placeholders)
```latex
% SOURCE (selection): evidence/outputs/w12-single-design-crossover/scored/run1_seed0_K32/scored.json (per_arm.*.per_corner)
% SOURCE (seed count of the rates, NOT of this figure): evidence/outputs/w12b-nr6-and-kstar/NR6-T7.md
% STATUS: PLACEHOLDER until figures/rendered/STATUS.json says RENDERED.
\begin{figure}[t]
  \centering
  \includegraphics[width=\linewidth]{figures/rendered/rc_set1_clu_composes_rivals_fail.pdf}
  \caption{\PLACEHOLDER{rendered set 1 --- held-out 2-way and six-factor requests, CLU beside every
  rival and the lookup floor through the one frozen decoder, judge verdict per panel; seed 0 only,
  n = 1 seed rendered, the rates are Table~T7's}. Corners were chosen by rule from the scored
  per-corner verdicts: the CLU valid on every request of the corner and every rival and the floor
  on none. Sets 2 (all arms succeed) and 3 (all arms fail) are in Appendix~\ref{app:rendered}.}
  \label{fig:rendered-set1}
\end{figure}
% Appendix:
% \includegraphics{figures/rendered/rc_set2_all_arms_succeed.pdf}
%   caption: \PLACEHOLDER{set 2: held-out requests every arm renders validly, OR SEEN requests
%   (labelled), OR a declared empty set with its counts}
% \includegraphics{figures/rendered/rc_set3_all_arms_fail.pdf}
%   caption: \PLACEHOLDER{set 3: held-out requests on which every arm, the CLU included, fails}
% \includegraphics{figures/rendered/rc_deletion_red_sphere.pdf}
%   caption: \PLACEHOLDER{red-sphere request before/after deleting every red-sphere fit row:
%   CLU store downdate (store-scoped exactness; frozen encoder/decoder outside the deletable
%   object; adaptation light and data-free) vs each rival as trained and retrained without.
%   The CLU may still compose the red sphere from "red" and "sphere" after deletion ---
%   composition working, not deletion failing. Sphere caveat: a degraded render is judged
%   sphere-like, so a sphere verdict after deletion is not proof of retention
%   (deletion.json sphere_caveat).}
```
