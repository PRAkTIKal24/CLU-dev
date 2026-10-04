# figures/rendered/: w17 spoke `rendered-compositions` (the only writer of this directory)

## STATUS: RENDERED. All four figures are real renders.
Every panel was rendered on batmac from the Hub-staged artefacts (`/Users/batmac/clu/stage/outputs`,
mirroring `C:/chlu/outputs`) by
`python -m w17_rendered_compositions.render --render` (branch `w17/rendered-compositions`).
Only seed 0 is rendered, at K\* = 32. A picture shows one seed. The rates over n = 3 seeds belong in Table T7.

## Same frozen decoder: ASSERTED in code (`SAME-DECODER.json`)
* One decoder object (`decoder_v1_s0.pkl`) and one external classifier are loaded once through the
  scorer's own `exp_gen_factor_wells.Judge`, which checks each hash against its manifest. The decoder
  hash is checked a second time against the **evidence base's** `gen-decoder-lift/winner/winner_hashes.json`
  (seed_0 `param_hash` `01ef91f7…`). The staged copy of that manifest is byte-identical to the evidence copy.
* Every panel carries a stamp with the decoder hash and the Python identity of the object that
  rendered it. `assert_same_decoder` refuses to proceed unless every panel carries ONE hash, ONE
  object and ONE seed. It passed on all panels of all four figures, across the CLU, the CLU after
  the downdate, 4 rivals as trained, 4 rivals retrained without the rows, and the lookup floor.
* **Positive control:** the render→classify path equals `Judge.read_back` element for element
  on real CLU features.
* **Reproduction:** before any figure is written, every rendered corner must reproduce the
  scored `n_valid` per arm from `scored.json`, or the run stops. It reproduced. The broader
  scan (`COVALID-SCAN.json`) re-rendered **every** held-out corner at seeds 0, 1 and 2 and reproduced
  every per-corner `n_valid` in the three T7 `scored.json` files, with zero mismatches.
* **Deletion:** the staged `features.npy` sha256 equals its manifest (`ef6359a4…`, generation 2).
  The downdated CLU reproduces the banked `deletion.json` *after* counts on the set-1 corners.

## Request tuples: `REQUESTS.json`
Each rendered request has its dataset row, exact 6-factor values, per-arm verdict,
acceptance, the judge's read-back tuple, the selection rules, the reproduction records and the
declarations.

## The four figures (PDF + PNG each)
| file | set | what it shows (seed 0, held-out unless stated) |
|---|---|---|
| `rc_set1_clu_composes_rivals_fail` | 1 | `pair-001/f4-hh` (2-way corner, large + tilted) ×2 and `nway-034` (6-factor box) ×2: CLU ✓, every rival and the floor ✗ |
| `rc_set2_all_arms_succeed` | 2 | **The all-six set is DECLARED EMPTY** (see below). The figure shows the nearest pattern: CLU, MLP, LSTM, Transformer and floor ✓, **continuous MLP ✗**. Rows come from `pair-000/f1-ll`, `pair-002/f2-hh`, `pair-002/f4-hh` and `nway-000`. |
| `rc_set3_all_arms_fail` | 3 | `pair-002/f1-lh` ×2, `pair-002/f3-lh` ×2, and one `nway-006` request found by scan: every arm ✗, **the CLU included** |
| `rc_deletion_red_sphere` | deletion | Fit row 10064 (red sphere, largest scale, then largest orientation). Top row: CLU and each rival as trained. Bottom row: CLU store downdate and each rival retrained without all red-sphere rows |

### Set 2: declared, with the reason
At **no** seed (0, 1 or 2) is any held-out test request valid for all six columns. At every seed,
every request with 5 of 6 columns valid fails only on the **continuous MLP** (`COVALID-SCAN.json`,
`hist_n_arms_co_valid` and `five_or_more`). A SEEN sample of 512 fit rows gives no all-six request either,
because the continuous MLP is valid on none of them (`/Users/batmac/clu/outputs/w17-rendered-compositions/diag/set2_diag.json`).
The figure therefore shows "every arm except the continuous MLP" with its ✗ in place and says so in its title.
The filename is unchanged so that `\includegraphics` paths stay stable.
⚠ These counts come from this spoke's own re-render, **not** from the evidence base. They are
recorded here and **not** in any caption as a number.

### Deletion panel: what it shows, and what it does not show
The CLU renders the deleted red-sphere row validly both **before and after** the downdate.
Each of the four rivals renders it invalidly both **as trained and retrained without**. The judge's read-back puts
their miss on scale and orientation, not on hue or shape (`REQUESTS.json` `deletion.verdicts.*.judged_tuple`).
The request was fixed **before** rendering, by the rule written into the placeholder `REQUESTS.json`:
largest scale, then largest orientation. Because it sits at the extreme of scale and orientation, it is also where the rivals are weakest.
**The panel is not evidence that anything was forgotten.** The banked deletion claim is store-scoped
exactness plus a light, data-free adaptation (`deletion.json`). The CLU can still compose a red sphere
from "red" and "sphere" in the surviving rows, which is composition working, not deletion failing.
Sphere caveat: a degraded render is judged sphere-like (`deletion.json`), so a sphere verdict after
deletion is not proof of retention. The lookup floor is left out because w12 has no
retrain-without comparator for it.

## LaTeX (drop-in)
```latex
% SOURCE (selection): evidence/outputs/w12-single-design-crossover/scored/run1_seed0_K32/scored.json (per_arm.*.per_corner)
% SOURCE (decoder hash): evidence/outputs/gen-decoder-lift/winner/winner_hashes.json (seed_0.param_hash)
% SOURCE (renders, tuples, same-decoder assertion): figures/rendered/{REQUESTS,SAME-DECODER,STATUS}.json
\begin{figure}[t]
  \centering
  \includegraphics[width=\linewidth]{figures/rendered/rc_set1_clu_composes_rivals_fail.pdf}
  \caption{Held-out 2-way (top two rows) and six-factor (bottom two rows) requests, rendered for the
  CLU, every rival and the lookup floor through one frozen decoder (hash-asserted per panel). The
  mark is the judge's verdict on that render. Seed 0 only; rates over seeds are in Table~T7. Corners were
  chosen by rule from the scored per-corner verdicts: the CLU valid on every request of the corner and
  every rival and the floor on none. The companion sets, where every arm succeeds and where every arm
  fails, are in Appendix~\ref{app:rendered}.}
  \label{fig:rendered-set1}
\end{figure}
% Appendix \label{app:rendered}:
% rc_set2_all_arms_succeed.pdf -- caption: "No held-out request is rendered validly by all six arms.
%   These are held-out requests that every arm except the continuous MLP renders validly; the
%   continuous MLP's failure is shown."
% rc_set3_all_arms_fail.pdf -- caption: "Held-out requests on which every arm fails, the CLU included."
% rc_deletion_red_sphere.pdf -- caption: "A fit row carrying a red sphere, before and after deleting
%   every red-sphere fit row. CLU: store downdate. Rivals: as trained / retrained without. The CLU
%   still renders the request after deletion, because it composes red and sphere from surviving rows;
%   the rivals miss this extreme-scale, extreme-orientation request both before and after. The panel
%   illustrates the operation; the deletion claim is the store-scoped exactness in Table~X."
%   % SOURCE: evidence/outputs/w12-single-design-crossover/deletion/test_seed0_K32/deletion.json
```
