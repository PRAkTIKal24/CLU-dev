# w22 `spoke/figures-redraw` PASS 2 — FINDINGS (batmac, 2026-09-24)

**Zero scored runs. Nothing was written to any ledger. No PREREG, and no bar or label was declared before a number.**
Every value drawn is read from an artefact that already existed. The artefact path and its SHA-256 are in `PROVENANCE.json`.

## Delivered
| # | brief item | result |
|---|---|---|
| 1 | Ladder + headline: add the w20 131,072-row set; print all rival sets; **pool AND steps in every header** | `fig_ladder` has three panels. The (a) and (b) panels are unchanged: **32,256 rows · 5,040 steps**, then **32,768 rows · 2,560 steps**. New panel (c) is the w20 set at **131,072 rows · 20,480/204,800 steps**. Each step count is asserted from its artefact on every rung, seed and family. The 32,768-row `ladder-rivals-32k` slot is a manifest entry. It was tested with a stand-in 4th set: the figure becomes a 2×3 grid with no code change. NEW `fig_headline_steps`: rows 3–4, rival score against steps at 16,384 / 32,768 / 131,072 rows. The MLP at 20,480 steps is within one tuple per corner of itself across all three pools. |
| 2 | `fig_maze_conditions`: six seeds, two configs, per-seed ceilings, hard/easy by ceiling | Panel (a) is the condition table plus the CLU-REL-A10 row (seeds 0–2, Head's rule: WIN; R\* = `rival@10x-steps: transformer`, asserted equal to `LABEL-BOTH-test.json`). Panels (b) A1 and (c) CLU-REL-A10 each show **six seeds**. Each seed has a ceiling bar, the best matched rival (max of 6 arms on that seed), the best **+rel** rival (REL panel only) and the highest control. Hard seeds 1 and 4 are shaded, because the A1 ceiling falls below 0.6407 (`MAZE-T-SEED4.json`). All 54 per-seed numbers recomputed from the run JSONs are float-equal to `REL-NEWSEEDS.json`. |
| 3 | Improved config's CMAZE-T panels, same hash gate | **Gate 1 PASSED:** 96/96 + 96/96 REL objects (390 files). The staged hashes also equal the peer replay's own records (0 mismatches). **Gate 2 could NOT run** (see below), so **no REL per-row panel is drawn**. The slot is `fig_cmazet_rel_per_arm` (not_landed), and `NEEDS-ARTEFACTS.json` is written. Drawn instead from existing artefacts only: NEW `fig_maze_rel_tuples`, per held-out tuple × 6 seeds, from the hash-linked zero-fit replay. |
| 4 | Deletion under the four-measurement rule, with (d) before→after and the random-1 % control | `fig_deletion` is redrawn from w20 `deletion_matched` (book **refit**, 131,072 pool). The panels are (a) the store equals its refit-without, (b) the book equals its refit-without, (c) the refusal display, (d) before→after, with **1 % items (control) beside red ∧ sphere**, (e) paired Δ with CI, and (f) against each rival retrained without the rows. |

## ⛔ The rows where the CLU does NOT win are drawn and marked
- **Seed 4 (CLU-REL-A10):** `control: random_atoms` = CLU = **0.9004149377593361** exactly. It equals the CLU on **16/16** held-out tuples (same valid and total per tuple). The best +rel rival (transformer, 8k and 80k) also equals it on 16/16. It is marked "control = CLU" in `fig_maze_conditions` (c) and printed bold in `fig_maze_rel_tuples`.
- **Seed 5 (CLU-REL-A10):** the best matched rival `rival@10x-steps: transformer` = CLU = 0.9393939…, on 16/16 tuples. Four +rel arms also tie. It is marked "rival = CLU".
- **Marked by the same rule, not in the brief:** A1 seed 1: the rival (0.5674) is above the CLU (0.5660). A1 seed 5: the rival (0.8667) is above the CLU (0.6818). The rule marks every seed where a rival or control ≥ the CLU, so nothing was hand-picked.
- **Seed 2:** the +rel rival does **not** close the gap (0.819 against 0.984). The caption prints all six +rel values beside the CLU's.
- **Ladder:** the (c) margin is the smallest on every rung. The store stays above every drawn rival on every rung and seed; this is *recorded* in provenance, and no longer a build gate that would refuse to draw.
- **Deletion:** red ∧ sphere costs the store −6.13 / −6.43 / −6.30. The 1 % item control moves −0.02 / −0.11 / −0.01. Against the MLP (seeds 1, 2) and the LSTM (seed 2), the CI for red ∧ sphere crosses zero. The caption says so.

## Acceptance
- **Two builds byte-identical:** `SHA256-w22-build1.txt` == `build2` (27 files). The build is installed into **both** `w17-paper/figures/` and `paper-visuals/`, and every installed file is re-hashed at its destination (`INSTALLED BYTE-IDENTICAL` ×2).
- **Drafts:** edits were staged (`staged-pass3/`, `tools/apply_pass3.py`) and the drafts were `cmp`-equal to the snapshot before install. The lock was re-checked (no peer live), and the install is byte-identical. The installed drafts compile with exit 0, 0 undefined citations, 0 undefined references, 0 errors, 0 overfull boxes, 0 "Float too large" and 0 "Figure pending". They run to **51 / 52 pp** (baseline 48 / 49 with the same figure dir), with 55 bibitems.
- **All four style diffs EMPTY** (`STYLE-DIFF-CHECK-pass3.md`). **PLACEHOLDER 15 = 15**, with the ordered arguments byte-identical (asserted in the apply script).
- **Provenance checker:** findings identical to baseline except for line shifts. All 54 new caption number tokens are found in their cited artefact.
- **Edits:** three captions were rewritten (`fig:ladder`, `fig:deletion`, `fig:maze-wm`) and two slots added (`fig:headline-steps`, `fig:maze-rel-tuples`).

## ⚠ Discrepancies / things a reader must know
1. **The replay's per-tuple format is `[n, correct]`, not `[correct, n]`.** The first element sums to `n_nway_rows` on every seed, and the reader asserts this. The first draft of the reader assumed the opposite and its own gate caught it.
2. **The draft's 16,384-row comparators' 1,280 steps are *derived*:** 20 epochs (`BASELINES-READY build_scale.epochs`, which that file labels BUILD/SMOKE scale) × ⌈16,384/256⌉ batches. The w20 LOG states the same 1,280. No artefact stores it as a field. The derivation is written in PROVENANCE.
3. **The deletion figure now shows the w20 book-refit protocol.** The draft's Table `tab:del` still prints the w13/w15 book-carried (d) rows. The caption says the two differ. Reconciling the table text is a draft-owner job, not this spoke's.
4. **The CLU-REL-A10 3-seed row reads WIN under the Head's rule.** Seeds 3–5 of the same config include both ties, and they sit directly below in panel (c). The six-seed panels carry **no** rule label, because the rule is written for 3 seeds; declaring a 6-seed rule would be a new bar.
5. The brief says "+rel recovers 97–100 % of the margin on seeds 3–5". Recomputed: seed 3 (0.9215−0.7230)/(0.9274−0.7230) = 97 %, seed 4 = 100 %. **Seed 5 is 0/0**, because the matched rival already ties, so the ratio is undefined there. No percentage appears in any caption.

## Did NOT run / not done
- **CLU-REL-A10 per-row panels (gate 2).** `clu.pkl`, the controls and the deletion pickles reference module `w21_clu` (scanned with pickletools; nothing was executed). That module and the `+rel` input builder live on the w21 branches on forgis-14 and are not on batmac. SSH from batmac to forgis-14 fails with `Host key verification failed`, and host keys were not overridden. → `NEEDS-ARTEFACTS.json`.
- **The 32,768-row ladder rival set** is still running. Only its slot exists.
- The per-corner breakdown of the red ∧ sphere loss (−32.1 on red-with-other-shapes corners) is in the w20 FINDINGS prose only, not in the JSON. It is not drawn.
- `fig_maze_wm.*` and the pass-1 figures of other spokes were left untouched (`rendered/` belongs to another spoke). They are mirrored into `paper-visuals/` verbatim, with their SHA-256 in its README.
