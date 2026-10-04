# w21 `spoke/figures-redraw` — FINDINGS (batmac, 2026-09-24)

**Zero scored runs. Nothing was written to any ledger.** No number here is new evidence. Every
figure is traced to an artefact that already existed, and the Part C panels are a re-read of
persisted objects that reproduces the scored artefact exactly.

## Delivered
| part | result |
|---|---|
| A | `fig_crossover`: mean ± seed sd at **every** K, n = 3, with no single-seed line. `fig_deletion`: 3 seeds, store and all 4 rivals, 3 scopes. `fig_ladder`: one panel per rival set, driven by the data manifest, including the data-matched MLP. |
| B | `fig_maze_conditions` replaces `fig_maze_wm` in the draft (label `fig:maze-wm` kept). It has per-seed markers on every row, Head's-rule labels, and **no margin bar**. The w16 row is labelled ASYMMETRIC in the row itself. The improved-config row is a pending manifest entry. |
| C | Hash gate: 132/132 objects pass. Reproduction gate **passes**: 0.800 / 0.566 / 0.808 (0.7246); controls 0.000 / 0.225 / 0.244 / 0.166; all 16 arms × 3 seeds, plus deletion rows, equal to the float. Four panel sets drawn (families, door-then-tint, per-arm success/failure (new slot), deletion). No hatched column remains. |
| D | Banners removed. `tag_placeholder` is a no-op, and a forbidden-word check runs on every rendered string of the new figures. Both drafts' captions and the placeholder index are corrected, and the stale "Eleven PLACEHOLDERs" comment is fixed. PROVENANCE.json is merged; FIGURES.md is extended. |

## Acceptance
- **Byte-identical rebuild:** yes. `build.sh` builds twice, and `SHA256-w21-build1.txt` = `build2`, for 11 stems plus PROVENANCE.json.
- **Drafts:** both compile with exit 0, 0 undefined citations, 0 undefined references, 0 errors and 0 overfull boxes, at 48 / 48 pp (baseline 47 / 47).
- **Style diffs:** all four are EMPTY (numbers; PLACEHOLDER args, byte-identical and ordered; artefact paths; section/label order).
- **PLACEHOLDERs:** 15 = 15, and no argument changed (asserted in both apply scripts).
- **Provenance checker:** its findings are identical to the baseline. Every new caption number was found in its cited file.
- **Installs:** each draft edit was staged, compiled and checked, then installed byte-identically. Before each install, the drafts were cmp-checked unchanged since the snapshot, and the lock was re-checked.

## ⚠ Discrepancies reported (the artefact wins over the brief)
1. **Observation row:** the Head's rule reads **COMPARABLE**, not LOSS. For VAL, |Δ̄| = 0.1053 and σ(paired Δ, ddof 1) = 0.1068. The label is drawn as the rule gives it.
2. **Words + observation (TEST):** the rule reads **WIN**. R\* = LSTM at 80,000 steps; the store is ahead on 3/3 seeds against R\* and the gate holds. However, w19's own pre-registered B1 bar failed (+0.099 < 0.10; behind the *per-seed best* rival on seed 1). The caption says so.
3. **"Byte-identical to the scored w16 object":** this cannot be hash-checked, because w16 persisted no object. The evidence shows that its TEST reads equal w16 A1's on every seed, and that is what the caption says.
4. **The 131,072-row rival set is not on this machine.** It is declared in `data/ladder_rival_sets.json` as not landed, and it is not drawn.

## Did NOT run / not done
- The 10×-read-floats diagnostic is not drawn in the Part C panels, because it is not a matched rival. It *was* reproduced in the gate.
- For deletion, only the MLP rival has a persisted as-trained/retrained pair; that is all w19 persisted.
- `fig_maze_wm.*` and `rendered/rc_w19_cmazet_*` were left on disk untouched (`rendered/` belongs to another spoke) but are no longer referenced by the drafts.
- There was no visual check of the compiled PDF pages (no poppler on this box). Every figure PNG was inspected, and the logs confirm every slot included its PDF (0 "Figure pending").
- Second pass pending: `w21-maze-t-clu-improve`. To add it: a row in `data/maze_conditions.json`, an arm entry in `data/cmazet_panels.json`, and its own `cmazet_extract.py` re-read (which must pass the gate), followed by `bash build.sh` and a caption line.
