# w22 `spoke/figures-redraw` PASS 2 — BRIEF (batmac)
**Wave-plan §7. ZERO scored runs. ⛔ Never write to any ledger.** Load the `dataviz` skill first.

## THE ONE JOB
**Bring every figure up to the wave-20/21 evidence, and install it into BOTH `w17-paper/figures/` and a new `outputs/paper-visuals/`.**

Your pass 1 (`w21/figures-redraw`, `cc17f159`) left a documented recipe for exactly this: a row in `data/maze_conditions.json`, an arm entry in `data/cmazet_panels.json`, a `cmazet_extract.py` re-read **that must pass the hash gate**, then `build.sh` and a caption line. ⭐ Follow your own recipe.

## FOUR ADDITIONS
1. **Ladder + headline figures: add the w20 131,072-row rival set.** ⭐ Print **all three** rival sets when they exist — 32,256-row (spoke-A `@ref`), **131,072-row (w20)**, and **32,768-row** (`w21/ladder-rivals-32k`, ⚠ **still running — leave the manifest slot so it drops in as a data file, not a rewrite**). ⛔ **Pool size AND step budget in every column header** — the w20 finding is that the operative variable was **steps**, not data.
2. ⭐⭐ **`fig_maze_conditions` becomes SIX seeds and TWO configs.** Add **CLU-REL-A10** and **seeds 3–5**, with **per-seed ceiling markers** and **hard/easy seed marking** (hard = seeds 1 and 4, classified by measured ceiling, cut 0.641 — ⛔ never by the CLU's own score).
3. **Draw the improved config's CMAZE-T panels** from its persisted objects, **through the same hash gate as pass 1** (which passed 132/132). Objects staged for you at `~/clu/stage/rel-persist/w21-maze-t-clu-improve/` (seeds 0–2) and `~/clu/stage/rel-persist/w21-rel-newseeds/` (seeds 3–5), 390 files.
4. **Deletion figure under the FOUR-MEASUREMENT rule:** value / conjunction / item, with **(d) before→after** and ⭐ **the random-1 % control beside it** — that control is what proves the conjunction collapse is *content*, not refit instability.

## ⭐⭐ THE NUMBERS THESE FIGURES MUST SHOW — read them from the artefacts, not from here
| seed | A1 CLU | CLU-REL-A10 | REL best matched rival | REL best `+rel` | REL max control |
|---|---|---|---|---|---|
| 0 | 0.800 | 0.908 | 0.803 | 0.915 | 0.778 |
| 1 | 0.566 | 0.894 | 0.567 | 0.869 | 0.563 |
| 2 | 0.808 | 0.984 | 0.685 | 0.819 | 0.477 |
| 3 | 0.674 | 0.927 | 0.723 | 0.921 | 0.698 |
| 4 | 0.533 | 0.900 | 0.624 | 0.900 | ⛔ **0.900 (control ties the CLU)** |
| 5 | 0.682 | 0.939 | ⛔ **0.939 (ties the CLU)** | 0.939 | 0.827 |
⛔⛔ **Two rows MUST be visible, not smoothed away:** on **seed 4** a `random_atoms` control **equals** the CLU (0.9004, exact on all 16 held-out tuples); on **seed 5** the best matched rival **equals** the CLU. ⭐ Mark both.
⭐ The **`+rel`** series is the mechanism: handing rivals the relational columns recovers **97–100 %** of the margin on seeds 3–5. It must be plottable beside the CLU, ⛔ never omitted.

## ACCEPTANCE
- **Two builds byte-identical** (SHA256 manifest). ⛔ Exit 0 proves nothing.
- Every figure traces to a named artefact; every caption number appears in that artefact.
- Hash gate passes for every persisted object you draw from; ⛔ **one mismatch = STOP and report, do not draw.**
- Installed into **both** `~/clu/outputs/w17-paper/figures/` and `~/clu/outputs/paper-visuals/`.
- Both drafts still compile: exit 0, 0 undefined citations, 0 undefined references, **all four style diffs EMPTY**, `\PLACEHOLDER` count unchanged and byte-identical in order.
- `PROVENANCE.json` and `FIGURES.md` updated.

## ⛔ THE `.tex` LOCK
You hold it. Refuse to start if a peer is live:
```bash
for g in w22-draft-rulings w22-seeds-and-tables w21-registry-fold w22-registry-fold; do
  tmux has-session -t $g 2>/dev/null && { echo "ABORT: $g live"; exit 1; }
done
```
⚠ `w22-registry-fold` may be running — it does **NOT** touch `.tex` and is **not** a conflict; check it is not editing a draft and proceed.
⭐ Stage every edit, verify, then install byte-identically.

## OUTPUT
`~/clu/outputs/w22-figures-redraw2/` + the two install targets.
