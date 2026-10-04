# w22 `spoke/figures-redraw` PASS 3 — BRIEF (batmac)
**ZERO scored runs. ⛔ No ledger write. You hold the `.tex` lock.** Load `dataviz` first.
⛔ **PRE-REGISTRATIONS ARE BANNED** (Head, 2026-09-24). Print everything measured.

## ⭐⭐ A. THE HEADLINE FIX — the CMAZE-T panels hide the fact that decides the tick
**The Head found this in Figure 14 and it is the most visible defect in the paper.**
⛔ **The panel draws the resulting STATE; the judge scores the FACT VECTOR** `[legal, displacement, event, pocket-next]`. ⇒ **Any fact that does not change the state is INVISIBLE.** **Legality** is the main one, and it vanishes exactly when **displacement = 0**: "legal but nothing happens" and "illegal" produce the *same* next state.

**⭐ THE TEST CASE — any fix must make these two panels visibly different:**
**seed 2, item 482** (`rule_a` · pocket 3 · door 1 · tint none · east → **illegal**), truth `[0,0,0,3]`:
| arm | emitted | mark | judge reason |
|---|---|---|---|
| **CLU** | `[1, 0, 0, 3]` | ⛔ **CROSS** | `legality` |
| `rival: mlp` | `[0, 0, 0, 3]` | ✅ **TICK** | `ok` |
⭐ Both predict displacement 0 and no key event ⇒ agent stays put, pocket 3, **identical picture, opposite marks.**
**Scale: 3,740 arm-pairs across the 3 seeds render identically with opposite marks** (of 92,628 opposite-mark pairs).

⭐ **THE FIX: surface the PREDICTED LEGALITY on every arm panel** — a border, a glyph, or a word. ⛔ **Do NOT re-render the pocket**: it is already correct (agent disc centre = held key's colour, white = empty) and is identical in both panels here. ⚠ An earlier Hub diagnosis blamed the undrawn `pocket-next` field — ⛔ **that was wrong; discard it.**
⭐ Apply to **every** figure sharing `draw_state`/`grid`: `fig_cmazet_families`, `fig_cmazet_per_arm`, `fig_cmazet_door_then_tint`, and the deletion panel.
⭐ Update the captions to say what the tick means: the judge's verdict over **all four facts**, of which the picture shows the state.
⭐⭐ **Print the count of same-picture-opposite-mark pairs in your FINDINGS** after the fix — it should be **0** once legality is visible.

## B. THE REL PER-ARM PANELS — gate 2 is now UNBLOCKED
Pass 2 passed gate 1 (192/192 hashes) but **could not run gate 2**: the REL pickles reference module `w21_clu`, absent from batmac.
⭐ **Two branch bundles are now staged at `~/clu/ship/`: `w21-rel-newseeds.bundle` and `w21-maze-t-clu-improve.bundle`** — they carry `peer_w21_clu_improve_e7fdb9b/w21_clu.py`, `CLU-REL-A10.json` and the `+rel` input builder. Fetch one into a worktree and import from there.
⛔ **The gate still applies:** re-read must reproduce the scored artefact **exactly** before any panel is drawn. ⛔ One mismatch ⇒ STOP and report; do not draw.
⭐ Objects: `~/clu/stage/rel-persist/{w21-maze-t-clu-improve,w21-rel-newseeds}/` (390 files, seeds 0–5).

## C. THE 32,768-ROW LADDER SET — leave the slot
`w21/ladder-rivals-32k` is **still running**. ⭐ Keep the manifest slot exactly as pass 2 built it (a data-file drop-in, ⛔ not a code change). Do **not** wait for it.

## ACCEPTANCE
- ⭐⭐ **seed 2 item 482: CLU and mlp panels visibly distinguishable.** State it explicitly.
- Same-picture-opposite-mark count after the fix: **0** (report the number).
- **Two builds byte-identical**; installed into **both** `w17-paper/figures/` and `paper-visuals/`, re-hashed at each destination.
- Both drafts compile: exit 0, 0 undefined citations/refs, 0 overfull; **all four style diffs EMPTY**; `\PLACEHOLDER` count and order unchanged.
- `PROVENANCE.json`, `FIGURES.md` and the `paper-visuals/README.md` index updated for any figure you change.

## OUTPUT
`~/clu/outputs/w22-figures3/` + the two install targets.
