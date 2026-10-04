# ⛔ FIGURE DEFECT — the CMAZE-T panels hide the fact that decides the tick
**Found by the Head in `fig_cmazet_families` (Figure 14), 2026-09-24. Diagnosed and quantified by the Hub from `paper-visuals/data/cmazet_items.json`.**
⭐ **No number is affected. The judge is correct. This is a RENDERING defect.**

## THE WORKED EXAMPLE — test any fix against this
**seed 2, item 482** · `rule_a` · pocket 3 · door 1 · tint none · east → **illegal**
Truth `[legal, displacement, event, pocket-next] = [0, 0, 0, 3]`

| arm | emitted | mark | judge reason |
|---|---|---|---|
| **CLU** | `[1, 0, 0, 3]` | ⛔ **CROSS** | `legality` |
| `rival: mlp` | `[0, 0, 0, 3]` | ✅ **TICK** | `ok` |

⭐ The CLU predicted the move was **legal**; it is **illegal**. It is correctly marked wrong.
⛔ **But both predict displacement 0 and no key event, so the agent stays put with pocket 3 in BOTH — the two panels render IDENTICALLY.**

## THE MECHANISM
`cmazet_panels.py` draws the **resulting state**: `gen, _ = T.apply_outcome(inst, st, *emitted[:3])`, then `draw_state(inst, gen.pos, gen.ks, …)`.
⛔ **The judge scores the FACT VECTOR; the panel draws the STATE.** Any fact that does not change the state is **invisible**.
⭐ **Legality is the main one**, and it goes invisible exactly when **displacement = 0**: "legal but nothing happens" and "illegal" produce the same next state.
⚠ The pocket is **not** the problem — it is drawn correctly (agent disc centre = held key's colour, white = empty) and is identical in both panels here.

## SCALE (all 3 seeds, all 16 arms, pairwise per item)
| | count |
|---|---|
| arm-pairs on one item with **opposite marks** | 92,628 |
| ⛔ …differing **only** in facts that cannot change the picture (`disp=0`, `event=0`) | **3,740** |
| …visibly different predictions (figure is fine) | 88,888 |

## WHAT IS **NOT** WRONG — all three tested
- ⭐ **Not a judge error:** **0** cases where the same emitted vector receives different verdicts. Self-consistent.
- ⭐ **Not a mark/arm misalignment:** the draw loop takes image and mark from the **same tuple** (`ok, em = arms[key]`).
- ⭐ **Not a scoring problem:** the reported scores are the judge's verdicts over all four facts; only the rendering drops information.
- ⚠ **A Hub misdiagnosis to discard:** an earlier reading blamed the **fourth fact (pocket-next)** being undrawn. ⛔ **That was wrong** — item 482 has pocket-next = 3 in both arms. The invisible fact is **legality**.

## THE FIX
⭐ **Surface the predicted legality on each panel** — a border, a glyph or a word — so the reader can see the fact that decided the mark. ⛔ Re-rendering the pocket does not help; it is already right.
⚠ Applies to every figure sharing the drawing function: **`fig_cmazet_families`, `fig_cmazet_per_arm`, `fig_cmazet_door_then_tint`**, and the deletion panel.
⭐ **Acceptance for any fix: on seed 2 item 482 the CLU and MLP panels must be visibly distinguishable.**
