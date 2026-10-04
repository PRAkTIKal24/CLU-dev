# Task: gen-decoder-lift — two decoder variants, one number: can `shape` on held-out data clear 0.8?

- **Agent:** `experiment-engineer` · **Output:** `.claude/outputs/gen-decoder-lift.md` + artefacts under `.claude/outputs/gen-decoder-lift/`
- **Read first:** `.claude/AGENT_PROTOCOL.md` (all, **especially the §5 pre-registration rule**) · **`.claude/tasks/gen-clu-phase1-rulings.md` — R15, R16, R17, R18 are the whole context** · `.claude/outputs/gen-feature-bridge.md` **§2, §6 and §11** · `chlu/core/gen_bridge.py`, `chlu/eval/gen_probe.py`, `chlu/experiments/exp_gen_bridge.py`.
- ⛔ Read no campaign ledger, charter or handover — **EXCEPT `.claude/handover_context.md` §7 Known Issues** (§7 only). ⚠ **§7.23 will bite you** — see below.
- **Base:** `main @ fcdca43` · **Branch:** `agent/experiment-engineer/gen-decoder-lift` · **Worktree:** fresh (`../CHLU-gen-declift`). ⛔ `CHLU-bprime-family2`, `CHLU-wt2`, `CHLU-wt3`, `CHLU-merge` are parked, not free.
- **Precondition:** none beyond the base. ⚠ Runs in parallel with `gen-store-grouping` and `gen-wide-split` — ownership split below.
- **Budget: bounded by Head ruling — a day of effort, TWO decoder variants maximum.** ⛔ Not a research programme.

## ⭐ DIAL DECLARATION (protocol §7)
- **Dial:** *none — instrument repair, plus one gating measurement.* ⛔ No store, no well, no arm, no CLU number.
- **Laundering control:** the **matched-seen** control (R16) — same hue corner, seen shape — reported beside every number.
  It is now a permanent companion row, not an optional extra.
- **Falsifies:** `shape` ratio still `< 0.8` after both variants ⇒ **option 2 activates** (accept the band, pre-register
  the floor at the measured ceiling, widened split mandatory). ⭐ **That is a pre-registered outcome, not a failure.**
- **Does NOT falsify:** no movement on `scale`. `scale` already passes at **1.007** against the matched-seen control —
  its deficit is photometric and there is nothing there to fix. ⛔ Do not chase it.

## The one job
Determine whether a **decoder-only** change can lift held-out `shape` from **0.5926** (3-seed, random-A normaliser;
0.500 seed-0 against matched-seen) to at or above **0.8 × its seen value**.

## ⛔ THE HARD CONSTRAINT — decoder only
**The encoder and the cached features are FROZEN and must not change.** `gen-store-grouping` is training the store on
that exact cache right now. ⭐ **If you change the encoder, the cache changes and every downstream spoke's work is
voided.** So: same encoder weights, same feature cache, same checksums — assert all three. ⛔ If you conclude the fix
*requires* an encoder change, **stop and report that as the finding.** It is a Head decision, not yours, and it is a far
larger one than this task.

## ⭐ Pre-register FIRST (before you train anything)
Write `PREREG.md` into your output directory naming, for each of your two variants, the predicted `shape` ratio and how
you derived it. **The Hub's standing expectation, which you should record and then test:**
> The `cls` diagnostic pretext already reaches `shape` **0.773** — better than `recon`'s 0.653 and **still below 0.8**.
> That is the whole pipeline under a *more favourable* pretext. ⇒ **A decoder-only fix is predicted NOT to clear 0.8.**
⭐ If you beat that, it is a pre-registered win. If you do not, option 2 activates cleanly with no appearance of retreat.
⛔ Do not write a prediction chosen to be safe; a hedge is worth nothing when the number lands.

## What to try — exactly two variants, declared before running
Pick **two**, state why, and ⛔ do not silently try five and report the best. Reasonable candidates:
capacity (wider/deeper decoder) · longer training with the same schedule · a skip-free architecture change ·
a loss that weights object pixels rather than uniform MSE (⚠ the current decoder reaches MSE 0.0027–0.0036 on **both**
A and B, so **reconstruction error is already low and is not what is failing** — B is hard to render *identifiably*,
not hard to reconstruct. ⭐ **That is the single most useful clue you have: a pixel-MSE objective is not being asked to
preserve shape identity.**)

## Non-negotiables
- **Train-split only.** The decoder never sees held-out-combination images; they are evaluation-only. **Re-run the
  R3.3 no-leak assertion and report its count** (the shipped one reports 72/72 excluded, 0 leaked).
- **Frozen after training**, parameter hash emitted. Whatever wins becomes THE shared decoder for every arm.
- **3 seeds**, spread reported. Same probe, same protocol, same splits.
- **Report per-factor and aggregate, and the matched-seen control row**, for every variant.
- ⛔ **Report `cls` as well as `recon`** — it is the fault-isolation arm and it is now load-bearing evidence.

## Files you own
`chlu/core/gen_bridge.py` — **the decoder class and its config only.** ⛔ Do not touch the encoder, the cache, or the
probe. Report exact line ranges. `tests/test_gen_decoder_lift.py` — new.
`chlu/config.py` — ⚠ **shared with two live spokes.** You add **fields inside the existing `GenBridgeConfig` block
only**. ⛔ `gen-store-grouping` appends a new dataclass after it; ⛔ `gen-wide-split` edits `GenDataConfig`. Touch
neither. Rebase onto `main` before finishing and resolve **only** your own hunk.
⛔ You own nothing under `chlu/eval/`, `chlu/data/`, `chlu/core/gen_store.py`, or `scripts/csf3/`.

## ⚠ Traps that have already cost two spokes time
- ⛔ **No `uv sync` in your worktree** (w6 lesson). Reuse the main venv:
  `PYTHONPATH=<worktree> /Users/user/Desktop/CHLU/.venv/bin/python -m …`, `CHLU_GEN_DATA_ROOT` set (worktrees have no
  `.claude/`). h5py is there via the `gen` extra.
- ⚠ **§7.23:** six test modules set `jax_enable_x64=True` at **module import** and sort before yours; an
  `eqx.nn.Conv2d` fed an explicitly-float32 array dies **in the full suite while passing alone**. ⭐ Cast the input to
  the module's own parameter dtype — the shipped decoder already does this; **keep it that way in any variant.**
- ⛔ **CSF3 needs 2FA and is unreachable from an agent session.** Run locally on CPU; emit the submit line; report any
  cluster item as **NOT RUN**.

## Acceptance
1. `PREREG.md` filed **before** any training, with a derived prediction per variant.
2. Exactly two variants, declared up front, both reported — ⛔ including the one that lost.
3. **`shape` ratio for each variant, 3 seeds, against both normalisers** (random-A and matched-seen), with the verdict
   `CLEARS 0.8` / `DOES NOT CLEAR`.
4. The encoder-and-cache-unchanged assertion passes (weights hash, cache checksum).
5. No-leak assertion re-run and counted.
6. Suite green (before/after, main venv); ref verified from the main repo before worktree removal.

## What would prove this spoke wrong
- ⭐ **You improved the aggregate and not `shape`.** The aggregate is four factors at ~1.0 and is not what is being
  tested. Look at `shape` first, every time.
- You changed the encoder or the cache to get there. Then the number is real but the phase pays for it downstream, and
  the store spoke's work is void.
- You reported the better variant and quietly dropped the other.

---

# ⟲ AMENDMENT 2026-09-10 — your winner must be a RECIPE, not one fitted artifact
A new experiment (the **factorisation ladder**) adds rungs in which **the encoder is retrained** (an unsupervised
decorrelation penalty at the penultimate layer, and one layer earlier). Those rungs have a **different feature space**,
so ⛔ **one fitted decoder cannot serve them** — each rung fits its own decoder and is scored against **its own**
ceiling.

⭐ **Therefore: whatever wins here becomes THE decoder RECIPE — architecture plus training schedule — refit per rung**,
not a single `.pkl` every rung reuses. Emit it that way: a loadable config that reproduces the winning decoder from
scratch given any feature cache, plus the fitted artifact for R0.
⛔ Do not hard-code anything to the current feature width (`D = 2048`); a later rung may differ. State explicitly which
of your hyperparameters are width-dependent.

Everything else stands: decoder-only, encoder and cache frozen, two variants maximum, prereg first.
