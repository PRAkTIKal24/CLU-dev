# ERRATA — `c2w8p3-capture-strong-phi`

Dated addenda to this spoke's `PREREG.md`. The PREREG is **not edited**.

---

## §1 — 2026-08-09, filed AFTER the nine census cells and BEFORE the scale-control cell ran

**What is added:** the **§4 scale-only control cell** (`randconv`, seed 0, address scale `a = 0.8`,
**payload co-scaled** `payload_scale 9 → 11.25`), which my `PREREG.md` §2 did not list among its
predictions because I had priced only the 3 arms × 3 seeds grid.

⚠ **This is an addition, and I mark it as one.** Its **reading is not mine and was not invented
here**: `PREREG-C2W8-PASS3` §4 registered it long before this spoke existed —

> *"a SCALE-ONLY control arm (identical φ, scale multiplied by a declared constant) must move G-ADDR
> by ≈0"*, Hub **Q8** prior 0.90, and `c2w8p3-gate-addr` operationalised it as **`|ΔA1| ≤ 0.05`**.

**The payload is co-scaled** because `c2w8p3-gate-addr` §6(c) measured that an **address-only**
rescale is *not* a covariant rescale on this rig — the payload channel is absolute, so an address-only
rescale walks the store across its own payload wall. The covariant version is the one that tests the
leg rather than the wall.

⚠ **Declared limitation, before the number:** the rescaled cell is a **complete re-run** (re-written
store, re-measured spacing, re-derived atom width), so `ΔA1` contains run-to-run variability as well
as scale sensitivity. It is an **upper bound** on the leg's scale-dependence, not a decomposition.

## §2 — 2026-08-09, filed AFTER the nine cells: what `PREREG.md` §1 did NOT anticipate

`PREREG.md` §1 registered that `cue_sigma / spacing_ref ≡ κ_q = 1.0` **by construction**, and
concluded that the cue is equally hard on every arm. **That is true of `spacing_ref` and FALSE of the
spacing the read must actually beat.** `spacing_ref` is the rig's task-1 **sizing** spacing (200
keys); the **codebook** spacing (the 16 live items) is a different number, and their ratio is
**arm-dependent** (1.08 on `pca`, 1.14 on `simclr`, **1.41** on `randconv`). ⇒ measured
`cue_sigma / codebook_spacing` = **0.927 / 0.875 / 0.710** — a **30 % spread in cue difficulty across
the arms being compared.** Filed here as a correction to my own registered §1, and carried into the
report's reconciliation list as item 4; it is **not** a post-hoc reinterpretation of a prediction.
