# Task: mia-decay-measurement — what does an adversary see at one half-life? (the open cell) (w25)

- **Agent:** `results-analyst` · **Output:** `.claude/outputs/mia-decay-measurement.md` · **Branch:** none expected (analysis harness on the existing store machinery; scratch + outputs only)
- **Read first:** `.claude/AGENT_PROTOCOL.md` **§7 (dial declaration)** · `.claude/outputs/unlearning-recon.md` §Item-3 ("the scoping issue") + recon item 6 (**the spec of this task**) + §1.1 (U-LiRA / per-example discipline, Hayes et al.) · `.claude/outputs/controller-mvp.md` §3(b) (the decay machinery you measure) · `negative_results.md` N99 · `claims_matrix.md` CM-22(m/n/o) (naming rules)

## ⭐ DIAL DECLARATION (protocol §7)
- **Dial:** lifetimes — the measurement that makes the retention dial *quotable against an adversary*, converting R1's hardest referee question ("what does a U-LiRA adversary see at t = one half-life?") into a figure.
- **Laundering control:** the same measurement on a **TTL-dict store** (item present until expiry, then row-deleted) — the boolean baseline whose distinguishability curve is a step function; CLU's claim is the *graded, physical* curve, and it must actually differ.
- **Falsifies:** if distinguishability does NOT decay with amplitude (an item at amp 0.06 is as detectable as at 1.0 — e.g. via residual curvature), the "forgetting is physical" story loses its privacy-adjacent leg and the finding is reported as the negative it is.
- **Does NOT falsify:** distinguishability persisting *longer* than retention (retention crossing before MIA does is an expected, publishable asymmetry — "the store stops answering before it stops leaking" is a finding either way, and its direction is the result).

## Why — genuinely open, and cheap
The recon found **no prior work** on adversarial distinguishability of a *partially decayed* memory (searched; absence-of-evidence flagged as such). Every unlearning eval treats presence as boolean; CLU's amplitude decay creates a state — the half-deleted item — that the field's instruments have never measured. This is the program's chance at a novel contribution in that literature **without touching the DP word**: a measured curve, not a certificate.

## Item 1 — the two curves on one axis (the deliverable)
On a designed store with the MVC-0 decay machinery (`leak` per item, the measured `exp(−leak·t)` law):
- **Retention curve:** per-item retrieval success vs `leak·t` (reproduces controller-mvp §3(b); the anchor).
- **Distinguishability curve:** an MIA-style, **per-example** score vs `leak·t` — the adversary must distinguish `Store(S ∪ {i})-after-n-ticks` from `Store(S)-never-written`, given query access (and, as a stronger arm, given the potential's values/gradients — state both threat models). Follow the **per-example U-LiRA discipline**, not population MIA (Hayes et al. 2024: population-level drastically overstates forgetting) — many paired store instantiations per item, likelihood-ratio-style separation, report the score distribution not just the mean.
- **⭐ The figure:** both curves on the `leak·t` axis; **report where they cross** and which decays first.

## Item 2 — what remains after `evict` (the recon's second unanswered question)
After amplitude passes the floor and the slot is freed: is anything about the item recoverable from the store state (residual curvature at the old site, allocator traces, neighbours' relocations)? Measure the same distinguishability score post-evict. ⚠ Under the current (history-dependent) placement, neighbours *can* carry traces — quantify it; this number is exactly what `order-independent-placement` would drive to zero, so it doubles as that task's baseline. Coordinate nothing; just cite each other's task files.
- ⚠ **The partially-decayed worst case the recon predicted:** an item at amp ≈ floor (0.05–0.06) is "neither present nor absent" — sample this region densely.

## Item 3 — the TTL-dict laundering line
Same protocol on a TTL-dict: distinguishability is a step (1 until expiry, 0 after — modulo the retain-set). If CLU's curve is not measurably *graded* relative to this, the "physical amplitude vs bookkeeping flag" differentiator (the answer to "so what — a TTL field does this") has no measurement behind it. This line is what makes the figure honest.

## Acceptance
PREREG before running: predicted crossing direction (register your genuine prior: does MIA outlive retention or vice versa?), the threat models, the per-example protocol, and the falsifiers above. The two-curve figure + the post-evict table + the TTL line, ≥3 seeds / many paired instantiations, all numbers re-derived from a saved metrics JSON. Echo the DIAL DECLARATION. Flags defects for the engineer; does not modify `chlu/` core.

## ⚠ Standing traps
- ⛔ Words: *scheduled forgetting / retention / distinguishability*. **Never "certified", never "unlearning", never "privacy guarantee"** (CM-22 m/n/o; the Hopfield-1983 naming collision). The claim is a **measured curve**, and the paper sentence pre-approved by the recon applies: *"we make no (ε,δ) claim; our guarantee is structural and algorithmic, not statistical."*
- The store-level scope: φ and payload channels are separate leak surfaces (recon §Item-3) — state the scope, don't claim system-level erasure.
- N94 discipline: state every fit's epoch/tick count.
