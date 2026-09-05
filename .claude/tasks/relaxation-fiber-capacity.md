# Task: relaxation-fiber-capacity — how many bits sit in the fiber, and are the channels independent? (w21, small)

- **Agent:** `physics-theorist` · **Output:** `.claude/outputs/relaxation-fiber-capacity.md` · **Branch:** none (no production code)
- **Read first:** `.claude/AGENT_PROTOCOL.md` · your own `relaxation-addressing-theory.md` **Item 5 (Prop 11)** and its **OQ-4**, which this task closes · `.claude/outputs/address-space-dimension-scaling.md` §1 (the `4·2^d` spatial law) · `.claude/outputs/clu-memory-architecture.md` (Prop 6)
- ⚠ **Small, sharply-scoped task.** It gates `clu-autoencoder` (task A) and should be cheap. Do not re-derive Prop 11.

## Why — the Head found a gap in Prop 11's scope
Prop 11 wrote the payload into **curvature** and read it as trajectory frequency. **Head's observation: two items at one location also differ under different `M`, `γ`, `μ²`, temperature — so why only one channel?** He is right, and one of the untested channels is better than the one tested: `ω = √(k/M)`, so **mass reaches the same frequency channel while being *per-launch*** (curvature is not), **and Prop 6 already proved mass ratios exactly learnable (2.2e-14).**

Your own OQ-4 — *"how many jet coefficients are linearly readable from a length-N rollout at energy E in d dims"* — is now load-bearing, because capacity may be **multiplicative**: `address-space-dimension-scaling` measured `K_max = 4·2^d` from **spatial packing alone**, and a fiber at each location multiplies it.

## Item 1 — ⛔ the degeneracy, first, because everything else depends on it
`ω = √(k/M)`: a stiff well with a heavy particle and a soft well with a light one give the **same period**. This is your own **OQ1 V↔M gauge**, and if it is exact then mass and curvature are **one channel, not two**.

**Hub's proposed resolution, to prove or refute: mass is a pure time-reparameterization; landscape shape is not.** Varying `M` rescales time uniformly; anharmonicity makes frequency amplitude-dependent and changes the orbit's **waveform**. ⇒ the channels separate if the read is waveform-sensitive rather than period-sensitive.
**Deliver:** the exact statement of what is and is not degenerate, the observable that separates them, and how many samples/what read length it costs. ⚠ **If the degeneracy is exact for the harmonic case and only breaks anharmonically, say so precisely** — it means the fiber's mass channel is worth nothing in a harmonic well, which would be a sharp design constraint.

## Item 2 — the channel inventory, with capacity per channel
For each of **`M` · `p₀` · curvature `k` · anharmonic coefficients · `μ²` · `γ`**, state: **address-side or landscape-side** · **per-item at a fixed location, or shared** · **learnable (and by which of the three Prop-7-compliant routes)** · **approximate bits readable from a length-`N` rollout**.
⚠ Two constraints to respect rather than rediscover: `γ` is global or a *spatial* field, so it is **not** per-item at a fixed location; temperature is **unbuilt** and **stochastic**, so it is a noise channel, not a deterministic code — treat both accordingly.

## Item 3 — independence / multiplexing
Are the channels **independent codes** (capacity multiplies) or **partially degenerate** (capacity adds, or collapses)? Give the composition rule for total capacity, in the form `K_total = K_spatial × K_fiber` **with its validity conditions**.
⚠ **State plainly if they do not multiplex.** A single honest number beats an optimistic product, and `clu-autoencoder` is explicitly instructed not to multiply them into a headline until you have shown independence.

## Item 4 — the cost side
The fiber is read **through time**, so a read that resolves `b` bits needs some rollout length `N(b)`. Give the scaling. ⚠ **This is the honest price of the fiber claim** — if `N` grows exponentially in the bits, the fiber is a curiosity rather than a capacity mechanism, and we need to know that before it enters a paper.

## Acceptance
The degeneracy verdict (Item 1) with its separating observable, the channel inventory with per-channel bits, the composition rule with validity conditions, and the read-length cost. **Label every result proven / verified-numerically / conjectured.** Small numerical checks where they settle something; a minimal harmonic-plus-anharmonic toy is sufficient and preferred over anything elaborate.

⚠ **"The fiber carries ~1 useful channel and does not multiply spatial capacity" is a perfectly acceptable outcome** and would save the program from building a flagship experiment on a channel that is not there. Report it plainly if so.
