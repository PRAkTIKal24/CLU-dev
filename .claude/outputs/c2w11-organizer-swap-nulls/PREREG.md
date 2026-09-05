# PREREG — `c2w11-organizer-swap-nulls` (C2W11 spoke C, the ORGANIZER SWAP's NULL side)

**Filed by the spoke, 2026-08-11, BEFORE the first cell of `exp_c2w11_nulls.py` ran and before the
module existed.** Base `main @ 168a892`, worktree `../CHLU-c2w11c`, branch
`c2w11-organizer-swap-nulls`.

⛔ **Nothing here is a result.** Every number is a pre-registered prediction with its derivation, or a
quotation of a banked measurement with its provenance. The scorecard at the end of
`.claude/outputs/c2w11-organizer-swap-nulls.md` scores every row of this file, right or wrong.

---

## 0. What I am predicting, and the substrate I am predicting it on

The substrate MOVED between C2W5 (`orgdiv-null-arms`, my direct ancestor) and this wave, and the two
moves point in **opposite directions**, which is why my numbers are not just C2W5's numbers again:

| quantity | C2W5 (banked) | C2W11 spoke A (measured, `FROZEN-INTERFACES-C2W11.json`) | consequence for me |
|---|---|---|---|
| distinct wells the launch reaches | **2.202** of 4 | ⭐ **3.998** of 4 (`K0` frac ≥ F = **0.9967**) | the *addressability* cap that killed C2W5 is **GONE** |
| occupancy precision at launch | 0.4061 | ⚠ **0.2303** | the particles reach 4 distinct wells but ~1 of 4 is in `A(x)` |
| K6 (asserted set already exactly right) | 2/2560 | **0.00065** | the launch head is **not** answering the question |
| `tol` | 0.47827 | **0.28696** (payload_radius 1.0 → 0.60) | scale-homogeneous; the family's difficulty is **unchanged** |

⭐ **The prediction that follows, and it is the load-bearing one.** A launch that reaches 4 **distinct**
wells at **23 %** precision is a launch that has *swapped* the C2W5 failure mode for a different one:
C2W5 could not visit 4 wells, C2W11 visits 4 wells and visits the **wrong** ones. Every arm in my
brief reads by (i) quantising each particle to one unit of its own codebook and (ii) summing the
payloads. Both failure modes are fatal to that read, so **I predict the arm-side numbers barely move
from C2W5's**, while the *attribution* moves a great deal (§4).

---

## 1. V1 — held-out exact-set accuracy on `Q_unseen` (the headline null side)

Metric grain `1/512 = 0.001953`. `chance_per_seed` (frozen) `= [0.0, 0.001953, 0.0]` on seeds 0–2.
Bar (for context only — ⛔ **I compute no swap verdict**) `= chance + 0.05 ≈ 0.0507`.

| # | quantity | **point prediction** | band | derivation |
|---|---|---|---|---|
| **P1** | **N1** unseen, best reader, selected config | **0.0020** | [0.000, 0.010] | banked 0.0000; the launch now reaches 4 distinct wells so a *few* queries may have an exactly-right asserted set — but K6 = 0.00065 caps that channel at ~1/1536 |
| **P2** | **N2** (VQ) | **0.0020** | [0.000, 0.010] | banked 0.00039; same cap |
| **P3** | **N3** (fitted static-geometric) | **0.0020** | [0.000, 0.010] | banked 0.0000 |
| **P4** | **N4** (kNN) | **0.0020** | [0.000, 0.012] | banked 0.00078; `set_code` key is the noiseless-key variant and is the strongest |
| **P5** | **N5** (Titans) | **0.0020** | [0.000, 0.012] | banked 0.00117 (the banked `null*` holder) |
| ⭐ **P6** | **`null*` = grid-max over ALL arms × ENTIRE registered grid × 5 score seeds** | **0.0039** | [0.001, 0.020] | the max of ~600 configs of near-chance noise at grain 0.00195 lands 2–3 grains up; banked 0.00117 at grain 1/2560 with a 5× larger query pool |
| **P7** | **does ANY arm clear `chance + 0.05`?** | ⛔ **NO**, `P(no arm clears) = 0.92` | — | the read-protocol cap of §4 binds every arm identically; the 8 % is the possibility that 4-distinct-well launches let a fitted assignment rule find a compositional channel C2W5 could not |

⚠ **P7's 8 % is a real prediction, not a hedge.** If an arm clears, that is a *finding about the
repaired launch*, and I report it without softening (dial declaration: an arm beating the physics arm
is a legitimate outcome).

## 2. The four internal-validity anchors (L1–L4) — what makes a null a statement about the PROBLEM

| # | anchor | bar | **prediction** | derivation |
|---|---|---|---|---|
| **P8 / L1** | N1 fits its own training items | ≥ 0.50 | **1.0000** (`P(≥0.90) = 0.85`) | banked 1.0000 with 28 672 free floats; `tol` and the payload scale moved together (degree-1 homogeneous) so the fit problem is **scale-identical** |
| **P9 / L2** | N4 `k = 1` memorises SEEN | ≥ 0.95 | **1.0000** | banked 1.0000; exact-key lookup |
| **P10 / L3** | shuffle-φ launder | ≤ chance + 0.005 | **≤ 0.0020** | banked ≤ 0.00039 |
| **P11 / L4** | N1 capacity flatness across `a ∈ {12,32,64}` | ≤ 0.02 | **0.0000** | banked 0.0000; capacity is not the binding constraint |
| **P12** | ⚠ **N3's in-sample fit** (banked **0.0799**, the weakest arm-side number in C2W5, and my brief says give it a budget that reaches a real optimum) | — | **0.35** | band [0.10, 0.80]. I add `payload_source="fitted"` at every level plus a 3× longer fitting budget; the arm has `N_a(d+2)+N_a·m = 448` free floats against 96 training rows, so it is *not* structurally capacity-starved once the payloads are free |

## 3. V2 — the matched confidence channel (VALUE leg ii's null side)

**Construction (mine, declared because it is NOT in `FROZEN-INTERFACES-C2W11.json` — see the rider):**
`n_novel = 4` of the `N_a = 32` wells are **reserved and never written**; SEEN is `K` rule-4-valid
`F`-subsets of the 28 written wells; the eval set carries **0 / 1 / 2 novel channels**. φ is untouched
(φ carries a code for every well, written or not), so the launch head is unchanged.

| # | quantity | **prediction** | derivation |
|---|---|---|---|
| ⭐ **P13** | **V2a** per-feature novelty AUROC, **N2** (distance-to-codebook) | **0.88**, band [0.60, 0.99] | ⭐ **The nulls should be STRONG here and I say so in advance.** A never-written well is a region no fitted codebook has mass in, so "novel" is *literally* "far from everything I fitted" — the easiest possible detection problem for a geometric arm |
| **P14** | V2a, **N1** (read-objective residual / max logit) | **0.82**, [0.55, 0.98] | same mechanism through the atom logits |
| **P15** | V2a, **N3** (the fitted rule's margin) | **0.75**, [0.50, 0.95] | a margin is a weaker novelty statistic than a raw distance |
| **P16** | V2a, **N4** (neighbour distance) | **0.85**, [0.55, 0.98] | raw-key distance, no fitting to blur it |
| **P17** | V2a, **N5** (the surprise gate) | **0.65**, [0.45, 0.90] | the gate is a *loss*, which mixes novelty with plain difficulty |
| **P18** | **`null*_V2a` = max over arms and grid** | **0.92** | max of five strong channels + grid |
| **P19** | **V2b** set-level ECE, best (lowest) arm | **0.12**, [0.02, 0.60] | ⚠ **DEGENERATE-BY-CONSTRUCTION RISK, registered in advance:** if every arm's set accuracy is ≈ 0, ECE collapses to *mean confidence* and a maximally under-confident arm "wins". I will report ECE **with the accuracy beside it** and label the degeneracy if it fires (`P(fires) = 0.80`) |
| **P20** | designed negative: permuted payloads ⇒ AUROC ≈ 0.5 | **0.50 ± 0.05** | payload permutation destroys the novel/known correspondence but not the geometry ⇒ chance |

## 4. ⭐⭐ The decodability ceiling, recomputed on the FEATURE-FACTORED launches — and I predict it MOVES

This is the number that turns "no arm clears" into an **attributable** finding, and it is the one
place where I expect a **large** departure from the banked value.

| # | condition | banked (C2W5, `P=4` designed offsets) | **prediction** | derivation |
|---|---|---|---|---|
| **P21** | **noiseless** (exact `φ(x)` set-code, matched filter over all 35 960 combos) | 1.0000 | **1.0000**, [0.98, 1.00] | same φ family, same `d = 4`, `N_a = 32`, `F = 4`; unchanged by the launch swap |
| ⭐⭐ **P22** | **as-launched** (matched filter over the launch points every arm sees) | **0.2719 ± 0.0126** | ⭐ **0.90**, band [0.30, 1.00] | ⭐ **The mechanism, stated so it can be checked rather than believed.** C2W5's launch was `set_code + o_p + σ_q ξ`, so the launch *mean* recovered a **noisy** set-code and the ceiling measured how much `σ_q` destroyed. The C2W11 launch is `R·e_{j_c} + σ_q ξ` with `σ_q = 0.15` against a code-to-code separation of `R·‖e_i − e_j‖ ≈ 2.8` — **the channel indices `{j_c}` are recoverable essentially exactly**, so the matched filter reduces to *"which `A` deflates to this pick-tuple"*. `A ↦ picks` is a deterministic map into 32·31·30·29 = 863 040 ordered tuples from 35 960 sets, so it is injective unless it collides. **The band's lower edge is the collision rate, which is exactly what is unmeasured.** |
| ⭐ **P23** | the gap this creates | — | **`null*` ≈ 0.004 against a ceiling ≈ 0.90** | ⇒ the registered §7.3-style branch fires **harder** than in C2W5: the information is not merely *present* in the launch, it is **almost fully present**, and the per-particle-quantise-then-sum read expresses ~0.4 % of it. ⛔ I will not state "no arm clears" without this number in the same sentence |

⚠ **The honest counter-hypothesis, registered so it cannot be a post-hoc excuse:** if the deflation map
collides heavily (many `A` sharing a pick-tuple), P22 lands near 0.30 and the finding becomes "the
feature-factored launch buys addressability but *destroys* set information" — the **opposite** reading,
and a more interesting one. Either outcome is a result; I commit to both branches now.

## 5. V3 — the anytime curve on the null side (leg iii's null side)

Read at the **frozen** budget grid `[50, 100, 200, 400, 800, 1200]` total Verlet steps, 4 particles,
split `address = round(b/3)`, `read = b − round(b/3)`, `γ_addr = 0.05 → γ_read = 0.02`, `dt = 0.05`
(`FROZEN-INTERFACES-C2W11.json::v3_budget_grid`, quoted, never re-derived).

| # | quantity | **prediction** | derivation |
|---|---|---|---|
| **P24** | **N1** instantiated as a landscape, score at `b = 1200` | **0.0020**, [0.000, 0.010] | N1 *is* a `FactoredStore` by construction; the settle re-quantises to the same wells |
| **P25** | **N2 / N3** instantiated (codebook → placed wells), score at `b = 1200` | **0.0020** each | the instantiation is F3-tuned over (atom width × amplitude × atom budget) and I predict **the tuning does not move the score** — the cap is the read, not the landscape |
| **P26** | **`null*_V3` = max over {N1, N2, N3} × budget × grid** | **0.0039** | same noise-max arithmetic as P6 |
| **P27** | the null-side curve is **FLAT** (max − min over the budget axis ≤ 0.004) | `P = 0.75` | N199's reference: a store that carries nothing reads flat. Here the store *carries* payloads, but the read cannot address them, so I predict flat-at-the-floor — ⛔ which is **not** the same claim as "the store is empty", and I will say so |
| **P28** | **N4 / N5** | ⛔ **DECLARED NOT-RUN for V3** (no landscape exists) — reported as flat reference lines at their static V1 scores | registered by the Hub in `PREREG-C2W11.md` §5 / NOT-RUN 9 |
| **P29** | **read-compute RATIO** (physics read ÷ N1's matched-capacity static read), on this cell's ledger | **≈ 1 000×**, band [300, 5 000] | banked **3 360×** (6.88e7 vs 20 480) at `a = 32`; this cell's `a = 12` cuts the physics numerator ~2.7× and the static denominator with it, and the budget grid tops out at 1 200 total steps rather than 1 200 per phase |

## 6. What would make me WRONG in a way that matters (registered falsifiers of my own work)

1. **L1 fails** (N1 cannot fit its own training items). ⇒ my optimiser is the story, the audit is void,
   and ⛔ **no "nothing works" statement may ship** (§A: no L1, no verdict).
2. **L3 fires** (an arm scores above chance + 0.005 on shuffled φ). ⇒ the score is a fitting artifact.
3. **P22 lands < 0.30.** ⇒ the feature-factored launch destroyed set information and the wave's
   structural change bought addressability at the price of decodability.
4. **Any arm clears `chance + 0.05`.** ⇒ P7 is refuted and the read-protocol refutation does not
   generalise to the repaired launch.

## 7. ⛔ Declared NOT-RUNs (never to be reported as nulls)

1. **`OD`, `OD_min`, any swap verdict, any tier-ii verdict, any paper number** — ⛔ not mine. I produce
   the `null*` side only.
2. **N4 / N5 for V3** — no landscape exists (Hub-registered).
3. **The γ axis** — no null arm has a rollout except through the V3 instantiation, where γ is taken
   from the frozen grid and is not an axis.
4. **Attention-ψ, ψ of any kind, the novelty head, the organization loss** — spoke B's.
5. **Any K-verdict re-adjudication** — spoke A's; quoted, never re-scored.

## 8. ⚠ Riders handed back to the Hub (things I must choose because they are not frozen)

- **R1 — the VALUE launch key is not in `FROZEN-INTERFACES-C2W11.json`.** The file freezes
  `launch_keys` for `k0/k6_k7cap/k3_k4_k5/m6/coverage` but registers no key for the V-legs. I adopt
  **`PRNGKey(7000 + seed)` (the `k3_k4_k5` key), SEEN with the key itself and `Q_unseen` with
  `fold_in(key, 1)`** — the exact pattern `stage_k3_k4_k5` uses, because that is the stage that fits
  the reader class and scores unseen exact-set accuracy. ⚠ **Spoke B must use the same or V1/V3 are
  scored on different launches.** I emit the launch-point hash so it is byte-checkable.
- **R2 — the novel/known feature split for V2 is not frozen.** I register `n_novel = 4`, chosen by
  `np.random.default_rng(20260811 + seed)`, wells never written. Spoke B must use the same rule.
- **R3 — the physics arm the oracle-imitation null imitates.** The trained organizer is spoke B's, so
  my oracle target is the **written but un-organized** physics store's own assignments (spoke A's
  `build_arm` + `multi_particle_read`). Declared as such, never as "the physics arm".

*Filed before `chlu/experiments/exp_c2w11_nulls.py` existed. Git: branch `c2w11-organizer-swap-nulls`
@ `168a892` (no commits yet at filing time).*
