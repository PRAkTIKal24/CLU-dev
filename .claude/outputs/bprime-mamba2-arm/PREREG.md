# PREREG — `bprime-mamba2-arm` (the Mamba-2 rival row for the B′ audit)

**Filed 2026-08-01, BEFORE any measured run of the arm** (protocol §5 pre-registration rule).
Written after the arm's *code* existed but before `run_rivals_cell` was ever invoked on it; the only
inputs to every number below are (a) the banked n=9 rival rows in `.claude/outputs/bprime-rivals-f3.md`
§1 / `draft-r2.md` §4.1.1, (b) the equations of Mamba-2 as pinned in `.claude/outputs/rival-recon.md`
§1.4, and (c) integer arithmetic on the iso-state budget. ⛔ No score of any Mamba-2 configuration was
looked at before this file was written.

Branch `agent/experiment-engineer/bprime-mamba2-arm`, worktree `../CHLU-mamba2`.

---

## 0. Dial declaration (echoed from the task)
- **Dial:** none — tier-i audit coverage (a rival row, not a CLU claim).
- **Control:** the full B′ column set (projected launder AND the raw +0 B table — the pre-registered
  R4/R5 distinction), blank store, same-keys null, rescue gate.
- **Falsifies:** nothing of ours. ⛔ Selection on the eval split would invalidate the row.
- **Does NOT falsify:** Mamba-2 beating the raw table would be the audit's first positive rival row and
  is reported as such; losing to the raw table is the metric-native-ceiling theorem, not news.

---

## 1. The derivation the predictions come from (stated first, so a hit is evidence)

**Mamba-2's SSD update at `n_head = 1`, written in this rig's notation** (Dao & Gu, ICML 2024,
arXiv:2405.21060; the SSD restriction is `A_t = a_t I`, a *scalar* times identity):

```
a_t = exp(-Δ_t · exp(A_log)),   Δ_t = softplus(w_Δ·x_t + Δ_bias)
h_t = a_t · h_{t-1} + B_t (Δ_t v_t)ᵀ        h ∈ R^{N×P},  B_t ∈ R^N,  v_t ∈ R^P
o_q = h_Tᵀ C_q
```

⭐ **This is exactly Gated DeltaNet's Eq. 6 with the delta-erase term deleted** — and that is not our
characterisation: the Gated DeltaNet paper itself (Yang, Kautz & Hatamizadeh, ICLR 2025) presents
*"Mamba2 as `S_t = α_t S_{t−1} + v_t k_tᵀ`"*, i.e. as the degenerate fast-weight rule that the delta
arms improve on. **Every prediction below is derived from that one relation:** Mamba-2 should sit in
the delta-arm cluster, displaced in the direction "no erase" and "no L2 normalisation on `B`/`C`"
(Mamba-2, unlike GDN-2 §3.5, does not normalise its key/query paths).

Two consequences we commit to *before* measuring:
1. **No erase ⇒ the state superposes without correcting collisions.** On a 5–6-item stream with an
   `aggregate` reader this should cost little (the delta arms' erase buys them ≤0.02 of `full` over
   each other), so `full` should land in the delta cluster, not in the TTT cluster.
2. **No L2 normalisation ⇒ `arg-min ‖q−k‖ ≢ arg-max q·k` in its own projected key space.** The
   projected (P5) table is therefore read by a *worse-matched* reader for Mamba-2 than for the delta
   arms, so Mamba-2's launder and its projected `+0 B` readers should be **worse** than the delta
   arms' — which mechanically pushes its **`+0 B` margin up** and its **dividend up**, while leaving
   the **raw-metric margin** (which does not use its projections at all) in the delta cluster.
   ⭐ This is the sharpest, most falsifiable thing in this file: **the +0 B margin and the raw margin
   are predicted to move in opposite directions relative to `gdn`.**

---

## 2. The registered numbers — `aggregate@base`, seeds 0–8 (n = 9), full F3 grid, 400 outer steps

| # | quantity | **registered point** | **band** | falsified if |
|---|---|---|---|---|
| **M1** | `full` (neg_mae) | **−0.42** | [−0.55, −0.33] | outside the band |
| **M2** | **raw-metric +0 B margin** | **−0.26** | [−0.40, −0.15] | outside the band, or > 0 |
| **M2b** | sign of M2 | **≤ 0 by > 2 SE** | — | margin > 0 by > 2 SE ⇒ the audit's FIRST positive rival row |
| **M3** | **projected +0 B margin (R5)** | **+0.06** | [−0.05, +0.20] | outside the band |
| **M3b** | M3 vs `gdn`'s (−0.0102) | **strictly greater** | — | M3 ≤ gdn's ⇒ the no-normalisation mechanism (§1.2) is wrong |
| **M4** | projected (arg-min) launder | **−1.0** | [−2.0, −0.40] | outside the band |
| **M5** | dividend vs own projected table | **+0.60** | [+0.00, +1.60] | outside the band |
| **M6** | same-keys null | **−1.0** | [−2.2, −0.40] | outside the band |
| **M7** | blank store | **−1.1** | [−2.4, −0.40] | outside the band |
| **M8** | lift over own blank | **+0.70** | [+0.10, +1.60] | outside the band |
| **M9** | **RESCUED at n = 9** (lift > 2 SE) | **YES** | — | NOT RESCUED ⇒ ⛔ no margin against Mamba-2 is quotable and I must say so |
| **M10** | P5-vs-raw gap (§4 finding) | **+0.55** | [+0.15, +1.20] | outside the band |

**Where the point values come from.** M1/M2: the n=9 delta cluster (`full` −0.4205/−0.4073/−0.4065;
raw margins −0.2732/−0.2600/−0.2592), taken at its centre with a band wide enough to also contain
`deltanet` (the no-decay end) and half the distance to the TTT cluster. M3: `gdn`'s −0.0102 displaced
**up** by the §1.2 mechanism, by roughly the amount that separates `gdn2` (+0.047, channel-wise
gates ⇒ a *shaped* metric) from `gdn` — i.e. one "metric-degradation step" ≈ +0.06. M4/M6/M7: the
`gdn`/`gdn2` n=3 values (−1.42/−1.27 launder, −1.22/−1.13 null, −1.32/−1.66 blank), rounded toward
`deltanet`'s milder values. M5 = M1 − M4. M8 = M1 − M7. M10 = raw-best (−0.21 banked) − M4.

## 2.1 The byte ledger — registered as EXACT INTEGERS (arithmetic, not a guess)

Budget = the CLU's banked `aggregate@base` full-byte figure, **1364 float32 = 5456 B**, unchanged.
Sizing law for Mamba-2 at `n_head = 1`, `d_state = d_head = d` (declared): state `= d²`, identical to
the delta arms' `n_head·d_k·d_v`.

| quantity | registered value |
|---|---|
| `d_head` (= `d_state`) | **36** (`36² = 1296 ≤ 1364 < 37² = 1369`) |
| F2 state | **1296 floats = 5184 B** — byte-identical to `deltanet`/`gdn`/`gdn2` |
| own matched table | `floor(1296/(36+36)) = 18` rows = **1296 floats = 5184 B**, ratio **1.000** |
| `table_is_lossless` | **True** (the `aggregate` stream is ≤ 18 tokens) |
| F1 parameters | **2095 floats = 8380 B** (`θ_K,θ_Q,θ_V` 3·36·5 = 540 · `θ_O` 36 · `S₀` 1296 · `w_Δ` 5 · `Δ_bias` 1 · `A_log` 1 · `D` 36 unused · `W_z` 180 unused) |
| state/param | **0.619** |

⛔ **If any integer in this sub-table comes out different, the sizing rule was mis-derived and I say so
in the report rather than restating the rule.**

## 2.2 Faithfulness claims registered as pass/fail (not bands)

| # | claim | check |
|---|---|---|
| **F1** | the chunked SSD state pass equals the sequential recurrence to fp32 tolerance | test, `Q ∈ {1,2,3,16,256}` |
| **F2** | the chunk length `Q` is **provably inert** (unlike TTT's `b`) ⇒ it is NOT a tuning axis | test |
| **F3** | the quadratic (attention/duality) read equals the recurrent read | test — this *is* state-space duality |
| **F4** | setting the decay to 1 (`A_log → −∞`) recovers plain linear attention `h = Σ B v ᵀ` | test |
| **F5** | masked/padded tokens are exact no-ops (Δ·mask = 0 ⇒ `a = 1`, input 0) | test |
| **F6** | Mamba-2's metric-native verdict is **weaker** than the delta arms': the read is a dot-product kernel smoother with exponential recency decay, but **without** L2 normalisation `arg-min‖q−k‖ ≢ arg-max q·k` | declared at equation level, measured against its own table |

## 2.3 The `overload` byte-frontier column (labelled, never a dividend family)

| # | registered | band |
|---|---|---|
| **M11** | `full` decode at every head width is **within noise of its own blank** ⇒ **NOT RESCUED, non-informative**, exactly as all 5 incumbents were | decode ∈ [0.00, 0.35] at every head width (chance = 0.1667) |

⛔ If M11 holds I will **not** draw a curve and **not** quote any CLU-vs-Mamba-2 margin there.

## 2.4 Selection rule (the ⛔ in the dial declaration)

Two selections are scored **from the same fits** and both are reported:
- **`f3`** — best-of-grid on the **fit split** (auxiliary streams from seeds `seed+101`, `seed+102`).
  This is the shipped rule and the one the banked n=9 rival rows use ⇒ **it is the primary, for
  comparability**.
- **`f3_val`** — best-of-grid on a **held-out** auxiliary stream (`seed+103`), the F3 discipline
  `bprime-rivals-f3` §9.3 recommends.
- **`f3_lite_control`** — C2W4's 3-lr sub-grid re-selected from the same fits (the tuning-effect price).

**Neither selection ever sees the eval split**, which is what the ⛔ forbids.
**M12 (registered):** no verdict in §2 differs between `f3` and `f3_val` — same rescue verdict, same
sign on M2 and M3. *Falsified if any does; that would be a finding about the selection rule, not
about Mamba-2.*

---

## 3. Alternatives explicitly considered and NOT registered (so a post-hoc pick is impossible)
- *"Mamba-2 lands in the TTT cluster (`full` ≈ −0.60)"* — prior ≤ 15%. It has no inner-loop
  generalisation gap; its state is a plain outer-product sum.
- *"Mamba-2 beats the raw table (M2 > 0)"* — prior ≤ 5%, and it is the outcome the dial declaration
  says to report **as the audit's first positive rival row**, not to explain away.
- *"NOT RESCUED"* (M9 = NO) — prior ≈ 25%. `ttt_mlp` is not rescued in any configuration and
  `ttt_linear` is unstable, so a fourth non-rescued arm is entirely possible.

## 4. Declared NOT-RUNs (never nulls)
`recency` / `manifold` (protocol-invalid, FB4) · Titans / SDM / GRU / SWA (D5 and §A14.2 rulings) ·
the deletion probe (**no rival family has a deletion verb**, Mamba-2 included) · a language-model
scale run (⛔ nothing here transfers to an LM claim) · Mamba-1 and Mamba-3 (only Mamba-2 is funded).
