# PREREG — B′, the audit paper
**Filed 2026-07-30 by `track2-admissibility` (web-scout), Campaign 2 wave C2W2, BEFORE the C2W2 gate is adjudicated.**
Charter §A3 ("B′ is a contribution, not a retreat") · §A5/C2W2 gate. This file exists so that B′, if activated at the wave review, is a **pre-registration and not a rationalisation**. Full evidence and citations: `.claude/outputs/track2-admissibility.md` (§2).

---

## 1. The claim (one sentence)
> **When does test-time dynamics buy anything over a table at matched bytes?** — one protocol (matched-byte launder + two-sided byte ledger + a **+0 B** substitute audit + same-keys null) applied uniformly to the modern neural-memory family (Mamba-2, DeltaNet, Gated DeltaNet, TTT, Titans, Sparse Delta Memory) **and to CLU**, reporting for each family the dividend of its learned dynamics over a byte-matched non-parametric store.

## 2. The table the paper is (have / need)

| family | matched-byte table launder | **+0 B** substitute | two-sided byte ledger | same-keys null | metric-native verdict | deletion / lifetime probe | anytime shape curve |
|---|---|---|---|---|---|---|---|
| **CLU** | have | have (**0-for-4**) | have (`ratio = 1.4·A + 0.8`, 1e−9, floor 2.20×) | have | have | have (AUC **0.5000±0.0000**, byte-equal **3072/3072**) | need |
| GRU | need | need | have (`d_hidden`) | need | have (weak) | n/a | need |
| sliding-window attention | need | need | have (`2·n_kv·d_head·w`) | need | have | n/a | need |
| Mamba-2 | need | need | have (`264·d_model + 1024`) | need | have (SSD duality) | need | need |
| DeltaNet / Gated DeltaNet | need | need | have (`n_head·d_k·d_v`) | need | have (GDN Eq. 8) | need | need |
| TTT-Linear / TTT-MLP | need | need | have (`d_head²` / `8·d_head²` + `b=16` buffer; **W₀ = parameters**) | need | have (Lin yes / MLP weakly) | need | need |
| Titans (MAC) | need | need | ⚠ **UNPINNED** — `2·|M_θ|` is **our reconstruction; the paper states no convention** | need | have (`L_M=1` yes, `≥2` weakly) | need | have (Titans-Revisited chunk sweep) |
| Sparse Delta Memory | need | need | have (Eq. 6) + **learned `M₀` = parameters** | need | have (PKM top-k *is* kNN) | need | need |

## 3. Sizing — reimplementability at this weight class (pinned 2026-07-30)
- **Free (published code, portable):** Mamba-1/2 (`state-spaces/mamba`, FLA), DeltaNet / GDN / GLA (FLA — 41-model table), TTT (`test-time-training/ttt-lm-pytorch` + JAX training repo + kernels repo), GRU / sliding-window attention (trivial).
- **⚠ SDM:** official code `facebookresearch/sparse-delta-memory` (model def + **Triton and CUDA kernels**, CC-BY-NC 4.0, `debug_sdm.yaml` single-GPU / `sdm_flagship.yaml` 1.4 B / `sdm_7B.yaml`) but **requires Torch ≥2.8, Triton ≥3.4, SM 80+ (Ampere/Hopper)** ⇒ **cannot run on this machine**. If included, it is **our reimplementation of Eqs. 3–5**, captioned as such.
- **⛔ Titans:** **no official code** (paper: *"we intend to make the code … available soon"*); **not in FLA**; **chunk size `b` never given a numeric value**; **no seeds reported**. Any Titans arm ships captioned *"reimplemented from the paper's description; the chunk size is not stated in the paper"*, citing Titans-Revisited (arXiv:2510.09551) on reproducibility.

## 4. Byte conventions B′ adds (on top of `rival-recon` §F2)
1. ⭐ **Learned-initial-state rule (general).** For any memory with a learned init (TTT `W₀`, SDM `M₀`, our `V_θ` init): **the initialisation is PARAMETERS (F1); only the per-sequence deviation is STATE (F2). Both declared.** Counting the init as state inflates; counting the deviation as parameters launders. SDM's own abstract makes the parametric role explicit.
2. **SDM:** `M_size = (d_qk^tot)²·d_v^tot/(4H²)` (Eq. 6); `α_t` is **per-head, not per-slot**; top-W/top-R index sets are per-read transients (F4, not F2). ⛔ **SDM Table 1 state/param ratios are quarantined** — two independent extractions disagree (156 % vs 168 %; 111 % vs 98 %).
3. **Titans:** the `2·|M_θ|` momentum accounting **remains our reconstruction and is captioned every time**.
4. **CLU:** every dividend/byte claim inherits the **≥2.20× ratio caveat** until the shared substrate lands (§A3).

## 5. ⭐ Pre-registered predictions (committed before measurement)
- **P1.** On real-data LM, the **+0 B pure-kNN** substitute over the store's own (key, value) pairs is **degenerate** (unbounded loss). Smoothed (add-λ / backoff), it loses to the tuned neural arm by **≥ 0.3 bpc** (≥ 20 % relative ppl). *Derivation:* Xu, Alon & Neubig (ICML 2023) — λ=1 gives ∞ perplexity because unretrieved targets get zero mass; kNN-LM's tuned λ = 0.25 ⇒ retrieval carries a minority of the probability mass.
- **P2.** Of the four families with an explicit (k,v)-shaped state (Mamba-2, DeltaNet, GDN, SDM): **≥ 3 lose to their own byte-matched table on a metric-native probe**, and **0 of 4 lose to it on real-data LM bpc.**
- **P3.** The two **function-valued** memories (TTT-MLP, Titans `L_M ≥ 2`) show the **largest positive dividend** over their matched-byte tables on real-data LM — a nonlinear readout being the field's only current escape from literal metric-nativeness.
- **P4.** **No published rival paper runs a non-parametric matched-byte control.** *Status at filing:* none found (Zoology/Based App. E.2 report state bytes but vary state by hyperparameters; MAD normalises iso-state across **neural** architectures only; SDM reports isoFLOP + isoParam with no table baseline). **Absence of evidence, medium confidence.**
- **P5.** The launder **transfers to all five** rival state types (explicit (k,v) → table directly; weight-valued memories → a byte-equal table of the `(θ_K x, θ_V x)` pairs). Predicted failures: **0 of 5.**

## 6. ⭐ Falsifiers of B′ ITSELF
- **FB1 — "not news."** ≥1 established paper in the family already runs a matched-byte non-parametric control and reports the same verdict ⇒ B′ is a replication. *(Kills P4 and B′'s novelty. Cheap to test.)*
- **FB2 — "not apples-to-apples."** For **≥2 of 5** rival families no byte-matched table is definable without an arbitrary modelling choice ⇒ the cross-family comparison is invalid and B′ collapses to a CLU-only negative.
- **FB3 — "the finding inverts."** Every rival shows a large positive dividend and only CLU does not ⇒ B′ is a *different paper* ("test-time dynamics pays, except for ours"). **We pre-commit to saying so rather than re-framing.**
- **FB4 — "the instrument is invalid."** The +0 B substitute is at ceiling for **every** family *including full attention* ⇒ the protocol measures the task, not the memory. **⭐ Run this first — it is cheap and it validates the protocol before it is spent on six families.**
- **FB5 — the theoretical neighbour.** Wang, Shi & Fox, *"Test-time regression: a unifying framework…"* (arXiv:2501.12352) already unifies linear attention, SSMs, fast-weight programmers, online learners and softmax attention as special cases. B′ must differentiate on the **empirical byte-matched audit** (they unify mechanisms; we price them). If that line adds a non-parametric matched-byte baseline, **FB1 fires.**

## 7. Banked evidence B′ reuses (do NOT re-measure)
matched-bytes launder record (28 cells; `matched=False` is architectural) · byte-floor theorem **`ratio = 1.4·atoms_per_item + 0.8`** verified to **1e−9** in all 28 cells, floor **2.20×**, measured min **2.28×** · substitute audit **0-for-4** (insertion order **0.776** vs 0.302; echo **1.0000** vs −0.180) · **Prop D2a**, three independent confirmations · byte-exact deletion **AUC 0.5000 ± 0.0000**, byte-equal **3072/3072** · `D` is the dividend's variance, not its magnitude · accuracy-vs-bytes curve (`decode 0.972 → 0.097` as ratio falls `478× → 2.28×`).

## 8. Standing caveats that travel with B′
- ⚠ **Pillar-4 narrowing:** MUNKEY (arXiv:2603.15033, ICML 2026) publishes key-deletion-as-unlearning with MIA AUROC≈0.5; it is **not exact** (avg gap **0.56 ± 0.21**) and is a ViT classifier, not a sequence memory. B′'s deletion column must be phrased on **verified byte-exactness**, never on "we alone delete."
- ⚠ **A table deletes exactly by construction** — say it ourselves before a referee does. Exact deletion is a result only for a *learned/superposed* store.
- ⚠ **Zero synthetic Track-2 candidates survive the substitute audit** (MAD `compression` fails by arithmetic; RULER VT/CWE/FWE fail to union-find/counters). B′'s real-data column is therefore the *only* admissible performance venue at this weight class — which is itself part of B′'s argument.
