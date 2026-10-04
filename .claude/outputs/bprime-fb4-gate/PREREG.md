# PREREG — `bprime-fb4-gate` (D0 FB4 · D1 write-mask hook · D2/D3 soft certificate)

**Filed 2026-07-31 by `experiment-engineer`, C2W3, BEFORE any measured run.**
Base local `main @ 6ff4c1d`, branch `agent/experiment-engineer/bprime-fb4-gate`, worktree `../CHLU-fb4`.
Protocol §5 pre-registration rule (the acceptance criterion is a measured ratio ⇒ this file is mandatory).

---

## 1. The D0.1 saturation rule, VERBATIM from the task file (ratified by the Head, non-tunable)

> Let `M(f)` = the metric's exact maximum (**1.0** for `decode`/`acc`/`r2`; **0.0** for `neg_mae`),
> `blank(f)` = the blank-store control, `sub(f)` = the best +0 B substitute, `attn(f)` = the attention arm.
> **Saturation** `S(f) = (sub(f) − blank(f)) / (M(f) − blank(f))`.
>
> **Family `f` is SUBSTITUTE-SATURATED iff `S(f) ≥ 0.95` AND `sub(f) ≥ attn(f) − 2 SE`** (3 seeds,
> `SE = sd/√3`, sample sd `ddof=1`).
>
> - ⛔ **FB4 FIRES iff ALL FOUR families are substitute-saturated.**
> - ◐ **FB4 PARTIAL (1–3 families saturated)** — each saturated family is struck from B′'s cross-family
>   audit as protocol-invalid; the wave proceeds on the survivors.
> - ✅ **FB4 CLEARS (0 families, or only the expected `manifold`).**

`0.95` and the 2-SE attention leg are constants and are **not** tuned after seeing data.

### 1.1 Declared computational choices the rule's text leaves open (fixed HERE, before the run)

**(a) Which SE.** `SE` in the attention leg = the SE of the **difference** `sub − attn` across the 3 paired
seeds (`sd(sub_s − attn_s, ddof=1)/√3`). Paired, because both arms are read on the same store from the
same seed. Also reported: the unpaired `SE(sub)` and `SE(attn)`; if the two legs disagree the verdict is
reported under both and flagged.

**(b) Aggregation order.** `S(f)` is computed from the **3-seed means** of `sub`, `blank` (and `M` exact),
and additionally per-seed (`S_s`) with its mean ± sd reported. The rule's verdict uses the
mean-of-arms form (`S(mean sub, mean blank)`), declared here.

**(c) ⭐ What counts as a "+0 B substitute" — the one genuinely load-bearing choice.**
The task's arm table gives `launder` = the settle-deleted arg-min table read at **table** bytes and
`substitute` = "the family's strongest +0 B substitute" at **table + 0 B** — i.e. *the same bytes*. The
substitute arm is therefore **a different reader of the same table**, and FB4's own text
("the +0 B substitute is at ceiling for every family *including full attention*") is a statement about
**any zero-extra-byte reader of the launder's table**. Accordingly, **pre-registered PRIMARY:**

> `sub(f) = max over ALL +0 B readers of the launder's own (key, payload) table`, **including the
> settle-deleted arg-min launder itself**.

For `overload` this is the "best of the frozen table readers at +0 B" the task asks me to declare:
the candidate set is `{settle_deleted (arg-min), knn2_mean, knn2_idw}`.
Per family the +0 B candidate sets are:
| family | +0 B candidates |
|---|---|
| `overload` | `settle_deleted`, `knn2_mean_+0B`, `knn2_idw_+0B` |
| `aggregate` | `settle_deleted`, `knn2_mean_+0B`, `knn2_idw_+0B` |
| `recency` | `settle_deleted` (pair-restricted, per the D4 fix), `order_aware_+0B` (shipped, unrestricted k=2), ⭐ `order_aware_pair_+0B` (**new here**: pick the *newer* of the query's **own two candidates** — the same pair information `restrict_index_to_pair` hands the CLU arm, so it is +0 B and it is the honest strongest substitute once every arm answers the 2-way question) |
| `manifold` | `settle_deleted` (constant 0), `echo_+0B` |

**Declared secondary (reported beside the primary, never instead of it):**
`sub_excl_launder(f)` = the same max **excluding** `settle_deleted`, with its own `S` and its own verdict.
Choosing the inclusive form as primary makes saturation *easier* to reach, i.e. it is the choice that
cannot flatter the program.

**(d) The attention arm, exactly.** Over the launder's own `(key, payload)` table:
`w = softmax(q·kᵀ / (τ·√d))`, and the read is `w`-weighted over the table's value column:
* `value` families (`overload`, `aggregate`): `pred = Σ_i w_i · payload_i` → `score_value`;
* `index` family (`recency`): the logits are restricted to the query's own pair (`restrict_to_pair`,
  the same fix every other arm gets) and the prediction is the arg-max — note this makes the arm
  **τ-independent** for `recency` (a positive scalar cannot reorder two logits); that will be stated;
* `coord` family (`manifold`): `pred = Σ_i w_i · spectator_i`, and the table's spectator column is
  **written zero for every row** (a table stores one point per item), so the arm is predicted to be
  identically 0.
`τ` is fitted by grid search (`τ ∈ logspace(−2, 2, 41)`) on the family's **own train split** — an
independent draw of the same query law with `rng = default_rng(seed + 20260731)` — maximising the
family's primary metric. Ledger: **table bytes + 4 B** (one float32 temperature). No other parameters.
⚠ The arm is a **table reader** and never sees a trajectory; it is **not** `AttentionPsi` and inherits
none of its quarantine.

**(e) The anchors.** `overload` at `load1x_shipped` (the 478× cell; reconciliation 6). `aggregate`,
`recency`, `manifold` at `base`. `recency` runs with `restrict_index_to_pair=True` (C2W2 D4), and the
pre-fix coverage is emitted alongside so the switch is auditable. Seeds `{0,1,2}` on all four.

---

## 2. ⭐ Predicted `S(f)` per family (committed before measurement)

Derivation source: `memory-gym-v0.md` §3.2 (the C2W1 28-cell artefact) for `sub`/`blank`, plus the C2W2
D4 recency fix. `M` is exact by definition.

| family | metric | `M` | predicted `blank` | predicted `sub` (primary, incl. launder) | **predicted `S`** | range | saturated? |
|---|---|---|---|---|---|---|---|
| `overload@load1x_shipped` | `decode` | 1.0 | 0.1667 | **1.0000** (`settle_deleted`; C1W1 measured the launder at 1.0000 at this anchor, vs `knn2` 0.7083) | **1.000** | [0.93, 1.00] | ⭐ **YES** |
| `aggregate@base` | `neg_mae` | 0.0 | −0.4221 | **−0.2081** (`knn2_idw_+0B`; launder −0.4472) | **0.507** | [0.40, 0.70] | **NO** |
| `recency@base` (pair-restricted) | `acc` | 1.0 | ~0.50 | **1.0000** (`order_aware_pair_+0B`, correct **by construction**) | **1.000** | [0.98, 1.00] | ⭐ **YES** |
| `manifold@base` | `r2` | 1.0 | −0.0001 | **1.0000** (`echo_+0B`, exact by construction) | **1.000** | [0.99, 1.00] | ⭐ **YES** |

**Reasoning.**
1. `overload` at the **shipped anchor** is measured where the arg-min table is *already perfect*
   (`launder = 1.0000`, C1W1). A family whose table reader is at the metric's exact maximum cannot
   discriminate readers. The Hub's prior ("overload does not saturate") was formed against `sub` =
   the *non-launder* substitutes (0.7083) — under the **secondary** definition I predict
   `S_excl = (0.7083 − 0.1667)/0.8333 = 0.650` ⇒ **not saturated**. This family's verdict is therefore
   *entirely* a function of choice (c), and I say so in advance.
2. `aggregate`'s target is a convex combination that is **not** any stored payload and is dropped at
   construction if it lands within `payload_tol` of one — so no table reader can be exact, and the
   +0 B ceiling is a genuine interior value. This is the family I predict survives.
3. `recency` after the D4 fix: once every arm chooses **between the query's own two candidates**, a
   reader that knows the table's row order answers the question **exactly** — insertion order *is* the
   ground truth. Predicted `sub = 1.0000` on 3/3 seeds. (The shipped, unrestricted `order_aware_+0B`
   measured 0.7764 only because the two nearest keys to a jittered midpoint are not always the
   intended pair; that failure mode is removed by the same fix that removes it for the CLU.)
4. `manifold`: `echo` = 1.0000 at +0 B by construction (intervention §8.3), C1W1-measured.

**⭐ PREDICTED VERDICT: `PARTIAL = {overload, recency, manifold}`; the sole surviving family is
`aggregate`.** (Secondary definition: `PARTIAL = {recency, manifold}`, survivors `{overload, aggregate}`
— the Hub's own prior, plus recency.) **FB4 does NOT fire** under either, because `aggregate` saturates
under neither. I am deliberately predicting *worse than the Hub's prior* (3 struck, not 1): the Hub's
prior was formed before choice (c) was fixed and before the recency pair-restriction was applied to the
substitute as well as to the CLU.

## 3. Predicted attention-arm ordering vs the +0 B substitute

| family | predicted `attn` | ordering | note |
|---|---|---|---|
| `overload` | **0.95 – 1.00** | `attn ≈ sub` (both at/near ceiling) | dot-product arg-max ≠ nearest-key arg-min only through the `‖k‖²/2` term; at 478× the CLU also reads 0.9722 |
| `aggregate` | **−0.25 … −0.15** | `attn ≳ sub` (attention is a *soft* kNN with a fitted bandwidth, a strict superset of the 2-NN mean at τ→0/∞ limits) | if `attn > sub + 2SE` the second leg fails and the family is not saturated regardless of `S` |
| `recency` | **≈ 0.50** (chance) | `attn ≪ sub` | the payload table carries no time column; pair-restricted arg-max of `q·k` at a midpoint is a coin flip |
| `manifold` | **exactly 0.0000** | `attn ≪ sub` | the table's spectator column is all zeros ⇒ any convex combination is 0 |

⚠ **Does NOT falsify anything:** attention winning on a metric-native family is the
metric-native-ceiling theorem (intervention §6 criterion 4), confirmed four times.

## 4. D1 / D2 / D3 bit-identity predictions (EXACT, not statistical — a partial pass is a fail)

* **D1** `store_write_mask_factory = None` (the default) ⇒ `chlu/core/clu_system.py` behaviour is
  **bit-identical to `6ff4c1d`**: predicted **exact** equality of the written store parameters and of
  every monitor trip-state on the C2W1 shipped anchor. A toy store family supplying its own mask
  preserves C3 locality where the unmasked leaf breaks it (the existing failure-asserting test gains a
  passing partner).
* **D2** soft certificate **default-OFF** ⇒ bit-identical to `6ff4c1d`; predicted **exact**.
  With SC-1 ON (`d_safe = ζ·sep_expected`, `ζ = 0.6`) the identification `d_safe := 2s_max + κ′σ_q` is
  broken and `R_cert` is still computed and reported but is no longer the gate; SC-3's budget
  `B = 0.33` (domain `s/sep ∈ [0.15, 0.30]`; `sep/2` is never-quote as a certified inradius) **TRIPS**
  monitor #3 when exceeded and never refuses.
  Predicted price, carried unsoftened from the theorist: `ρ_ex` up to **6.3×** at a `λ_min` cost of
  **2.2–6.0×**, and **the dividend in that region stays ≈0** (+0.0043 … −0.0067) — a **precondition**,
  not a result.
* **D3** monitor #3's correlation leg → C3 first-order calibration leg: predicted spearman **+0.914**
  (vs +0.412) and **0/12 sign flips** (vs 1/12) at zero extra cost.
* **Trip-state acceptance:** every changed trip on the re-run C2W1 shipped anchor maps **1:1** to a named
  repair, diffed against the **on-disk** C2W1 artefacts (`.claude/outputs/{memory-gym-v0,full-clu-harness}/*.json`),
  never against a freshly generated baseline; everything else bit-identical. Diff is against the
  **post-C2W2** state (monitor #6: **27 post-repair**, 58 **pre-repair**; #6 artefact count **31 of 58**).

## 5. Declared NOT-RUN (never reported as nulls)
* `overload` at the base atom budget (0/18 admissible incl. the Gaussian control — reconciliation 6).
* The annealed second read variant (`memory-gym-v0`'s +0 B read lever) on the FB4 arms — the FB4 rule is
  written against the shipped read; the annealed read is orthogonal to the substitute question.
* Any rival family (`bprime-rivals` owns them, and it is gated on this verdict).
* A learned/trained attention reader (more than a scalar temperature would break the byte commensurability
  leg of FB4 and is the falsifier "the attention arm cannot be given a commensurate ledger").
