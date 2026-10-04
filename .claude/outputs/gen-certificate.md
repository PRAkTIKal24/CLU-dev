# gen-certificate — experiment-engineer report

**Task + acceptance criterion:** compute, for a requested factor combination, a composition energy/barrier
**from quantities the store already exposes**, and report how well it separates impossible requests from
possible ones — with **provably zero invalid-example supervision**, asserted mechanically.

**Status: done.** 3 seeds {0, 1, 2} × 2 groupings, all six arms complete. The zero-invalid-supervision
assertion ran on every one and found **0 C-record ids in 15 fits / 78,848 ids** each time. Suite green on
both sides of the rebase (**2166 → 2204, +38 = exactly my new tests**, 0 failed). ⛔ **No kill-rule verdict is
rendered here** — that is `gen-cell-run`'s, against its own prereg (task instruction, R7).

⭐ **The one-line result.** The certificate separates impossible requests from valid-but-unseen ones at
**AUROC 0.9569 ± 0.0099** (R23's primary, B-vs-C) with **zero invalid-example supervision, asserted rather
than promised** — and at a threshold fitted on valid data alone it refuses **90.3 % of impossible requests
while refusing 1.4 % of the held-out valid ones**.
⛔ **The one-line caveat.** Half of the C set (`out_of_range`) is separable by a grid lookup and the other
half (`duplicate_factor`) carries a **seven-terms-vs-six giveaway worth AUROC 1.0000 to a statistic with no
model in it**. Against the term-count control I built to remove that, the derived barrier holds at
**0.9146 ± 0.0086** while the naive distance-to-store control collapses to **0.5630 ± 0.0396** — a margin of
**+0.3516 ± 0.0372, positive in 3/3 seeds**. That is the number the claim rests on.

> ## ⛔ DOWNSTREAM RECONCILIATION LIST — 5 items, needs a Hub owner (protocol §5 corollary)
> 1. ⭐⭐ **The C set's `duplicate_factor` records carry SEVEN terms; A and B carry six.** A statistic that
>    does nothing but *count terms* scores **AUROC 1.0000 exactly** on C-dup-vs-B and **0.5000 exactly** on
>    C-oor-vs-B (measured, all 3 seeds, both groupings). ⛔ **Every arm in the cell — kNN, LACE, CLU —
>    inherits this giveaway**, and it is invisible in an aggregate C number. I ship the fix (`A7`: valid
>    requests carrying a *redundant* term, so they are 7 terms long and still satisfiable) and report every
>    dup number in both forms. **`gen-baselines` and `gen-cell-run` need this ruling before they score C.** (§3, §5)
> 2. ⚠ **`scripts/csf3/job_gpu_gen.sh` hard-rejects any `STAGE` outside `encoder|decoder|store|cell`**, so
>    there is no `certificate` stage on the cluster. A working submit line exists (`STAGE=cell` +
>    `EXTRA_ARGS="--stage certificate"`; verified that argparse takes the later value) but it writes into a
>    `cell/` workspace path. ⛔ `scripts/csf3/` is not mine to edit. **Owner: `gen-csf3-launcher`.** (§10)
> 3. ⚠ **`gen_data.data_root` means two different things in the shipped tree** — the *dataset* directory for
>    `shapes3d_root()`, the *parent* of it for `default_gen_data_root()`. `exp_gen_store.load_c_records`
>    uses the parent spelling, so setting `--data-root .../gen/shapes3d` makes it log
>    "⚠ no C set … SKIPPED" and continue. A latent silent-skip. **Owner needed.** (§9.3)
> 4. ⭐ **§7.23 has a mirror image and it is not in the ledger: a test that restores `jax_enable_x64` to a
>    hard-coded `False` turns x64 OFF under the six modules that enable it at import** — which happens during
>    *collection*, before any test runs, so sort order does not protect anyone. It cost me a full-suite run
>    (9 failures in `test_goldstone.py`, invisible in isolation). Fixed here; ⛔ **the ledger entry belongs in
>    §7 so the next spoke does not repeat it.** (§9.6)
> 5. ⚠ **My pre-registered primary `clu` includes two components that are term-count contaminated**
>    (`curv`, `energy`). The term-count-clean combination is `clu_pair_reach = reach + z(pair)`, which is
>    **better** on the controlled metric (0.9328 vs 0.9146). ⛔ Which one the *cell* reports is a Hub/Head
>    decision, not mine to take after the fact. (§4, §6)

---

## ⭐ DIAL DECLARATION (echoed, protocol §7)
- **Dial:** *admission* — whether a request can be refused from the store's own geometry.
- **Laundering control:** ⭐ a plain **distance-to-store** threshold on the same features (`distance`), and
  the same thing plus the grid-reach term (`distance_reach`, the **hard** control). Both ran as full arms on
  every number below; ⛔ neither is a footnote. A third, `count`, measures the syntactic giveaway directly.
- **Falsifies:** certificate AUROC at chance — **or any threshold that touched invalid data.** Neither fired:
  the audit certified 6/6 (seed, grouping) runs, and the certificate is 0.9569 ± 0.0099 on the primary metric.
- **Does NOT falsify:** losing to a **label-supervised** post-hoc detector (LACE). Reported from phase 1
  onward, never phase-killing (R7). ⛔ That arm is `gen-baselines`'; nothing here is a LACE comparison.

## FLAG-PROVENANCE TABLE (protocol §5, mandatory)
| | value |
|---|---|
| branch / commits (post-rebase) | `agent/experiment-engineer/gen-certificate` — `0ae9975`, `dd4a859`, `00fbc59`, `eed1fa5`, `7ff5a00`, `24492e2`, `ac22220`, `5bd8d2b` (pre-rebase: `e6c60cc`…`01d4d8d`) |
| base | `main @ 9a4457d` (the amendment's base). ⚠ **`main` moved to `a0034d4` mid-session** (gen-dream-sampler merged); every result above was produced at `9a4457d` and the branch was rebased afterwards — §11.1 |
| dataset / pretext / condition | `shapes3d` / `recon` (R1 primary) / `recomb2element` (Montero). ⛔ no `cls`, no dSprites, no wide cells |
| seeds | **{0, 1, 2}**, all complete. `±` = population std over seeds (ddof = 0), matching `gen-store-grouping`; per-seed values printed under every headline |
| request sets | **A** = 1,024 held-back `eval_a` rows · **B** = all **72** held-out rows · **C** = **72** records read VERBATIM from `.claude/data/gen/shapes3d/c_invalid.json` (schema v1, 36 `duplicate_factor` + 36 `out_of_range`) · **A7** = 1,024 term-count controls built from A alone |
| encoder (FROZEN) | hash asserted == `bridge_manifest.json`: s0 `98d326028316ae2c…`, s1 `fc05e2c5f39694b2…`, s2 `154d498610648fac…` |
| feature cache (FROZEN) | s0 `d6728f01d102b0d0…` (scope **`full`**, 480,000 rows), s1 `484b446c4dbb9fc4…`, s2 `8c0a69e88beb46b8…` (scope **`used`**, 9,288 rows). ⭐ `_assert_cache_reencodes` **max\|Δ\| = 0.0** on all 3 seeds |
| cache extension | s0 0 rows; s1 **32,199**; s2 **32,241** rows encoded on the fly through the frozen encoder (`gen-store-grouping` §10.1 pattern). ⛔ no cache rewritten on disk |
| decoder | ⛔ **NONE. Nothing is decoded.** `assert_no_decoder_in_path()` ran first in every seed and passes two legs (runtime signatures + module AST). ⭐ This is why the `dec_recipe="baseline"` trap cannot reach a number here — see §9.1 |
| store | ⭐ **RE-MATERIALISED, not retrained**, identical key chain; `relative_mse` asserted **`abs_delta == 0.0`** against `gen-store-grouping`'s shipped manifest on **all 6** (seed, grouping) arms: s0 `0.28266923706753433`/`0.4185249337141173`, s1 `0.26975176078844887`/`0.41951284319427884`, s2 `0.28414548978230525`/`0.45214615079541975` |
| grouping | `K = 5` **loaded from `store_plan.json`, never re-derived**; re-derived `labels` asserted element-for-element equal to the shipped vector on all 6 arms. Wells s0 `[303,305,573,501,366]`, s1 `[397,300,307,300,744]`, s2 `[283,283,390,616,476]` |
| store config (all `gen_store` defaults) | `n_modes 64` · `n_store_train 32768` · `n_corr 16384` · `train_steps 1200` · `train_batch 128` · `train_lr 3e-3` · `commitment_beta 0.25` · `tau 1.0 → 0.1` geometric · `min_well_frac 0.10` · `grouping_method auto → ward` · `eval_batch 256` · `encode_chunk 16384` |
| **new** `gen_certificate` config | `laplace_alpha 1.0` · `reach_radius_frac 0.5` (the factor's own Voronoi radius) · `tau_curvature None → gen_store.tau_end = 0.1` · `accept_quantile 0.95` · `groupings [correlation, random]` · `term_count_control True` · `save_store True` · `assert_store_bit_identity True` · `n_requests_a None` |
| realised reach radii | `[0.05, 0.05, 0.05, 0.0357142857142857, 0.5, 2.1428571428571423]` (half the median grid spacing, per factor) |
| mode usage | **64.0 / 64** modes used in every well of every arm (no dead codebook) |
| min rows per (factor, value) | **2,115** (s0) / 2,106 (s1) / 2,108 (s2) — the histograms are not sparse |
| JAX / env | JAX **0.9.0**, **main venv** `/Users/user/Desktop/CHLU/.venv` (protocol §4). ⛔ **no `uv sync` in the worktree** |
| wall clock | s0 1,383 s + 877 s; s1 1,114 s + 1,495 s; s2 1,107 s + 1,493 s — CPU only, under load averages 30–430 from four concurrent foreign spokes and a post-reboot Defender scan (§9.5) |

---

## 1. Preconditions, checked mechanically first

| Check | Result |
|---|---|
| `git cat-file -e main:chlu/core/gen_store.py` | ✅ exit 0 |
| store artefact exists (`store_plan.json`, `K=5` pinned) | ✅ |
| `c_invalid.json` present, schema v1, 72 records, factor names match the loaded grid | ✅ read verbatim, never regenerated |
| worktree | fresh `../CHLU-gen-cert` off `main @ 9a4457d`. ⛔ `CHLU-bprime-family2`, `CHLU-wt2`, `CHLU-wt3`, `CHLU-merge`, `CHLU-gen-store` untouched |
| shared checkout | ⚠ was on another agent's branch `agent/experiment-engineer/pilot-ttt-nan-and-d5-wiring @ 7fcef50` the whole session; ⛔ I never edited it (protocol §3.2) |
| concurrent spokes | `gen-r0-wide-refit` (×3 processes), `gen-baselines`, and two foreign pytest runs observed. I share only `chlu/config.py` and `chlu/cli/experiment_cmd.py`, both **append-only** |

I read `AGENT_PROTOCOL.md`, `gen-clu-phase1-rulings.md` (R2/R5/R6/R7 + R8–R13, R23 via the amendment),
`gen-store-grouping.md`, `gen-r0-rescore.md`, `chlu/core/soft_certificate.py`, and — per the task —
**`handover_context.md` §7 Known Issues only**.

**On reusing `soft_certificate.py` (the task said "reuse what fits, do not edit it").** ⛔ I edited nothing in
it, and in the end I **imported nothing** from it either, which needs justifying rather than passing over.
Its objects are a *merge* certificate over a C3 physics store: `cert_radius(s_max, sigma_q)`,
`cert_margin(sep_after, R_cert)`, `capture_radius(relax_fn, site)`, `budget_state(deficits)` — every one of
them is a statement about **whether two written sites will collide**, parameterised by a key spacing, a
payload scale and a relaxation dynamic. R0's store has no sites, no writes, no relaxation and no key
spacing: it is a channel partition with a per-well codebook. The one verb that did transfer is the
soft-occupancy softmax, and I reach it the same way `gen_store` does. ⭐ **What I took instead is the
module's *discipline*** — a declared admission radius that is not the same object as the guarantee radius
(SC-1), and a falsifier recorded in the artifact rather than in prose (SC-7). Forcing `cert_radius` onto a
codebook would have meant inventing a `sigma_q`, i.e. inventing the answer. ⚠ If the Hub wants the
certificate expressed in SC-1…SC-7 language, that is a scope change and a decision, not a silent substitution.

---

## 2. What the certificate is (declared in `PREREG.md` before a line of it existed)

For a request `R = {(factor j, value u)}`, from quantities the store already exposes:

| component | definition | what it can see |
|---|---|---|
| **`reach`** | `Σ_t (dmin_t / radius_j)²`, `dmin` = distance from `u` to the nearest **realised** grid value, `radius_j` = half the median grid spacing (the factor's own Voronoi radius) | ⛔ **identically 0** for every A, every B **and every `duplicate_factor` C record.** A detector of `out_of_range` and nothing else |
| **`pair`** | `χ(t,t') = −Σ_k log Σ_m q_t[k,m]·q_t'[k,m] / p̄[k,m]`, max over unordered pairs. `q` = the per-well mode histogram of **the store's own addressing** over valid train rows (Laplace α=1); `p̄` = the marginal mode usage | ⭐ the **only** component that can see a `duplicate_factor` contradiction |
| **`curv`** | `max_k(−λ_min)`, `λ_min = 2 − (2/τ²)·λ_max(Cov_w(μ_k))` — the **closed-form Hessian** of the store's own soft energy `Φ = −2τ² log Σ_m exp(−‖x−μ_m‖²/2τ²)` at the product-of-experts composed point `x̂_k = Σ_m r[k,m] μ_{k,m}` | a composed point sitting on a ridge *between* modes |
| **`energy`** | `Σ_k min_m ‖x̂_k − μ_{k,m}‖² / D` — the store's own quantisation energy of the composed point | how far off-codebook the composition lands |

`cert_total` (arm **`clu`**) `= reach + z(pair) + z(curv) + z(energy)`, every `z` fitted **on the A set alone**
through the audited fitter. Higher = refuse.

⭐ **The Hessian is not autodiff and not approximate.** `∇²Φ = 2I − (2/τ²)Cov_w(μ)` is exact; `λ_max(Cov_w)`
comes from the `(M, M)` Gram, so no `(d, d)` matrix is ever formed. It is checked against a four-point finite
difference in `test_hessian_closed_form_matches_finite_differences` (agrees to `2e-3` absolute).

**Controls, all scored in the same job on the same store and the same features:**
`distance` (naive kNN-style prototype composition → plain distance to store) · `distance_reach` (the same
plus the identical reach term — the **hard** control) · `count` (the number of terms) · `reach` alone ·
each derived component alone.

---

## 3. ⛔⭐ THE STRUCTURAL FINDING, and it is not about our method

Two facts about the C set decide how every number in this phase must be read. Both are properties of
`c_invalid.json` as `gen-data-splits` built it, and both are **measured, not argued**:

**(a) `out_of_range` is separable by a grid lookup.** `reach` is 0 for every on-grid request and > 0 for
every out-of-range one, so `reach` **alone** scores **B-vs-C[oor] = 1.0000 exactly**, on every seed, in both
groupings. ⛔ Any arm that embeds a request at all will get ~1.0 there. That number carries no information
about any mechanism and must never be quoted as certificate performance.

**(b) ⭐ `duplicate_factor` records carry SEVEN terms; A and B carry six.** So the arm `count` — a statistic
that does nothing but count terms — scores:

| `count` arm | measured, all 3 seeds, both groupings |
|---|---|
| B-vs-C[`duplicate_factor`] | **1.0000 ± 0.0000** |
| B-vs-C[`out_of_range`] | **0.5000 ± 0.0000** |
| B-vs-C (aggregate C) | **0.7500 ± 0.0000** |

⛔ **A pooled C number of 0.75 is available to a one-line function with no model in it.** This is not a
criticism of the C set — it is a faithful consequence of "request two values of the same factor" — but it
means **the dup question must be asked against a term-count control**, or every arm in the cell is scored
partly on request length. That control is `A7`.

**`A7`** (prereg amendment A2): each A request gets one **redundant** extra term — a factor repeated *at the
value it already has*. It is still satisfiable (it is the same image) and it is seven terms long. ⛔ It is
built from **valid requests alone**, so no invalid example is used and the supervision claim is untouched.

⭐ **`pair` is provably immune to this, and the composed-point components provably are not.** By
Cauchy–Schwarz `Σ_m q²/p̄ ≥ 1`, so `χ(t,t) ≤ 0`; every other pair a repeated term adds is a copy of one the
request already had. Hence `max over pairs` cannot move when a term is repeated — **measured: 0 of 16 `pair`
scores moved, while `curv` moved on all of them** (`test_the_pair_barrier_is_invariant_to_a_redundant_term`).
The component means show the size of the contamination directly (seed 0, correlation):

| set | `pair` | `curv` | `energy` | `reach` | n_terms |
|---|---|---|---|---|---|
| A | 0.2889 | 12.04 | 0.006641 | 0 | 6 |
| **A7** (same requests + a redundant term) | **0.2889** | **2.773** | 0.006004 | 0 | **7** |
| B | 0.3649 | −2.000 | 0.001356 | 0 | 6 |
| C | 2.107 | 0.568 | 0.009004 | 17.44 | 6.5 |

⚠ `curv` moves by **9.3** on a request whose *meaning* did not change at all.

---

## 4. ⭐ THE RESULT — all three AUROCs, separately and with equal prominence (R7/R23)

**Correlation grouping (the R0 headline store), 3 seeds, mean ± population std.**

| arm | **A-vs-C** | ⭐ **B-vs-C** (R23 primary) | **A∪B-vs-C, class-balanced** | B-vs-C[dup] ⚠ | B-vs-C[oor] | ⭐ **A7-vs-C[dup]** (controlled) | AURC(B∪C) |
|---|---|---|---|---|---|---|---|
| ⭐ **`clu`** (pre-registered primary) | **0.9537 ± 0.0048** | **0.9569 ± 0.0099** | **0.9553 ± 0.0060** | 0.9138 ± 0.0199 | 1.0000 ± 0.0000 | **0.9146 ± 0.0086** | 0.1998 ± 0.0277 |
| `clu_pair_reach` (A1.2 secondary) | 0.9664 ± 0.0066 | 0.9383 ± 0.0074 | 0.9523 ± 0.0046 | 0.8766 ± 0.0148 | 1.0000 ± 0.0000 | **0.9328 ± 0.0132** | 0.2124 ± 0.0046 |
| `pair` alone | 0.7145 ± 0.0140 | 0.5846 ± 0.0317 | 0.6495 ± 0.0193 | 0.8766 ± 0.0148 | 0.2926 ± 0.0537 | **0.9328 ± 0.0132** | 0.5103 ± 0.0143 |
| `curv` alone | 0.5454 ± 0.0180 | 0.5981 ± 0.0192 | 0.5718 ± 0.0180 | 0.6111 ± 0.0063 | 0.5851 ± 0.0385 | 0.5658 ± 0.0137 | 0.4471 ± 0.0106 |
| `energy` alone | 0.6089 ± 0.0246 | 0.8005 ± 0.1785 | 0.7047 ± 0.0894 | 0.8652 ± 0.1332 | 0.7357 ± 0.2240 | 0.7271 ± 0.0352 | 0.3111 ± 0.1361 |
| `reach` alone | 0.7500 ± 0.0000 | 0.7500 ± 0.0000 | 0.7500 ± 0.0000 | 0.5000 ± 0.0000 | 1.0000 ± 0.0000 | 0.5000 ± 0.0000 | 0.3600 ± 0.0000 |
| ⭐ `distance` (**the laundering control**) | 0.5264 ± 0.0209 | 0.7886 ± 0.0255 | 0.6575 ± 0.0038 | 0.8372 ± 0.0434 | 0.7400 ± 0.0382 | **0.5630 ± 0.0396** | 0.3586 ± 0.0352 |
| ⭐ `distance_reach` (**the hard control**) | 0.7846 ± 0.0206 | 0.9186 ± 0.0217 | 0.8516 ± 0.0163 | 0.8372 ± 0.0434 | 1.0000 ± 0.0000 | **0.5630 ± 0.0396** | 0.2452 ± 0.0113 |
| `count` (**syntactic**) | 0.7500 ± 0.0000 | 0.7500 ± 0.0000 | 0.7500 ± 0.0000 | 1.0000 ± 0.0000 | 0.5000 ± 0.0000 | 0.5000 ± 0.0000 | 0.3750 ± 0.0000 |

Per-seed, `clu`: **B-vs-C** 0.9660 / 0.9616 / 0.9431 · **A7-vs-C[dup]** 0.9093 / 0.9268 / 0.9077.
Per-seed, `distance_reach`: **A7-vs-C[dup]** 0.6007 / 0.5801 / 0.5082.

⛔ **The pooled unbalanced A∪B-vs-C number is computed and shipped, labelled
`pooled_UNBALANCED_do_not_quote`, precisely so nobody has to recompute it to see why.** For `clu` it is
0.9539 — close to the balanced 0.9553, because A and B behave alike for it. ⚠ For `distance` it is **0.5436**
against a balanced **0.6575** and a B-vs-C of **0.7886**: a **0.245 spread** across three numbers that a
single pooled AUROC would have collapsed into one. That is R7's hazard, measured on our own control.

⭐ **The balanced identity holds numerically:** `AB_vs_C_balanced == ½(A-vs-C + B-vs-C)` to
**max 1.67e-15** across all arms and seeds (weighted Mann–Whitney; asserted at run time and in the tests).

**Random grouping (the store-level laundering control), 3 seeds:** `clu` B-vs-C **0.9277 ± 0.0453**,
A7-vs-C[dup] **0.7989 ± 0.0326**; `distance_reach` A7-vs-C[dup] 0.5944 ± 0.0674. Full table in `tables.md`.

### ⭐ The operating point — and the number I did not expect to be this clean
Threshold = the **95th percentile of A** (valid data only, through the audited fitter). Correlation grouping,
`clu`, 3 seeds:

| refused | A (valid, seen) | ⭐ B (valid, **unseen** — the cost) | C (invalid) |
|---|---|---|---|
| rate | 0.051 ± 0.000 | **0.014 ± 0.020** (per-seed 0.000 / 0.042 / 0.000) | **0.903 ± 0.011** |

⭐ At a threshold that refuses 5 % of *seen* valid requests, the certificate refuses **1.4 % of the held-out
valid combinations** and **90.3 % of the structurally impossible ones**. `clu_pair_reach` refuses **0.000 ±
0.000** of B at 0.912 ± 0.007 of C.

---

## 5. ⭐⭐ THE LAUNDERING CONTROL — the question the spoke actually lives on

⚠ **Read the contaminated column first, because on its own it would have been an equivocal result.**

| margin (correlation grouping) | mean ± std | per-seed | verdict |
|---|---|---|---|
| `clu` − `distance` on **B-vs-C** | **+0.1683 ± 0.0331** | +0.1732 / +0.2062 / +0.1256 | 3/3 positive |
| `clu` − `distance_reach` on **B-vs-C[dup]** ⚠ contaminated | **+0.0766 ± 0.0465** | +0.0455 / +0.1424 / +0.0421 | ⛔ **below my pre-registered +0.10 threshold** |
| ⭐ `clu` − `distance_reach` on **A7-vs-C[dup]** (controlled) | **+0.3516 ± 0.0372** | +0.3087 / +0.3467 / +0.3994 | ⭐ **9.5× the seed spread, 3/3** |
| ⭐ `pair` − `distance_reach` on **A7-vs-C[dup]** | **+0.3698 ± 0.0292** | +0.3356 / +0.3668 / +0.4069 | ⭐ 12.7× spread, 3/3 |

**What happened, plainly.** On the *contaminated* dup metric the naive distance control reaches 0.8372 and
the derived barrier 0.9138 — a +0.077 margin that misses the bar I set. **Both numbers are inflated by the
same seven-terms giveaway.** Remove it and the control collapses to **0.5630** — barely above chance — while
the derived barrier holds at **0.9146**. ⭐ The derivation is **not** decoration over naive geometry; the
appearance that it might be was an artefact of scoring 7-term requests against 6-term ones.

⛔ **Honest statement of what this does and does not settle.** `distance_reach` is *my* in-house trivial
control, built to the same supervision class. The **byte-matched kNN arm that kill rule (iv) actually names
is `gen-baselines`'**, and it is not run here. This result predicts that arm will also collapse on `A7`, but
⛔ that is a prediction, not a measurement, and `gen-cell-run` must make it.

⚠ **And the control is weaker than even these margins suggest** — §7.1 shows `distance` is at **chance**
against A (0.5264 ± 0.0209) and **below chance** on out-of-range (0.4836 ± 0.0275). Its B-vs-C number is a
property of the B set, not evidence of a refusal ability. ⛔ Do not quote "the naive baseline reaches 0.79"
without that column beside it.

### The mechanism fires for the right reason — attribution, which no term count can fake
For each C record the certificate names a culprit: for `duplicate_factor` the two terms of the max-conflict
pair; for `out_of_range` the max-reach term. Scored against the record's own `violated_factor` field:

| grouping | `duplicate_factor` | `out_of_range` | chance |
|---|---|---|---|
| **correlation** | **28/36, 29/36, 29/36 = 0.797 ± 0.013** | **36/36 = 1.000** (3/3 seeds) | 0.167 |
| random | 24/36, 25/36, 23/36 = 0.667 ± 0.024 | 36/36 = 1.000 | 0.167 |

⭐ **4.8× chance**, and the correlation grouping beats random by **+0.130 ± 0.026, 3/3 seeds.** ⛔ This
statistic is independent of the term count and of the accept threshold entirely.

---

## 6. ⭐ The audit — the criterion the task said it lives on

Every threshold and every normaliser in the harness is fitted through one function,
`ValidOnlyFitter`, which records the **sha256 of exactly the sorted ids it saw**. `assert_no_invalid` then
re-reads every record and asserts no C id appears in any of them.

| per (seed, grouping) run | value |
|---|---|
| fits audited | **15** (2 row registrations + 4 z-standardisers + 9 per-arm quantile thresholds) |
| ids audited | **78,848** |
| C ids checked against every fit | **72** |
| ⭐ **C ids found in any fit** | **0** |
| runs certified | **6 of 6** (3 seeds × 2 groupings); `certificate_results.json.CERTIFIED_no_invalid_supervision = true` |

**Why this is evidence and not a promise:**
- ⭐ the assertion has a **positive control** — `test_the_audit_fires_when_a_C_id_reaches_a_fit` hands it a
  fit that *did* see a C record and requires it to raise. An assertion never seen to fail is not evidence;
- it **refuses an empty invalid set** (`test_the_audit_refuses_an_empty_invalid_set`) — a check that cannot
  fail is not a check;
- a component with zero variance on the fitting set (`reach` is identically 0 on A) is recorded as
  `passthrough_zero_variance: true` rather than silently patched;
- the store's own training rows and the prototype rows are registered on the same footing, so the audit
  covers the *upstream* fit too, not just my thresholds;
- it is **structurally impossible** for a C record to have contributed data anyway: a C request has no rows.
  The audit is there for the *thresholds*, which is where the failure would actually hide.

⛔ **No learned refusal head exists anywhere in this branch.** The criterion is a fixed function of the
request and the store; the only fitted quantities are four z-scales and nine thresholds, all from A.

---

## 7. Pre-registration scorecard

`PREREG.md` written **before `chlu/core/gen_certificate.py` had a line** (design fixed from `gen_store.py`,
`c_invalid.json` and the rulings alone). Amendments A1 (after the plumbing smoke) and A2 (disclosed: after
seeing seed 0 / correlation's non-A7 numbers, before any full-scale A7 number existed) are dated in the file.

| # | prediction | threshold | measured (correlation, 3 seeds) | verdict |
|---|---|---|---|---|
| P1 | `clu` B-vs-C ≥ 0.90 (point 0.96) | 0.90 | **0.9569 ± 0.0099** | ⭐ **HELD**, point hit |
| P1a | `clu` B-vs-C dup ≥ 0.80 (point 0.92) | 0.80 | **0.9138 ± 0.0199** | ⭐ **HELD** |
| P1b | `clu` B-vs-C oor = 1.000 exactly | exact | **1.0000 ± 0.0000** | ✅ **HELD** |
| P2 | \|A-vs-C − B-vs-C\| ≤ 0.05 | 0.05 | **0.0032** | ✅ **HELD** |
| P3 | balanced = ½(A-vs-C + B-vs-C), \|Δ\| ≤ 1e-12 | 1e-12 | **max 1.67e-15** | ✅ **HELD** |
| P4 | `distance` B-vs-C ≤ 0.70 (point 0.55) | 0.70 | **0.7886 ± 0.0255** | ⛔ **FALSIFIED** — the naive control is *stronger* than I predicted (§5) |
| P4a | `distance` oor ≤ 0.70 (point 0.50) | 0.70 | **0.7400 ± 0.0382** | ⛔ **FALSIFIED** — see the finding below |
| P5 | `clu` − `distance_reach` on dup ≥ +0.10 | +0.10 | **+0.0766 ± 0.0465** | ⛔ **FALSIFIED as stated** — and A2-P5 shows why the metric, not the claim, was wrong (§5) |
| P6 | `count` dup = 1.000 exactly | exact | **1.0000 ± 0.0000** | ✅ **HELD** (by construction — and it is the finding) |
| P6a | `count` oor = 0.500 exactly | exact | **0.5000 ± 0.0000** | ✅ **HELD** |
| P7 | refusal rate on B ≤ 0.25 (point 0.10) | 0.25 | **0.014 ± 0.020** | ⭐ **HELD**, 7× better than the point |
| P8 | AURC on balanced B∪C ≤ 0.15 (point 0.06) | 0.15 | **0.1998 ± 0.0277** (random = 0.500) | ⛔ **FALSIFIED** — well below the uninformative 0.5, but above my bar |
| P9 | corr − rand dup AUROC > 0 in 3/3 | 3/3 | ⚠ **+0.0585 ± 0.0942**, per-seed +0.1802 / **−0.0494** / +0.0448 | ⛔ **FALSIFIED on the contaminated metric**; ⭐ **HELD on A7: +0.1157 ± 0.0319, 3/3 positive** |
| P10 | C ids in any fit = 0 | 0 | **0**, 6/6 runs | ⭐ **HELD** |
| A1-P1 | `clu_pair_reach` A7 dup ≥ 0.80 (point 0.90) | 0.80 | **0.9328 ± 0.0132** | ⭐ **HELD** |
| A1-P2 | `clu_pair_reach` dup ≥ `clu` dup (on B) | ≥ 0 | −0.0372 | ⛔ **FALSIFIED on B**; ⭐ **HELD on A7 (+0.0182)** |
| A2-P1 | `count` A7 dup = 0.5000 exactly | exact | **0.5000 ± 0.0000** | ✅ **HELD** |
| A2-P2 | `reach` A7 dup = 0.5000 exactly | exact | **0.5000 ± 0.0000** | ✅ **HELD** |
| A2-P3 | `clu` A7 dup ≥ 0.75 (point 0.85) | 0.75 | **0.9146 ± 0.0086** | ⭐ **HELD**, beat the point |
| A2-P4 | `clu` A7 dup < `clu` B dup | — | 0.9146 vs 0.9138 (**+0.0008**) | ⚠ **essentially a tie, direction not confirmed** |
| A2-P5 | ⭐ `clu` − `distance_reach` on A7 dup > 0, 3/3 | 3/3 | **+0.3516 ± 0.0372, 3/3** | ⭐⭐ **HELD**, 7× the point estimate |
| H0 | counter-hypothesis: the derivation is decoration (`clu` A7 dup ≤ 0.60) | — | 0.9146 | ⛔ **H0 REJECTED** |

**Five predictions failed and they are the useful part.** P5 and P9 failed **on the contaminated metric and
held on the controlled one** — which is exactly the evidence that the contamination is real and material.
A2-P4 was a coin-flip I called wrong and it is a tie. P8's AURC missed my bar while sitting at 0.40× the
uninformative baseline, so the bar was optimistic, not the result. **P4/P4a deserve their own section**, and
my first explanation for them was wrong — see §7.1.

### 7.1 ⭐ Why P4 "failed", and why the answer needed the A-vs-C column to exist
`distance` scores **0.7886 ± 0.0255** on B-vs-C, which looks like a competent refusal criterion and is what
falsified P4. It is not one. Broken out:

| `distance` (the laundering control) | measured, 3 seeds |
|---|---|
| **B-vs-C** | 0.7886 ± 0.0255 |
| **A-vs-C** | **0.5264 ± 0.0209** — chance |
| A-vs-C[`out_of_range`] | **0.4836 ± 0.0275** — ⛔ *below* chance |
| A-vs-C[`duplicate_factor`] | 0.5692 ± 0.0412 |

⭐ **The control cannot detect an impossible request at all.** Its whole B-vs-C signal is that the mean
`dist` of **B** is lower than that of **A** in every seed (0.07693 / 0.07171 / 0.08068 vs 0.08013 / 0.07385 /
0.08542) — B is a *single tight photometric corner*, so its composed prototype sits closer to the store than
an average seen request does. The criterion is ranking "how corner-like is this request", and B wins that by
construction.

⚠ **I first explained this as out-of-range values snapping to grid endpoints, and that explanation is
false.** All 36 oor records do snap to an endpoint (36/36, checked), but endpoint-ness does not raise `dist`
for valid requests: AUROC(≥3 endpoint terms vs 0 endpoint terms, **both valid**) = **0.4401**, per-seed
0.2903 / 0.5306 / 0.4994 — no effect and not even consistent in sign. ⛔ Recording the refuted version
because a conjecture that survives into a report as fact is exactly what the prereg rule exists to stop.

⭐ **This is the clearest argument in the whole spoke for R23's three-AUROC rule.** With only B-vs-C on the
page, `distance` reads as a 0.79 baseline that the certificate beats by 0.17. With A-vs-C beside it, it reads
as **a criterion at chance whose headline number is a property of the B set**. For comparison `clu` scores
0.9537 ± 0.0048 on A-vs-C and 0.9569 ± 0.0099 on B-vs-C — the same on both, which is what an actual admission
criterion looks like. (⚠ Precision note: `clu`'s oor separation is 1.0000 against B and **0.9994 ± 0.0002**
against A — a handful of the 1,024 A requests outrank an out-of-range record.)

---

## 8. Risk–coverage and AURC (⛔ the mixture is part of the number)

Accept in increasing certificate order; risk = invalid share among accepted. **Ties are collapsed** — one
curve point per *distinct* score, the whole tied block admitted together.

(correlation grouping, 3 seeds; lower is better, and every mixture carries its own baseline)

| mixture | ⭐ `clu` | `clu_pair_reach` | `distance_reach` | `reach` | `count` | uninformative |
|---|---|---|---|---|---|---|
| **B ∪ C** (72/72, the primary) | **0.1998 ± 0.0277** | 0.2124 ± 0.0046 | 0.2452 ± 0.0113 | 0.3600 | 0.3750 | **0.5000** |
| A ∪ C (1024/72) | **0.0073 ± 0.0014** | 0.0054 ± 0.0011 | 0.0283 ± 0.0054 | 0.0346 | 0.0350 | **0.0657** |
| balanced A∪B vs C | **0.0908 ± 0.0118** | 0.0889 ± 0.0044 | 0.1490 ± 0.0137 | 0.2138 | 0.2222 | **0.3333** |

⛔ **A metric bug I found in my own code and fixed before any full-scale number existed** (prereg A1.1):
ties were resolved in *input order*, which admitted the valid requests of a tied block first. The `reach` arm
ties **every** A, every B and every `duplicate_factor` request at exactly 0.0. Recomputed both ways on the
**shipped seed-0 scores** (not the smoke run): collapsed-ties AURC **0.3600** over **9 distinct scores**,
in-order-ties AURC **0.1552** — i.e. the bug would have reported the most degenerate arm in the table as the
second best. A regression test pins it. ⚠ No AUROC was affected (average ranks already).

---

## 9. Findings a later spoke needs

**9.1 ⭐ The `dec_recipe` trap cannot reach this stage — structurally, and it is checked.** The amendment
warned that an arm decoded through the superseded `baseline` recipe is void and looks normal in the logs.
The certificate **decodes nothing**: it is a function of the request, the store's modes, and the store's own
addressing over cached features. `assert_no_decoder_in_path()` runs first in every seed and enforces that on
two legs — no public entry point takes a `decoder`/`probe` argument (catches a re-bound function), and the
module **AST** contains no reference to `decod*`, `dec_recipe` or `load_model` (docstrings and comments
excluded, so the warning text in my own docstring does not trip it). It has a positive control in the tests.
⭐ For provenance I record `gen-decoder-lift`'s winner hashes without loading them; the winning-recipe
assertion is **not applicable** to this arm rather than skipped.

**9.2 ⚠ `curv` and `energy` are term-count contaminated; `pair` and `reach` are not.** §3 quantifies it.
This is why reconciliation item 4 exists: the honest primary for the cell is arguably `clu_pair_reach`, which
is **better** on the controlled metric (0.9328 vs 0.9146) and refuses **0.000** of B. ⛔ I am not swapping my
pre-registered primary after the fact; I am reporting both and flagging the decision.

**9.3 ⚠ `gen_data.data_root` is two different things in the shipped tree.** `shapes3d_root(root)` treats it
as the **dataset** directory; `default_gen_data_root()` is its **parent**, and
`exp_gen_store.load_c_records` uses the parent spelling. Passing `--data-root .../gen/shapes3d` therefore
loads the images fine and silently logs "⚠ no C set … SKIPPED". My loader accepts both spellings and
**fails loudly** if neither resolves. Reconciliation item 3.

**9.4 ⛔ An unattended run lost a completed 40-minute arm to a `PermissionError`.** Seed 0 finished both
groupings; the final `cert_manifest.json` write raised `[Errno 1] Operation not permitted` because the
detached process had outlived its parent shell's sandbox grant. The arm's numbers survived only in the log.
Two fixes shipped: `run_seed` now **merges** into an existing manifest when every provenance key agrees (and
**refuses** to merge when one does not), so a narrowed `--groupings` completes a partial seed instead of
overwriting it; and `aggregate_runs` no longer dies when a seed lacks a grouping — it aggregates over the
seeds that have it and **records which ones**, so a mean is never silently over fewer seeds than it claims.
⚠ **Any spoke launching long jobs via `nohup`/`setsid` from a tool shell on this machine should expect this**
— and note `setsid` does not exist on macOS at all.

**9.5 ⚠ The machine rebooted mid-run (2026-09-12 ~00:28) and lost ~10 h of overnight wall clock.** Seeds 1–2
were re-run from scratch afterwards. ⭐ **The re-run is its own reproducibility proof:** every non-A7 number
for seed 0 reproduced the pre-reboot run **exactly** (`0.9660 / 0.9321 / 1.0000 / 0.9495 / 0.9578 / 0.1778`,
both groupings), across a reboot, a different machine load and a different process. The pre-reboot manifest
is kept at `preA7_seed_0/` for audit.

**9.6 ⛔⭐ §7.23 HAS A MIRROR IMAGE, and it bit my own full-suite run.** §7.23 warns that six modules enable
`jax_enable_x64` at module import and *sort before yours*, so your conv module gets a float64 world. The
converse is not written down and is worse: **module import happens during collection — before any test runs,
and regardless of sort order** — so those six have already set the flag globally by the time *any* test
executes. My `test_certificate_is_stable_under_x64` toggled x64 on and restored it in a `finally` to a
**hard-coded `False`**, which turned x64 **off underneath all six**. Result: **9 failures in
`test_goldstone.py`**, invisible in isolation, and reproducible with just two files:

```
pytest tests/test_gen_certificate.py tests/test_goldstone.py -q -p no:randomly   ->  9 failed, 45 passed
```

⭐ **Fix: save and restore the OBSERVED value, never a literal** — `prior = bool(jax.config.jax_enable_x64)`
… `finally: jax.config.update("jax_enable_x64", prior)` — and assert the restore. Verified: the same two
files now give **54 passed**, and my module plus **all six** x64 modules together give **130 passed**.
⚠ Note the consequence for the test's own strength: inside the full suite the flag is *already* True, so the
toggle is a no-op there and the test only genuinely toggles when run in isolation. It is still a real check
in both regimes, but it is not the same check.
⛔ **Any spoke writing an x64 test must restore the prior value, not `False`.** A float32-pinning fixture is
not the fix (§7.23) and neither is a literal restore.

Apart from that, §7.23's original direction did **not** bite: the modes are explicitly `float32` and every
entry point casts its input to the store's own parameter dtype (the `gen_bridge` exemplar). All 38 new tests
pass in the **full suite**, not only in isolation.

**9.7 §7.12** (`uv run chlu` broken by the hidden `.pth`) is why every command here is
`PYTHONPATH=<worktree> /Users/user/Desktop/CHLU/.venv/bin/python -m …`.

---

## 10. ⛔ CSF3 — NOT RUN, with the submit line beside it

**Standing constraint 2026-09-10: CSF3 needs 2FA and is unreachable from an agent session.** Everything above
ran **locally on CPU**. ⛔ No job was submitted, no job id exists, and no cluster acceptance item is reported
as passing.

| cluster acceptance item | status |
|---|---|
| certificate stage run on CSF3, 3 seeds | ⛔ **NOT RUN** |
| `$SCRATCH/CHLU-gen/...` artefacts pulled back | ⛔ **NOT RUN** |

⚠ **The launcher has no `certificate` stage** (`job_gpu_gen.sh` line 95 hard-rejects anything outside
`encoder|decoder|store|cell`). Until `gen-csf3-launcher` adds one, the working form rides on `STAGE=cell`
with an `EXTRA_ARGS` override — **verified** that the payload's argparse takes the *later* `--stage`:

```
export GEN_GIT_COMMIT=$(git rev-parse HEAD)      # 2fa0631 on agent/experiment-engineer/gen-certificate
cd ~/scratch/CHLU && mkdir -p logs

# plan first (prints the -a range and every realised path)
DATASET=shapes3d PRETEXT=recon STAGE=cell ARMS=clu SEEDS="0 1 2" PLAN=1 \
    bash scripts/csf3/job_gpu_gen.sh

sbatch --export="ALL,DATASET=shapes3d,PRETEXT=recon,STAGE=cell,ARMS=clu,SEEDS=0 1 2,\
PAYLOAD_MODULE=chlu.experiments.exp_gen_certificate,\
EXTRA_ARGS=--stage certificate,GEN_GIT_COMMIT=$GEN_GIT_COMMIT" \
    -a 0-2%2 -t 04:00:00 scripts/csf3/job_gpu_gen.sh
```

**Realised output paths it will produce** (one per array task):
```
$SCRATCH/CHLU-gen/workspaces/shapes3d_recon/cell/clu_s0/artifacts/seed_0/{cert_manifest.json,scores_correlation.json,scores_random.json,store_correlation.npz,store_random.npz}
$SCRATCH/CHLU-gen/workspaces/shapes3d_recon/cell/clu_s1/artifacts/seed_1/...
$SCRATCH/CHLU-gen/workspaces/shapes3d_recon/cell/clu_s2/artifacts/seed_2/...
   + logs/stage.log and run_manifest.json per task
```
⚠ The workspace path says `cell/` because the launcher derives it from `STAGE`; the artefacts are
certificate artefacts. ⛔ That mislabelling is the reason reconciliation item 2 needs an owner.

⭐ **A validated dry run plus a correct submit line is the complete deliverable here** (task instruction), and
the local run is not a dry run: it is the full 3-seed, 2-grouping, 9-arm result.

---

## 11. Suite

| | count |
|---|---|
| **before** — detached worktree `../CHLU-gc-base @ 9a4457d` (my named base), main venv | **2112 passed, 0 failed** (2:27:24) |
| **after** — `agent/experiment-engineer/gen-certificate @ 01d4d8d` (pre-rebase), main venv | **2150 passed, 0 failed** (0:46:10) |
| delta | **+38 = exactly the 38 new tests in `tests/test_gen_certificate.py`** |

⭐ The *before* number **2112** reproduces `gen-bridge-wide`'s reported post-merge figure exactly, so the
baseline is the state the Hub recorded and nothing drifted between that merge and mine. Collection on the
branch counts **2150** tests = 2112 + my 38.

Both `-q -p no:randomly --no-cov -p no:cacheprovider`, matching `gen-bridge-wide`'s flags so the numbers are
comparable with the one it recorded. ⛔ No `uv sync` in any worktree (w6), so the resolver hazard does not
apply; JAX **0.9.0** in both.

⛔ **An earlier *after* run is retained as `suite_after_x64bug.log` and is NOT the result.** It was the run
that exposed §9.6: it was stopped once the cause was identified, its tree was edited underneath it, and the
number in it describes code that no longer exists. Reporting it as the result would have been the easy and
wrong thing to do.

⚠ **`main` MOVED during this session** — `9a4457d` → **`a0034d4`** ("[hub] merge gen-dream-sampler"), which
appended its own `GenDreamConfig`/`GenR0WideConfig` blocks at the same `chlu/config.py` anchor as mine and
added 220 lines to `chlu/core/gen_store.py`. The before/after pair above is therefore a self-consistent pair
at **my named base `9a4457d`**, not at current `main`. See §11.1 for what I did about it.

`tests/test_gen_certificate.py` alone: **38 passed**. My module + all six §7.23 x64 modules: **130 passed**.

### 11.1 Post-rebase verification
⭐ **The branch was rebased onto `main @ a0034d4` and the suite was re-run on both sides, so the Hub has a
pair at the tree it will actually merge — not only at my named base.**

| | count |
|---|---|
| **before (current main)** — `../CHLU-gc-base`, detached at **`a0034d4`** | **2166 passed, 0 failed** (0:55:15) |
| **after (rebased)** — `agent/experiment-engineer/gen-certificate @ 5bd8d2b` | **2204 passed, 0 failed** (0:55:13) |
| delta | **+38 = exactly the 38 new tests** |

⭐ `main` was at `a0034d4` **before and after both runs** (checked at each end — a concurrent Hub merge
would have invalidated the comparison). Both runs finished within 2 s of each other on an otherwise quiet
machine (load ≈ 7).

**The conflict, and how it was resolved.** As the task amendment predicted, `chlu/config.py` and
`chlu/cli/experiment_cmd.py` conflicted at the shared end-of-block anchor with `gen-dream-sampler`'s
`GenDreamConfig` and `gen-r0-wide-refit`'s `GenR0WideConfig`.

⚠ **A mechanical "keep both sides of every hunk" resolution is WRONG here and I nearly shipped it.** Git's
hunks split *inside* expressions: in `load_config` the two blocks came to share one closing `),`, and in the
CLI the two appended **functions** (`cmd_exp_gen_dream` and `cmd_exp_gen_certificate`) were interleaved into
one broken hybrid that would have silently destroyed a peer's command. `ruff` caught it as a syntax error; a
reviewer might not have. ⭐ **The correct resolution for "two spokes appended to the same anchor" is not a
line-level merge at all** — it is: take the peer's file **wholesale**, then re-apply your own additions at
their anchors. That is what shipped (`git checkout main -- <file>` + a re-apply script), and it is verifiable:

| check | result |
|---|---|
| `GenR0WideConfig` / `GenDreamConfig` / `GenCertificateConfig` all present in `chlu/config.py` | ✅ 4 / 4 / 5 references |
| `cmd_exp_gen_store` / `cmd_exp_gen_dream` / `cmd_exp_gen_certificate` all present in the CLI | ✅ 3 of 3 |
| ⭐ **lines deleted vs `main` across the whole branch** | ✅ **0** (`git diff main...HEAD --numstat`) |
| `ruff check` on both shared files | ✅ All checks passed |
| targeted cross-spoke run: my tests + `gen_dream` + `gen_r0_wide` + `gen_store` + `gen_bridge` + `gen_bridge_wide` + `config` + all six §7.23 x64 modules | ✅ **290 passed** (5:03) |

⛔ **Unresolved conflicts: none.** ⛔ No peer's block was deleted, renamed, reordered or tidied.

---

## Git footprint
- **Branch:** `agent/experiment-engineer/gen-certificate` (worktree `/Users/user/Desktop/CHLU-gen-cert`),
  base `main @ 9a4457d`.
- **Commits**, post-rebase (`main..HEAD`, ⭐ **verified from the MAIN repo**: the ref resolves to `5bd8d2b`
  and all 8 commits are listed there; the worktree is left in place for review, not removed):
  - `0ae9975` derive a refusal criterion from the store's own geometry
  - `dd4a859` exp-gen-certificate: the harness, the controls and the CLI hook
  - `00fbc59` tests: the audit's positive control and the analytic Hessian
  - `eed1fa5` add the term-count control the C set's own construction demands
  - `7ff5a00` plot the certificate and both controls on the same axes
  - `24492e2` pin why the pair barrier is the term-count-clean statistic
  - `ac22220` resume a seed instead of restarting it, and aggregate what exists
  - `5bd8d2b` stop this module turning x64 OFF under six shipped ones
- **Files touched:** `chlu/core/gen_certificate.py` (new), `chlu/eval/gen_cert_eval.py` (new),
  `chlu/experiments/exp_gen_certificate.py` (new), `tests/test_gen_certificate.py` (new),
  `chlu/config.py` (**append-only**), `chlu/cli/experiment_cmd.py` (**append-only**).
  ⛔ **Not one existing line modified** in either shared file. ⛔ Nothing under `scripts/csf3/`,
  `chlu/core/gen_store.py`, `chlu/core/gen_bridge.py` or `chlu/core/soft_certificate.py` was touched.
- ⚠ **Scope note the task did not anticipate:** the task's "files you own" list names the core, eval and test
  modules but **no experiment module**, while also requiring a CLI subparser and a `PAYLOAD_MODULE=
  chlu.experiments.exp_gen_<stage>` submit line — both of which need one. I created
  `chlu/experiments/exp_gen_certificate.py` (new file, no one else's lines) as the minimal reading that
  satisfies the CLI contract and the submit-line requirement. Flagging rather than assuming.
- **`chlu/config.py` placement:** the task said "append at the **end of the file**". ⛔ Taken literally that
  breaks — `CHLUConfig`'s `field(default_factory=GenCertificateConfig)` is evaluated at class-creation time,
  so a dataclass defined after it raises `NameError`. I appended `GenCertificateConfig` at the end of the
  **gen block**, immediately after `GenStoreConfig` and before `ProjectConfig` — the same position
  `GenStoreConfig` itself occupies — and wired the three house sites (`CHLUConfig` field, `load_config`,
  `save_config`). ⭐ Still a pure addition: **zero deletions**.
- **Rebased onto local `main` @ `a0034d4`** (not `origin/main`, §7.21/protocol §3.5). ⚠ **The predicted
  conflict DID materialise** with `gen-dream-sampler` and `gen-r0-wide-refit` at the shared `chlu/config.py`
  and `chlu/cli/experiment_cmd.py` anchors — resolved by keeping every block, **0 lines deleted**, with the
  method and its verification in §11.1. ⛔ Unresolved conflicts: none.
- **Artifacts** (all under `.claude/outputs/gen-certificate/`): `PREREG.md` (+ amendments A1, A2) ·
  `runs/shapes3d_recon/{certificate_results.json, certificate_summary.png, seed_{0,1,2}/{cert_manifest.json,
  scores_{correlation,random}.json, store_{correlation,random}.npz}}` · `tables.py` + `tables.md` (the full
  9-arm tables for **both** groupings) · `seed{0,1,2}.log`, `seed0_random.log`, `aggregate.log` ·
  `suite_before.log` (9a4457d), `suite_after.log` (pre-rebase), `suite_before_a0034d4.log`,
  `suite_after_rebased.log`, `suite_postrebase.log` (the targeted cross-spoke run),
  `suite_after_x64bug.log` (⛔ the stopped run that exposed §9.6 — kept as evidence, **not** a result) ·
  `preA7_seed_0/` (the pre-reboot seed-0 manifest, kept for the reproducibility check in §9.5).

## Open questions / follow-ups / risks
1. ⛔ **Reconciliation items 1–4 need a Hub owner before `gen-cell-run` scores anything against C.** Item 1
   is the load-bearing one: it changes how *every* arm's dup number must be computed.
2. ⚠ **The byte-matched kNN arm of kill rule (iv) is not measured here.** My `distance_reach` is an in-house
   proxy. I predict it collapses on `A7` the same way; ⛔ that is a prediction.
3. ⚠ **`out_of_range` is a grid lookup.** If the cell wants a refusal claim that is not partly a lookup, the
   C set needs a third kind — *combinatorially* impossible requests whose every term is on-grid and whose
   term count is six. ⛔ Not mine to build (R2: the C set is single-sourced, fixed before grouping ran).
4. ⚠ `energy`'s B-vs-C carries a **±0.18 seed spread** (0.8005 ± 0.1785) — by far the least stable component.
   Consistent with it reading a term-count-sensitive quantity. Not investigated further.
5. ⚠ The shipped `scores_*.json` for seeds 0–2 hold A/B/C rows only; the patch that also dumps A7's
   per-request rows landed after those runs. A7's **aggregates** are in every manifest. Re-running any seed
   regenerates the full dump.
6. ⚠ I never tested the **wide** condition. The code is parameterised for it (`--condition`, `artifacts_tag`
   for both the bridge and the store-plan directory) and fails loudly if the wide store plan is absent, but
   ⛔ "it is parameterised" is not "it has run".

## Proposed handover updates (for the Hub)
- **§3 config:** new `GenCertificateConfig` (dataclass + `CHLUConfig` field + `load_config` + `save_config`,
  append-only) and the new `chlu exp-gen-certificate` subcommand. Notable defaults: `groupings
  ["correlation","random"]` (⛔ the control is never optional), `accept_quantile 0.95` (fitted on A only),
  `reach_radius_frac 0.5`, `term_count_control True`, `save_store True`.
- **§7 NEW known issue — the C set's `duplicate_factor` records are 7 terms against A/B's 6.** A pure
  term-count statistic scores **1.0000** on dup-vs-B and **0.5000** on oor-vs-B. Every arm inherits it.
  Score dup against a term-count control (`A7`, shipped in `chlu/core/gen_certificate.redundant_requests`).
- **§7 NEW known issue — `gen_data.data_root` is the dataset dir for the loaders and its parent for
  `default_gen_data_root`.** `exp_gen_store.load_c_records` silently skips the C set under the other spelling.
- **§7 NEW known issue — §7.23 has a MIRROR IMAGE.** The six x64 modules set the flag at **module import**,
  i.e. during collection, before any test runs and regardless of sort order. A test that restores
  `jax_enable_x64` to a literal `False` therefore turns x64 **off underneath all six** (measured: 9 failures
  in `test_goldstone.py`, reproducible with two files, invisible in isolation). ⭐ Save and restore the
  **observed** value, and assert the restore.
- **§7 NEW discipline note — a detached job can outlive its permission to write.** A completed arm was lost
  to `PermissionError` on the final manifest write. Write incrementally *and* make the run resumable; ⛔
  `setsid` does not exist on macOS.
- **§7 NEW discipline note — "two spokes appended at the same anchor" is NOT a line-level merge.** Git's
  conflict hunks split inside expressions and inside function bodies; resolving hunk-by-hunk interleaved two
  peers' CLI *functions* into one broken hybrid. Take the peer's file wholesale, then re-apply your own
  additions at their anchors, and check `git diff main...HEAD --numstat` sums to **0 deletions**.
- **§7 NEW discipline note — collapse ties before integrating a risk-coverage curve.** Resolving ties in
  input order flatters any arm with a degenerate score (measured: AURC 0.1552 → 0.3600 for `reach`).
- **Rulings ledger:** the *admission* dial has a first measurement — certificate **B-vs-C 0.9569 ± 0.0099**
  with **zero invalid-example supervision, asserted**; the term-count-controlled dup margin over naive
  geometry is **+0.3516 ± 0.0372, 3/3 seeds**; the store is bit-identical to R0's on all six arms.
