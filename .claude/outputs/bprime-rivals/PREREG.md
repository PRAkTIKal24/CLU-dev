# PREREG — `bprime-rivals` (C2W4, experiment-engineer)

**Filed 2026-07-31, BEFORE any rival arm was run.** Protocol §5 pre-registration rule. Base `main @ d4f56c8`,
branch `agent/experiment-engineer/bprime-rivals`, worktree `../CHLU-rivals`.
Governing documents: `.claude/tasks/bprime-rivals.md`, `PREREG-Bprime.md` §2/§4/§5/§6/§7, charter ADDENDUM 3
§A13/§A14.2/§A15, `bprime-fb4-gate.md` §A3, `bprime-theory.md` T1, `rival-recon.md` §F1–F9.

⭐ **Nothing measured had been run when this file was written.** The only numbers used to derive the
predictions below are the **banked** C2W1/C2W3 values (`PREREG-Bprime.md` §7), which I am forbidden to
re-measure and am reusing verbatim.

---

## 0. DIAL DECLARATION (echoed)
- **Dial / pillar:** none in the CHLU sense — cross-family **AUDIT**, claim-architecture **TIER i**.
- **Laundering control:** the launder *is* the deliverable — per family × per arm: matched-byte table
  launder · **+0 B** substitute (signed margin) · two-sided byte ledger · same-keys null · blank-store
  control · identical φ and φ-bytes on every arm (enforced in code, raises).
- **Falsifies:** FB2 (≥2 of 5 families have no definable byte-matched table) · FB3 (every rival shows a
  large positive dividend and only CLU does not) · FB1 (an established paper already runs this control).
- **Does NOT falsify:** a rival beating CLU · a rival losing to its own table on a metric-native probe
  (that is the metric-native-ceiling theorem, predicted by P2) · a family being hard · one arm failing
  to train (reported with evidence and budget).

---

## 1. What I am building and running (declared before the fact)

### 1.1 The arms (5 rival state types + the banked CLU column)
| arm | equations implemented (verified from the paper this session) | state convention |
|---|---|---|
| `ttt_linear` | Sun et al. arXiv:2407.04620 **Eq. 1, 2, 4, 5** + §2.4 mini-batch (`b = 16`) + §2.7 (`f_res(x) = x + LN(f(x))`, learnable `W₀`, learnable `η`) | `d_head²` + `b`-token buffer |
| `ttt_mlp` | same + §2.7 `f_MLP` (2 layers, hidden `4×`, GELU) | `8·d_head²` + buffer |
| `deltanet` | Yang et al. 2024, as restated in GDN-2 **Eq. 5** | `n_head·d_k·d_v` |
| `gdn` (ablation, GDN-1) | Yang et al. 2025, as restated in GDN-2 **Eq. 6** | `n_head·d_k·d_v` |
| `gdn2` (**reference delta-rule arm**, §A14.2) | Gated DeltaNet-2 arXiv:2605.22791 **Eq. 10** (boxed), with **Eq. 8** (`e_t = b_t⊙k_t`, `z_t = w_t⊙v_t`), **Eq. 9**, **Eq. 11** (`b_t = σ(W_b x_t)`, `w_t = σ(W_w x_t)`), **Eq. 12** (`g_t = −exp(a)⊙softplus(W_f x_t + δ)`, `α_t = exp(g_t)`) | `H·d_k·d_v` — **Eq. 90 of the paper states the per-layer recurrent state is `H d_k d_v` floats**, i.e. the −2 revision **preserves** GDN's accounting. Verified, not inherited. |
| CLU | **banked, not re-measured** (`PREREG-Bprime.md` §7) | — |

⛔ **NOT built, declared NOT-RUN with reason:** Titans (D5 Hub ruling — no official code, chunk size never
numeric, no seeds ⇒ our-reconstruction-audited-against-our-reconstruction), Sparse Delta Memory (D5 — needs
Torch ≥2.8 / Triton ≥3.4 / SM 80+, cannot run on this machine), Mamba-2, GRU, sliding-window attention
(not in §A14.2's family set).

### 1.2 The families
- **`aggregate@base`** — the **sole dividend family** (§A14.2; `S = 0.5068`, the only survivor of FB4).
- **`overload@load1x_shipped`** — **BYTE-FRONTIER COLUMN ONLY**, labelled at every appearance, with the
  declared secondary reading `S_excl = 0.6500`. Never a dividend family, never a headline.
- ⛔ `recency`, `manifold` — **DO NOT RUN** (saturated, protocol-invalid).

### 1.3 The protocol, per (family × rival × seed), seeds **0, 1, 2**
1. `full` — the rival's learned test-time dynamics.
2. `launder` — its **own byte-matched table** of `(θ_K x_t, θ_V x_t)` pairs, arg-min read (P5's construction).
3. `+0 B` substitutes — 2-NN mean and 2-NN IDW over that same table (the `aggregate` family's declared
   +0 B reader set, `chlu/eval/dividend.py`), **signed margin** `full − best(+0B)` reported.
4. `same_keys_null` — same table keys, payload column permuted.
5. `blank` — the identical rival with **nothing written** (state = `W₀`, i.e. the learned init only).
6. two-sided byte ledger (F1 parameters / F2 state), **learned-initial-state rule applied to CLU too**.
7. `admissible/total` cell coverage, first-class.
8. metric-native verdict at equation level, then **measured** against its own table.

### 1.4 The byte conventions I commit to (before measuring)
- **Learned-initial-state rule (`PREREG-Bprime.md` §4.1):** `W₀` (TTT), `S₀` (delta-rule) and CLU's `V_θ`
  **initialisation** are **PARAMETERS (F1)**; only the **per-sequence / per-stream deviation** is **STATE
  (F2)**. Applied to CLU by *measuring* which `V_θ` leaves the masked write actually moved.
- **Iso-state sizing rule:** `d_head` per rival = the largest head width whose **state floats ≤ the CLU's
  banked `aggregate@base` full-byte figure of 5456 B = 1364 float32**. Derived values (arithmetic, filed
  before running): `ttt_linear d=29` (`29²+16·29 = 1305`), `ttt_mlp d=12` (`8·12²+16·12 = 1344`),
  `deltanet/gdn/gdn2 d=36` (`36² = 1296`).
- **Matched-byte table:** `n_rows = floor(state_floats / (d_k + d_v))`, rows drawn from the write stream in
  order (the rival's own `(θ_K x, θ_V x)` pairs). Same `θ_K, θ_V, θ_Q, θ_O` on **every** arm (identical φ).
- **Byte law:** the **corrected** `ratio = [A(D+2) + d]/(d+m)` (floor **2.20×** gauss, **2.40×** at
  `n_spec = 1`). ⛔ Never "verified to 1e-9 in all 28 cells" (it is 24/28).
- **F2a guard (binding).** The rival's outer parameters are fitted on **auxiliary streams built from
  different seeds** (different sites, different payloads) and never on the cell's own stream, so no
  item-specific content can hide in pool (1). Declared because it is the guard that makes F2 meaningful.
- **Baseline tuning (F3-lite).** lr ∈ {1e-3, 3.16e-3, 1e-2}, Adam, 400 outer steps, best-of-grid on the
  fit split. ⚠ This is a **reduced** grid vs `rival-recon` F3's 6×2 — declared as a budget choice, not
  presented as F3 compliance.

---

## 2. ⭐ PRE-REGISTERED PREDICTIONS (committed; derivations included)

### 2.1 The task's three inherited predictions
- **P2** (*of the four (k,v)-shaped-state families, ≥3 lose to their own byte-matched table on a
  metric-native probe; 0 of 4 lose on real-data LM bpc*). ⚠ **I test the FIRST HALF ONLY** — the real-data
  half belongs to `cluformer-pilot` and I say so. ⚠ I adjudicate **3 of the 4 by measurement**
  (DeltaNet, GDN, GDN-2); **Mamba-2 and SDM are reasoned from their equations only** and are labelled as
  such — never blurred.
  ⭐ **Scoring rule, declared now:** "its own byte-matched table" = the family's own **strongest +0 B
  reader** of that table (the protocol's own definition — the +0 B reader set is part of the launder set,
  `bprime-fb4-gate` §A3 `zero_byte_candidates`). **Prediction: ≥2 of the 3 measured LOSE ⇒ P2 SUPPORTED
  on its measured part.** The arg-min-only reading is reported beside it and is predicted to go the
  **other** way (see 2.2), and I pre-commit to printing both.
- **P3** (*the two function-valued memories show the largest positive dividend*). ⛔ **Pre-declared
  NOT-RUN**: no Titans arm (D5), so the pair cannot be formed. TTT-MLP alone is run and reported as a
  single-arm datum, never as P3. **NOT-RUN is not refuted.**
- **P5** (*the launder transfers to all five rival state types; predicted failures 0 of 5*).
  **Prediction: 0 of 5 failures.** The 5 state types I actually run are TTT-Linear, TTT-MLP, DeltaNet,
  GDN, GDN-2 — i.e. `d_head²`, `8d_head²`, and three `n_head·d_k·d_v` variants.

### 2.2 ⭐ My own per-cell numeric predictions on `aggregate@base` (metric `neg_mae`, 3-seed means)
**Derivation anchors (all banked, none re-measured):** CLU `full` **−0.5261**, CLU arg-min `launder`
**−0.4472**, CLU `blank` **−0.4221**, best +0 B substitute (2-NN IDW) **−0.2081**, `S = 0.5068`,
byte ratio **54.56×** (5456 B / 100 B), `K = 5` live rows.
From the launder value: an arg-min reader returns a *stored* payload, so its MAE is
`E|λ|·E|Δa| ≈ 0.5·E|Δa| = 0.4472` ⇒ **`E|Δa| ≈ 0.894`** for a neighbouring pair. Every prediction below
is built from that one estimated constant.

| # | quantity | predicted value | band | derivation |
|---|---|---|---|---|
| R1 | rival `launder` (arg-min over the **projected** table), mean over 5 arms | **−0.42** | [−0.55, −0.25] | same structural bound as the CLU's launder (`0.5·E|Δa| = 0.447`); a *learned* `θ_K` can only sharpen the arg-min, never make it interpolate, so it may improve modestly |
| R2 | rival `full`, linear-read arms (`ttt_linear`, `deltanet`, `gdn`, `gdn2`) | **−0.15** | [−0.30, −0.05] | the read `o = Σ_s v_s (k_s·q)` **is** an aggregator: it can express a convex combination. The residual error is the mismatch between the learned kernel weights and `λ`; the +0 B IDW reader, which approximates the same thing with a hard 2-NN, achieves **0.2081**, so a *trained* soft kernel should be at least as good |
| R3 | rival `full`, `ttt_mlp` | **−0.13** | [−0.30, −0.03] | nonlinear readout ⇒ weakly metric-native; the only arm that can in principle beat a kernel smoother |
| R4 | **dividend vs own arg-min table launder**, per arm | **+0.27** | [+0.05, +0.45] | R2 − R1 |
| R5 | **signed +0 B margin** = `full − best(+0B on its own table)`, per arm | **−0.02** | [−0.15, +0.08] | the +0 B 2-NN IDW reader of the *same* table is the same estimator class; ⭐ **≥3 of 5 arms predicted ≤ 0** |
| R6 | `blank` (rival with `W₀` only) | **−0.75** | [−1.2, −0.45] | with no state the read is `θ_O·f(θ_Q x; W₀)`, uncorrelated with the payload ⇒ MAE ≈ E\|target\| plus bias; strictly worse than the CLU's blank (−0.4221), which still sees the address geometry |
| R7 | `same_keys_null` | **−0.45** | [−0.60, −0.30] | permuting the payload column of the table leaves an arg-min over keys returning a *random* stored payload ⇒ MAE ≈ E\|a − a'\| ≈ `E|Δa|`·(a constant near 0.5–0.7) |
| R8 | rival **state/own-table byte ratio** | **1.00** | [1.00, 1.06] | matched by construction, up to one row's floor rounding |
| R9 | **`table_is_lossless`** (n_rows ≥ n_tokens) on `aggregate` | **True for 5 of 5** | — | 22 / 56 / 18 rows vs ≈10 stream tokens |
| R10 | CLU two-sided ledger, learned-initial-state rule applied: **state bytes < param bytes** | **True** | — | the masked write moves only the *written* slots' atom groups: 5 live × 32 atoms × 7 floats = 1120 + codebook 20 = **1140 floats state** vs **1344 floats params** ⇒ predicted `state/param = 0.848` |

### 2.3 ⭐ The falsifier predictions (this is what the prereg is for)
- **FB2 — predicted DOES NOT FIRE.** A byte-matched table is definable **without an arbitrary modelling
  choice** for all 5 arms I run (each has an explicit float state and an explicit `(θ_K x, θ_V x)` stream).
  I will state which of the 5 §A14.2/§2 families I adjudicated *by measurement* and which by equation.
- **FB3 — ⚠ PREDICTED TO PARTIALLY FIRE on the arg-min reading, and NOT to fire on the protocol's own
  reading.** R4 says every rival beats its arg-min table (+0.27) while the CLU does not (−0.0789). ⭐ **I
  pre-commit, in writing, to reporting that plainly if it happens** — including the sentence *"test-time
  dynamics pays for the rival families and does not pay for ours, against the arg-min control."* R5 says
  that dividend is erased by the family's own **+0 B** reader for ≥3 of 5 arms — which is the audit's
  actual finding and is *not* a re-frame, because R5 is registered **here, before measurement**.
- **FB1 — predicted DOES NOT FIRE** (inherited from `bprime-fb1-recon`: 0 HIT / 2 PARTIAL out-of-family).
  I will report the same hour if implementing one of these papers reveals an in-family matched-byte
  non-parametric control I can see from inside the code and a sweep could not.
- **FB5 — predicted DOES NOT FIRE** (arXiv:2501.12352 is purely theoretical, no experiments/baselines).

### 2.4 Byte-frontier column (`overload@load1x_shipped`) — **labelled, never a dividend**
- F1: the rivals' accuracy-vs-state-bytes curve is **monotone increasing and saturating**, with the knee
  where `n_rows` first exceeds the stream length (19 tokens) — predicted at `d_head ≈ 8–10` for
  `ttt_linear` (`d=8` ⇒ 12 rows < 19 = budget-limited; `d=16` ⇒ 33 rows ≥ 19 = lossless).
- F2: the CLU's banked curve (`decode 0.972 → 0.097` as the ratio falls `478× → 2.28×`) is **reused, not
  re-measured** (§7).
- F3: ⛔ every appearance carries the label *"byte-frontier column; the declared secondary reading is
  `S_excl = 0.6500`"*.

### 2.5 What would make me report the same day
FB2 firing · FB3 firing in the strong form (R5 positive for ≥4 of 5 arms, i.e. rivals beat even their own
+0 B readers while CLU does not) · FB1 firing from inside the implementation · finding myself building a
replacement family or sizing a language-model run (§6.2 hard stop).

---

## 3. Declared limitations, written before the results exist
1. ⭐ **One surviving dividend family.** Two rival families audited against **one** synthetic family is a
   **thin cross-family audit**, and it goes in the paper's Limitations verbatim (§A14.2). Stated in my
   report's first screen.
2. **Minimal faithful reimplementations, not vendored stacks** — faithful to the update equation and the
   state size, minimal in everything else (no conv branch, no SWA hybrid, single head, no multi-layer
   backbone). Captioned in every table.
3. **No deletion verb exists for any rival.** The gym stream's delete row is skipped for the rival arms,
   which is a protocol asymmetry **in the rivals' favour** (they are never asked to forget). Declared.
4. **Reduced lr grid** (3 points, not F3's 6×2).
5. The rivals are trained on this gym's query law, at this weight class, with `d_in = 5`. Nothing here
   transfers to an LM claim; the real-data leg is `cluformer-pilot`'s and is **tier iii**.
