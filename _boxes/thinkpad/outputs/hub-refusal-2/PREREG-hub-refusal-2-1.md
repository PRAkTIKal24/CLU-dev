# PREREG-1 — `hub/refusal-2` · COERCIVE-COMPONENT SCREENING
**Filed:** `2026-09-17T10:58:00Z` · **Branch:** `hub/refusal-2` @ `945a3e9` (`base/iter2`)
**Machine:** FORGIS_14, Windows AMD64, CPU, 4 threads · **START_UTC:** `2026-09-17T10:49:57Z`
⛔ **Filed BEFORE any statistic was computed under this criterion**, and before any invalid row
of any kind — synthetic or benchmark — was scored by it. Amendments are APPENDED, dated, ⛔ never edited.

---

## 0. ⛔⛔ MANDATORY DISCLOSURE — verbatim, as my task addendum requires

> *"This criterion was chosen on a structural argument formed after reading `hub/refusal`'s analysis of
> invalid examples on a fit-derived synthetic analogue. No benchmark invalid row was seen. The
> zero-invalid-supervision claim is therefore 'structural, not blind', and the paper says so."*

⭐ I restate in my own words what that concedes, because a formality that is not understood is worthless.
My predecessor **looked at invalid examples** (synthetic ones) and published what it saw: the argmax read
sits exactly on an atom, `out_of_range` has the sign backwards, the store is decorative for `S_max`.
**I read that.** So my choice of statistic is downstream of information derived from invalid data, even
though **no benchmark invalid row has ever been seen by me or by my predecessor**, and even though the
argument that selected this criterion is structural (a geometric property of the read point) rather than
empirical (a number that was better on invalids).
⛔ **The honest label is "structural, not blind", and every number I report carries it.** ⛔ It is NOT
the same claim as "zero invalid supervision", and I will not let the two be conflated in my report,
in the ledger, or in any line a paper could lift.

---

## 1. DIAL DECLARATION (AGENT_PROTOCOL §7)

- **Dial:** **admission** — refusing a request the store cannot honour.
- **Laundering control:** the **physics-free readout margin** (§4.2) — a plain address-book confidence
  margin with no store geometry and no energy in it. If the landscape energy does not beat it, the
  physics is decorative and the claim is laundering.
- **Falsifies the claim:** `store_removed` failing to collapse `S`'s AUROC to ~0.5 (§4.1); or the
  readout margin matching `S` within the *Certified Access* §4.1 residual of **+0.024 AUROC**; or
  `min(AUROC_oor, AUROC_dup) < 0.75` at the single val-only threshold; or `accept_A7 < accept_B − 0.05`.
- **Does NOT falsify the claim:** losing to `knn_proto` on `out_of_range` **alone** (kNN scores 0.97
  there). ⛔ My addendum states this is *"not required and not expected"*. kNN's R8 rule moves its probe
  off the feature manifold, so unbounded `out_of_range` detection is a property of metric-native
  extrapolation — the metric-native-ceiling theorem, not news. **The claim is the PAIR.**

---

## 2. ⭐ THE CRITERION — ONE, landscape-derived, and the exact object it is computed on

### 2.1 The statistic
```
S  =  E_store( x_read )
```
- `E_store` is the **shipped** `chlu.core.gen_landscape.store_energy`, called, ⛔ never re-implemented:
  `E_store(x) = Σ_k −σ_k² · logsumexp_i( −‖x_k − a_ki‖² / 2σ_k² )` — the store's kernel free energy over
  its own atoms, evaluated at the reference bandwidth `sigma_scale = sigma_scale_end = 1.0`.
  ⭐ It is a **soft squared distance from the state to the nearest stored atom**, per well, summed. It is
  therefore a function of **what the atoms contain**, which is what makes `store_removed` a real control.
- `x_read` is the **LINEAR read point**: `x_read[:, cols_k] = P[:, k, :] @ atoms[k]`, the
  posterior-weighted combination of well `k`'s atoms.

### 2.2 ⭐ THE LINEAR READ IS SHIPPED, NOT INVENTED — and this is the whole hinge
`gen_landscape.launch_state` already implements exactly this as `launch_mode="posterior_mean"`
(`gen_landscape.py:1028`, one of the four `LAUNCH_MODES` at `gen_landscape.py:123`), and it is reachable
through the **shipped config knob** `GenDynamicalReadConfig.launch_mode` (`config.py:6196`, default
`"clamp_atom"`). ⛔ **No manifest-pinned file is edited to obtain it.** I flip a declared config value.

⭐ **Why this is not the dead criterion, stated as geometry.** Under the shipped **argmax** read
(`clamp_atom`) `x_read` **is** an atom, so `‖x_k − a_ki‖ = 0` for some `i` in every well and `E_store`
is pinned at its floor — my predecessor measured that distance as **0.0000**. `E_store` at the argmax read
carries **zero** information about validity, by construction. Under the **linear** read the state is a
*combination*, so it can and does leave every basin: an invalid request whose posterior is spread over
**mutually inconsistent** atoms lands in the void *between* them, where `E_store` is large.
⭐ **The read change is what makes refusal possible at all.** ⛔ This is a different object from the dead
`S_max = max_t [E_t − E_store]` — that was a *request-energy residual* at the argmax/MALA point, and it
died because the address book is a simplex over stored atoms and a simplex has no outside.

### 2.3 ⭐ `x_read` IS THE LINEAR READ POINT ITSELF — 0 settling steps. Declared now, and why.
The criterion is evaluated at the read point the linear addressing produces, i.e. **`steps = 0`**.
⛔ It is **not** evaluated after MALA descent. The reason is declared before measurement: `settle_chain`
descends `E_total`, which **contains `E_store` as a summand**, so settling optimises my own statistic
downward and is expected to launder exactly the signal I am measuring — my predecessor measured that
laundering at **−0.169 / −0.152 AUROC, 74.8 % of it inside the first TEN of a thousand steps**.
⭐ A statistic taken *after* descending it is not "the energy at the read point"; it is the energy at the
minimiser, which the addendum's phrase *"the store's own energy at the **linear** read point"* excludes.

### 2.4 ⛔ THE RULE — one threshold, chosen on `val` only, not swept
```
refuse  iff  S > threshold ,   threshold = quantile( S on val , q = 0.95 )
```
- `q = 0.95` is the harness's **own shipped** `refusal_valid_quantile`
  (`config.py:4594`, `GenBaselinesConfig`) — the same constant, to the digit, that every rival's
  published threshold uses. ⛔ **NOT swept. Not mine. Not tuned.**
- The threshold is taken by the **shipped** `chlu.eval.rivals.gen_knn.threshold_from_valid`, and refusal
  rates by the **shipped** `refusal_rates`. ⛔ Neither is re-implemented.
- `val` is the repaired stratified `val` split (`n_val = 512`, `gen_rowsplit.split_fit_val`), which is
  **seen-valid** — so setting the threshold spends **zero** held-out budget.
- ⛔ **No invalid row of any kind participates in choosing or adjusting the threshold.** If that sentence
  ever becomes false, the run is void and I say so.

---

## 3. ⛔⛔ COMPANIONS — COMPUTED AND LOGGED, **REJECTED IN ADVANCE**, so none can become a second bite

Declaring these **now**, before any of them has a number, is what stops a better-looking one being
promoted later.

| # | companion | status |
|---|---|---|
| **C1** | `d(x_read, nearest atom) / atom spread` — the §C.4 statistic | ⛔ **REJECTED IN ADVANCE.** Named by my addendum. Computed and logged on every row; ⛔ **never adopted**, whatever it scores. |
| **C2** | `S` at the **settled** state (MALA from the linear launch) | ⛔ **REJECTED IN ADVANCE** (§2.3). Computed and logged on every row; ⛔ never adopted, whatever it scores. |
| **C3** | `S_sum`, `S_max`, `S_mean` — my predecessor's dead request-energy residuals | ⛔ **REJECTED IN ADVANCE.** Logged for continuity with the inherited analysis only. |
| **C4** | the readout-margin control (§4.2) | ⛔ **A CONTROL, NOT A CANDIDATE.** It exists to be beaten, never to be adopted as the criterion. |

⛔ **If I adopt any of C1–C4 after seeing its number, this PREREG is void and the report says so in its
first ten lines.** The only statistic that may be reported as THE result is §2.1's `S`.

---

## 4. ⭐ MANDATORY CONTROLS — all three, ON EVERY ROW

### 4.1 `store_removed` must collapse `S`'s AUROC to ~0.5
Charter §C.2, via my predecessor's `shuffled_store`: every channel column of every well's atom matrix
permuted independently among the `N` atoms. **Byte count unchanged, every channel marginal unchanged,
no atom a train row any more.** Exactly one declared quantity differs from the full arm.
⛔ **If `store_removed` does not collapse `S`'s AUROC to ~0.5, it is the address book again and THE
RESULT IS VOID.** ⭐ I state the expected mechanism so the control is a real test: `E_store` is a distance
to atom *content*, and the blend's distance from the atoms it blends is only large when the blended atoms
are **far apart in feature space**, which is a property of what they contain. Shuffling destroys that and
should leave only the posterior's entropy — which is §4.2's control.
⚠ **This is the control that killed my predecessor's criterion** (`0.7809 ≥ 0.7511`). I expect it to be
the binding risk here too, and I am not assuming it passes.

### 4.2 The physics-free READOUT-MARGIN control
*Certified Access* §4.1 measured the relaxation residual at **≤ +0.024 AUROC over a plain margin** — a
reviewer will ask, so I measure it. Two physics-free scores, computed from the **address book only**, with
⛔ no store content, no distance and no energy:
```
M_conf   = − mean_k  max_i P[:, k, i]              (top-1 posterior confidence, negated so "high = refuse")
M_margin = − mean_k ( P_(1),k − P_(2),k )          (top-1 minus top-2 posterior margin)
```
Both go through the **same** §2.4 rule (threshold = `quantile(·, 0.95)` on `val`). ⭐ **Reported beside
`S` on every row.** ⛔ If `S` does not beat the better of them by **more than +0.024 AUROC** on the pair,
the landscape is decorative and I say so.

### 4.3 `accept A7` printed beside `accept B` on EVERY row
Inherited fact: three of five rivals falsely refuse **~60 %** of the valid `A7` control
(`knn_proto` accepts `B` at 1.0000 and `A7` at **0.4120**) and the published table hides it.
⭐ **Every row I report prints both.** A refuser that buys rejection by refusing a valid redundant
request has not refused anything.

---

## 5. ⭐ THE BAR — set before any result exists against it

A **WIN** requires **all four**, at the single val-only threshold, on seed 0:
1. `min( AUROC_oor , AUROC_dup ) ≥ 0.75`
2. `accept_A7 ≥ accept_B − 0.05`
3. ⭐ **beating every rival on the PAIR** — best rival pair is `knn_proto`'s **(0.8426, 0.3519)**
   (reject-oor, reject-dup). ⭐ **No rival refuses both kinds.** The pair is the claim.
4. ⛔ **all three §4 controls pass**: `store_removed` collapses to ~0.5; `S` beats the readout margin by
   **> +0.024 AUROC**; `accept_A7 ≥ accept_B − 0.05`.

⛔ Beating kNN on `out_of_range` alone (0.97) is **not required and not expected** — ⛔ I do not chase it.
⛔ **Head ruling R1 applies to any Δ against a rival:** *tie* := paired 95 % CI on Δ includes 0 **or**
|Δ| ≤ m = 0.020; *win* := Δ > 0 with CI excluding 0; *loss* := Δ < −m with CI excluding 0. A tie is banked
and reported, and I keep pushing with the remaining time.

---

## 6. ⛔ THE GO / NO-GO — the gate in front of any scored spend

**Object:** `S` on my predecessor's **fit-derived synthetic analogue** (`scorer/synthetic_probe.py`'s four
declared constructions from 256 uniformly-sampled `fit` rows, RNG `20260916`), at **rung A and rung B**,
**seed 0**, under the linear read, with all three §4 controls.
⛔ **`c_invalid.json` and `a7_redundant.json` are NOT opened by the go/no-go.**
⚠ Fit rows are **SEEN**, so the valid side here is easier than `B`. ⛔ These numbers are **NOT the
benchmark** and may never be quoted as the refusal result.

```
IF  min( AUROC_oor , AUROC_dup ) < 0.6  on the analogue
THEN  ⛔ DO NOT SPEND.  Report the number and stop the refusal line for this iteration.
```
⭐ **That is a complete and acceptable outcome**, and I record now, before I know the number, that I will
honour it. ⛔ I will not re-run the gate with a different construction to get past it.

**`AUROC_dup` is the A7-CONTROLLED one**: `AUROC(syn_dup vs syn_a7)`, ⛔ never `syn_dup vs syn_valid` —
otherwise a term counter passes it. **`AUROC_oor`** is `AUROC(syn_oor_k vs syn_valid)`, and the gate uses
the **WORST** of `k = 1, 2, 3`, so severity cannot be cherry-picked.

---

## 7. THE SCORED RUN — at most ONE, seed 0 only

- **Budget:** COMMON §1c — **AT MOST ONE scored run this iteration, SEED 0 ONLY**; 0/3 spent on this
  target on any machine to date. Reported as *"scored 1/3 — seed 0 only, 3-seed pending full caches"*.
- **Object:** C-set scoring on seed 0 from the **linear** read: sets `val` (threshold), `B`
  (accept-unseen-valid), `A7` (valid redundant control), `C` (`out_of_range` and `duplicate_factor`),
  `A7D` — every arm re-run on this box so the comparison is same-device (Head ruling R3).
- **Preconditions, BOTH required:** (i) this PREREG names the object, bar and controls — it does; and
  (ii) the go/no-go clears `min(oor, dup) ≥ 0.6` **with the §4 controls passing on non-held-out corners**.
- ⛔ **Seeds 1 and 2 are NOT attempted.** Their caches are `scope="used"` (9,288 rows); covering the cell
  needs ~32,200 new rows. That is a re-encode and it is forbidden.
- ⛔ **A shortfall > 0 means STOP and report. It never means encode.** ⛔ I will not raise `ULP_BOUND`,
  widen `FRAC_BOUND`, or pass `will_extend=False` with a shortfall > 0.

---

## 8. FALSIFIERS — the measured outcomes that kill this, stated in advance

| # | falsifier | kills |
|---|---|---|
| **F1** | `store_removed` does not collapse `S` to ~0.5 | ⛔⛔ the whole claim — it is the address book again, **VOID** |
| **F2** | `S` beats neither readout-margin control by > +0.024 AUROC | ⛔ the physics — the criterion is a plain margin in costume |
| **F3** | `min(AUROC_oor, AUROC_dup) < 0.75` at the val-only threshold | ⛔ the bar |
| **F4** | `accept_A7 < accept_B − 0.05` | ⛔ the accept half — rejection bought by refusing valid |
| **F5** | `S` does not beat `knn_proto`'s pair `(0.8426, 0.3519)` under R1 | ⛔ the novelty (each rival already covers one kind) |
| **F6** | `out_of_range` AUROC < 0.5 (sign backwards, as it was for `S_max`) | ⛔ the "one criterion, both kinds" claim |
| **F7** | the linear read's `x_read` turns out to sit on an atom (`d → 0`) | ⛔ §2.2's hinge — it would be the argmax read again |

---

## 9. FLAG PROVENANCE (mandatory, AGENT_PROTOCOL §5)

| flag | value | source |
|---|---|---|
| commit | `945a3e9` (`base/iter2`) = pin tree + reference-read files + §2.2(b) gate patch | `git diff --stat 8e36bc9 945a3e9` = 952 insertions / 2 deletions, 6 files |
| seeds | **0 only** | COMMON §1c |
| dataset / pretext | `shapes3d` / `recon` | `refusal_lib.DATASET/PRETEXT` |
| rungs | `A` (64 atoms/well, rank 16, 524,288 B) · `B` (512 atoms/well, rank 128) | `GenDynamicalReadConfig` |
| ⭐ `launch_mode` | ⭐ **`posterior_mean`** (⛔ shipped default is `clamp_atom`) | `config.py:6196`; **THE one changed knob** |
| `steps` | ⭐ **0** for the criterion (§2.3); 1000 (A) / 2000 (B) for companion **C2** only | `DYNREAD_STEPS` |
| `sigma_scale_end` (= `sigma_ref`) | `1.0` | dynread config |
| `sigma_scale_start` | `4.0` | dynread config |
| `step_size` | `0.2` | dynread config |
| `temp_start` / `temp_end` | `0.0` / `0.0` | dynread config |
| `metropolis` | `True` | dynread config |
| `clamp_threshold` | `0.5` (⚠ reporting-only; the linear read does not branch on it) | dynread config |
| `request_strength` / `coupling_strength` | `1.0` / `1.0` | dynread config |
| `descent_window` / `descent_tol` | `100` / `0.001` | dynread config |
| `request_chunk` | `256` | dynread config |
| `row_seed_offset` | `20260914` | dynread config |
| `n_fit` / `n_val` | `32256` / `512` | `gen_cell` |
| `grouping_source` | `per_seed` | `gen_cell` |
| `refusal_valid_quantile` (`q`) | `0.95` | `config.py:4594`, **shipped, not swept** |
| threads | `OMP=MKL=OPENBLAS=4` | COMMON §3 |
| jax / jaxlib / numpy / python | `0.9.0` / `0.9.0` / `2.4.1` / `3.11.16`, **CPU** | measured |
| GPU | ⛔ **none** — CPU only | COMMON §2 |

---

## 10. ⛔ WHAT WOULD MAKE THIS HUB'S ANSWER WORTHLESS

- ⛔⛔ The threshold tuned, even slightly, on an invalid example.
- ⛔ Adopting C1, C2, C3 or C4 after seeing its number.
- ⛔ Two criteria, one per kind. ⭐ **One criterion covering both** is the entire novelty.
- ⛔ Quoting `gen-arms-a7-wiring/refusal_scores.json` (the SUPERSEDED broken-split file). My scorer
  asserts the **directory**, not the basename, before reading any rival row.
- ⛔ Reporting AUROC alone. **The (reject-invalid, accept-unseen-valid) pair at ONE threshold is the
  result**; AUROC is a companion.
- ⛔ Re-encoding the feature cache, or loosening the re-encode gate, to reach seeds 1–2.
- ⛔ Manufacturing a result to fill the 12 h window. ⭐ **"Not solved — here is exactly how far I got and
  what blocked me" is a SUCCESS.**
