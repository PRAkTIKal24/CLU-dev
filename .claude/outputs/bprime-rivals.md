# bprime-rivals — experiment-engineer report (C2W4)

**Task + acceptance criterion:** build B′'s rival rows — a TTT-class and a delta-rule memory on the gym
harness — and close every `need` cell of `PREREG-Bprime.md` §2 in the rows I own, with one uniform
protocol (matched-byte table launder · two-sided byte ledger · +0 B substitute · same-keys null · blank
store · identical φ). **Status: done.**

## ⚠ RECONCILIATION LIST (protocol §5 corollary — needs an owner, in my first 10 lines)
1. ⭐ **`PREREG-Bprime.md` §5's P5 construction is a WEAKER control than it looks, and B′ must say so.**
   Reading a weight-valued memory's byte-matched table *through the memory's own projections* costs the
   table **0.19–1.02 neg_mae** against the raw-metric table at the same bytes. Every "dividend" I measured
   against the registered P5 launder is erased by the raw-metric table. **Owner needed:** `bprime-draft`
   (prose) + whoever amends the protocol description. Details §4.
2. **`PREREG-Bprime.md` §2's Titans row** → declared **NOT-RUN** with reason (D5); its `2·|M_θ|` stays
   ⚠ UNPINNED. **P3 is NOT-RUN, not refuted.** Owner: `bprime-draft`.
3. **A store-locality observation, not a claim:** under the learned-initial-state rule the CLU's *measured*
   write-time deviation covers **192/192 atom centres but only 160/192 widths/amps** (§5.3). Benign and
   fully explained (the free slot's re-draw), but it is the first time anyone has measured it. Owner: Hub
   → possibly `harness-debt`.

---

# 1. ⭐ THE AUDIT TABLE (the paper) — `aggregate@base`, 3 seeds, mean ± SE

**`PREREG-Bprime.md` §2 columns. Every `need` I closed is marked `have`; every one I did not is NOT-RUN
with its reason.** ⭐ **The one-family thinness, in my own words: two rival families audited against ONE
surviving synthetic family is a thin cross-family audit, and the rival rows below cannot carry more weight
than that.** It belongs in Limitations verbatim.

| family / arm | matched-byte table launder | **+0 B** substitute (signed margin) | two-sided byte ledger | same-keys null | metric-native verdict | deletion probe | anytime / frontier |
|---|---|---|---|---|---|---|---|
| **CLU** (banked, §7 — **not re-measured**) | **have** −0.4472 | **have**, margin **−0.3180 ± 0.0804** | **have** 5456 B / 100 B, **54.56×** | have −0.8175 | have | **have** (AUC 0.5000 ± 0.0000, byte-equal 3072/3072) | **have** (banked curve) |
| **TTT-Linear** | **have** −0.4245 | **have**, **−0.0523** | **have** F1 5592 B / F2 5220 B | **have** −0.4577 | **have** metric-native | ⛔ NOT-RUN — no deletion verb exists in the family | have (frontier, **non-informative**, §3) |
| **TTT-MLP** | **have** −0.4108 | **have**, **−0.2284** | **have** F1 5736 B / F2 5376 B | **have** −0.4285 | **have** *weakly* metric-native | ⛔ NOT-RUN — as above | — |
| **DeltaNet** | **have** −0.6658 | **have**, **−0.0047** | **have** F1 9956 B / F2 5184 B | **have** −0.6480 | **have** metric-native (Eq. 5) | ⛔ NOT-RUN | — |
| **Gated DeltaNet** (ablation) | **have** −1.4158 | **have**, **+0.0448** | **have** F1 9956 B / F2 5184 B | **have** −1.2202 | **have** metric-native (Eq. 6) | ⛔ NOT-RUN | — |
| ⭐ **Gated DeltaNet-2** (§A14.2 **reference**) | **have** −1.2735 | **have**, **+0.0445** | **have** F1 9956 B / F2 5184 B | **have** −1.1341 | **have** metric-native (Eq. 10) | ⛔ NOT-RUN | have (frontier, **non-informative**) |
| Titans (MAC) | ⛔ **NOT-RUN** | ⛔ NOT-RUN | ⚠ **UNPINNED** (our reconstruction; left unpinned) | ⛔ NOT-RUN | (positioning) | ⛔ NOT-RUN | ⛔ NOT-RUN |
| Sparse Delta Memory | ⛔ **NOT-RUN** | ⛔ NOT-RUN | (Eq. 6, positioning only) | ⛔ NOT-RUN | (positioning) | ⛔ NOT-RUN | ⛔ NOT-RUN |
| Mamba-2 / GRU / SWA | ⛔ NOT-RUN — outside §A14.2's ruled set | | | | | | |

**NOT-RUN reasons, stated once and never reported as nulls.** *Titans:* Hub ruling D5 — NeurIPS 2025,
peer-reviewed (⛔ never "a preprint"); **no official code, chunk size `b` never given a numeric value, no
seeds** ⇒ an arm would be our reconstruction audited against our reconstruction's table. *SDM:* official
code needs **Torch ≥2.8 / Triton ≥3.4 / SM 80+** ⇒ cannot run on this machine. *Deletion column:* no rival
family has a deletion verb at all, which is the point of D6 (⚠ and **a table deletes exactly by
construction** — exact deletion is a result only for a *learned/superposed* store).

## 1.1 The measured audit, in full (the numbers behind the table)

| rival | `d_head` | F1 param B | F2 state B | own table B | **full** | own arg-min table | **dividend** | **+0 B margin** | ⭐ **RAW-metric +0 B margin** | blank | lift over own blank | **RESCUED?** |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| ttt_linear | 29 | 5592 | 5220 | 5104 | −0.4546 ± 0.0312 | −0.4245 | −0.0302 | **−0.0523** | **−0.2465** | −0.8426 | +0.3879 ± 0.0869 | ✅ |
| ttt_mlp | 12 | 5736 | 5376 | 5376 | −0.6324 ± 0.2036 | −0.4108 | −0.2216 | **−0.2284** | **−0.4242** | −0.6031 | −0.0293 ± 0.1090 | ⛔ **NO** |
| deltanet | 36 | 9956 | 5184 | 5184 | −0.4652 ± 0.0402 | −0.6658 | +0.2006 | **−0.0047** | **−0.2571** | −0.5657 | +0.1004 ± 0.1296 | ⛔ **NO** |
| gdn | 36 | 9956 | 5184 | 5184 | −0.3961 ± 0.0208 | −1.4158 | +1.0197 | **+0.0448** | **−0.1880** | −1.3220 | +0.9259 ± 0.2387 | ✅ |
| **gdn2** | 36 | 9956 | 5184 | 5184 | −0.3964 ± 0.0220 | −1.2735 | +0.8771 | **+0.0445** | **−0.1883** | −1.6618 | +1.2654 ± 0.4968 | ✅ |
| **CLU** (banked) | — | 5376 | 5200 | 100 | **−0.5261 ± 0.0863** | −0.4472 | **−0.0789** | **−0.3180** | (= its own) | −0.4221 | — | ✅ |

**Rescue gate** (`rival-recon` F3's sanity gate, applied per cell and reported first-class): an arm within
**2 SE of its own blank-store control** is **NOT RESCUED** and ⛔ **no margin against it is quotable.**
Rescued on `aggregate`: `ttt_linear`, `gdn`, `gdn2`. **Not rescued: `ttt_mlp`, `deltanet`.** Only rescued
arms enter the FB2/FB3 adjudications.

## 1.2 ⭐ The finding, in one sentence
> **At byte-matched state, no memory family in this audit — neither the two rival families nor the CLU —
> beats a zero-extra-byte reader of a raw table holding the same bytes: 0 of 5 rivals (margins −0.19 to
> −0.42) and the CLU (−0.3180 ± 0.0804).** The large positive "dividends" the delta-rule arms show
> (+0.88 to +1.02) exist **only** against the arg-min control **read through their own projections**, and
> they were *pre-registered to exist* (R4) and *pre-registered to vanish* under the +0 B reader (R5).

## 1.3 Admissible-cell coverage, first-class (C2W2 standing rule)
| family | seed | admissible / attempted queries | store admitted / offered |
|---|---|---|---|
| aggregate | 0 / 1 / 2 | **58/72 · 66/80 · 55/80** (0.806 · 0.825 · 0.688) | **5/8** on all three seeds |
| overload@load1x_shipped | 0 / 1 / 2 | **24/24** (1.000) on all three | **6/6** on all three |

Admissibility filtering is **not** gutting coverage (0.69–1.00). The `aggregate` drops are the family's own
construction rule (a query whose target lands within `payload_tol` of a stored payload is dropped, which is
exactly what keeps the arg-min launder from being accidentally right).

---

# 2. ⭐ THE PREREG SCORECARD (registered → measured → verdict)
`PREREG.md` was filed at `.claude/outputs/bprime-rivals/PREREG.md` **before any measured run** (the only
inputs to its predictions were banked §7 numbers).

| # | registered | measured | verdict |
|---|---|---|---|
| **P2** (measured half) | ≥2 of the 3 measured (k,v)-state families lose to their own byte-matched table's strongest **+0 B** reader | **1 of 3** (deltanet only) | ⛔ **REFUTED as registered.** ⭐ Second reading, pre-committed and printed beside it: **3 of 3** lose to the **raw-metric** +0 B table at the same bytes. ⚠ **First half only** — the real-data-LM half (*0 of 4 lose on bpc*) is `cluformer-pilot`'s and is **not tested here**. ⚠ **Mamba-2 and SDM were adjudicated from their equations only**, never blurred with the measured three. |
| **P3** | the two **function-valued** memories show the largest positive dividend | — | ⛔ **NOT-RUN** (no Titans arm ⇒ the pair cannot be formed). **NOT-RUN is not refuted.** TTT-MLP alone is a single-arm datum and it is **not rescued** on `aggregate`. |
| **P5** | the launder transfers to all five rival state types; **0 of 5** failures | **5 of 5 carry a byte-matched table**; 0 failures | ✅ **SUPPORTED** |
| **R1** | rival arg-min launder ≈ −0.42, band [−0.55, −0.25] | −0.4245 · −0.4108 (TTT) · −0.6658 · −1.4158 · −1.2735 (delta) | ◐ **2 of 5 in band; the 3 delta arms fall far below it** — the finding in §4 |
| **R2/R3** | rival `full` ≈ −0.15, band [−0.30, −0.05] | −0.40 to −0.63 | ⛔ **OUT OF BAND (all 5).** I over-predicted the rivals: a byte-matched linear memory at `d_in = 5` does not interpolate as well as its own 2-NN reader does. |
| **R4** | dividend vs own **arg-min** table = **+0.27**, band [+0.05, +0.45] | **mean +0.3691** | ✅ **IN BAND** |
| **R5** | signed **+0 B** margin = **−0.02**, band [−0.15, +0.08], **≥3 of 5 ≤ 0** | **mean −0.0392; 3 of 5 ≤ 0** | ✅✅ **IN BAND and the count is exact** |
| **R5-raw** | *(not banded — added after the band was fixed; reported as a second reading, never substituted)* | −0.1880 … −0.4242 | **5 of 5 ≤ 0** |
| **R6** | rival `blank` ≈ −0.75, band [−1.2, −0.45] | −0.57 … −1.66 | ◐ **4 of 5 in band** (gdn2 −1.66 below it) |
| **R7** | same-keys null ≈ −0.45, band [−0.60, −0.30] | −0.43 … −1.22 | ◐ **2 of 5 in band**; the delta arms' null tracks their launder, as their equations imply |
| **R8** | rival state / own-table bytes = **1.00**, band [1.00, 1.06] | **1.000 · 1.000 · 1.000 · 1.000 · 1.023** | ✅ **IN BAND, 5 of 5** |
| **R9** | `table_is_lossless` **True 5 of 5** | **True 5 of 5** | ✅ |
| **R10** | CLU state bytes < param bytes under the learned-init rule; ratio **0.848** | **state 5200 B < param 5376 B, ratio 0.967** | ◐ **direction CONFIRMED, magnitude WRONG** — I forgot the free slot's re-draw (§5.3) |
| **§1.4** | iso-state head widths **29 / 12 / 36** from the 1364-float budget | **29 / 12 / 36** | ✅ **exact** (asserted in `tests/test_rivals_ledger.py`) |
| **§2.4 F1** | frontier knee at `d_head ≈ 8–10` (19 stream tokens) | knee at `d=16` for gdn2; ttt_linear lossless from `d=2` | ⛔ **WRONG, and the error is mine:** I derived it from `overload`'s **18** offers, but `load1x_shipped` sets `n_offer = 6` ⇒ **7** stream tokens. |

**Score: 6 confirmed (2 exact) · 4 partial · 3 wrong-direction · 2 NOT-RUN.** The two sharpest predictions
(R4's band and R5's band *and* count) both survived; the two I got most wrong (R2/R3, the frontier knee)
are reported as findings, not smoothed.

---

# 3. ⛔ THE BYTE-FRONTIER COLUMN (`overload@load1x_shipped`) — **labelled at every appearance**

> ⛔ **BYTE-FRONTIER COLUMN — not a dividend family, never a headline. Its defensibility is the declared
> secondary reading `S_excl = 0.6500` (the arg-min launder excluded from the +0 B reader set).**

CLU's banked curve is **reused, not re-measured**: `decode 0.972 → 0.097` as the ratio falls `478× → 2.28×`.
Rivals measured beside it (3 seeds, 24 queries, 7 stream tokens, 6 live items, chance = 0.1667):

| rival | `d_head` | state B | table rows | lossless | **full** | own table | blank |
|---|---|---|---|---|---|---|---|
| gdn2 | 2 | 16 | 1 | no | 0.1389 ± 0.1002 | 0.1667 | 0.1806 |
| gdn2 | 4 | 64 | 2 | no | 0.0694 ± 0.0367 | 0.1944 | 0.0972 |
| gdn2 | 8 | 256 | 4 | no | 0.1944 ± 0.0845 | 0.1806 | 0.1528 |
| gdn2 | 16 | 1024 | 8 | **yes** | 0.2222 ± 0.0556 | 0.1111 | 0.1944 |
| gdn2 | 36 | 5184 | 18 | yes | 0.1667 ± 0.0241 | 0.2083 | 0.2083 |
| ttt_linear | 2 | 144 | 9 | yes | 0.0833 ± 0.0481 | 0.2222 | 0.0833 |
| ttt_linear | 8 | 768 | 12 | yes | 0.3333 ± 0.0722 | 0.1667 | 0.1667 |
| ttt_linear | 36 | 7488 | 26 | yes | 0.0694 ± 0.0501 | 0.1111 | 0.1111 |
| **CLU** (banked) | — | 57384 | — | — | **0.9722** | **1.0000** | 0.1667 |

⛔ **I report this column as NON-INFORMATIVE for the rivals and I do not draw a curve from it.** Every
rival point is **within noise of its own blank-store control** (`RESCUED = False`, 0 of 5) — they are at
chance. ⚠ **This is a NOT-RESCUED verdict, not a result about the rivals:** under `rival-recon` F3's sanity
gate **no margin against them here is quotable**, and I explicitly do **not** claim "the CLU beats the
rivals 0.972 vs 0.10".

**Evidence that it is not under-training** (I checked before declaring it): at **5× the outer budget**
(2000 steps vs 400) gdn2 goes 0.0417 → 0.0000 and ttt_linear 0.2083 → 0.1250, while ttt_linear's *fit-split*
loss reaches **MAE 0.024**. It is a **generalisation failure across item geometries** (fit MAE 0.024 → eval
MAE 0.75, 31×), forced by F2a's guard that the outer parameters never see the eval stream's items. The
payload alphabet is spaced **0.4** apart, so an eval MAE of 0.58–0.75 decodes at chance by arithmetic.

---

# 4. ⭐⭐ THE METHODOLOGICAL FINDING (reconciliation item 1, and the strongest thing in this report)

**`PREREG-Bprime.md` §5's P5 says the byte-matched table for a weight-valued memory is "the byte-equal
table of the `(θ_K x, θ_V x)` pairs". I implemented exactly that — and it is not a neutral control.**

| arm | its own **projected** table (P5, as registered) | the **raw-metric** table at the same state bytes | the control's cost |
|---|---|---|---|
| ttt_linear | −0.4245 | −0.2081 (best +0 B) | **0.216** |
| ttt_mlp | −0.4108 | −0.2081 | **0.203** |
| deltanet | −0.6658 | −0.2081 | **0.458** |
| gdn | −1.4158 | −0.2081 | **1.208** |
| gdn2 | −1.2735 | −0.2081 | **1.065** |

Two mechanisms, both at equation level: (i) `θ_K, θ_V` are trained for the **recurrence**, not for a table,
so arg-min in the projected space is a *worse* metric than the raw address space; (ii) a table row decoded
by the memory's own output head `θ_O` is **out of distribution** for `θ_O`, which was trained on
kernel-averaged reads (`o = Σ_s v_s(k_s·q)`), not on single stored `v`s.

**Consequence for B′, stated plainly:** a paper that ran only the registered P5 launder would report
dividends of **+0.88 to +1.02** for Gated DeltaNet-2 and Gated DeltaNet — and **all of them vanish
(margins −0.188)** against a raw table holding the same bytes. Since B′'s entire claim is *"the field has
never measured this, and here is the honest way to measure it"*, **the honest way must include the raw
control.** I ran both, reported both, and never substituted one for the other.

⭐ **This is a contribution, not a caveat:** it is exactly the "launder by omission" failure the gym-side
`+0 B` callers were built to prevent, appearing on the *rival* side of the audit for the first time.

---

# 5. THE TWO-SIDED BYTE LEDGER (D3.3), with the learned-initial-state rule applied to the CLU too

## 5.1 The rule and its arithmetic
**`W₀` / `S₀` / `V_θ`(init) are PARAMETERS (F1); only the per-stream deviation is STATE (F2).** Counting
the init as state inflates; counting the deviation as parameters launders. Enforced structurally: every
ledger's breakdown must sum to its total or `LedgerError` is raised.

| arm | F1 parameters | F2 state | state/param | own table bytes | state/table |
|---|---|---|---|---|---|
| ttt_linear (`d=29`) | 5592 B (incl. `W₀` = 870 floats) | **5220 B** = `d²+16d` | 0.933 | 5104 B | **1.023** |
| ttt_mlp (`d=12`) | 5736 B (incl. `W₀` = 1164 floats) | **5376 B** = `8d²+16d` | 0.937 | 5376 B | **1.000** |
| deltanet / gdn / gdn2 (`d=36`) | 9956 B (incl. `S₀` = 1296 floats) | **5184 B** = `n_head·d_k·d_v` | 0.521 | 5184 B | **1.000** |
| **CLU** (`aggregate@base`) | **5376 B** = `V_θ` init, 1344 floats | **5200 B** = 1300 floats **measured** | **0.967** | **100 B** | ⛔ **52.0×** |

⭐ **The asymmetry that is itself a finding:** every rival's state **can be** byte-matched to its own table
(**1.000–1.023**); the CLU's **provably cannot** — T1's corrected floor
`ratio = [A(D+2)+d]/(d+m) ≥ 2.20×` (**2.40×** at `n_spectator = 1`) makes matched bytes unreachable under a
masked write, and the shipped cell sits at **54.56×**.

## 5.2 D7 — the structural ledger identity (theorist C3), as a blocking check
`chlu/eval/dividend.py::assert_ledger_identity` (append-only; the C2W1 signatures are untouched) asserts
**as integers**: `full == 4[N_at(D+2)+K·d]` and `launder == 4K(d+m)`. Fired green on **every cell**:
`N_at = 192, D = 5, K = 5, d = 4, m = 1, floats/atom = 7, A = 38.4` ⇒ `full = 4(192·7 + 5·4) = 5456`,
`launder = 4·5·5 = 100`, `ratio_corrected = 54.56` — **digit-for-digit the banked ledger.** A drifted store
raises (tested). ⛔ I use the **corrected** law and do not touch `memory_gym.byte_ratio_law`; a regression
test asserts the known live disagreement of **+8.6667** at `n_spectator = 1` (**24/28, not 28/28**) so
`harness-debt`'s fix will flip a test rather than pass silently.

## 5.3 The CLU deviation, measured (reconciliation item 3)
`clu_two_sided_ledger` diffs `V_θ` before/after the stream: **192/192 atom centres moved (960 floats) but
only 160/192 widths and amps (160 + 160)**, plus 20 codebook floats = **1300**. Fully explained: 5 written
slots × 32 atoms = 160 atoms move all three leaves; the **6th (free) slot is re-drawn** by the allocator, so
its 32 centres change while its widths/amps are re-set to their *initial constants* and register no change.
Benign — but it is the first measurement of the write's actual footprint, and it is why my R10 magnitude
(0.848) was wrong while its direction held.

---

# 6. METRIC-NATIVE VERDICTS — argued at equation level, then **measured** (D3.6)

| arm | verdict | equation-level argument | measured against its own table |
|---|---|---|---|
| DeltaNet (Eq. 5) | **metric-native** | `o = Sᵀq`; `S` is a sum of outer products ⇒ `o = Σ_s z_s(k_s·q)`, a linear kernel smoother; `q,k` are **L2-normalised** (§3.5) so `argmin‖q−k‖ ≡ argmax q·k` **exactly**. The only non-metric ingredient is the scalar `β_t`. | loses to the raw +0 B table by **0.257** |
| Gated DeltaNet (Eq. 6) | **metric-native** | adds a scalar decay `α_t` — a scalar reweighting | loses by **0.188** |
| **Gated DeltaNet-2 (Eq. 10)** | **metric-native** | erase `b_t` (key side) and write `w_t` (value side) become **channel-wise**, so the effective metric is a learned **diagonal, token-dependent Mahalanobis** shape rather than the identity. It is still a metric ⇒ criterion 4 still closes — and the table it is audited against is entitled to the same shape, which is why the +0 B readers run on the **same projected keys**. | loses by **0.188** |
| TTT-Linear | **metric-native** | with gradients at `W₀` (§2.6 equivalence) the read is `W₀q − 2η Σ_s(W₀k_s − v_s)(k_s·q)`; the paper's **Theorem 2** makes it general — the nonparametric TTT learner **is** the Nadaraya–Watson estimator with kernel `exp((θ_K x)ᵀθ_Q x')` | loses by **0.247** |
| TTT-MLP | **weakly** metric-native | `f_MLP`'s GELU means the read is *not* a kernel average of stored values, so criterion 4 does **not** close at equation level — the only arm in this task for which it does not | loses by **0.424**, and is **not rescued** |

⭐ This is `rival-recon` **F9** discharged with measurement: *"the rivals' reads are metric-native too"* is
now a shared property of the field with numbers attached, not an assertion.

---

# 7. FALSIFIER ADJUDICATION (each with its evidence)

**FB2 — "not apples-to-apples" — ⛔ DOES NOT FIRE.** A byte-matched table is definable **without an
arbitrary modelling choice** for **5 of 5 state types I adjudicated by measurement** (ttt_linear, ttt_mlp,
deltanet, gdn, gdn2): each has an explicit float state and an explicit `(θ_K x, θ_V x)` stream, so
`n_rows = floor(state_floats/(d_k+d_v))` is *forced*, not chosen. ⚠ **Which of the 5 §2 families I did NOT
adjudicate by measurement, stated and never blurred: Mamba-2, Sparse Delta Memory, Titans** — reasoned from
their equations only. On that basis the "≥2 of 5" bar is not reached in the part I measured; a full verdict
on §2's five families needs Mamba-2 and SDM run, which this task did not.

**FB3 — "the finding inverts" — ⛔ DOES NOT FIRE in the strong form. ⚠ It DOES fire in the weak form, and I
say so plainly rather than re-framing (pre-committed, `PREREG.md` §2.3).**
- **Weak form, measured:** against the **arg-min** control, **`gdn` (+1.02) and `gdn2` (+0.88) show large
  positive dividends while the CLU shows −0.0789.** In that reading, *test-time dynamics pays for the
  delta-rule family and does not pay for ours.* **That sentence is true as measured and is in this report.**
- **Strong form, measured:** **0 of 5** rivals beat the raw-metric +0 B table at the same bytes, and neither
  does the CLU (−0.3180). B′ is therefore **not** a different paper — but *only because* the distinction
  that decides it (R4 vs R5) was registered **before** measurement. Had I not pre-registered R5, adding the
  raw control after seeing R4 would have been indistinguishable from a re-frame.

**FB1 — "not news" — ⛔ DOES NOT FIRE.** Inherited from `bprime-fb1-recon` (14 candidates: 0 HIT · 2 PARTIAL
both out-of-family · 7 NEAR-MISS · 5 NO), and **nothing I saw from inside these implementations changes
it** — which was the specific thing I was asked to watch for. I carry the **narrowed** sentence, not the
original: the audit-at-equal-bits discipline **is standard outside the family** (learned Bloom filters,
learned indexes, SOSD — **B′'s cited methodological ancestry, never suppressed**); a **token-matched trivial
control was published 7 days before filing** (arXiv:2607.21962); the substitute-audit **idea** in general
form is **Poliak et al. 2018 / Feng, Wallace & Boyd-Graber, ACL 2019** — conceded. What survives is stronger
than silence: *seven independent groups built the adjacent instrument and none closed the loop.*

**FB5 — ⛔ DOES NOT FIRE.** arXiv:2501.12352 is purely theoretical (softmax attention as the nonparametric
special case **analytically**, no experiments, no baselines). They unify mechanisms; B′ prices them.

**FB4** — not mine to re-adjudicate; I consume its ruling (§A14.2) unchanged.

---

# 8. ⭐ WHAT I VERIFIED ABOUT GATED DELTANET-2 MYSELF (the task required this, not inheritance)

Fetched and read arXiv:2605.22791 (*"Gated DeltaNet-2: Decoupling Erase and Write in Linear Attention"*,
NVlabs) this session. **Equation numbers I implemented:**
- **Eq. 10** (the boxed recurrence): `S_t = (I − k_t(b_t ⊙ k_t)ᵀ) D_t S_{t−1} + k_t(w_t ⊙ v_t)ᵀ`
- **Eq. 8** `e_t = b_t ⊙ k_t`, `z_t = w_t ⊙ v_t`; **Eq. 9** `S̄_t = D_t S_{t−1}`, `r_t = S̄_tᵀe_t`,
  `S_t = S̄_t + k_t(z_t − r_t)ᵀ`
- **Eq. 11** `b_t = σ(W_b x_t)`, `w_t = σ(W_w x_t)`; **Eq. 12**
  `g_t = −exp(a) ⊙ softplus(W_f x_t + δ)`, `α_t = exp(g_t)`
- **§3.1** the negative-eigenvalue variant scales **only the erase gate to `[0,2]^{d_k}`**, the write gate
  stays in `[0,1]^{d_v}` (implemented, `erase_scale = 2.0`)
- **§3.5** block design: L2 normalisation on the `q` and `k` paths, SiLU on `v` (implemented)
- Its ablations: **Eq. 5** = DeltaNet, **Eq. 6** = Gated DeltaNet (both implemented as named arms)

⭐ **State-size convention — VERIFIED, NOT INHERITED (the specific instruction):** the paper's **Eq. 90**
states *"a per-layer recurrent state of `H d_k d_v = 16·128·128 = 262,144` floats per batch element"*.
**The −2 revision preserves DeltaNet/GDN's `n_head·d_k·d_v` accounting**, so `PREREG-Bprime.md` §2's ledger
row stands unchanged for GDN-2. A unit test asserts the citation so it cannot silently drift.

A test also encodes the paper's **own reduction** (§3.1: collapsing both channel-wise gates to a shared
scalar recovers the scalar-gated update), so a future edit cannot quietly change what we implemented.

**TTT** (arXiv:2407.04620) equations implemented: **Eqs. 1, 2, 4, 5**; **§2.4** mini-batch
(`G_t = ∇ℓ(W_{t'}; x_t)`, `t' = t − mod(t,b)`, `b = 16` *"for all experiments in this paper"*); **§2.7**
`f_res(x) = x + LN(f(x))`, learnable `W₀` (*"shared between all sequences"* — the sentence the
learned-initial-state rule rests on), learnable `η`, `f_MLP` = 2 layers / 4× hidden / GELU.

---

# 9. FLAG-PROVENANCE TABLE (mandatory, protocol §5)

| item | value |
|---|---|
| commit (results produced at) | **`8862577`**, branch `agent/experiment-engineer/bprime-rivals`, base local `main @ d4f56c8` |
| worktree / venv | `../CHLU-rivals`, **main venv reused** (`/Users/user/Desktop/CHLU/.venv`), **no `uv sync`** |
| **JAX / Equinox / Optax** | **0.9.0 / 0.13.4 / 0.2.6** (identical to `main`'s venv — the w6 worktree-drift trap avoided) |
| seeds | **0, 1, 2** (3 seeds, every cell). SD convention: sample sd `ddof=1`, `SE = sd/√3` |
| fit-stream seeds (F2a guard) | `seed + 101`, `seed + 102` — **different sites, different payloads, never the eval stream** |
| families run | `aggregate@base` (dividend) · `overload@load1x_shipped` (**frontier column only**). ⛔ `recency`, `manifold` **NOT RUN** |
| gym non-default flags | `family=aggregate`, `capacity=6`, `consolidate_every=2`, `clu_overrides={stage_admission: True}` |
| CLU non-default flags | `capacity=6`, `budget=6`, `min_atoms=192`, `min_atoms_base=192`, `min_atoms_c=1.0`, `stage_admission=True` (i.e. the **shipped** `aggregate@base` cell, unmodified) |
| rival outer loop | Adam, **400 steps**, lr ∈ {1e-3, **3.16e-3**, 1e-2}, best-of-grid **on the fit split**; TTT also `b ∈ {1, 16}`. ⚠ **F3-lite** — a reduced grid vs `rival-recon` F3's 6×2, declared as a budget choice, **not** presented as F3 compliance |
| chosen configs (aggregate) | s0: ttt_linear (d29, lr1e-2, b16) · ttt_mlp (d12, lr1e-2, b16) · delta arms (d36, lr3.16e-3) — s1/s2: ttt_linear flips to (d36, lr1e-2, **b1**); ⚠ **`b` changes the head width** because the buffer is in the state budget (declared, iso-state rule) |
| iso-state budget | **1364 float32 = 5456 B**, the CLU's **banked** `aggregate@base` full-byte figure; head widths **29 / 12 / 36**, filed in PREREG §1.4 before any run |
| byte law used | **corrected** `ratio = [A(D+2)+d]/(d+m)`; floors **2.20×** (`n_spec=0`) / **2.40×** (`n_spec=1`). ⛔ never *"verified to 1e-9 in all 28 cells"* (it is **24/28**) |
| dtype | float32 throughout, on both sides of the ledger |
| wall clock | 6 audit cells **260 s** total + frontier; whole run < 8 min |
| reproducibility | **verified**: re-running `aggregate/base@s1` at the committed hash reproduces **every rival arm bit-identically** and the same `phi_id` |
| artifacts | `.claude/outputs/bprime-rivals/run/exp_bprime_rivals_metrics.json` (+ `.png`, `run.log`) |

⚠ **Reproducibility bug I introduced and caught in review, disclosed because it briefly produced numbers:**
the per-rival fit key used Python's `hash(name)`, which is **salted per process** (`PYTHONHASHSEED`) — two
runs at identical seeds differed. Fixed to `RIVALS.index(name)` **before any number in this report was
recorded**; the reported run is the post-fix one and its reproduction is bit-identical.

---

# 10. HOW I VERIFIED (commands + observed output)

| check | command | observed |
|---|---|---|
| CLU fidelity vs the **banked** column | the audit run's `clu_reproduction` | ⭐ **digit-for-digit identical**: full `−0.682608 / −0.384693 / −0.511032`, launder `−0.496261 / −0.413103 / −0.432255`, blank `−0.438906`, ledger `5456 B / 100 B / 54.56×`; overload `1.0 / 0.958333 / 0.958333` and launder `1.0` |
| unit tests | `pytest tests/test_bprime_rivals.py tests/test_rivals_ledger.py -q` | **56 passed in 20.35 s** |
| full suite (shared-file regression) | `pytest tests/ -q` | see §11 |
| lint | `ruff check chlu/ tests/` | **All checks passed** |
| smoke | `python -m chlu.experiments.exp_bprime_rivals --quick --seeds 0 --families aggregate …` | exit 0, 29 s |
| the run | `python -m chlu.experiments.exp_bprime_rivals` (3 seeds, 5 rivals, 2 families + frontier) | 6/6 cells, 3/3 frontier cells, **0 degenerate, 0 errors** |
| D7 ledger identity | in-code assert on every cell | green 6/6; a drifted store raises (tested) |
| identical-φ invariant | in-code assert across **12 arm rows per cell** | green; a 1e-9 perturbation raises `PhiMismatchError` (tested) |
| under-training probe (before declaring NOT-RESCUED) | 2000-step re-fit on `overload` | gdn2 `0.0417 → 0.0000`, ttt_linear `0.2083 → 0.1250` — **not under-training** |
| reproducibility | re-run `aggregate/base@s1` at `8862577` | **all rival arms bit-identical**, same `phi_id` |

---

# 11. GIT FOOTPRINT

**Branch** `agent/experiment-engineer/bprime-rivals` (worktree `../CHLU-rivals`), base local `main @ d4f56c8`.
⛔ Not pushed. Rebase onto `main`: **no-op** ("up to date"), no conflicts.
**Verified from the MAIN repo** (`git -C /Users/user/Desktop/CHLU log --oneline main..agent/experiment-engineer/bprime-rivals`) — the w4 lesson:

| commit | subject |
|---|---|
| `7ac7264` | add `chlu/eval/rivals`: minimal faithful TTT + delta-rule memories (B′ D1/D2) |
| `1ffe144` | D7: assert the T1 ledger identity structurally, as integers |
| `82abc1e` | `exp_bprime_rivals`: the cross-family matched-byte audit (B′ D3/D4/D6) |
| `8862577` | tests: rival equation-faithfulness + the two-sided byte ledger |

**Files touched — all inside my declared ownership, nothing else:**
`chlu/eval/rivals/{__init__,ledger,ttt,deltanet,fit}.py` (new package; `fit.py` is the fifth file in **my
own** new package — the outer loop and the five-arm protocol, kept out of `__init__.py`) ·
`chlu/eval/dividend.py` (**append-only**: `LedgerIdentityError`, `ledger_identity`,
`assert_ledger_identity`, + 3 `__all__` entries; the C2W1 signatures are untouched) ·
`chlu/experiments/exp_bprime_rivals.py` (new) · `tests/test_bprime_rivals.py`,
`tests/test_rivals_ledger.py` (new).
⛔ **`memory_gym.py`, `exp_memory_gym.py`, `monitors.py`, `attribution.py`, `config.py`, `race.py`,
`fb4_gate.py`, `clu_system.py` — NOT touched** (imported read-only). No conflicts with any concurrent
spoke. The gym's `byte_ratio_law` bug was left for `harness-debt`, as instructed, with the disagreement
recorded in a test.

**Full-suite result:** `PYTHONPATH=. python -m pytest tests/ -q` → **`1117 passed, 31 warnings in 853.55 s`,
0 failed** (1061 pre-existing + 56 new). No regression from the `dividend.py` append-only edit.

---

# 12. OPEN QUESTIONS / FOLLOW-UPS / RISKS

1. ⭐ **The registered P5 launder is a weak control (§4).** The protocol description in
   `PREREG-Bprime.md` §5 should be amended to require **both** the projected and the raw-metric table.
   This is the single most consequential thing I found. **Owner needed.**
2. ⛔ **Two of five arms are NOT RESCUED on the dividend family and none is rescued on the frontier
   family.** If B′ wants quotable margins against TTT-MLP/DeltaNet it needs a bigger outer budget or the
   full F3 grid — my F3-lite is declared, not defended. **A cross-family audit whose rivals are at their
   own blank floor on one of its two families is thinner than the family count suggests**, and that is a
   second, independent thinness on top of §1's.
3. **The `overload` frontier column is currently an argument for the CLU that I refuse to make.** The
   CLU reads 0.972 there and every rival is at chance — but they are not rescued, so no margin is
   quotable. If the Hub wants that comparison it must be *earned* with a rescued baseline.
4. **What would strengthen tier i most cheaply:** Mamba-2 and a GRU/SWA arm (both trivial to implement and
   both would move FB2 from "5 of 5 measured, 3 reasoned" to a genuine 5-of-5 §2 verdict) — **not** a new
   synthetic family, which §A14.2 defers to this wave's review.
5. **Risk I want on the record:** every rival number here is at `d_in = 5`, `K = 5–6` items, ~10-token
   streams. ⛔ Nothing here transfers to a language-model claim. I did **not** size an LM run (§6.2's hard
   stop was never approached).
6. My results **suggest** things about where a dividend could live that are **not** tier-i claims; per the
   dial declaration I am not writing them as findings, and I have not mentioned the §A13 reframe anywhere.

---

## Proposed handover updates (for the Hub)

1. **§10 running log / `claims_matrix.md`:** B′'s rival rows exist. Headline, quotable with its provenance
   table: **at byte-matched state on `aggregate`, 0 of 5 rival state types and the CLU all lose to a
   zero-extra-byte reader of a raw table holding the same bytes** (rivals −0.188…−0.424; CLU
   **−0.3180 ± 0.0804**), while the delta-rule arms show **+0.88…+1.02** against the *projected* arg-min
   control — pre-registered to appear (R4) and pre-registered to vanish (R5).
2. **New never-quote candidates:** (a) ⛔ any rival **dividend measured only against the projected
   `(θ_K x, θ_V x)` table** (§4 — it costs the control up to **1.208**); (b) ⛔ any margin against
   **TTT-MLP or DeltaNet on `aggregate`** or **any rival on `overload`** — **NOT RESCUED**, within 2 SE of
   their own blank store.
3. **Erratum to carry forward:** `PREREG-Bprime.md` §5's **P5 construction** needs the raw-metric table
   beside it (reconciliation 1). `PREREG-Bprime.md` is deliberately not edited; the amendment belongs in
   B′'s protocol section.
4. **§2 registry:** **Gated DeltaNet-2's state convention is now VERIFIED from its own Eq. 90**
   (`H·d_k·d_v`, 262,144 floats/layer) — the row is no longer inherited from GDN(-1). **P3 is NOT-RUN**
   (no Titans arm); the **Titans `2·|M_θ|` cell stays ⚠ UNPINNED**.
5. **§7 Known Issues — a new entry to consider:** the CLU's *measured* write footprint is **192/192 atom
   centres but 160/192 widths/amps** (the free slot's re-draw). Benign, explained, first measured here —
   but anyone reasoning about C3 write locality from the mask alone will get it wrong.
6. **Config defaults:** none changed. No `chlu/config.py` edit (it is standing read-only to C2 engineers).
7. **`cluformer-pilot`'s gate:** my audit columns landed on `aggregate` with the protocol holding
   (identical φ enforced in code, ledger identity green on every cell, 3 seeds everywhere) — the §A14.3
   checkpoint is satisfiable on my side. ⚠ But note follow-up 2: the *rival* side of the audit is thinner
   than the family count suggests.
