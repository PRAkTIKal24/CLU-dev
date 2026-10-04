# v1-pj-fidelity — paper-referee report

**Task + acceptance criterion:** fidelity audit of the Head's `pj_sub.tex` condensation against the Advisor-accepted base `submission.tex` — "no information lost or misrepresented during the editing." Acceptance: `pj_sub.tex` byte-identical (md5 twice, matching); `submission.tex` and `papers/v1-short/**` untouched; every registry citation checked on disk; every negative positive-controlled.
**Status:** done.
**DIAL DECLARATION (echoed):** Dials touched **NONE**. Read/grep/report only. No experiment, no paper-file edit, no new measurement.
**Reconciliation list owner (protocol §5 corollary):** this report contains a downstream reconciliation list (§7, "missing-experiment / wiring list"). **It needs an owner at the review that accepts this report.** 26 of its 27 items are *wiring* (evidence that exists in the base and was deleted), not new science.

---

## ⛔ 0. BYTE-IDENTITY BLOCK — and the finding that outranks the rest of the report

```
=== md5 AT START OF PASS ===
MD5 (.claude/NIPSsubmission/v1-ttcl/pj_sub.tex)    = bb98439d4dfdbfc279aa2988e0ecc5b8
MD5 (.claude/NIPSsubmission/v1-ttcl/submission.tex) = caef2272f9dc96d349b46486563d24ee

=== md5 AT END OF PASS ===
MD5 (.claude/NIPSsubmission/v1-ttcl/pj_sub.tex)    = bb98439d4dfdbfc279aa2988e0ecc5b8   ← IDENTICAL ✅
MD5 (.claude/NIPSsubmission/v1-ttcl/submission.tex) = caef2272f9dc96d349b46486563d24ee   ← IDENTICAL ✅
```

**⛔⛔ FINDING 0 — THE OBJECT IS NOT THE OBJECT THAT WAS SCOPED. `pj_sub.tex` changed between task-scoping and this pass.**

| | at scoping (task file) | measured at 22:1x, this pass |
|---|---|---|
| md5 | `301ecdf5ed117544cfb12d346fbb7d91` | **`bb98439d4dfdbfc279aa2988e0ecc5b8`** |
| lines | 240 | **410** (+71 %) |
| tex-words (`wc -w`) | 3 981 | **6 450** (+62 %) |
| survival vs base | ≈32 % | **52.3 %** |

`ls -lT` shows `pj_sub.tex` mtime **Aug 26 22:10:42**, while `pj_sub.pdf` is **21:49:15**. The Head was editing live. Consequences, all load-bearing:

1. **The task's own shape premise ("at 32 % the dominant risk is omission") is measured at 52 %.** It is still the right premise — every serious finding below is an omission — but the audit ratio is not the one scoped.
2. **⛔ Three of the Advisor's pre-flight zero-hit measurements are REFUTED, and the task file says a refutation outranks the rest of the report.** All three point the same way: **§4.4 (the CM-23(r) matched-compute anytime read) now EXISTS in the file and did not at scoping.**

| Advisor pre-flight claim | re-measured (`/usr/bin/grep -o … | wc -l`, per-file) | verdict |
|---|---|---|
| `no mask oracle` = 0 hits | **1** (§4.4, l.154) | ⛔ **REFUTED** |
| `anytime` = 0 hits | **1** (§4.4, l.154) | ⛔ **REFUTED** |
| the nine `ties` hits are all false friends (`properties`/`capacities`/`quantities`) | **13** hits; 11 false friends, **2 GENUINE** — *"the confidence-gated anytime read **ties** the matched-compute feedforward floor"* and *"it **ties** rather than wins"* (both l.154) | ⛔ **REFUTED** |
| `external benchmark` / `headline metric` = 0 | 0 / 0 | ✅ upheld |
| `govern the store` / `measured separately` = 0 | 0 / 0 | ✅ upheld |
| `compute-adaptive` / `dead flat` / `directed` / `1.40` = 0 | 0 / 0 / 0 / 0 | ✅ upheld |
| B3, B4 and the 400-ep rider survive | survive | ✅ upheld |

3. **The built PDF (`pj_sub.pdf`, 21:49) predates the current `.tex` (22:10).** Any page-budget statement read off that PDF — including the task's "main text pp. 1–8, appendix p. 9, references p. 10" — is **not a measurement of the current object**. I did not re-build (no edits, and building writes files). **The page budget is UNVERIFIED and, given +62 % of words landed after that build, is very likely violated.** Flagged, not judged (Prohibition 2).

**Manifest — untouched files, verified on disk:**
`submission.tex` md5 `caef2272f9dc96d349b46486563d24ee` = the scoping value ✅.
`.claude/papers/v1-short/**` — all 11 files mtime **Jul 20 01:33**, byte-untouched ✅ (`draft.tex` `208797d1…`, `draft.md` `00d703d5…`, `draft.pdf` `141f2c37…`, `CHANGELOG.md` `a2bc48c0…`, `draft.log` `86d8f80c…`, + 6 PNGs `679647f6…`/`b0cfbf53…`/`708b6fae…`/`bcc5f32d…`/`8b1dfdd c…`/`fc372ae5…`). Note: `papers/v1-short` does not exist at repo root; the path is `.claude/papers/v1-short`.

---

## 1. Verdict (simulated venue review)

### **BORDERLINE — leaning reject on the current text; weak-accept after the MUST-FIX list.**

**Meta-review.** The condensation is *numerically* clean to a degree that is genuinely unusual: **of 210 distinct numeric tokens in `pj_sub.tex`, exactly two have no literal ancestor in the base, and neither is a fabricated result** (one is a seed-list expanded from an en-dash range, one is `1.91→≈1.9`). Not a single surviving `±` is mis-transcribed. The Head's question — "are the numbers intact?" — answers **yes**. But the paper the numbers now sit in is a materially different paper, and it is different in exactly the direction the task predicted and a hostile reviewer punishes. **The abstract now contains zero negatives where the base's contained all three** (Hopfield is cheaper *and* more noise-robust; energy-gating *loses*; the noise wall). **All six contribution-list reporting-grade tags (`[proven]`/`[verification…]`/`[evidence]`) are gone, and so is the sentence "three of our six contributions are boundaries or negatives."** Two section headings that carried their own concessions ("*…and where energy-gating it **loses***", "*A matched-compute anytime read: **it ties***") were renamed to neutral labels. The **§4 grade line and the mandatory scoreboard sentence are deleted**; the **§5 "Scope (stated, not buried)" paragraph — the paper's only "no claim here is at scale" statement — is deleted**; the **App-F grade line ("theory-complete on toy EBMs; no runs on trained CLU checkpoints are claimed") is deleted**, so four MCMC design rules now read as results about the CLU. And in App E, **the only two of six analytic checks that did *not* match to machine precision (ζ*=0.2356 vs measured 0.27; re-absorption 56 vs predicted 49.5) are the two that were removed**, and replaced by the prose "*accurately tracked the theoretical … across all swept rapidities*" — while the intro still promises verification "to machine precision." That last one is not an omission; it is the one place where the edit produced a **claim that the deleted evidence contradicts**. Separately, the appendix that carried the noise wall's *data* (C.4.c, six σ-cells) was deleted while the flattering ρ=0.9 table was kept — the noise wall survives as prose, its table does not. This is, precisely, *"a materially less-qualified paper without a single number having been edited."*

Two genuinely good things, recorded so the review is not soft-by-omission in the other direction: (i) **B6 (C-6's three fine prints) is fully intact** — matched-quadratic scope, "volume alone is not the latch receipt", "a free ledger does not buy BIBO" all present in main text and in the contribution list; (ii) **MF-A (decision/transport) and B2 (the 400-ep rider + the convergence retraction) survive in §3.2.1 and §4.1 respectively**, which are the two relocations the task named as highest-risk. The condensation did not commit the two errors most likely to be fatal.

---

## 2. PART A — the Head's question: are the numbers intact?

### A.0 Headline

**⛔ NO number in `pj_sub.tex` lacks an ancestor in `submission.tex`.** The most serious finding available in this pass **did not fire**. Method: `re.findall(r'\d+(?:\.\d+)?')` over both files → 210 distinct tokens in `pj_sub.tex`; set-difference against the base yields **2** candidates, both adjudicated below as ancestor-present.

### A.1 The two orphan tokens

| # | token | location | ancestor | six-axis verdict | triage |
|---|---|---|---|---|---|
| **A-1** | `45` | App A.4, `Seeds & $\{42,43,44,45,46\}$` | base A.4: `seeds & $\{42\text{--}46\}$ (Item~1) / $\{42,43,44\}$ (Item~2) / $\{42,43\}$ (anchor item~2, 2 episodes/cell)` | **value ✓ · precision ✓ · units ✓ · ± n/a · ⛔ SEED COUNT WIDENED · ⛔ SCOPE CLAUSE DELETED.** The token has an en-dash ancestor; the **claim** does not. A per-item seed schedule of **5 / 3 / 2** became a flat **5 for all of §4.3**. Item 2 is the epoch-scaling frontier — the source of §4.3's `4000 ep`, `0.975`, `Δ+0.03` and `1.00→0.89` — measured at **n=3**. Cross-checked against `claims_matrix.md` CM-8 provenance cell on disk: *"regime-remap-2000ep (COMPLETE: **n=8 capacity, n=5 stress, n=3 frontier**)"* — the matrix confirms three different n's. | **MUST-FIX** |
| **A-2** | `1.9` | App B.2, *"The escaping arms present a linear growth rate $r^\ast(2T)/r^\ast(T)\approx1.9$"* | base B.2 item 1: *"$=1.000$ for every bounded arm/exit (saturated) vs **$1.96/1.94/1.91$** for the escaping ones"* | **⛔ PRECISION CHANGED (three per-exit values → one rounded scalar) · ⛔ the paired contrast `1.000` for the bounded arms is DELETED** — the diagnostic is a *ratio against a saturated control*, and the control is gone. `≈1.9` also understates: `1.96` rounds to `2.0`. | SHOULD-FIX |

### A.2 Six-axis audit of every headline number that survived

**±:** programmatic check — **zero `X\pm Y` pairs in `pj_sub.tex` disagree with the base.** No error bar was altered. Values that carry a `±` in the base and appear bare in `pj_sub.tex`: `0.01, 0.04, 0.09, 0.24, 0.41, 0.71, 0.82, 0.89, 0.90, 0.975`. All but one are false friends (the base quotes them bare in its own main text or table). The genuine one is `0.975` (A-3 below).

| # | number (pj) | ancestor (base) | value | prec. | units | ± | seeds | scope clause | verdict |
|---|---|---|---|---|---|---|---|---|---|
| A-3 | `0.975` (App C.3 prose, kv96 @4000 ep) | C.4.b `0.975\pm0.00`, **n=3** | ✓ | ✓ | ✓ | ⛔ **dropped** | ⛔ **n=3 printed under a heading that says "8 pooled seeds"** | **⛔ WIDER — MUST-FIX** |
| A-4 | `4.81\pm0.44\times` (contrib. 4 + §4.1) | §4.1 pt-2 identical | ✓ | ✓ | ✓ | ✓ | §4.1 ✓ ("5 seeds" in the subsection lead); **contribution bullet ⛔ drops "5 seeds, MQAR vocab-256, laptop"** | PARTIAL — SHOULD-FIX (C-5) |
| A-5 | `0.894\pm0.021` vs `0.847\pm0.037` | identical | ✓ | ✓ | ✓ | ✓ | ✓ | ⚠ base also gives the **step counts** `@629\pm60` vs `@3000`; deleted ⇒ the `×4.81` has no absolute denominator anywhere in the paper. SHOULD-FIX |
| A-6 | `1.57\pm0.07\times` / `1.14\pm0.06\times` | identical | ✓ | ✓ | ✓ | ✓ | ✓ | ⛔ the **accuracies at those levels — `0.547\pm0.039` (kv24) and `0.286\pm0.037` (kv32) — are DELETED.** See F-9. |
| A-7 | `0.647\pm0.063` @ `ε=0.05` | identical | ✓ | ✓ | ✓ | ✓ | ✓ | ⛔ **measured risk `0.030` deleted; `30/30` cells → "all evaluated cells"**; ⛔⛔ **ECE `0.100\pm0.021` deleted entirely** (0 hits, positive-controlled). See B7. |
| A-8 | `0.18\to0.88` / `0.43\to0.87` | identical | ✓ | ✓ | ✓ | n/a | ⛔ base A.2 carries a dedicated provenance row *"Hopfield-transfer source: `minus-the-physics` Part B, 5 seeds … CM-2"* — **DELETED** ⇒ the memory-agnostic number has **no provenance** | SHOULD-FIX (C-7) |
| A-9 | router `1.000/0.948` @`8.81e7` vs gated `0.887/0.715` @`1.18e8` | identical | ✓ | ✓ | ✓ | table ✓ | ⛔ §4.2 drops "**across all mixes {50/50,80/20,95/5} and both N, over 5 seeds**" (A.3/C.2 retain mixes + seeds) | SHOULD-FIX (C-5) |
| A-10 | `449`-param | ✓ | ✓ | ✓ | ✓ | n/a | §4.2 drops "**2-layer**" (A.3 has `hidden 32`) | NICE |
| A-11 | `1.18e8` / `1.76e8→2.94e8` / `0.41→0.28` | identical | ✓ | ✓ | ✓ | n/a | ⛔ **the FLOPs accounting model (base A.3: `flops_grad_factor 6.0`, `flops_verlet_grads 2.0`, "routed leg = 2 units (flat in N), chain = N units") is DELETED** ⇒ *no FLOP number in this paper is reproducible.* | **MUST-FIX (C-7)** |
| A-12 | `9.9\times, 9.5\times, 6.2\times` | identical | ✓ | ✓ | ✓ | n/a | ✓ **intra-CLU adjacency held at all 4 sites** (§4.3, Fig-2 caption, C.3 column header, C.4 column header) | ✅ CLEAN |
| A-13 | `0.947` to `0.979` (Hopfield band) | §4.3 identical | ✓ | ✓ | ✓ | n/a | ⛔ **"at `β\ge5`; extra iterations change accuracy `\le0.003`" DELETED, and App C.3 (Hopfield iteration parity, 3 seeds, the whole table) DELETED** ⇒ *"Hopfield reaches its ceiling in ≈1 matvec"* — the load-bearing premise of "Hopfield is the cheaper retriever" — now has **zero evidence in the document**, while A.4 still advertises the sweep `{1,2,3,5,10}` | **MUST-FIX** |
| A-14 | `Δ+0.02`, `Δ+0.03`, `1.00→0.89` | identical | ✓ | ✓ | ✓ | n/a | see A-1/A-3; ⛔ **"kv128 only ties (`Δ+0.004`)" DELETED** — see F-6 | **MUST-FIX** |
| A-15 | `0.36` vs `0.71` @ `σ=0.6`/kv32, fidelity `1.0` | identical | ✓ | ✓ | ✓ | base table `0.36\pm0.06`; base *main text* quotes bare — so bare is faithful | ✓ | ⚠ base writes fidelity **`≈1.0`**, pj writes **`1.0`** (exact for kv32 per C.4.c; harmless) | ✅ |
| A-16 | `+0.8\pm1.6` pp / `+1.1\pm1.4` / `+4.6\pm2.2` (§4.4) | CM-23(r) verbatim, checked on disk | ✓ | ✓ | ✓ | ✓ | ⛔ `3/6 positive`, `6/6 seeds`, `+4.4` ensemble, `ungated → 0.000`, **`1.40\pm0.20\times` auto-stop** all DELETED; ⛔ the **entire C-5 scope clause** DELETED | **MUST-FIX** — see B10 |
| A-17 | squeeze injection tuples `(0.25,1.13,1.65)…(2.0,27.5,54.6)`; `\det S=1.000\ (\pm4e{-6})` | identical | ✓ | ✓ | ✓ | ✓ | ✓ | ✅ CLEAN |
| A-18 | `\Delta Q=0.2500`, `0.0803`, `1.2e-7`, `7.5e-10`, `2.05`, `0.2793`, `0.2465`, `0.0972`, `0.0379` | identical | ✓ | ✓ | ✓ | n/a | ✓ | ✅ CLEAN (the §3.2.1 table is transcribed exactly) |
| A-19 | reconstruction *"exactly recoverable ($q_{\rm in}=q_{\rm out}-\Delta$)"* | base: *"exactly recoverable … (**max err $2.2\times10^{-8}$**)"* | ⛔ | ⛔ **BOUND → BARE ASSERTION.** The task names this class explicitly: *"a bound becoming a point estimate is a claim change, not a rounding."* Here it is worse — the bound became **no number at all**, upgrading a measured `2.2e-8` to unqualified exactness. | — | — | — | **MUST-FIX** |
| A-20 | BIBO `0.09/102.13/103.43/104.83`, `\Delta H=2.88`, `x_b=3.5355/3.536`, `V_b=3.125`, `T=2000` | identical | ✓ | ✓ | ✓ | n/a | ⛔ **"the receipt predicts BIBO blow-up on 6/6 exits" DELETED**; ⛔ T-values `52.12/53.43/54.83` deleted; ⛔ the bounded-arm ratio `1.000` deleted (A-2) | SHOULD-FIX |
| A-21 | App E: `2.0e-12`, `2.2e-16`, `1.8e-15` | identical | ✓ | ✓ | ✓ | n/a | ⛔⛔ **checks (D) `ζ^\star=0.2356` vs measured `0.27` and (F) re-absorption `56` vs predicted `49.5` DELETED**; ⛔ check (B) `q=-1.068` deleted; ⛔ `E_{\rm inj}/E^\star=2.718=e^{2\cdot0.5}` deleted; ⛔ the provenance line (`checks.py`, numpy float64, seed 0, $V=\beta(q^2-a^2)^2$, $a=\beta=1$, $\varepsilon=0.05$) DELETED | **⛔⛔ MUST-FIX — see F-1** |
| A-22 | MCMC `0.0995\to0.0065`; `\sigma_i^\star=\sqrt{M_{\rm eff,i}T\gamma(2-\gamma)}` | identical | ✓ | ✓ | ✓ | n/a | ⛔ `L_1=0.0095` (π-reversibility verification), `T_{\rm eff}:1.0\to0.61`, `D=1.29e{-3}` vs `\half s^2=1.25e{-3}`, `N_{\rm erode}\approx(\Delta_{\rm read}/s)^2` all DELETED — yet App F still says *"**proving** that without projection … will progressively erase"* | **MUST-FIX** |

---

## 3. PART B — the do-not-cut list, item by item

Every registry citation below was read **on disk at the moment of use** (`claims_matrix.md`, `critique_register.md`, `philosophy-synthesis.md` §"Positioning Charter"), never quoted from the task file.

| # | mandatory object | verdict | evidence (measured, positive-controlled) | ⛔ the claim now standing unqualified |
|---|---|---|---|---|
| **B1** | C-2 grade label — *"Reporting grade: verification… oracle channel placement, dim 2/4, 5 seeds, laptop-CPU, γ=0"* | **PARTIAL → effectively ABSENT at three of four sites** | `Reporting grade` **pj=0 / base=4** (control: `oracle` pj=8). §3's *scope list* survives in prose (l.69: "dimensions 2 and 4, utilizing 5 random seeds, oracle channel placement, laptop-CPU execution, a sharp boundary γ=0 rollout, mass band [4.0,0.25]") ✓ and §4's transition to trained memories is stated ✓. But: **all six contribution-list grade tags deleted** (`\[proven` **0/3**, `\[verification` **0/2**, `\[evidence` **0/3**); §4's grade line deleted; §4.4's grade line deleted; **App F's grade line deleted** (`theory-complete` **0/2**, `toy EBM` **0/5**); **the abstract's scale qualifiers deleted** (base: *"On a designed analytic testbed (dim 2 and 4, 5 seeds, oracle placement, laptop-CPU) we verify…"* → pj: *"We verify this certificate stack on a designed analytic testbed"*). | **Two of six contributions and the entire headline read as empirical results.** Worst instance: **§5's "Design rules for certified Markov kernels" + App F now assert four operating rules for the CLU's retry kernel with no statement that they are theory-complete on *toy EBMs* and that **no runs on trained CLU checkpoints are claimed**.** Second-worst: the abstract. |
| **B2** | ⛔⛔ MF-C — *"at a 400-epoch budget"* + *"at convergence… **matches** full-budget… the payoff is **rationing, not accuracy**"* | **PRESENT** (§4.1) / **PARTIAL** (contribution list) | `400-epoch` **pj=2** (contrib. 4 + §4.1). §4.1 pt-2 verbatim: *"At an **under-converged 400-epoch** training budget… **At full convergence (2000 epochs), the gating mechanism achieves strict parity with the full-budget baseline rather than exceeding it.** Therefore, the core asset of the gate is **compute rationing** on clean retrieval tasks, **not generalized accuracy dominance**."* ✓ App-A note repeats the epoch split ✓. Contribution 4 carries "at a 400-epoch budget" ✓ and asserts **no** accuracy gain ✓. | **The highest-risk relocation held.** ⛔ **But its evidence was deleted:** base App C.4.a's **`full-budget acc` column** (which shows gate `1.00` = full `1.00`, `0.99`=`0.99`, `0.91`=`0.91` at 2000 ep) is **gone from pj's C.3 table**, and pj §4.1 carries **no cross-reference** where the base said "(§4.3, App. C.4.a)". *The convergence retraction is now an unsupported assertion.* → MUST-FIX (wiring). |
| **B3** | ⛔ THE NOISE WALL — 0/6, gate `0.36` vs Hopfield `0.71` @σ=0.6/kv32 despite fidelity ≈1.0 | **PRESENT in prose · ⛔ ABSENT as data** | CM-8 read on disk, verbatim: *"**THE NOISE WALL (dominant negative, travels with every reversal claim)**: under cue noise σ∈{0.3,0.6} NO cell closes at any capacity (gate 0.36 vs Hop 0.71 at σ=0.6/kv32) despite fidelity ≈1.0."* pj carries it at **four** prose sites: contribution 6, §4.3, Fig-2(c) caption, App C.4, App D ✓✓. | ⛔⛔ **But base App C.4.c's six-cell σ table was DELETED** while the ρ-axis table was kept — and the kept table is the flattering one (`+0.16`, `+0.08`, `−0.16`). **pj's only stress-axis table shows CLU ahead in 2 of 3 rows; the table where it loses 6 of 6 was removed.** Also deleted: the `closes?` column and the CLOSE criterion (Δ≥−0.01), the ρ=0.5 rows (incl. the **kv96 ρ=0.5 non-close, Δ=−0.05**), the tallies **`6/15`** / `6/9` / `0/6` (`6/15` **pj=0/base=3**; `0/6` **pj=0/base=3**), the σ-axis Δ range `−0.05…−0.35`, and the σ-axis fidelity column. **Optics: a referee who reads the appendix tables sees a win table and no loss table.** |
| **B4** | ⛔ *"intra-CLU… never a cost win over Hopfield"* | **PRESENT** | `intra-CLU` **pj=4 / base=8**; every one of the four sites is an adjacency to a savings figure (§4.3, Fig-2 caption, C.3 column header, C.4 column header). §4.3: *"…represent intra-CLU rationing against a full-budget CLU baseline, **not a comparative computational win over the Hopfield network**. The Hopfield baseline remains the strictly cheaper retriever at matched accuracy."* ✓ Forbidden form `9–10× savings vs Hopfield` **0 hits** (control `9.9` present). | ✅ **Clean.** ⚠ minor: base adds *"a curve that **falls with load**, not a constant"* in §4.3; pj leaves that to the C.3 table. |
| **B5** | ⛔ *"energy-**gating** it **loses** to a 449-param physics-free router"* (CM-7) | **PRESENT in body · ⛔ ABSENT from title and abstract** | CM-7 read on disk: *"the energy-gated wormhole **LOSES** to a 449-param physics-free router in FLOPs AND accuracy… **FORBIDDEN: energy-as-routing-signal superiority**."* pj contribution 5 ✓ (*"we explicitly report that energy-gating this mechanism is outperformed…"*); §4.2 body ✓ with the full number set. `loses` **pj=1 / base=13**. | ⛔ **The load-bearing heading is gone:** base *"§4.2 The one-hop non-local edge, **and where energy-gating it loses**"* → pj **"Boundary Analysis for Non-Local Routing."** ⛔ **And the abstract's concession is gone:** base *"energy-gating it **loses** to a 449-param physics-free router"* → pj *"**evaluating scaling boundaries** for non-local routing against a physics-free baseline."* A ToC/abstract reader now sees a boundary study, not a loss. |
| **B6** | ⛔ C-6's three fine prints (Prop-12 matched-quadratic · *"volume alone is not the latch receipt"* · *"a free ledger does not buy BIBO"*) | ✅ **PRESENT — all three, main-text-adjacent** | (i) §3.1: *"the $e^{2|\zeta|}$ bound is a **matched-quadratic certificate**; applied to a quartic well, the raw ratio can naturally exceed this limit."* ✓ (ii) §3.2.1 + Fig-1 caption + contribution 3: *"volume preservation alone is not the latch receipt, as the random-shift baseline maintains detJ=1 but still scrambles the topological spread."* ✓ (iii) §3.1 BIBO ¶ + contribution 3 + App B.2 ✓. Charter C-6 read on disk: *"appear in the main text **next to** the claim they qualify."* Satisfied. | ✅ **The paper's inversion-proofing survives.** ⚠ Losses: the `Prop-12` and `Prop-A2` labels, the `App. D N31` pointer, and — in App D — the **two dedicated "Fine print of Payoff A / Payoff B" paragraphs and the "Payoff scope" paragraph** (*"Both payoffs are on the designed testbed… they do **not** claim a real-data or learned-memory win"*; `Payoff scope` **pj=0/base=1**, `no training` **pj=0/base=1**). |
| **B7** | LTT exchangeability + ECE `≈0.100 ± 0.021` | **PARTIAL — ⛔ the C-6-named number is ABSENT** | Exchangeability ✓ (**pj=1/base=1**): *"LTT relies on exchangeability between write-time probes and deployment queries, meaning the calibration operates slightly under-confident when transitioning across task domains."* ⛔ **`0.100` = 0 hits in pj** (positive control: `0.021` **does** hit, but at `0.894\pm0.021`, a false friend — read in context). Also deleted: *"which our jittered protocol only approximates"*, *"the guarantee holds **within the stated probe-to-deployment scope**"* (C-6's literal required form: *"cannot destabilize" → "certified within [stated scope]"*), `30/30`, measured risk `0.030`. | ⛔ **C-6 VIOLATION.** The register's P20/G5 rule is explicit: *"LTT exchangeability caveat **+ ECE≈0.10** … in main text next to the claim, not appendix."* The paper now claims a distribution-free coverage certificate with **no measured calibration shift anywhere in the document**, and the "under-confident" direction is asserted without the number that establishes it. **This is the surviving CLU-side asset if escalatability is scoped down, and it is now the least-evidenced claim in §4.** |
| **B8** | The measured score sentence (*external benchmarks won on their own headline metric = ZERO*) — Head-ruled IN 2026-08-26 | ⛔ **ABSENT** | `external benchmark` **pj=0 / base=1**; `headline metric` **pj=0 / base=1** (positive control on the same file: `Hopfield` pj=32). Base §4 lead: *"**The scoreboard sentence, stated up front: external benchmarks won on their own headline metric = ZERO.**"* | ⛔ **Mandated by CM-23 ("the program's scoreboard sentence — mandatory in any performance section"), CM-31, CM-37, all read on disk.** Without it, §4's four subsections — a 4.81× saving, a flat-in-N edge, a `Δ+0.02` reversal, and a tie — are the paper's entire performance record with **no statement of what the program has actually won externally.** Nearest surviving substitute is the contributions closer (*"rather than claiming a general leaderboard advantage"*) — a *disclaimer*, not the *measured score*. |
| **B9** | The §A20.5 substrate-scope sentence — Head-ruled IN, same session | ⛔ **ABSENT** | `govern the store` **pj=0/base=1**; `measured separately` **pj=0/base=1**. Base §5: *"And the substrate scope, stated once, in our own voice: **these laws govern the store; end-to-end performance additionally depends on the encoder, measured separately, φ-bytes ledgered.**"* | ⛔ **§4.4 is the exact claim this sentence exists to bound** — it runs a *designed store* behind a *learned frozen encoder φ*, and it is the paper's only matched-compute result. pj states the substrate ("a designed store addressed through a frozen encoder") but **never states that the encoder is a separate, separately-measured, byte-ledgered channel.** A reviewer reads §4.4's tie as a property of the memory. |
| **B10a** | N103's tie — **CM-23(r)**: *"ties"*, never *"wins"* | ✅ **PRESENT (and the Advisor's 0-hit pre-flight is REFUTED)** | CM-23(r) read on disk: *"⭐ THE R3-NATIVE READ — a **TIE**, and the wording is **"ties"**, never "wins"."* pj l.154 uses **`ties` twice**, once explicitly defensively: *"We note explicitly that it **ties rather than wins** against the matched-compute feedforward baseline."* ✓ And N95's same-section obligation is discharged **in the same subsection**: *"where a mask oracle can be constructed, the gated read does not beat the machine-learning-optimal masked-erasure oracle in any evaluated cell, establishing a firm metric-native ceiling."* ✓ | ✅ The wording rule is honoured. ⛔ **But the scope is not** — see B10a′. |
| **B10a′** | CM-23(r)'s *"Scope that travels"* | ⛔ **ABSENT** | Deleted: *"One dataset (MNIST-class images, φ fit on the first task's classes only and frozen), one corruption level (pixel dropout **p=0.8**), one crowded store (**199–200 wells**, end-of-stream), 6 seeds, laptop-CPU"*; the **p=0.5 saturation** counter-cell (*"first pass 0.99, floor 1.000 — every line is flat"*); the **CIFAR counter-cell** (*"the 1-shot kNN-in-φ floor is 1.000 and the same read sits **below** it, `−0.055±0.019`"*); *"nothing here generalises past it"*; `1.40\pm0.20\times` auto-stop (`1.40` **pj=0/base=1**). Base §4.4's heading itself carried the concession — ***"…: it ties"*** — pj heading is **"Matched-Compute Anytime Read."** | ⛔ **C-5 + the task's drift mode 6.** §4.4 now reads as *the* matched-compute result rather than *one cell of one dataset at one corruption level, adjacent to a cell where the same read loses.* A reviewer asking "does this generalise?" finds nothing to stop them assuming yes. |
| **B10b** | The trilemma's third corner — **CM-23(y)** + N119's *"neither fix may be described as available"* | **PARTIAL** | CM-23(y) read on disk. pj §5 ¶2 states the impossibility ✓ and keeps a form of the third corner (*"dictating that a faded memory mathematically requires more integration steps to read"*). ⛔ Deleted: **"dropping amplitude-independent latency **is** the compute-adaptive-read dial"** (`compute-adaptive` **pj=0/base=2**); ⛔ **N119's mandatory clause "*Both proposed fixes to that corner are refuted, and neither may be described as available*"**; ⛔ *"The shipped store drops the second… the recommended gated-stiffness channel drops the third"*; ⛔ `r=−0.85`; ⛔ **the entire provenance/substrate parenthetical** (*"Proved and measured on this program's decaying-store instantiation — a designed store with a quadratic payload channel and a fixed read budget, 3 seeds, laptop-CPU — **not on the §3 unit**"*). | ⛔⛔ **This is the sharpest instance of drift mode 1 in the paper.** An impossibility result about a *different substrate at 3 seeds* now sits, unattributed and unscoped, in §5 immediately after the §3 unit's certificates. A reviewer will read it as a theorem about the object of §3. And CM-23(y)'s explicit prohibition (*neither fix may be described as available*) is simply not in the document. |
| **B10c** | N90's mechanism attribution — **CM-23(g)** — with **N95**'s same-section obligation | ⛔ **ABSENT (the claim, not just the caveat)** | `directed` **pj=0/base=3**; `dead flat` **pj=0/base=1**. Base §5 design-rule 2 carried the full measured result: *"the lift is the **directed** symplectic re-launch: equal-energy random kicks and ensembles of k independent restarts are **dead flat in all 8 cells**, while confidence-gated CLU retry rises **+6.6–+76.2 pp** and **auto-stops at ×1.2–1.8**; ungated retry-all collapses **0.96→0.004** at 9× compute"* + the three companions (the NN floor `−3.9…−20.7`/`−9.7…−48.2` pp, **negative in 40/40**; the mask/noise protocol split; *cosine-to-nearest-well is a ranking, never an acceptance, signal*) + N95's decision-grade NO restated in-section. **All of it is gone.** | ⚠ **Deleting a positive claim is safe under CM-23** — the N95 obligation is now vacuous in that section because the claim it companions no longer exists. **But:** (i) this was the program's strongest *measured mechanism* result and the paper is now weaker for a venue asking "what does the physics buy?" (G2); (ii) with it went the paper's own falsification record (`40/40` negative), a **C-9** loss; (iii) pj's surviving design-rule 2 (*"the squeeze family orbit is reducible and non-ergodic, it must be layered onto a MALA step"*) is now pure theory presented, per B1, **without the App-F grade line**. |

---

## 4. PART C — drift-mode audit of the surviving claims

| # | mode | base form (quoted) | pj form (quoted) | ruling |
|---|---|---|---|---|
| C-a | **1 — designed testbed reading as general** | *"**Reporting grade: evidence.** §4 leaves the designed testbed for **trained** memories… **The scoreboard sentence, stated up front: external benchmarks won on their own headline metric = ZERO.**"* | *"To determine the practical viability of these formalisms, we evaluate **three applications** on trained associative memories."* | ⛔ **WIDER.** Grade label + scoreboard both gone. **And an internal contradiction is introduced: the §4 lead says "three applications" while §4 contains four subsections** — base flagged §4.4 in its heading as *"(beyond the three pillars)"*; pj deleted that, so §4.4 now silently reads as one of the three, erasing the base's careful separation of its different substrate and its own 6-seed protocol. |
| C-b | **1** | §5: *"**Scope (stated, not buried).** The §3 verifications use **oracle channel placement, dim 2/4, 5 seeds, laptop-CPU**; the §4 learned-memory pillars are MQAR-style, vocab-256, kv/N small, laptop-CPU. **No claim here is at scale, and none uses learned placement.**"* | *(no equivalent paragraph exists)* | ⛔⛔ **WIDER — MUST-FIX.** The paper's single consolidated "no claim here is at scale" statement is deleted. Charter **C-5** on disk: *"Every generalizing claim carries its scale qualifier… Draft-review checklist item: **grep for scope-free plurals**." *Swept:* the abstract, §4.2, §4.4 and §5's trilemma are all now scope-free. |
| C-c | **1** | App F header: *"**Grade: theory-complete on toy EBMs. No runs on trained CLU checkpoints are claimed anywhere in this paper.**"* + §5 header parenthetical *"(theory-complete on toy EBMs; rule 2's mechanism controls are measured on a designed store; no runs on trained CLU checkpoints are claimed)"* | App F: *"To secure a stationarity certificate for test-time retries, we derive the four operating constraints for the proposed Markov kernel."* §5: *"The MCMC implementation necessitates four strict design rules."* | ⛔⛔ **WIDER — MUST-FIX.** *Both* grade statements deleted. Four rules about the CLU's retry kernel now stand with no substrate. Compounded by App F.6's deletion, which carried *"**What remains unrun is this specification itself** … No trained-CLU-checkpoint result is claimed."* |
| C-d | **1 / genre** | l.2 comment: *"V1 workshop short … **--- position/theory paper**"*; abstract: *"**We take a position**"*; §5: *"**The position, restated.**"* | l.2: *"V1 workshop short (ML4PS / NeurReps class) paper."*; abstract: *"we **propose a theoretical and empirical framework**"*; §5: *"**We establish that** on a conservative memory…"* | ⛔ **CHANGED IN KIND.** The genre label is deleted in three places and *"position"*→*"establish"*. C-2's whole designed-verifies/learned-evidences architecture is scaffolded on the position framing; removing it while also removing the grade labels (C-a/C-c) leaves nothing marking the split. |
| C-e | **2 — intra-CLU reading as a Hopfield win** | *"…is **intra-CLU** rationing vs a full-budget CLU, **never a cost win over Hopfield**. … **We state this next to every accuracy-improvement claim.**"* | *"…represent intra-CLU rationing against a full-budget CLU baseline, **not a comparative computational win over the Hopfield network**."* (×4 sites) | ✅ **IDENTICAL in force.** Adjacency discipline held at every site. |
| C-f | **3 — 400-ep reading as converged** | §4.1 pt-2 + the C-7 cross-section note | §4.1 pt-2 (verbatim-equivalent) + the shortened App-A note | ✅ **NARROWER/safe.** ⚠ but see B2 — the supporting `full-budget acc` column was deleted. |
| C-g | **4 — "priced" reverting to "cannot reach"** | §3.2 heading: *"The discriminating experiment: **squeeze reach is priced**, the wormhole is flat-priced…"*; dagger footnote; *"observed edge d≈3.2, not a knife-edge at L=2.5"*; *"the swept ζ≤2.0 reaches d≲3.6 (≤e⁴≈55 H)"* | Heading: **"Differentiating Access Mechanisms"**; body: *"reach via squeeze is **exponentially priced** in distance, whereas reach via wormhole is **flat-priced**"*; *"**priced out** of the swept rapidity budget (ζ≤2.0), **not that it is fundamentally incapable of reaching them**"*; table diagnostic column *"Reach beyond $C_T$ is exponentially priced"*; C.1 *"priced out past the swept rapidity grid"*; abstract *"**prices reach exponentially rather than capping it strictly**"* | ✅ **IDENTICAL in substance — MF-B's repair SURVIVED at five independent sites.** ⛔ **But the repaired heading itself was overwritten** with a neutral one, and the `d≈3.2` observed edge / `d≲3.6` / `≈55 H` anchors are deleted. **NICE-to-restore, not a claim change.** |
| C-h | **5 — decision/transport conflation (MF-A)** | §3.2.1: *"**Decision is not transport — state this once, sharply.** … §4.2's `router_mlp` **is** a learned decision head… **A learned gate bolted onto a certified channel is not a counterexample to the receipt; it is a consumer of it.**"* + the abstract sentence + C.1's "two router objects" note | §3.2.1: *"It is critical to distinguish between an analytic transport map and a learned decision head. A learned decision router **may** select whether to utilize a non-local edge, but it transports the data through the certified detJ=1 wormhole channel."* + C.1 note ✓ | ✅ **PRESENT — NARROWER.** ⛔ Deleted from the **abstract** (base: *"Deciding whether to take a certified edge is orthogonal: a learned decision head (§4.2) routes through the detJ=1 channel and inherits its receipt — the certificate prices only transport"*). ⚠ *"**may** select"* is hypothetical where the base was factual ("**is** a learned decision head") — a small softening of a guard that exists to be unambiguous. |
| C-i | **6 — a tie reading as a win** | §4.4 heading *"…: **it ties**"*; *"**It ties; it does not win**"* | *"**ties**"* ×2 incl. *"it **ties rather than wins**"* | ✅ **IDENTICAL.** (Heading concession lost — see B10a′.) |
| C-j | **forbidden-form re-opening** | §4.2: *"The direct edge is the mechanism; **energy is not the routing signal**."* | §4.2: *"…relying on the relaxation energy as a routing signal is computationally sub-optimal. The direct edge remains the required transport mechanism, but **raw kinetic** energy is not the optimal routing signal **in this regime**."* | ⛔ **WIDER — SHOULD-FIX.** **Two hedges added to a CM-7/CM-3 *prohibition*:** (i) *"in this regime"* re-opens other regimes; (ii) *"raw kinetic"* re-opens non-kinetic energy signals. CM-7 on disk: *"**FORBIDDEN: energy-as-routing-signal superiority** (third instance of the CM-2/CM-3 pattern)"* — unconditional. ⚠ *Honest counterweight:* CM-3's own corollary notes *"the 'energy' read-out **is** a KINETIC read-out,"* so "raw kinetic" is descriptively defensible; *"in this regime"* is not. (The base's own regime hedge attached to the **router's dominance**, not to the prohibition, and pj deleted **that** one: *"a harder band is the untested fair stress test"* is gone.) |
| C-k | **C-1 (the reference-unit clause)** | *"**Nothing below uses a property specific to the reference training objective except where stated.**"* | *"The empirical results in this work **assume the reference training objective of the CLU** but evaluate generalized properties of the learned Hamiltonian."* | ⛔ **CHANGED IN KIND.** These assert opposite things about objective-dependence. The pj form is also self-tangled (results both *assume* the objective and evaluate *generalized* properties). Direction is safe (it narrows), but it is now inconsistent with §4.1's memory-agnostic finding and with §3, which uses **no training at all** (A.5: `training: none`). |
| C-l | **C-7 (cross-section reproducibility)** | App A closing: *"…Appendix B.2's BIBO battery uses **γ=0.02 because a bounded arm must be able to settle** (at γ=0 a conservative orbit never converges, so 'bounded' would be untestable); §4.3 uses `langevin_noise=legacy`."* | *"…The calibrated gate evaluation utilizes **γ=0.3** and 400 epochs… the regime mapping sweeps up to 4000 epochs."* | ⛔ **MUST-FIX.** The γ reconciliation is deleted: the paper now runs §3 at **γ=0** and App B.2 at **γ=0.02** with **no explanation**, and `langevin_noise=legacy` is deleted from A.2 and A.4. Charter C-7 on disk: *"apparent contradictions between differently-flagged runs must be **impossible to construct**."* Two are now constructible. ⚠ Also: base said *"γ/friction 0.3 **at write**"*; pj drops "at write", so `γ=0.3` now reads as the §2 governed-map damping during §4.1 retrieval — which it is not. |
| C-m | **exactness upgrade** | *"the pre-jump state is exactly recoverable, $q_{\rm in}=q_{\rm out}-\Delta$ (**max err $2.2\times10^{-8}$**)"* | *"the pre-jump state remains **exactly recoverable** ($q_{\rm in}=q_{\rm out}-\Delta$)"* | ⛔ **WIDER** (A-19). A measured bound became unqualified exactness. |
| C-n | **evidence deleted, claim kept** | App E lists **six** checks, two of which do **not** match to machine precision: `(D) ζ*=0.2356 vs measured 0.27` and `(F) re-absorption 56 vs predicted 49.5` | App E lists four, both discrepant checks removed; (F) replaced by *"the governor re-absorption mechanics **accurately tracked** the theoretical exponential injections **across all swept rapidities**"* — while §1 still says *"we state each certificate as a theorem and **verify it to machine precision**"* | ⛔⛔ **WIDER — the single most quotable defect in the document.** In the base, "machine precision" coexisted with a printed 13 % and 15 % discrepancy; the reader could check. In pj the discrepancies are gone and the prose asserts accuracy. **This is not an omission; it is a claim the deleted evidence contradicts.** |

### 4.1 Forbidden-form sweep — positive-controlled

Method: `/usr/bin/grep -o -i -- "<pat>" pj_sub.tex | wc -l` (never `grep -c`), per-file, every hit read in context.

| forbidden form | hits | adjudication |
|---|---|---|
| *"beats feedforward via test-time compute"* / `cannot draw` / `structurally cannot` / `test-time compute beats` / `outperforms the feedforward` / `beats the matched-compute` | **0/0/0/0/0/0** | ✅ CM-23(b)'s retracted absolute-dominance reading does not appear in any form. The only feedforward comparison is the §4.4 **tie**, correctly worded. |
| the anytime curve as a **uniqueness** claim (`unique` / `uniquely` / `only … anytime`) | **0/0/0** | ✅ no uniqueness claim. ⚠ **but** the base's *explicit* disclaimer is deleted (*"**no uniqueness is claimed for the anytime shape**: that figure is an occupied venue (deep-equilibrium models, energy-based transformers, recurrent-depth architectures)"*), together with the flat-curve C-6 paragraph and its oracle-addressing control (`0.0223→0.8219→0.8711` vs shipped-read flat `0.0004`). Compliant, but the guard is gone. |
| *"the anytime read wins"* / `wins` | **0 / 1** | ✅ the single `wins` is *"it ties rather than **wins**"*. |
| *"9–10× savings vs Hopfield"* (`9--10`, `9-10`, `savings vs Hopfield`, `cheaper than Hopfield`) | **0/0/0/0** | ✅ (control: `9.9` present, 4×, each intra-CLU-adjacent). ⚠ registry note: `claims_matrix.md` CM-8 records the intra-CLU figure as **"6–10×"**, not the task file's "9–10×"; the paper quotes the per-cell `9.9/9.5/6.2×`, consistent with both. |
| energy-as-superior-confidence/routing (`better confidence`, `energy is a better`, `superior routing`, `superior energy`) | **0/0/0/0** | ✅ and **actively disclaimed** three times: contribution 4 (*"rather than baseline energy signal superiority"*), §4.1 pt-1 (*"we do not assert that the CLU possesses an inherently superior energy signal"*), App D (*"the raw energy signal adds negligible predictive value over a standard readout margin"*). ⛔ **but** the disclaimers lost their number — `ΔAUROC ∈[−0.004,+0.024]` is deleted (`0.004` **pj=0/base=5**) — and §4.2's disclaimer acquired two hedges (C-j). |
| `SOTA` / `state of the art` / `leaderboard` (as a claim) | **0/0/1** | ✅ the single `leaderboard` is *"rather than claiming a general **leaderboard** advantage."* |
| `dominat*` | **1** | ✅ *"they do not universally **dominate** simpler neural routing heuristics or baseline Hopfield retrievers."* |
| **Positive controls (all must be > 0):** `ties` 13 · `Hopfield` 32 · `intra-CLU` 4 · `detJ` 29 · `wormhole` 29 · `noise` 13 · `0.36` 3 · `449` 2 · `400-epoch` 2 · `oracle` 8 | ✅ | grep is live on this file; every zero above is a true negative. |

**Sweep verdict: the forbidden-form register is CLEAN. Not one forbidden claim, hedged or implied, appears in `pj_sub.tex`.** Every defect in this report is an omission, a scope deletion, or an evidence deletion — never a prohibited assertion.

---

## 5. Deliverable 6 — the structural map (report, do not judge)

| base | pj_sub | status |
|---|---|---|
| §1 Introduction · §2 Setup · §3 Certificate stack · §4 Learned memories · §5 Position | §1–§5, same order | ✅ 5 main sections retained |
| §3.1 · §3.2 · §3.2.1 · **§3.3 "Why a prior null does not bear on this claim"** | §3.1 · §3.2 · §3.2.1 | ⛔ **§3.3 DELETED from main text.** Its content (the N1 squeeze-retry null: *"tested selection among already-reachable attractors, not crossing to unreachable"*) survives **only as App D's first sentence.** Under C-10 (*"short-paper reviewers aren't mandated to read appendices"*) the "didn't your own retries fail?" preemption is effectively lost. |
| §4.1 · §4.2 · §4.3 · §4.4 (*"beyond the three pillars: **it ties**"*) | §4.1 · §4.2 · §4.3 · §4.4 | 4 subsections retained; **both concession-bearing headings renamed**; lead-in still says "three applications" |
| App A.1–A.5 + **"Cross-section reproducibility note (C-7)"** | App A.1–A.5 + a 3-sentence note | ✅ all five tables retained; ⛔ **rows deleted throughout** (see F-4); ⛔ the C-7 note gutted |
| App B.1, B.2 (+ **Fig. 4 `fig4_bibo.png`**) | App B.1, B.2 | ⛔ figure + 4 of 6 B.2 discussion items deleted |
| App C.1 (+ **C.1.b latch-payoff full grid + dim×seed replication**), C.2, **C.3 (Hopfield iteration parity)**, C.4 (a/b/c) | C.1, C.2, **C.3 = base C.4.a + frontier prose**, **C.4 = base C.4.c ρ-rows only** | ⛔ **C.1.b DELETED** · ⛔ **base C.3 (iteration parity) DELETED ENTIRELY** · ⛔ **C.4.b table DELETED** (prose only) · ⛔ **C.4.c σ-table DELETED, ρ=0.5 rows DELETED** · ⚠ **subsection letters silently re-used** (pj's "C.3" is not the base's C.3) — a citation hazard if any note or rebuttal refers to "App. C.3" |
| App D Negatives (N1, N2, **N2b**, N3, **N24**, **N30**, **N31**, **N23**) + 3 fine-print/scope paragraphs | App D — one paragraph | ⛔ `N30` `N31` `N23` `N24` **0 hits each** (base 2/2/1/3). **N30 (the V(data)-anchor does not transfer to memory fidelity) is deleted from the whole document**, incl. its §4.3 mention. **N23 (smooth z-gate mis-routes ~10–12 %) is gone entirely.** ⛔ the *Payoff A/B fine-print* and *Payoff scope* paragraphs deleted. **C-9 exposure.** |
| App E Analytic verifications — 6 checks + provenance | App E — 4 checks, no provenance | ⛔ see C-n / A-21 |
| App F F.1–F.6 + grade line | App F — 4 paragraphs | ⛔ grade line, **F.5** (*"the parsimony argument is weak"*), **F.6** (the specified discriminating experiment + *"No trained-CLU-checkpoint result is claimed"*) deleted |
| **4 figures** (fig1, fig2, **fig_frontier_clean**, **fig4_bibo**) | **2 figures** (fig1, fig2) | ⛔ 2 of 4 deleted. Both source PNGs are still on disk in `figs/`. |
| References — all cited in body | References — **6 of 17 cited nowhere in the body** | ⛔ `Lieb & Robinson`, `Duane`, `Neal`, `Roberts & Tweedie`, `Wales & Doye`, `Geifman` — **0 body hits each** (base 3/4/4/3/1/1). Cause: the **"Prior-art honesty" paragraph was deleted** (`Prior-art` **pj=0/base=1**, `novelty` **0/1**, `Mermin` **0/1**), which was where all six were cited and where the paper scoped its own novelty: *"we cite these for the bound and lineage and claim **only the design consequences** — not novelty of nonlocality, stochastic escape, MCMC itself, or the causal bound."* |

---

## 6. Reviewer-hat attack pass — the register's composites against THIS draft, plus fresh ones

| ID | attack | lands? | why |
|---|---|---|---|
| **G1** (*unit test on a testbed built to satisfy the theory*) | **LANDS HARDER THAN ON THE BASE.** The base pre-empted it with four "Reporting grade" labels, six bracketed contribution tags, a "Scope (stated, not buried)" paragraph and a "Payoff scope" appendix paragraph. **pj has none of the four.** The §3 scope list survives in one sentence; everything else that marked designed-vs-learned is gone. |
| **G2** (*which component buys what*) | **LANDS, newly.** The base's C.2 routing table carried a **`calibrated (energy head)` ablation arm** (`0.860±0.122` / `0.677±0.174` @ `1.34e8`/`1.49e8`) between the raw energy gate and the learned router. **pj deleted that row**, so the paper now shows only the two endpoints. Compounded: the CM-23(g) mechanism controls (kick / ensemble-of-restarts / ungated) — the program's cleanest *which-component* evidence — are deleted (B10c). |
| **G3** (*toy scale*) | **LANDS.** dim 2/4, kv ≤ 96, N ≤ 8, laptop-CPU — unchanged from the base, which is fine. What changed is that the base **said so in one place, in its own voice** ("No claim here is at scale"), and pj deleted that paragraph and the abstract's qualifiers. **G3 now lands with nothing to absorb it.** |
| **G5** (*certificate fine print*) | **PARTIALLY REPELLED.** C-6's three fine prints survive intact and main-text-adjacent (B6) — genuinely good. **But the fourth, ECE ≈ 0.100 ± 0.021, is deleted** (B7), which is the exact item the register's P20 names. And App E now claims uniform machine precision after deleting the two checks that were not (C-n). |
| **G6** (*foundational-paper falsifications*) | **NEUTRAL/COMPLIANT.** C-1 is honoured: no audit-confession paragraph; J&P 2026 cited for the primitive's introduction only; no legacy number load-bearing. |
| **M2/M3** (*salami / de-anon optics*) | **STANDING, unchanged by the edit — but worth the Head's eye.** `[AUTHORS PLACEHOLDER]` sits above *"Our reference memory is the Causal Learning Unit (CLU), **introduced as CHLU in Jawahar & Pierini (2026)**"* plus a self-citation to *"a companion theoretical note (Anonymous, 2026)"*. Base is identical, so **not a condensation defect** — but at a double-blind venue the first-person link to a named prior work is a de-anonymisation vector, and it is now the *only* remaining pointer to the program's scope (the prior-art paragraph that contextualised it is gone). |
| **NEW — F-A: "your appendix shows the wins and hides the losses."** | **LANDS HARD.** pj's only stress-axis table (C.4) contains **three rows, two of them CLU-positive**; the six-row table where CLU loses every cell was deleted (B3). The `closes?` column and the `6/15` tally went with it. A referee who diffs the two builds — or who simply notices that the "dominant negative" has prose but no data — will say this out loud. |
| **NEW — F-B: "your verification appendix deleted its own two failures."** | **LANDS HARDEST.** See C-n. `ζ*=0.2356` vs measured `0.27` (+15 %) and re-absorption `56` vs predicted `49.5` (+13 %) removed; replacement prose asserts accuracy; §1 still promises "machine precision." |
| **NEW — F-C: "the premise of your cost argument is unevidenced."** | **LANDS.** *"Hopfield reaches its ceiling in ≈1 matvec"* underwrites the entire §4.3 cost story, and App C.3's iteration-parity table (its only evidence) was deleted while App A.4 still advertises the sweep `{1,2,3,5,10}` (A-13). |
| **NEW — F-D: "your FLOP comparison is not reproducible."** | **LANDS.** The FLOPs accounting model rows are deleted from A.3 (A-11), so `8.81e7` vs `1.18e8` vs `2.94e8` cannot be checked. |
| **NEW — F-E: "§4 promises three applications and delivers four."** | **LANDS (craft).** C-a. |
| **NEW — F-F: "you cite six papers you never discuss."** | **LANDS (craft).** §5's structural map, last row. |
| **NEW — F-G: "at kv24/kv32 your escalatable memory is 55 %/29 % accurate and you don't say so."** | **LANDS.** The base printed `0.547±0.039` and `0.286±0.037` beside the `1.57×`/`1.14×` savings; pj prints only the ratios (A-6/F-9). The honest reading — *the rationing gate saves most where the memory is best, and the memory is poor at the harder levels* — is no longer available to the reader. |

---

## 7. Triaged findings — itemised

### MUST-FIX (blocks submission)

| # | location | finding |
|---|---|---|
| **F-0** | whole file | **The object changed between scoping and audit** (md5 `301ecdf5…`→`bb98439d…`, 240→410 lines). Three Advisor pre-flight zero-hit measurements refuted; the built PDF is stale by 21 minutes and the page budget is therefore **unverified**. **Re-baseline before any further pass; re-build before any page ruling.** |
| **F-1** | App E | **The two analytic checks that did not match to machine precision were deleted and replaced by an accuracy assertion**, while §1 still claims verification "to machine precision" (`ζ*=0.2356` vs `0.27`; `56` vs `49.5`). App E also lost all provenance (`checks.py`, float64, seed 0, potential + ε). C-7 + honesty. |
| **F-2** | abstract | **All three concessions deleted**: Hopfield-cheaper-and-more-noise-robust → *"mapping a settled performance regime"*; energy-gating-**loses** → *"evaluating scaling boundaries"*; and the scale qualifiers (dim 2/4, 5 seeds, oracle placement, laptop-CPU) simply removed. B3/B4/B5/C-5. |
| **F-3** | §4 lead | **The mandatory scoreboard sentence is absent** (B8; Head-ruled IN 2026-08-26; CM-23/CM-31/CM-37 read on disk). The §4 "Reporting grade: evidence" label is absent with it. |
| **F-4** | §5 | **The "Scope (stated, not buried)" paragraph is deleted**, taking with it *"No claim here is at scale, and none uses learned placement"* **and B9's substrate-scope sentence** (Head-ruled IN). C-5. |
| **F-5** | §5 + App F | **Both MCMC grade statements deleted** — four design rules now read as CLU results (C-c). |
| **F-6** | App C.4 / §4.3 | **The noise-wall σ-table (6 cells) deleted while the CLU-favourable ρ-table was kept**; ρ=0.5 rows, the `closes?` criterion, and the `6/15` / `6/9` / `0/6` tallies deleted. Also **"kv128 only ties (Δ+0.004)" deleted**, removing the stated ceiling of the epoch-budget argument. |
| **F-7** | App C.3 | **n=3 frontier numbers (`0.975`, `Δ+0.03`, `1.00→0.89`) printed under an "8 pooled seeds" heading**, `±` dropped; A.4's per-item seed schedule (5/3/2) flattened to a uniform 5 (A-1/A-3). |
| **F-8** | §4.1 pt-3 | **ECE `0.100 ± 0.021` absent**, together with *"within the stated probe-to-deployment scope"*, `30/30`, and measured risk `0.030`. **Direct C-6 / register-P20 violation** (B7). |
| **F-9** | §4.1 pt-2 / App C.3 | **The evidence for the convergence retraction was deleted** — App C.4.a's `full-budget acc` column is gone and §4.1 carries no cross-reference (B2). Also `0.547±0.039` / `0.286±0.037` deleted (F-G). |
| **F-10** | §5 ¶2 | **Trilemma stated with no substrate, no seeds, no attribution, and without CM-23(y)/N119's mandatory *"neither fix may be described as available"*** (B10b). |
| **F-11** | App A.3 | **FLOPs accounting model deleted** ⇒ no FLOP number in the paper is reproducible (A-11). C-7. |
| **F-12** | App A closing note | **The γ=0 vs γ=0.02 reconciliation deleted**; `langevin_noise=legacy` deleted from A.2 and A.4; `γ/friction 0.3 **at write**` → `γ=0.3`. **C-7's "impossible to construct" bar fails** (C-l). |
| **F-13** | §3.2.1 | *"exactly recoverable"* with the `2.2e-8` bound deleted (A-19/C-m). |
| **F-14** | §4.3 / App C | **App C.3 (Hopfield iteration parity) deleted**, leaving *"Hopfield reaches its ceiling in ≈1 matvec"* — the premise of the cost story — unevidenced (A-13). |

### SHOULD-FIX

| # | location | finding |
|---|---|---|
| S-1 | §4.2 heading, §4.4 heading | Both concession-bearing headings replaced with neutral labels (B5, B10a′). |
| S-2 | §4.4 | The whole CM-23(r) scope clause, the CIFAR counter-cell (`−0.055±0.019`), the p=0.5 saturation, `1.40±0.20×`, `3/6`, `6/6` deleted (A-16/B10a′). |
| S-3 | contributions | All six grade tags deleted; *"three of our six contributions are boundaries or negatives"* and *"§4 is the paper's honest perimeter"* deleted; contribution 4 lost "5 seeds, MQAR vocab-256, laptop" (B1, A-4). |
| S-4 | §4.2 | *"in this regime"* + *"raw kinetic"* hedges added to a CM-7 prohibition; the base's own scope caveat (*"a harder band is the untested fair stress test"*) deleted (C-j). |
| S-5 | App D | `N30`, `N31`, `N23`, `N24` labels gone; N30 and N23 gone from the document entirely; ΔAUROC `[−0.004,+0.024]` gone; the Payoff-A/B fine print and Payoff-scope paragraphs gone. **C-9 exposure.** |
| S-6 | §3.3 | Deleted from main text; the N1 preemption survives only in App D (structural map). |
| S-7 | App C.2 | The `calibrated (energy head)` ablation arm deleted (G2). |
| S-8 | App C.1 | C.1.b (latch-payoff full grid) and the dim×seed replication (`0.0803/0.0679/0.0533/0.0448`) deleted ⇒ *"across dimensions 2 and 4"* is now unevidenced. |
| S-9 | App B.2 | 4 of 6 discussion items deleted, incl. *"the receipt does not make an unsafe exit safe — it **refuses** it"* (`r*=0.09`), the `6/6` prediction, the bounded-arm `1.000` control, and the *"the coercive screen is an **oracle** too"* scope. `1.91`→`≈1.9` (A-2). |
| S-10 | §1 | The reference-unit objective clause reversed and self-tangled (C-k). |
| S-11 | §4 lead | "three applications", four subsections (C-a). |
| S-12 | App A.2 | The Hopfield-transfer provenance row deleted ⇒ `0.18→0.88` has no source (A-8). |
| S-13 | figures | `fig_frontier_clean.png` and `fig4_bibo.png` dropped though both are on disk; the paper's headline figure (Fig. 1) is retained and is still the right one. |

### NICE

N-1 §3.2 heading lost its repaired "priced" wording though the body kept it at five sites (C-g). N-2 `d≈3.2` observed edge / `d≲3.6` / `≈55 H` anchors deleted. N-3 `Prop-12`, `Prop-A2`, `App. D N31` cross-reference labels dropped. N-4 App F lost `L1=0.0095`, `T_eff 1.0→0.61`, `D=1.29e-3` vs `1.25e-3`, `N_erode`, and F.2(a)'s *"σ* is a proposal-tuning scale, not a correctness condition"*. N-5 `\S4.1` step counts `@629±60` vs `@3000` dropped. N-6 pj's appendix letters re-use base letters with different content (C.3/C.4) — a citation hazard. N-7 six uncited references. N-8 `[WORKING TITLE]` / `[AUTHORS PLACEHOLDER]` remain (C-10-legal at drafting; the Head's call at submission).

---

## 8. Missing-experiment / wiring list — for the Hub

**⛔ Category 1 — EXISTS in the base or in `.claude/outputs/*`, deleted from `pj_sub.tex` (wiring; not new science; the Head rules on restoration, per Prohibition 2 I do not propose page trades):**

1. App C.3 Hopfield iteration-parity table (`regime-remap-2000ep` Item 3, 3 seeds) — underwrites "≈1 matvec".
2. App C.4.a `full-budget acc` column — underwrites B2's convergence retraction.
3. App C.4.b epoch-frontier table (n=3) + `fig_frontier_clean.png` + the non-monotone fidelity-dip negative (kv96 `0.39→0.24→0.97`; kv128 `0.40→0.09→0.71`).
4. App C.4.c σ-axis table (6 cells) + ρ=0.5 rows + `closes?` + the `6/15`/`6/9`/`0/6` tallies.
5. App C.1.b latch-payoff grid + dim×seed replication.
6. App C.2 `calibrated (energy head)` arm.
7. `fig4_bibo.png`.
8. App E checks (B), (D), (F) + provenance line.
9. ECE `0.100±0.021`, measured risk `0.030`, `30/30`.
10. kv24/kv32 accuracies `0.547±0.039` / `0.286±0.037` and step counts `629±60`/`1919±82`/`2636±127`.
11. App A.3 FLOPs accounting rows + `n_seeds=5 (default 2)`.
12. App A.2 Hopfield-transfer provenance row; `langevin_noise=legacy` in A.2/A.4; A.4 per-item seed schedule; A.5 `dims`, JAX, tests, bit-identical-reproduction rows.
13. The C-7 cross-section note's γ reconciliation.
14. N23, N24, N30, N31 + ΔAUROC range + the Payoff-A/B fine print + Payoff scope.
15. CM-23(g)/N90 directed-relaunch result + its three companions + the 5-seed NN-gap ranges.
16. CM-23(y)/N119's *"neither fix may be described as available"*, `r=−0.85`, and the trilemma's substrate/seed attribution.
17. CM-23(r)'s scope clause, the CIFAR counter-cell, p=0.5 saturation, `1.40±0.20×`, `3/6`, `6/6`.
18. The scoreboard sentence (B8) and the substrate-scope sentence (B9).
19. §5's "Scope (stated, not buried)" paragraph.
20. App F grade line + F.5 + F.6 + its four verification numbers.
21. §3.3 (the N1 preemption) in main text.
22. The "Prior-art honesty" paragraph (and with it, six now-uncited references).
23. `2.2e-8` reconstruction bound.
24. `kv128 only ties (Δ+0.004)`.
25. All six contribution grade tags + "three of our six contributions are boundaries or negatives".
26. The base's §4.2 scope caveat (*"a harder band is the untested fair stress test"*).

**⚠ Category 2 — GENUINELY MISSING EXPERIMENTS (Hub task candidates; unchanged by the condensation, but pj deleted the sentences that named them as owed):**

27. **Noise-robustness cure for the rationing gate** (noise-aware `τ` / longer relax budget / denoising init). The base's App D named this as future work; pj deletes the naming while keeping the negative. *This is the paper's largest open flank and its reviewer-obvious next experiment.*
28. **Harder-band routing stress test** (non-linearly-separable cues) — CM-7's own caveat says the router's dominance is driven by the cue geometry; untested.
29. **Sub-level-set estimator for a learned, non-coercive `V_θ`** — retained in pj's future work ✓; still unrun.
30. **γ>0 governor re-absorption sweep** — retained ✓; still unrun (and App E's deleted `56 vs 49.5` was the only datum on it).
31. **App F.6's specified discriminating experiment** — certified mixture kernel at γ=0 with coset projection, on a **trained CLU checkpoint**, latch-erosion decay curve as the money plot. pj keeps the bullet, deletes the specification *and* the "no trained-CLU-checkpoint result is claimed" guard.
32. **Multi-seed for CM-23(r)'s τ sub-claim** — the matrix records it as **1 seed**, "multi-seed owed"; never in either draft.
33. **dim-4 evidence** for §3.2.1 — exists as base C.1.b; without it, *"across dimensions 2 and 4"* is an unevidenced plural.

---

## 9. The three sentences a hostile reviewer would quote

> **1.** *"Appendix E asserts that 'the governor re-absorption mechanics accurately tracked the theoretical exponential injections across all swept rapidities' and the introduction promises verification 'to machine precision' — yet the two checks in this same battery that disagreed with theory by 13 % and 15 % (re-absorption 56 steps vs a predicted 49.5; a squeeze threshold ζ\*=0.2356 vs a measured 0.27) appear nowhere in the paper, and Appendix E carries no provenance at all."*

> **2.** *"The authors call the noise wall their dominant negative, but the only stress-axis table in the paper is the one with two CLU-favourable rows; the six-cell table in which the gate loses every cell — along with the 6/15 close tally, the 'closes?' criterion, and the σ-axis fidelity column — is not printed, so the paper's sharpest result against itself exists only as a sentence."*

> **3.** *"Section 4.3's cost story rests entirely on 'Hopfield reaches its ceiling in approximately one matrix-vector multiplication', Section 4.2's headline rests on a FLOP count, and Section 4.1's convergence retraction rests on the gate matching the full-budget baseline — and the iteration-parity sweep, the FLOP accounting model, and the full-budget accuracy column have each been removed from the appendix that Appendix A still advertises."*

---

## 10. How I verified

- `md5` on `pj_sub.tex` / `submission.tex` at pass start and pass end (§0). `ls -lT` for mtimes. `ls -la` + `md5` manifest for `.claude/papers/v1-short/**`.
- `wc -l -w -c` on both objects; survival ratio computed as `6450/12331 = 52.3 %`.
- Numeric-token audit: Python `re.findall(r'\d+(?:\.\d+)?')` over both files, `Counter` set-difference → 210 distinct tokens, 2 orphans; separate `±`-pair extraction (`(\d+)[^\d]{0,12}\\pm[^\d]{0,12}(\d+)`) with a base↔pj cross-check → **0 mismatched error bars**, 10 bare-value candidates, 1 genuine (`0.975`).
- All greps via **`/usr/bin/grep`** (the shell `grep` here is `ugrep 7.5.0` and silently exits 0 on bounded-context patterns over long `.tex` lines). Counts via `grep -o … | wc -l`, **never `grep -c`**. Every zero-hit sweep paired with a positive control on the same file (§4.1). Every `ties` hit read in context (11 of 13 are false friends inside `properties`/`capacities`/`quantities`).
- Registry read on disk at the moment of use, per-file (directory-level grep over `.claude/` returns nothing — gitignored): `claims_matrix.md` (CM-3 l.561, CM-7 l.565, CM-8 l.566, CM-23 at byte 270321 — extracted and read in full), `critique_register.md` (G1/G2/G3/G5/G6, P15–P20, M1–M4), `philosophy-synthesis.md` l.581–600 (Positioning Charter C-1…C-10), `AGENT_PROTOCOL.md`.
- Section maps via `/usr/bin/grep -n "^\\\\section\|^\\\\subsection\|^\\\\subsubsection\|^\\\\appendix\|subsection\*"` on both files; figure inventory via `includegraphics` extraction + `ls figs/`.
- **PDF page counts could not be measured** (object streams compressed; `mdls` returns null; no `pdfinfo`/`pdftk` on this machine) — and the build is stale regardless (§0.3).

## Prohibitions honoured

1. **No paper file edited** — md5 identical at both ends of the pass; `submission.tex` and `papers/v1-short/**` byte-untouched.
2. **No page cuts or restorations proposed** — the wiring list is an inventory of what left, not a layout recommendation.
3. **C-8 hermetic** — V2's and V5's drafts were not opened; the only files read outside V1's directory are the shared registry documents.
4. **C3-era numbers treated as PENDING** — none quoted; the one C3-adjacent observation (CM-8 records "6–10×", the task file says "9–10×") is reported as a registry note, not used to score the draft.
5. **No intent judged** — every item is reported PRESENT / PARTIAL / ABSENT with its consequence. **The ruling on every cut is the Head's.**

## Proposed handover updates (for the Hub)

- **`pj_sub.tex` is a live-edited object.** Any future audit task must pin the md5 **at spawn time**, and the spoke must abort-and-report if it differs. This pass's three refuted pre-flight measurements are all attributable to that gap.
- **`v1-pj-fidelity` findings F-0…F-14 need an owner** (protocol §5 reconciliation-list corollary). 26 of 33 list items are pure wiring from the base; only items 27–33 are experiment candidates.
- **Appendix-letter collision:** pj's `C.3`/`C.4` are not the base's `C.3`/`C.4`. Any cross-document reference to "App. C.3" is now ambiguous between the two builds.
- **Registry discrepancy for the Hub to adjudicate:** `claims_matrix.md` CM-8 records the intra-CLU savings as **"6–10×"**; the `v1-pj-fidelity` task file's forbidden-form list says **"9–10× savings vs Hopfield"**. The draft quotes per-cell `9.9/9.5/6.2×` and is compliant under either, but the two ledger phrasings should be reconciled.
