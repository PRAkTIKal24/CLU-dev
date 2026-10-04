# pj-fidelity-v5 — doc-curator report

**Task + acceptance criterion:** fidelity audit of the Head's `NIPSsubmission/v5-palm/pj_sub.tex` against the Advisor-accepted base `submission.tex` (Part A: numbers + claim equivalence; Part B: what was lost, against the `v5-referee-v02` do-not-cut list), and render `pj_sub.pdf` (Part C).
**Status: partial.** Parts A and B are complete. **Part C is BLOCKED: this session has no shell/Bash tool, so `pdflatex` could not be run and `md5` could not be computed.** Everything Part C needs except the actual invocation has been prepared (§C).

**DIAL DECLARATION (echoed):** none — verification/audit pass; no performance claim; no laundering control applies to this report.

> ⛔ **THE ABSOLUTE CONSTRAINT HELD.** `pj_sub.tex` was opened read-only. No `Edit`/`Write` call was ever issued against it. The only file I created inside `v5-palm/` is the new `pj_sub_buildcopy.tex`. **I cannot print an md5 (no shell); the byte-identity claim rests on tool provenance, not on a hash — the Hub/Head should run `md5 pj_sub.tex` against the pre-pass value to close this formally.**

> ⚠ **DOWNSTREAM RECONCILIATION LIST — READ FIRST (protocol §5 corollary; needs a named owner):**
> 1. **A-C1 — `pj_sub.tex` §5 states "Certified removal is proven strictly at the store level."** The base states the exact opposite (*"we claim no certified $(\varepsilon,\delta)$ unlearning"*). This inverts a standing never-quote (N118 / CM-25(f) / charter: "certified" only in denial or literature-description form). **Highest-consequence finding in this audit.** Owner: Head (his file); Hub to route.
> 2. **A-C2 — the three stated deletion conditions and the recency exclusion appear NOWHERE in `pj_sub.tex`** (`budget >= n_cells`, `leak = 0`, priority/attribute eviction; recency excluded) — 0 hits, abstract and §3.3 both. N118 bans exactly this unqualified form. Owner: Head; Hub to route.
> 3. **A-C3 — the file will not compile as written**: 8 unescaped text-mode `&` and 15 table rows terminated with a single `\` instead of `\\` (measured, §C). These are edits the Head owes his own file.
> 4. **A-C4 — the score sentence ("External benchmarks won on their own headline metric = ZERO") is absent**, and so is the deletion section's trivial-substitute laundering control ("a flat table deletes exactly by construction"). Owner: Head.

---

## 0. Instrument and method

- **Files:** `pj_sub.tex` (192 lines, 2,684 words per Add.53) vs `submission.tex` (371 lines, 8,993 words). Every numeric token of `pj_sub.tex` was walked in file order against the base; base line numbers cited.
- **Sweeps: per-file, positive-controlled** (⚠ memory: directory-level Grep over `.claude/` returns false negatives — a directory sweep for `pj_sub|v5-palm` returned "no files found" while the files exist. Every sweep below is per-file.)
  - **Positive control:** combined pattern (`107.77|Blelloch|gamma|deletion|…`) on `pj_sub.tex` ⇒ **28 matching lines ⇒ instrument LIVE.**
- **Not done:** no TeX run, no md5, no PDF inspection (no shell). No number was re-derived from raw JSON; the base `submission.tex` is treated as the source of truth per the task, with `v5-referee-v02` as the do-not-cut checklist and `BUILD-NOTE.md` §6 as the prior 27/27 verification of that list on the base.

---

# PART A — is what survived FAITHFUL?

## A.1 — Numbers: the headline answer

**⭐ No number in `pj_sub.tex` lacks an ancestor in `submission.tex`.** Every numeric token — value, precision, ± — that survives is transcribed **exactly**. I found **zero digit changes, zero dropped ±, zero unit changes, zero rounding changes**. On the Head's literal question ("has the paraphrasing misrepresented any fact *numerically*"), the answer is **no**.

**The defect is not in the digits, it is in the scope riders attached to them.** Eleven numbers survived with a qualifier stripped; the number is right and its domain of validity is now larger than measured. Those are tabulated in A.2 and re-ranked in Part B.

### A.1.1 Full numeric ledger (pj token → base ancestor)

| # | `pj_sub.tex` token | site | base ancestor (`submission.tex` line) | verdict |
|---|---|---|---|---|
| 1 | $\gamma_{\rm crit}=2\varepsilon\mu$ | abs, §1, §3.1 | L32, L61 | exact |
| 2 | $\approx11$ orders / "eleven orders" in $\mu^2$ | abs, §3.1 | L32 ("$\approx11$"), L61 | exact ⚠ see A.2-N2 |
| 3 | $107.77\pm4.78\times$ | abs, §3.2 | L32, L75, L241 | exact |
| 4 | $13.28\pm0.12\times$ | §3.2 | L75, L241 | exact |
| 5 | $106.1\pm5.0\times$ | §3.2 | L32, L75, L286 | exact ⚠ see A.2-C4 |
| 6 | dim 4, hidden 64 | §1, §3.1, §5 | L32, L61 | exact |
| 7 | $\varepsilon=0.05$ | §3.1 | L61 | exact |
| 8 | log-slopes $-1.006$, $+1.23$–$+1.27$ | §3.1 | L61, L128 | exact |
| 9 | Hessian $\mu^2\approx10^{-15}$ | §3.1 | L61 | exact |
| 10 | argmin $0.902\pm0.003\times\gamma_{\rm crit}$ | §3.1 | L61 | exact |
| 11 | $-1.0020\pm0.0003$ / $+1.116\pm0.011$ | §3.1 | L61 | exact |
| 12 | $\mu^2\in[1.7\times10^{-12},\,7\times10^{-2}]$ | §3.1 | L61 | exact ⚠ see B-1 |
| 13 | $0.35\%$; $0.9001\pm0.0052$ vs $0.9032\pm0.0027$ | §3.1, App B | L32, L69, L207 | exact |
| 14 | drift $\le4.9\times10^{-12}$ rad / 200k steps | §3.2, App A | L73, L128 | exact (units kept) |
| 15 | $D_\theta=\varepsilon T(2-\gamma)/(2F^2\gamma)$ | §3.2 | L73 | exact |
| 16 | $\gamma:0.05\to0.2$, $3.77\pm0.23\times$ | §3.2 | L73, L128 | exact ⚠ see A.2-N1 |
| 17 | $T_{\rm local}=1.26\times10^{-4}$ vs $10^{-3}$ | §3.2, App C | L75, L241 | exact |
| 18 | $7.94\times$ refrigerator | App C | L241 | exact |
| 19 | hop fraction up to $43.0\%$ → $0.0000$ | §3.2 | L75 ("5.5/43.0/2.4%… to 0.0000, 3/3 seeds") | exact, narrowed to the max ⚠ A.2-C5 |
| 20 | 3-dimensional datastore | §3.3 | L32, L302 | exact |
| 21 | $0.29\times$–$1.71\times$ lattice capacity | §3.3 | L79 | exact |
| 22 | AUC $0.99985$ | §3.3 | L79 | exact ⚠ "at full load" dropped |
| 23 | AUC $0.5000\pm0.0000$, six statistics | §3.3 | L79 | exact |
| 24 | $1.000$ / $0.983$ (TTL vs decay, exact adversary) | §3.3 | L81, L315 | exact **and correctly paired** |
| 25 | retention $0.832$ at $A=0.051$ | §3.3 | L81 | exact ⚠ placement scope dropped |
| 26 | $R_{50}$ $1.146\to0.752$ | §3.3 | L81, L321 | exact ⚠ comparator dropped |
| 27 | designed $n=5$ / emergent $n=3$ | §5 | L61, L85 | exact |
| 28 | $10^3$ items | §5 | L85 | exact |
| 29 | $\hat D_\theta/D_\theta^{\rm pred}=1.0068\pm0.0219$ | App A | L73, L128 | exact ⚠ "25 cells, seed 44" dropped |
| 30 | $+0.9552\pm0.0422$; $+0.955\pm0.042$ | App A | L128, L136 | exact |
| 31 | $-0.956$ to $-0.979$ | App A | L128, L136 | exact |
| 32 | full emergent 4-instrument table (3 rows × 13 values) | App B | L179–181 | **all 39 values exact** ⚠ see B-4 |
| 33 | $\gamma_{\rm crit}$ $0.02334$/$0.01424$/$0.02265$ | App B | L179–181 | exact |
| 34 | vault table: $0.050$/$0.145$/$0.525$; $1.00000$/$0.36249$/$0.12591$; $0.99815\pm0.00257$/$0.36326\pm0.00067$/$0.12592\pm0.00055$; $0.9981$/$1.0021$/$1.0001$ | App C | L257, L258, L261 | **all exact**; 3 of 6 source rows kept |
| 35 | minimum separation $1.540000$ | App D | L306 | exact ⚠ relabelled "placement limit" |
| 36 | adversary table $1.000$/$1.000$/$0.000$; $0.983$/$1.000$/$0.017$; $0.559$/$0.996$/$0.437$ | App D | L314–316 | **all exact** |
| 37 | $\mathrm{AUC}=1.00000$ (overflow, no waitlist) | App E | L306, L340 | exact |
| 38 | $0.995$ recency decay factor | §2 | L49 | exact ⚠ citation dropped |
| 39 | reference metadata (18(11):e1010716; 24(5):1091–1103; 48th; 42nd; 1995/2007/2021/2022) | Refs | L91, L92, L95, L97 | exact (DOIs/arXiv ids dropped) |

**Numbers present in the base and dropped from `pj_sub.tex` (not errors — omissions, see Part B):** $9.483\times10^{15}$ step bound · $1.7\times10^{-14}$ / $\le1.1\times10^{-15}$ latch floors · $T^\star\approx3\times10^{-3}$ · capacity $\approx1$–$1.6$ bits · $1-|\lambda_{\rm coset}|=1.06$–$2.96\times10^{-3}$ · $8.11\pm0.37$ and $7.942$ · $110.25\times$ · $86.97\pm2.94\times$ · $0.9998\pm0.0019$ · $0.2235$ · $23.39\pm10.06$ · $297.8\pm196.8\times$ · $0.4586\pm0.1181$ · scalar-control hops $0.73/10.2/0.26\%$ · $61/64$ vs $43/64$, $2.836$ moves/delete · $1.52\times$ on $R_{50}$ · TTL lookup radius $0.75$–$0.77$ · $\Delta=0.5$ rad and every $\ell_\theta/\Delta$ · instrument offsets $1.33/1.80/1.82\times$, CV $2.2$–$3.7\%$, rate ratio $0.995\pm0.003$ · 150 epochs · 25 cells / 10-of-10 conditions / 30-of-30 cells / 24 orders / 200 orders.

## A.2 — Claim equivalence, both texts quoted

Verdict vocabulary: **IDENTICAL / NARROWER (safe) / WIDER (⛔) / CHANGED IN KIND**.

### ⛔ C1 — "certified", inverted from denial to affirmation *(most serious claim finding)*
- **Base (L79):** *"This is a store-level guarantee only --- the frozen encoder and any residue of past writes in a learned landscape are separate channels, measured separately, and **we claim no certified $(\varepsilon,\delta)$ unlearning**"*; (L51) *"**We make none of those claims**: ours is a store-level bit-exactness statement with the encoder excluded."*
- **pj (L105):** *"**Deletion boundary definitions:** **Certified removal is proven strictly at the store level**; broader systemic task-level performance payoffs and amortization optimizations are excluded."*
- **Verdict: WIDER ⛔ / CHANGED IN KIND.** The base's denial has become an affirmative claim of *proven certified removal*. "Certified removal" is a term of art (Guo et al. 2020 §2 Eq. (1), an $\varepsilon$-condition) that the program has ruled may appear **only** in denial or literature-description form. It now appears in the paper's Limitations — the section a reviewer reads to check scope — as a positive result, and the sentence that would have denied it is gone. Compounding: the base's Guo/Ginart/Sekhari/Bourtoule framing is cut, so nothing in `pj_sub.tex` defines "certified", while `Bourtoule et al. (2021). Machine unlearning` sits uncited in the bibliography.

### ⛔ C2 — store-level deletion, stated without its three conditions or the recency exclusion
- **Base (abstract, L32):** *"store-level deletion is exact, the post-deletion store byte-identical at every load measured to the store that never held the item, **under three stated conditions (`budget >= n_cells`, `leak = 0`, priority/attribute eviction; recency eviction excluded)**, on a designed store of **dim 3, capacity 8--64, no learning**."*
- **pj (abstract, L32):** *"store-level deletion is exact and yields a post-deletion state that is byte-identical to a store that never held the item, **evaluated under explicit operational constraints on a designed datastore**."*
- **pj (§3.3, L94):** *"When evaluated on a designed, non-learned 3-dimensional datastore, store-level deletion achieves exact byte-for-byte state parity across all operational loads tested ($0.29\times$ to $1.71\times$ lattice capacity)."*
- **Verdict: WIDER ⛔.** "Explicit operational constraints" **names none of them**, and the recency exclusion — the one condition the program made *permanent* — appears nowhere in the file (per-file sweep: `recency` = 1 hit, and it is the unrelated "0.995 recency decay factor" of §2). N118 bans precisely the unqualified form. The store-level/encoder scope survives (§1 bullet 2, §3.3 opening) — that half is intact; the *eviction-regime* half is gone.

### ⛔ C3 — the lifecycle reads as **evaluated**, not shipped
- **Base (L41):** *"A three-state memory lifecycle (PROTECTED $\rightleftarrows$ ACTIVE $\to$ TRASH, **7/7 legs**, kill-conditions committed before the verbs, **one leg unexercised**) … **shipped as a mechanics demonstration on a toy synthetic build: no value or benchmark number is claimed, and none was run**."*
- **pj (§1, L51):** *"**We evaluate under** a three-state memory lifecycle implemented on a synthetic build."*
- **Verdict: CHANGED IN KIND ⛔.** "We evaluate under" makes the lifecycle an *experimental setting under which the paper's results were obtained* — it was never run as an evaluation, and the base's "no value or benchmark number is claimed, and none was run" is the sentence that prevented exactly this reading. Leg counts (7/7, one unexercised) and the state names are gone; the two riders are gone (Part B-6). The word "toy" is gone.

### ⛔ C4 — the emergent vault: the *laws* transferring has become *the architecture* transferring
- **Base (L75):** *"**Both laws transfer** to an emergent register (evidence, 3 emergent seeds at $T=4\times10^{-3}$ and $8\times10^{-3}$, both above $T^\star$): the refrigerator law holds at $0.9998\pm0.0019$ of prediction … and the $\hat D_\theta\propto\gamma_{\rm eff}^{-2}$ law at $1.016$--$1.103$, **a law-referenced vault of $106.1\pm5.0\times$** beside the designed $107.77\pm4.78\times$."* — and (L286) *"A measured/measured ratio over the same cells reads $297.8\pm196.8\times$ …; **it is recorded and is never the vault number**."*
- **pj (§3.2, L88):** *"**Translating this architecture to an emergent register yields** a corresponding law-referenced vault of $106.1\pm5.0\times$."*
- **Verdict: WIDER ⛔ (borderline).** The adjective "law-referenced" survives — credit — but nothing in `pj_sub.tex` defines it, both constituent law-fits are cut, the seed/temperature scope is cut, and "translating this architecture … yields" states a *transfer of the vault*. A reader takes $106.1\times$ as a measured emergent vault. The base is explicit that the directly-measured emergent ratio is $297.8\pm196.8\times$ and is **never** the vault number.

### ⛔ C5 — the confinement result lost its laundering control
- **Base (L75):** *"the fraction of states outside the register ($|\theta|>1$ rad) falls from $5.5/43.0/2.4\%$ with no hole to $0.0000$ inside it, 3/3 seeds, **while a scalar friction of the same $\gamma_{\rm eff}$ still hops ($0.73/10.2/0.26\%$)**."*
- **pj (§3.2, L88):** *"while unconstrained states exhibit out-of-register hopping fractions up to $43.0\%$, the engineered vault reduces this fraction to $0.0000$ across all seeds."*
- **Verdict: WIDER ⛔.** The control arm — equal-$\gamma_{\rm eff}$ scalar friction, which still hops — is deleted, so "the engineered vault" is now contrasted only against *no friction at all*. The claim "the hole confines" is exactly the claim the scalar control exists to protect (this is the one vault result with no designed analogue). Restoring six numbers restores it.

### ⛔ C6 — App B: "four independent **rollout** instruments"
- **Base (L171):** *"**I-J**, the one-step Jacobian … **I-R1**, rollout first crossing … **I-R2**, rollout last crossing; **I-R3** (primary), rollout envelope-rate fit"*; (L69) *"Two instruments, one law."*
- **pj (App B, L139):** *"measurements were duplicated across **four independent rollout instruments**, keyed against identical fit windows."*
- **Verdict: CHANGED IN KIND ⛔ — a factual misstatement.** One of the four is the analytic one-step Jacobian, not a rollout. The whole point of the appendix (an analytic instrument cross-checked by *independent* rollouts; referee MF-3) is erased, and pj's own §3.1 contradicts it by naming "the Jacobian's $0.9032\pm0.0027$" — the same number this appendix attributes to a "rollout instrument". **Internal contradiction inside the surviving text.**

### ⛔ C7 — App B asserts as settled an explanation the base registers as REFUTED
- **Base, negatives table (L361):** *"Our microscopic explanation of the instrument offset — **Wrong**, though the offset itself is real, constant and quotable: the pre-registered single-slow-mode projection **fails** (amplitude fraction $0.638/0.644/0.327$ does not order like the angular overlaps) … multi-mode transient, not anharmonicity."* (The measured partial-amplitude *observation* does appear in the base's table caption, L189.)
- **pj (App B, L139):** *"discrepancies between analytical Jacobians and threshold instruments are **quantified strictly as constant scale offsets resulting from partial write amplitudes**, rather than fundamental deviations in decay rates."*
- **Verdict: WIDER ⛔.** The ancestor exists (the caption), but the base's own negatives entry says this micro-explanation *failed its pre-registration*. `pj_sub.tex` prints the failed explanation as the finding and drops the negative. This is a pre-registration-discipline violation of the kind AGENT_PROTOCOL §5 was written for.

### ⛔ C8 — two new third-party/market claims with no ancestor
- **Base (L51):** *"soft-deleted vectors stay reconstructible from raw index files in graph ANN databases (Chakraborttii et al., 2026)"*; *"The two literatures leave a seam."*
- **pj (§2, L61):** *"deleted vectors remain **fully** reconstructible from raw index files within graph databases, **producing significant data leakage vulnerabilities**. This highlights a clear conflict: **production environments require exact, unrecoverable deletion**, while current best-effort methods fail to eliminate historical embeddings."*
- **Verdict: WIDER ⛔.** "fully", "significant … vulnerabilities" and "production environments require exact, unrecoverable deletion" are **assertions about deployed systems with no ancestor in the base** — and the base's entire framing discipline (Add.52: *"V5's white space is named without claiming a system property"*) is that we describe a seam, not a market requirement. Both sentences are also uncited (Chakraborttii dropped).

### ⛔ C9 — "total information eradication"
- **Base (L79):** *"AUC $0.5000\pm0.0000$ on all six statistics, byte-equal $1.0000$ … This is a **store-level guarantee only** — the frozen encoder and any residue of past writes in a learned landscape are separate channels."*
- **pj (§3.3, L94):** *"post-deletion states display an AUC of $0.5000\pm0.0000$ across six independent adversarial statistics, **verifying total information eradication from the active network geometry**."* Also: *"our canonical placement **completely isolates the target data**."*
- **Verdict: WIDER ⛔.** "Total information eradication" is a system-level absolute; the base's measured statement is a store-state indistinguishability at zero tolerance under stated conditions. Directly in tension with the (surviving) encoder-exclusion bullet in §1.

### ⚠ N1 — the retention numbers no longer carry $\Delta$ and $\ell_\theta/\Delta$, against the file's own stated rule
- **pj (§1 Nomenclature, L55):** *"$n_{1/2}$ is **reported strictly alongside** $\Delta$ and $\ell_\theta/\Delta$."*
- **pj (§3.2, L86):** *"transitioning $\gamma$ from $0.05$ to $0.2$ successfully lengthens the half-life by a factor of $3.77\pm0.23\times$."* — no $\Delta$, no $\ell_\theta/\Delta$.
- **Base (L73):** *"$\gamma:0.05\to0.2$ lengthens the half-life $3.77\pm0.23\times$ **at $\Delta=0.5$ rad, $\ell_\theta/\Delta<0.05$**"*; App A header (L126): *"Every $n_{1/2}$ is quoted with its read tolerance $\Delta=0.5$ rad and its $\ell_\theta/\Delta$."*
- **Verdict: WIDER ⛔ + self-contradiction.** `\Delta=0.5` appears **zero** times in `pj_sub.tex`; `\ell_\theta` appears only in the Nomenclature sentence that promises it. The paper states a reporting rule and then violates it at every site. This is also a registered negative ("Never an $n_{1/2}$ without $\Delta$ and $\ell_\theta/\Delta$", base L335).

### ⚠ N2 — "three distinct regimes … at different $\mu$ values"
- **Base (L61):** *"latch, overdamped register and underdamped working memory are three regimes of one curve, **evaluated at two values of $\mu$ --- not two laws**."*
- **pj (§3.1, L69):** *"latching states, overdamped registers, and underdamped working memory represent **three distinct regimes of a single underlying curve evaluated at different $\mu$ values**."*
- **Verdict: NARROWER (safe) but hazardous.** It does not say "three values of $\mu$" (the referee's MF-2 error), so the fix is not undone; but "different $\mu$ values" re-opens the ambiguity MF-2 closed — the underdamped branch is *mass-independent*, so it is not a different $\mu$. Recommend the base's exact wording.

### ✅ Rulings in the Head's favour (stated so the diff is fair)
| claim | verdict |
|---|---|
| Abstract vault: *"…achieves a $107.77\pm4.78\times$ retention increase **on designed units**"* vs base *"$107.77\pm4.78\times$ designed and $106.1\pm5.0\times$ … on emergent"* | **NARROWER (safe)** — ⭐ the referee's MF-13(a) (vault reading as general in the abstract) is **fixed** here, better than the base's abstract. |
| Blelloch & Golovin attribution in the abstract, §2, §3.3 and App D | **IDENTICAL** — attribution present at 4/4 surviving deletion sites (Part B-3). |
| The V-curve law, $\gamma_{\rm crit}=2\varepsilon\mu$, $\mu^{-2}$ branch, mass-independent floor | **IDENTICAL**. |
| Conformal symplecticity $\det J=(1-\gamma)^d$ / $(1-\gamma\phi(q'))^d$ | **IDENTICAL**. |
| Sign flip: friction preserves, temperature erases | **IDENTICAL**. |
| Nomenclature Def-2 (inertial $M_i$ / spectral $\mu_k^2$; retention statements use $\mu$) | **IDENTICAL** (drops only "$\varepsilon$ … never a tilt"). |
| "the store stops answering before it stops leaking" (paraphrased) | **IDENTICAL in content** — see B-1 for the verbatim-form issue. |
| TTL-flag laundering control, $1.000$ vs $0.983$, correctly paired | **IDENTICAL** — ⭐ the referee's MF-9 control survives the condensation. |
| $\mu\to0$ corner, $\gamma_{\rm crit}\to0$ | **IDENTICAL**. |
| Future work list ($\mu^2$ spectrum, $10^3$-item deletions, temperature field) | **NARROWER (safe)**. |
| No emergent $\sigma_\theta$ ratio is quoted | **compliant** (the banned quantity is simply absent). |
| Scale/laptop framing in §1 and §5(i) | **NARROWER (safe)** but see B-11 for the missing "scale is a scope choice" sentence. |

## A.3 — The author-name rule (Add.51)

Per-file standalone sweep of `pj_sub.tex`: **`\bMo\b` = 0 · `Morse` = 0 · `Moser` = 0** in body, captions, labels and filenames — and **0 in the bibliography too**, because the arXiv:2605.03338 entry and every sentence citing it were cut. Also **0**: `CHLU`, `Jawahar`, `Pierini`, `companion`, `sibling`, `forthcoming`, `in preparation`, `Anonymous`.
**Ruling: COMPLIANT, and strictly stronger than the base on de-anonymization** (the base's two-step exposure — the CHLU continuity sentence plus the Anonymous theory note — is gone). ⚠ Two consequences the Head should weigh, both flagged not resolved: (a) the CLU↔CHLU continuity sentence was a *sanctioned* citation, and dropping it loses the lineage; (b) the equivariant-Lyapunov prior-art citation is now absent from a paper that still asserts symmetry-protected flat directions — a novelty exposure, not an anonymization one.
Inherited, not introduced: `\usepackage{neurips_2025_ml4ps}` — the venue string is in the `.tex` (as in the base); the base's build note records `ml4ps` = 0 in the compiled PDF.

## A.4 — Never-quote sweep, full, per-file, positive-controlled

**Instrument LIVE** (positive controls on `pj_sub.tex`: `107.77` ✓, `Blelloch` ✓, combined control pattern = 28 lines).

**Zero-hit list (all 0):** `13.9` · `≈14×` · `we alone` · `CLU-former` · `0 of 5` · `CSF3` · `prior mismatch` · `P=4` · `compositional family` · `residual protects` · `watch stayed green` · `state-of-the-art` · `SOTA` · `best-in-class` · `benchmark win` · `beats` · `wins` · `our fix-up cascade` · `deletion-compliant` · `0.272` · `Guo Def. 1/2` · `right-to-be-forgotten` · `memory provenance` · `companion` · `sibling` · `forthcoming` · `in preparation` · `github` · `zenodo` · `huggingface` · `.claude` · `chlu/` · `CHLU` · `Pierini` · `Jawahar` · `PALM` · `Mo` · `Morse` · `Moser` · `Anonymous` · `13.88`.

**Non-zero, adjudicated individually:**

| pattern | n | line | disposition |
|---|---|---|---|
| `Certified` | 1 | 105 | ⛔ **VIOLATION — affirmative form.** See A.2-C1. The base's 3 occurrences are all denial/literature. |
| `unlearning` | 2 | 96, 118 | ⚠ L118 = the Bourtoule reference title — permitted. L96 = *"rather than providing **cryptographic unlearning privacy**"* — denial form, so it satisfies the letter of the rule, but "cryptographic unlearning privacy" is a **new coinage with no ancestor**, and sitting nine lines from L105's "Certified removal is proven" it reads as a boast, not a denial. Flag. |
| `outperform` | 1 | 88 | ⚠ **NEW — the base has 0.** *"significantly outperforming a scalar friction control"*. The base says *"against $13.28\pm0.12\times$ for a scalar friction of equal $\gamma_{\rm eff}$"*. The comparison is internal (our own control), so it is not a benchmark-win claim — but `outperform` is a word-level never-quote and "significantly" is an unearned significance word (no test was run). |

---

# PART B — what was LOST (the `v5-referee-v02` do-not-cut list, walked item by item)

The base was verified **27/27 present** on this list (`BUILD-NOTE.md` §6). Below: present/absent in `pj_sub.tex`, **ranked by consequence**. ⛔ Flagged, never fixed.

## Tier 1 — a claim now stands without a mandatory rider (claims violations)

| # | do-not-cut item | status in `pj_sub.tex` | which claim now stands unqualified |
|---|---|---|---|
| **B-1** | **The exact-deletion form: three conditions + recency exclusion** (CM-25(f) verbatim; N118) | ⛔ **ABSENT — 0 hits, both sites** | The abstract's and §3.3's "store-level deletion is exact / exact byte-for-byte parity". This is the single form N118 was written to ban. |
| **B-2** | **"we claim no certified $(\varepsilon,\delta)$ unlearning"** | ⛔ **ABSENT, and INVERTED** into "Certified removal is proven strictly at the store level" (L105) | The whole deletion contribution now reads as certified unlearning at the store level. |
| **B-3** | **The designed-symmetry precondition, beside the claim** | ⛔ **ABSENT from §3.2.** A one-line trace survives in §5 ("rely explicitly on baseline design symmetries **discussed in §3.2**" — §3.2 does not discuss them: a *semantic* dangling reference) and a numberless echo in App E ("Symmetrical continuous cosets do not natively emerge…") | Everything in §3.2. The base calls this *"the boundary on everything above"*. Gone with it: $1-|\lambda_{\rm coset}|\approx10^{-3}$ vs designed $\le1.1\times10^{-15}$, complete relaxation of a written value, capacity $\approx1$–$1.6$ bits, "continuum flat directions must be designed in", $T^\star\approx3\times10^{-3}$. **The vault is measured on a register the base proves does not emerge — and `pj_sub.tex` no longer says so where the vault is claimed.** |
| **B-4** | **The `fdt` + Newtonian fine print *beside* the $T>0$ claim** (C-6) | ⚠ **PARTIAL** — present in §1 bullet 3 and §5 item 3 (`\texttt{fdt}` ✓); **absent from §3.2** and **absent from App A's header** (the base's *"the reference default is `legacy`, under which none of these laws hold"* is gone entirely) | Every $T>0$ number in §3.2 and App A. C-6's requirement is placement *beside the claim*, and that placement is lost. |
| **B-5** | **The score sentence** — *"External benchmarks won on their own headline metric = ZERO"* | ⛔ **ABSENT** (`ZERO` = 0 hits) | The paper's honest-posture anchor. Partially substituted by §1's "rather than end-to-end system benchmark superiority" and §5's "task-level performance payoffs … excluded" — but the sentence itself, which is the program's standing self-score, is gone. |
| **B-6** | **The lifecycle's two riders** (demotion is re-exposure, never the trash region; the trash criterion is keyed on read-hits, never depth) | ⛔ **ABSENT** — with the entire lifecycle description; only the phrase "three-state memory lifecycle" survives, in the **evaluated** framing (A.2-C3) | The lifecycle claim. Also gone: 7/7 legs, the unexercised leg, PROTECTED/ACTIVE/TRASH, "no value or benchmark number is claimed, and none was run". |
| **B-7** | **N108's sentence — *"the store stops answering before it stops leaking"*** | ⚠ **PARAPHRASED, not verbatim**: *"The system actively ceases answering queries prior to halting internal data leakage"* (L96). Content preserved; the required *form* is not. Its two riders **are** gone: the placement scope (*"placement-dependent, quoted for the controller-placed disk"*) and the denial (*"Against an exact adversary decay reduces the effect size, not the AUC, so we claim no reduction in distinguishability per se"*) | The retention $0.832$ number now travels with no $|c|$-distribution scope (N119/SF-3), and the effect-size-not-AUC denial is gone while the AUC numbers stay. |
| **B-8** | **The trivial-substitute laundering control for deletion** — *"A flat table deletes exactly by construction — the trivial substitute"* | ⛔ **ABSENT** | §3.3's exactness claim now has no trivial-substitute control at all. |
| **B-9** | **The equal-$\gamma_{\rm eff}$ scalar control on the confinement claim** | ⛔ **ABSENT** (A.2-C5) | "the hole confines". |
| **B-10** | **The $R_{50}$ differentiator's comparator** — *"where a TTL vector store's lookup radius is a constant hard step at $0.75$–$0.77$, independent of age"* | ⛔ **ABSENT** (and the $1.52\times$) | $R_{50}$ $1.146\to0.752$ is now a bare within-system change with no baseline — the referee's SF-2 failure mode ("a comparative claim with no number"), applied to the one differentiator the base says needs no adversary model. |
| **B-11** | **C-5 scale qualifiers** | ⚠ **PARTIAL.** Present: §1 "laptop-scale … (dim 4, hidden 64)", §5(i) with $n=5$/$n=3$, §3.3 "designed, non-learned 3-dimensional". **Absent:** capacity 8–64; "no learning anywhere"; **the scale-as-scope-choice sentence** (*"this short is sealed to laptop-scale evidence; no larger-scale result is quoted or should be inferred"*); **any scale qualifier in the abstract** except "a designed datastore" | The abstract's three headline claims carry no scale. |
| **B-12** | **The $\approx11$-decade instrument note** (referee MF-1, the fix V2 carries and V5 v0.2 dropped) | ⛔ **ABSENT** — *"That low endpoint is the ring-profile probe's resolution on a checkpoint whose Hessian $\mu^2$ is machine zero rather than a spectral mass, so eleven orders is one curve on one instrument."* | §3.1 now says "Hessian $\mu^2\approx10^{-15}$" in one paragraph and "spans eleven orders … $[1.7\times10^{-12},7\times10^{-2}]$" in the next. **The exact internal contradiction MF-1 was raised to fix is re-created**, and §5 no longer carries the base's "(iv) the eleven decades are one decade of emergent variation attached to the designed $\mu\to0$ corner". |

## Tier 2 — attributions, citations and captions

| # | item | status |
|---|---|---|
| **B-13** | **Blelloch–Golovin at EVERY deletion site (N118)** | ✅ **PRESENT 4/4**: abstract (L32), §2 (L61), §3.3 (L92), App D (L173). ⚠ The base's **no-priority clause** (*"We claim no priority over order-independent placement and no novelty for the displacement rule or its delete-time repair"*, and *"'Fix-up cascade' is our name for their repair"*) is **ABSENT**, as is the base's negatives-table row *"The placement algorithm is ours — No: Blelloch & Golovin (2007) own it outright"*. Attribution survives; the explicit disclaimer of priority does not. Mitigating: pj never claims the composition either, so no novelty claim stands unqualified. |
| **B-14** | **The corrected Guo citation form (§2, Eq. (1), $\varepsilon$-only)** | ⛔ **ABSENT** — the whole sentence is cut. Neutral in isolation (a dropped citation cannot be miscited) — **but it is the sentence that defined "certified"**, so its removal is what leaves L105's "Certified removal is proven" undefined and unopposed. Rank with B-2. |
| **B-15** | **The k-regime scope clause on the erosion horizon** | **N/A — the entire erosion study (App H) is cut from this condensation.** No unqualified chain-length claim exists in `pj_sub.tex`; the mandatory clause is therefore moot here. ✅ by absence of the claim. Same for the arXiv:2503.21536 / CD mis-citation (MF-10): gone with its host sentence. |
| **B-16** | **The substrate-scope sentence** (*"These laws govern the store; end-to-end performance additionally depends on the encoder, measured separately, $\varphi$-bytes ledgered"*) | ⛔ **ABSENT in its required form.** Two partial substitutes: §1 "Results reflect core mechanistic properties of the physical store rather than end-to-end system benchmark superiority" and §1 "excluding frozen encoder traces" (deletion-scoped only). The encoder-measured-separately / $\varphi$-bytes ledger content is gone. |
| **B-17** | **Emergent-arm caveats** | (a) **no $\sigma_\theta$ ratio quoted** ✅ compliant *by absence* — but the base's explanation (*its control returns $0.4586\pm0.1181$ where it must return $1.000$*) is gone, so a future editor has no record of why. (b) **The $\theta=\pi$-is-not-a-vacuum confound** ⛔ ABSENT — moot, since no emergent first-passage number survives. (c) **The contrast number is designed-only** — the number ($8.11\pm0.37$ / $23.39\pm10.06$) is **absent entirely**, so the rider is moot ✅; but its role is taken over by the unqualified $106.1\times$ sentence (A.2-C4). |
| **B-18** | **The estimator's name on $107.77\times$** (*"$107.77\times$ is the quoted number and travels with its estimator's name"*; $D_\theta$-estimator, pre-registered; first-passage reads $86.97\pm2.94\times$ and is boundary-layer biased) | ⛔ **ABSENT** at both sites (§3.2 and App C). The base *explicitly requires* this number to travel with its estimator; it now travels bare. |
| **B-19** | **Figure 1's caption scope labels** | ⛔ **LOST.** Base: "(circles, **verification**, $\mu^2_{\rm rad}=0.670$–$1.348$)", "(squares, **evidence**, $\mu^2_{\rm soft}=2.0$–$5.4\times10^{-2}$, **3/3 seeds**)", "against the $\mp1$ asymptotes", plus the flag-provenance parenthetical (dim 4, hidden 64, $\varepsilon=0.05$, laptop-CPU, `langevin_noise="fdt"`, Newtonian). pj keeps only "five designed radial modes (circles) and three emergent learned-MLP coset modes (squares) … Data collected via one-step Jacobian evaluation." **The figure file is unchanged**, so the caption is not *false* — it has lost its status labels and its provenance. |
| **B-20** | **C-2 two-layer status labelling (designed = verification / learned = evidence)** | ⛔ **ABSENT PROGRAM-WIDE in this file**: `verification` and `evidence` as status tags = 0 uses (the words appear only in ordinary prose). Every appendix scope header that carried them is cut. A reader cannot tell which results are exactness verifications and which are evidence. |
| **B-21** | **App B's instrument rider** — *"both threshold instruments fail below $\gamma_{\rm crit}$, so no rollout $n_{1/2}$ is ever quoted from a threshold there"* | ⛔ **ABSENT**, while the table that needs it survives **in full**. `pj_sub.tex` prints slope-below values of $+0.0725$ / $+0.0688$ / $+0.0455$ (which the base identifies as instrument *failures*, "where the law says $-1$") with no rider, no instrument names (I-J/I-R1/I-R2/I-R3), no mean±sd row, and no $\delta$/grid metadata. **A reader sees four instruments contradicting each other by a factor of 20 and is given nothing.** Compounded by A.2-C6's "four rollout instruments". |
| **B-22** | **App C's vault-table caption content** — 24 field cells at $0.9998\pm0.0019$; *"The coupled bath predicts $1.0$ everywhere and is measured at $0.2235$"*; 2048 walkers × 40 samples, 3 emergent seeds | ⛔ **ABSENT.** The table still prints a "Coupled Pred." column of $1.0$ with **no statement that the coupled bath is the rejected hypothesis** — the discriminator that makes the vault credible is now an unexplained column. |
| **B-23** | **The $\sigma_{\rm obs}$-is-our-own-modelling-choice admission** | ⛔ **ABSENT** (`modelling choice` = 0). The App D table still prints the $\sigma_{\rm obs}=0.1$ row ($0.559$ vs $0.996$) — now as a real graded protection result. The base's caption exists precisely to prevent that reading. |
| **B-24** | **"at full load"** on AUC $0.99985$ | ⛔ **ABSENT** — scope qualifier dropped from a comparative leakage number. |
| **B-25** | **The anonymization note** (*"Any supplementary or linked material, including code, is anonymized; only an anonymized snapshot may be linked."*) | ⛔ **ABSENT.** PALM's code-inclusive anonymization rule now has no carrier in the artifact. |

## Tier 3 — citations dropped from claims that need them; orphaned bibliography

- **~28 of 32 references were cut.** Surviving bibliography: Aitken 2022 · Andersson & Ottmann 1995 · Blelloch & Golovin 2007 · Bourtoule 2021.
- ⛔ **3 of the 4 surviving entries are orphans** — `Aitken`, `Andersson`, `Bourtoule` are **never cited in the body** (their citing sentences were cut). Only Blelloch & Golovin is used.
- ⛔ **Claims now standing with no citation** (each was cited in the base): MemGPT and Mem0 as named systems (Packer 2023; Chhikara 2025) · the $0.995$ recency decay factor (Park 2023) · Ebbinghaus decay (Zhong 2024) · learned expiration spans (Sukhbaatar 2021) · recurrent forget gates (Behrouz 2025 — and with it the Titans contrast, the paper's nearest published neighbour) · reconstructibility of soft-deleted vectors (Chakraborttii 2026) · timestamp invalidation of contradicted edges (Rasmussen 2025) · "benchmarks measure retrieval, not forgetting" (Yang 2026; Uddin 2026) · certified removal (Guo 2020) · exact-deletion-by-isolation (Bourtoule/Ginart/Sekhari — Bourtoule is in the bibliography but uncited) · the canonical-placement prior art chain (Snyder 1977; Andersson & Ottmann 1995 — in the bibliography, uncited; Micciancio 1997; Naor & Teague 2001; BGV 2008) · the exponential-slowdown price (Buchbinder & Petrank 2003) · dissipative Nambu–Goldstone modes (Minami & Hidaka 2018) · symmetry-protected Lyapunov neutral modes (arXiv:2605.03338) · LSTM/LEM comparators.
- **Consequence, ranked:** §2 is now a related-work section that names commercial systems and quotes a competitor's hyperparameter with zero citations, and the paper's two *closest* prior works (the learned-forget-gate neighbour and the equivariant-neutral-modes preprint) are invisible. This is a referee-facing defect independent of any claims issue.

## Tier 4 — content dropped with no rider consequence (recorded, not charged)

The trilemma and the **compute-adaptive-read dial** (a named program dial) · the packing-price-is-negative result and its numbers · the $\mu\to0$ model-side bound $n_{1/2}\ge9.5\times10^{15}$ · the "what would have falsified the law" sentence · the finite-write-amplitude robustness sweep · the left-branch mass-independence identity · the collapse statistics · the OU-estimator cross-validation · **18 of the base's 21 negatives** (App E keeps 3) · 10 of 11 figures · the whole prior-art paragraph · the composition-not-a-new-algorithm framing · limitations (iv) and the scale-as-scope-choice sentence.
⚠ **The negatives estate is the program's C-9 signature** (the referee called it "best in the portfolio"): 21 → 3. `pj_sub.tex`'s own framing sentence no longer claims completeness, so nothing false is asserted — but B-5, B-21, A.2-C7 and B-17 are all cases where the *specific* negative that protects a surviving claim is one of the 18 cut.

---

# PART C — render (BLOCKED, with everything but the invocation prepared)

⛔ **`pj_sub.pdf` was NOT produced.** This session has **no Bash/shell tool** (available tools: Read, Write, Edit, Grep, Glob). `pdflatex` cannot be invoked and `md5` cannot be computed. Reporting this rather than claiming a build is the honest outcome; the acceptance criterion "3. `pj_sub.pdf` exists" is **not met**.

## C.1 — Static build analysis: the file will NOT compile clean as written

Measured by per-file regex, not asserted:

| defect | count | lines | effect |
|---|---|---|---|
| **unescaped `&` in text mode** | **8** | 32, 61, 92, 115, 116, 117, 118, 173 | `! Misplaced alignment tab character &.` at each site. Under `-interaction=nonstopmode` TeX continues, but the ampersand is dropped/garbled — **including in the abstract's Blelloch & Golovin attribution**. |
| **table rows terminated with a single `\` instead of `\\`** | **15** | 129, 131, 132 · 145, 147, 148, 149 · 162, 164, 165, 166 · 179, 181, 182, 183 | Control-sequence error at every row end **and** no row terminator, so each `tabular` collapses toward one row ⇒ `! Extra alignment tab has been changed to \cr` cascades. **All four tables are affected; none will typeset correctly.** Measured: lines ending in `\\` = **0**; lines ending in a single `\` = **15**. |

**Everything else checks out statically:** all 4 `\ref` targets (`sec:vcurve`, `sec:vault`, `sec:deletion`, plus `sec:vault` in §5) are defined ⇒ **no dangling cross-references** (a real improvement over v0.2's nine); the single `\includegraphics` target `figs/fig1_damping_optimum.png` exists; all four `tabular` preambles match their cell counts once `\\` is restored (`lll`=3 ✓, `lcccc`=5 ✓, `lcccccc`=7 ✓, `lccc`=4 ✓); no stray `_`/`#`/`%`; math delimiters balanced at every site inspected; `neurips_2025_ml4ps.sty` present; `\@notice` suppression and the `\hypersetup` metadata scrub are byte-identical to the base.

## C.2 — `pj_sub_buildcopy.tex` — created, repairs listed, **not built**

I created **`.claude/NIPSsubmission/v5-palm/pj_sub_buildcopy.tex`**: a copy of `pj_sub.tex` with **exactly 23 minimum repairs and nothing else** (no reflow, no added/removed lines — a `diff` against the original shows 23 changed lines and no line-number drift).

**Every repair — these are edits the Head owes his own file:**

| # | line | repair | before → after |
|---|---|---|---|
| 1–8 | 32, 61, 92, 115, 116, 117, 118, 173 | escape the ampersand | `&` → `\&` |
| 9–11 | 129, 131, 132 | row terminator (budget table) | `…\` → `…\\` |
| 12–15 | 145, 147, 148, 149 | row terminator (emergent table) | `…\` → `…\\` |
| 16–19 | 162, 164, 165, 166 | row terminator (vault table) | `…\` → `…\\` |
| 20–23 | 179, 181, 182, 183 | row terminator (deletion table) | `…\` → `…\\` |

**Zero content edits.** No number, word, claim, caption or label was touched — every Part A/B finding above is left exactly as the Head wrote it.

**The build, for whoever has a shell** (run inside `.claude/NIPSsubmission/v5-palm/`):
```
/Library/TeX/texbin/pdflatex -interaction=nonstopmode pj_sub_buildcopy.tex   # ×3
```
⚠ **If the resulting PDF is to be presented as `pj_sub.pdf`, the report must state it was built from the build copy, not from `pj_sub.tex`.** Page split (main / references / appendices) is therefore **not reported** — it cannot be measured without the run. For context only (Add.52, not a target): PALM is 4 pp short / 9 pp full, references and supplementary excluded; the base measures 6 pp main / 19 pp total, and `pj_sub.tex` is ~30 % of the base's words with 1 of 11 figures, so a main text in the 2–3 pp range is the expectation — **an expectation, not a measurement.**

## C.3 — Acceptance criterion 4 (byte-identity)

`pj_sub.tex` was read and never written. **No `Edit` or `Write` call targeted it at any point in this session** — the only write into that folder was the new `pj_sub_buildcopy.tex`. ⛔ **I cannot print an md5 (no shell).** Requested of the Hub/Head: `md5 .claude/NIPSsubmission/v5-palm/pj_sub.tex` and compare to the pre-pass value.

---

## Findings summary, ranked by consequence

1. ⛔ **A.2-C1 / B-2** — "Certified removal is proven strictly at the store level" (§5). A standing denial inverted into an affirmative claim of certified removal.
2. ⛔ **A.2-C2 / B-1** — the three deletion conditions and the recency exclusion are absent from the whole file; "deletion is exact" stands unqualified in the abstract and §3.3.
3. ⛔ **B-3** — the designed-symmetry precondition is gone from §3.2, and §5 points at a section that no longer contains it; the vault claim stands without the boundary the base calls "the boundary on everything above".
4. ⛔ **A.2-C6 / B-21** — App B calls all four instruments "rollout" (one is the analytic Jacobian) and prints the threshold instruments' failure values with the rider that explains them removed.
5. ⛔ **A.2-C5 / B-9 / B-8 / B-10** — three laundering/trivial-substitute controls dropped (scalar-friction confinement control; flat-table deletion substitute; TTL lookup-radius comparator on $R_{50}$).
6. ⛔ **A.2-C4 / B-18** — $106.1\times$ reads as the vault transferring; $107.77\times$ no longer travels with its estimator's name.
7. ⛔ **B-12** — the eleven-decade instrument note is gone and the $10^{-15}$/$1.7\times10^{-12}$ contradiction (referee MF-1) is re-created.
8. ⛔ **A.2-C3 / B-6** — the lifecycle reads as evaluated, not shipped; both riders and the "none was run" sentence gone.
9. ⛔ **A.2-C8 / C9** — new system-level assertions with no ancestor ("production environments require exact, unrecoverable deletion"; "total information eradication").
10. ⛔ **A.2-C7** — a refuted micro-explanation asserted as the finding.
11. ⛔ **N1 / B-4 / B-11 / B-20** — $\Delta$ and $\ell_\theta/\Delta$ absent while the Nomenclature promises them; the `fdt`+Newtonian print no longer sits beside the $T>0$ claim; no scale qualifier in the abstract; the verification/evidence scheme is gone.
12. ⛔ **B-5 / B-25 / Tier 3** — score sentence, anonymization note, and ~28 references gone; 3 of the 4 surviving references are uncited orphans.
13. ⛔ **C.1** — 23 mechanical defects; the file does not compile clean.
14. ⚠ **A.4** — `outperform` (new; base has 0) and "cryptographic unlearning privacy" (new coinage).

**Credit where earned:** zero numeric errors in 39 checked quantities including three tables reproduced value-for-value; the abstract's vault claim is *better* scoped than the base's (MF-13(a) fixed); the TTL laundering control survived; Blelloch–Golovin attribution survived at 4/4 sites; no dangling `\ref`; the author-name rule is fully satisfied and the de-anonymization chain is shorter than the base's.

---

## Open questions / follow-ups / risks

1. **Part C is unfinished and needs a shell.** Whoever re-runs it should build `pj_sub_buildcopy.tex`, report the page split, and print the `pj_sub.tex` md5. If the Hub wants the PDF from a spoke, that spoke needs Bash.
2. **`pj-referee-v5` is gated on the PDF** (Add.53) and is explicitly barred from reading this report. It cannot start until someone builds. Flagging as a scheduling blocker, not resolving it.
3. **I did not adjudicate whether the Head's cuts are correct** — only whether they preserve the base's claims. Several Tier-4 losses (the trilemma / compute-adaptive-read dial; the negatives estate 21 → 3) are *scope decisions* that belong to the Head, and I have deliberately not scored them as defects.
4. **One thing I could not check:** whether the surviving Figure 1 PNG's own axis annotations still match its shortened caption (I cannot compare rendered figure content to caption text without the build).
5. **No disagreement between an output and the handover was found to flag** — the base, the referee report and the build note are mutually consistent on every item I used.

## Proposed handover updates (for the Hub)

1. **`pj-fidelity-v5` PARTIAL: Parts A+B done, Part C BLOCKED (no shell in the spoke session).** Report at `.claude/outputs/pj-fidelity-v5.md`; a repaired, unbuilt `pj_sub_buildcopy.tex` (23 mechanical repairs, zero content edits) is in `NIPSsubmission/v5-palm/`.
2. ⭐ **Part A answer to the Head: numerically clean.** 39 quantities checked, **zero digit/±/unit errors, zero orphan numbers**. The condensation's damage is entirely in dropped scope riders, not in misquotation.
3. ⛔ **Four items need a Head decision before this file goes anywhere:** (a) §5's "Certified removal is proven strictly at the store level" — a never-quote inversion; (b) the missing three deletion conditions + recency exclusion; (c) the designed-symmetry precondition's removal from §3.2 while §5 still cross-references it; (d) App B's "four independent rollout instruments" (factually wrong and self-contradicting with §3.1).
4. **Registry candidates for the Hub to rule on** (I did not write to any registry this pass — this task was audit-only): (i) *negatives registry* — nothing new tried-and-failed was produced here; (ii) *`future_work.md`* — no new scientific boundary surfaced; (iii) ⭐ **a process negative worth recording**: a hand-condensation to 30 % preserved 100 % of numbers and lost 12 of ~25 mandatory riders — i.e. **rider loss, not number drift, is the failure mode of condensation**, and a rider-checklist diff should be a standing gate on any future hand-edit of a paper file.
5. **Two mechanical defects the Head owes his own file** even if he rejects every claims finding: the 8 unescaped `&` and the 15 single-`\` row terminators.
6. **`pj-referee-v5` remains gated** on a built PDF; the referee is barred from this report, so the Hub should hand it a PDF and nothing else.

**Git footprint:** none — no tracked file touched. Files created: `.claude/outputs/pj-fidelity-v5.md`, `.claude/NIPSsubmission/v5-palm/pj_sub_buildcopy.tex`. `pj_sub.tex` untouched.
