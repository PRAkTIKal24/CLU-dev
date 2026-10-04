# LOG — w6 spoke/shorts-inventory

Spoke: `experiment-engineer`, documentation only. No scored run, no claim.
Task file: `/c/chlu/tasks-w6/w6-spoke-shorts-inventory.md`
Base: `base/iter6 = df025685`. Worktree: `/c/chlu/worktrees/CHLU-w6-shorts`.

- **2026-09-20T06:25:55Z** — START_UTC.txt written; no live one existed (output dir was empty).
- **2026-09-20T06:26Z** — Hashed all five PDFs in `/c/chlu/shorts/`. **The two "⚠ identical byte size" pairs are byte-identical files, not merely same-sized.**
  - `F_v1_LCFM_finalSub_2.pdf` and `F_v1_TTCL_finalSub_2.pdf` — sha256 `1a01cfb873bad918796c90fdf1111adf85fc6af7f8fc3e3aeff4105ddc561dd1` (identical).
  - `F_v5_PaLM_Subm3.pdf` and `F_v5_stody_Subm3.pdf` — sha256 `ff12667570ce5b4e24f5d0333d0515130dc4f23aa7e3c05cc4df6ebb5eea4514` (identical).
  - `F_v2MemShort_bento_final2.pdf` — sha256 `5a0fc126c07975a41ef4a8a139594b010ae95f2c9aa2e1e39d2ffbd2e48b4e11` (unique).
  - ⇒ **Three genuinely distinct papers (V1, V2, V5); the venue-variant filenames are the same bytes renamed.** Not "same content different typesetting" — the same file. No substance to compare.
- **2026-09-20T06:26Z** — Beginning text extraction.

- **2026-09-20T06:35Z** — ⚠ **EXTRACTION TRAP, recorded because it nearly produced a wrong law.**
  `pdftotext -layout` (Latin-1 default) **silently drops every Greek glyph** in these PDFs. Under that
  extraction V2's overdamped retention law reads `n1/2 ~ 2 ln 2/[(2 - )( )2]` — it has lost the
  **γ in the numerator** and both symbols in `(εµ)²`. The true law is `n₁/₂ ≈ 2γ ln 2/[(2−γ)(εµ)²]`.
  Re-extracted all three with `-enc UTF-8`; the PDFs do carry ToUnicode CMaps, so the UTF-8 recovery
  is the paper's own encoding, not a reconstruction. **Every formula in the deliverable is quoted
  from the UTF-8 recovery.** No page rendering was available (no pdftoppm/poppler-utils, no
  pypdf/pymupdf/pdfminer), so figures were read via their captions in the text layer.
- **2026-09-20T06:36Z** — ⛔⛔ **HEADLINE FINDING, FLAGGED LOUDLY FOR `hub/potentials` AND `hub/retention`:**
  **The "Mexican-hat width → interpolation floor" law the wave-6 plan assumes is NOT IN THE SHORTS.**
  - "interpolation" (any inflection) occurs **zero times** in all three papers.
  - "Mexican hat" occurs **exactly once** in the entire corpus — V2 App. M.4 — as a *numerical control
    potential for an autodiff-Hessian round-off check*. No width parameter, no separation parameter,
    nothing derived from its shape. "Sombrero" occurs zero times.
  - The nearest derived object is V2/V5's retention floor `n₁/₂ = 2 ln 2/(−ln(1−γ))`, which is
    **curvature-INDEPENDENT** — substituting it for a width-set floor would invert the papers' finding.
  - ⛔ Second alarm: **"the exact-decay law" is ambiguous** — no paper uses that phrase, and the corpus
    contains three non-interchangeable exact statements about decay (asymptotic overdamped rate;
    the exact `|λ_flat|=1` latch; "decay commutes with deletion"). A hub must name which.
- **2026-09-20T06:37Z** — **PARTIAL INVENTORY PUBLISHED** to
  `/c/chlu/outputs/w6-shorts-inventory/shorts-experiments-inventory.md`. Sections 0–4 final:
  paper identity, the absent-law alarm, 50 DERIVED laws (D1–D50) quoted verbatim with paper+section,
  8 assertions-without-derivation (A1–A8), and the hub→short mapping. Section 5 (per-experiment rows)
  in progress. **Hubs are unblocked on everything they pre-register against.**

- **2026-09-20T06:45Z** — Resolved six column-garbled tables by re-extracting in PDF **reading
  order** (`pdftotext -enc UTF-8`, no `-layout`): V1 App. C.1 certificate table and App. D.3 routing
  grid; V2 App. I (both tables) and App. J; V5 Tables 1, 3, 4 and 7. All resolved unambiguously
  **except** one cell — V5 Table 4's per-seed `Γ_jac/Γ_R3` values (0.9935 / 1.0000 / 0.9948) cannot
  be mapped to seeds 42/43/44 from the text layer. Recorded in the deliverable as
  **[alignment unrecoverable]** rather than guessed; the mean (0.995 ± 0.003) is unambiguous.
- **2026-09-20T06:52Z** — **SECTION 5 + 6 APPENDED. DELIVERABLE COMPLETE.** 63 experiment rows
  (E1.1–E1.21, E2.1–E2.19, E5.1–E5.23), each with what was run · on what object · the claim · the
  MEASURED number · the DERIVED law it tests (or an explicit "no derived law" mark). Status banner
  flipped PARTIAL → COMPLETE; **no earlier text was altered** beyond that banner.
- **2026-09-20T06:53Z** — Provenance check (recorded although this spoke ran no numbers):
  `chlu.__file__ = C:\chlu\worktrees\CHLU-w6-shorts\chlu\__init__.py` — resolves inside the worktree.
- **2026-09-20T06:54Z** — Code: **`git status` clean, zero commits.** This is a documentation spoke;
  the deliverable lives in `CHLU_HUB_OUTPUTS` (outside the repo), so there is no code to commit and
  `w6/shorts` sits at base/iter6 `df02568` unchanged. Bundled anyway and verified:
  `w6-shorts.bundle`, `git bundle verify` → *"The bundle records a complete history"*, carrying
  `refs/heads/w6/shorts` and `refs/heads/base/iter6` both at `df0256856c5792faf2ab99c4682b4ed75e117b48`.
  Outputs verified by directory listing (gitignored paths do not show in `git status`):
  `LOG.md`, `shorts-experiments-inventory.md` (66,194 bytes), `START_UTC.txt`, `w6-shorts.bundle`.

---

## FINAL REPORT — w6 spoke/shorts-inventory

**Deliverable:** `/c/chlu/outputs/w6-shorts-inventory/shorts-experiments-inventory.md` (66 KB).
**Dial: the record.** No scored run, no claim, as the task file requires.

### The falsifier fired. Say it plainly.

This spoke's declared falsifier was: *"a derivation the physics hubs need is not actually in the
shorts — in which case say so explicitly; an invented derivation is worse than a missing one."*

**It fired, on the law the wave-6 plan assumes for `hub/potentials`:**

> ⛔ **The "Mexican-hat width → interpolation floor" law is NOT IN THE SHORTS.**
> - **"interpolation" (any inflection) occurs zero times across all three papers.** There is no
>   interpolation floor, no interpolation experiment, no interpolation claim in this corpus.
> - **"Mexican hat" occurs exactly once in the entire corpus** — V2 App. M.4 — where it is a
>   *numerical control potential* used to show that a cancellation-free Hessian is accurate where the
>   autodiff Hessian has an `ϵ/δ` round-off floor. No width parameter, no separation parameter,
>   nothing derived from the hat's shape. "Sombrero" occurs zero times.
> - The nearest genuinely derived object is V2/V5's retention floor `n₁/₂ = 2 ln2/(−ln(1−γ))`, which
>   is **curvature-INDEPENDENT**. Substituting it for a width-set floor would *invert* the papers'
>   actual finding. **No formula was invented to fill the gap.**

> ⛔ **Second alarm: "the exact-decay law" does not exist under that name.** No paper uses the
> phrase. The corpus holds **three non-interchangeable** exact statements about decay: (1) the
> overdamped rate — which the papers write with `≈`/`≃`, **not exact**; (2) the exact latch
> `|λ_flat| = 1`, infinite half-life at µ=0, T=0; (3) "decay commutes with deletion" (V5 App. G.8).
> **A hub must name which one it pre-registers against.** They license different experiments.

### The variant question, settled

Not "venue variants" — **byte-identical files**. `F_v1_LCFM` ≡ `F_v1_TTCL`
(sha256 `1a01cfb8…`), `F_v5_PaLM` ≡ `F_v5_stody` (sha256 `ff126675…`). **Three genuinely distinct
papers** (V1 *Certified Access*, V2 *The Price of a Flat Direction*, V5 *Retention You Can Predict,
Scope and Delete*), not five and not four. There is no substance to compare between the pairs.

### The extraction trap — the single most important process finding

Default `pdftotext` (Latin-1) **silently drops every Greek glyph** in these PDFs. Under that
extraction V2's central retention law reads `n1/2 ~ 2 ln 2/[(2 - )( )2]` — it has **lost the γ in
the numerator** and both symbols in `(εµ)²`. The true law is **`n₁/₂ ≈ 2γ ln 2/[(2−γ)(εµ)²]`**.
Anyone pre-registering from a Latin-1 dump of these files would pre-register against a law that is
wrong by a factor of γ. The PDFs *do* carry ToUnicode CMaps, so `-enc UTF-8` recovers the paper's
own encoding — not a reconstruction. **Every formula in the deliverable is quoted from the UTF-8
recovery.** This is called out in a banner at the top of the deliverable.

### What was delivered

- **§0** — the three-papers-from-five-files finding, with hashes.
- **§1** — the two absent/ambiguous laws, stated loudly and first.
- **§2** — **50 DERIVED laws (D1–D50)**, quoted **verbatim with paper and section**: the shared
  conformally-symplectic map; the full retention band structure (latch `q∞ = q₀ + εp₀/(Mγ)`,
  overdamped `n₁/₂ ≈ 2γln2/[(2−γ)(εµ)²]`, curvature-independent floor `2ln2/(−ln(1−γ))`, crossover
  `h∗ ≈ γ/2`, stability edge `h = 2`, Jordan-block exceptional point, √ onset); the friction dial and
  `γ_crit = 2εµ` with **both** log-slopes; the FDT scale `σ⋆ᵢ = √(MᵢTγ(2−γ))`; coset diffusion
  `D_θ = εT(2−γ)/(2F²γ)` with `F² = Mr∗²`, the sign flip, and `ℓ_θ`; the vault
  (`γ_eff`, `T_local`, `vault = (γ_eff/γ)²`) plus the derived-then-falsified coupled-bath rival;
  "decay commutes with deletion"; and V1's full certificate stack (causal box `Lᵢ = Tεc/√Mᵢ`,
  `det S_ζ = 1`, `H(S_ζz) ≤ e^{2|ζ|}H(z)`, reach bracket, `det J = 1 + ∇g(q)ᵀΔ`, latch transport
  `Q′−Q = pᵀXΔ`); plus GMOR `µ²F² = δΣ`.
- **§3** — **8 assertions with no derivation (A1–A8)**, flagged so no hub pre-registers against an
  assertion. The two that matter most: **A1 the Lifetime Trilemma** (stated as fact in *both* V5 and
  V1, proved in neither) and **A2 `µ² ∝ A/ℓ²`** (a V1 certificate-table cell whose `A` and `ℓ` are
  **never defined anywhere in the paper**) — A2 is the closest thing in the corpus to a "width" law
  and must not be used as one. **A5** is the only "separation" quantity: V5's lattice spacing
  **1.540000**, a measured constant with no closed form.
- **§4** — the hub mapping: `hub/retention` ← **V2** primary + **V5** for the γ and T axes;
  `hub/potentials` ← **V2 App. B/D/M** + **V1 App. H** — each with an explicit "must NOT assume" column.
- **§5** — **63 experiment rows** across the three papers, every one separating **MEASURED** from
  **DERIVED**, and marking rows that carry *no* derived law (V1's rationing, regime map, noise wall,
  routing grid; V2's emergent arm, CD erosion, compute table; V5's packing price, trilemma geometry,
  laundering control).

### Three results the papers themselves disqualify — carried forward so no hub builds on them

1. **V5 E5.15** — the emergent-arm field-to-scalar contrast measured **23.39 ± 10.06** against a
   pre-registered bound of **[6.5, 9.5]**: V5 reports this as **triggering its own declared
   falsifier**. The *absolute* vault (106.1 ± 5.0× vs 110.25 predicted) still holds.
2. **V5 E5.16** — V5 declares the stationary-spread comparison **void**: the control that must return
   exactly 1.000 returned **0.4586 ± 0.1181**. No in/out spread ratio may be quoted from it.
3. **V5 E5.17** — the crossover-temperature result is an explicit **n < 3 cell**.

### Honest limits of this spoke

- **No page rendering was available** (no poppler `pdftoppm`, no pypdf/pymupdf/pdfminer), so figures
  were read through their captions in the text layer, not visually. Every formula quoted came from
  the text layer, which carries the papers' own ToUnicode mapping — but I did not see the rendered
  equations. Where a table cell could not be resolved from the text layer it is marked
  **[alignment unrecoverable]** (one cell, V5 Table 4) rather than guessed.
- This is a records spoke: **nothing here was executed or reproduced.** Every number is what the
  paper reports, attributed to the paper.
- Scope rider inherited from all three papers: every quantitative result is at **dim 4** (V1 also
  dim 2), **hidden 64**, **ε = 0.05**, **single-core CPU**, on **5 designed and/or 3 emergent seeds**,
  and every `T > 0` result requires **fdt noise plus a Newtonian kinetic mode**.

### Bottom line for the two waiting hubs

**`hub/retention` is fully supplied** — its entire pre-registration surface (band structure, the
critical point, the diffusion law, the vault) is derived in V2 App. C and V5 App. G and is quoted
verbatim in §2.

**`hub/potentials` is supplied for everything except the one law it was told to assume.** The causal
box, the injection certificate, the gate Jacobian, latch transport, GMOR and the spectral-mass
definition are all there. **The Mexican-hat width → interpolation floor is not, and no substitute
exists in this corpus.** That hub needs to re-plan that leg before it pre-registers, not after.
