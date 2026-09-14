# n1-fulltext-and-track-check — web-scout report

**DIAL DECLARATION (echoed): none — citation verification + venue fact-finding; no performance claim; no laundering control applies.**

Task + acceptance criterion: (Q1) read arXiv:2605.03338 in full and rule law-vs-prediction on the gap→lifetime relation; (Q2) establish the NeurReps 2026 Findings Track's archival status and four other facts, verbatim or declared unavailable.
Status: **done** (both questions answered; two items declared unrecoverable and flagged, not inferred).

**⚠ DOWNSTREAM RECONCILIATION LIST (needs an owner — see §Reconciliation):** any CHLU/V2 site that characterises N1 as containing a *lifetime law* must be corrected. **N1 contains no closed-form gap→lifetime relation anywhere in its text.** Retrieval date for everything below: **2026-08-21**.

---

## Answer first

**Q1.** N1 is a **theorem about zero Lyapunov exponents, plus a qualitative-then-empirically-validated prediction** — *not* a lifetime law. The paper has exactly **five numbered equations (1)–(5)**, none of which relates a pseudo-gap to a memory lifetime; the term "pseudo-gap" is **never formally defined** in the paper; and no formula for "predicted lifetime" appears in the main text or in Appendices A–D. The relation exists **only in the released code** (`exp31`): `T_pred = −ln(1 − θ*/|φ₀|)/λ_gap`, i.e. **T ∝ λ⁻¹** — the textbook inversion of exponential drift-to-threshold, with no derived exponent, no crossover, and no floor. The paper's quantitative claim is a **correlation**, not a law: "pseudo-gap log-lifetime correlation 0.9999999886 … median measured/predicted lifetime ratio about 1.013."

**Q2.** The Findings Track's **archival status is UNSTATED on every venue-owned surface I could reach** — the neurreps.org CFP states archival status explicitly for the other two tracks and *omits it for Findings*; the Dual Submission Policy names only Proceedings and Extended Abstract; the OpenReview venue group JSON carries no archival field; there is no FAQ; and the track is **new in 2026** (2024 and 2025 had only two tracks), so there is no prior-edition precedent. Under the program's standing rule, an unverified archival status is the finding.

---

## Q1 — arXiv:2605.03338, full text

**Bibliographic facts (verified, arXiv abs page + HTML v1):** Hanson Hanxuan Mo (sole author, Dept. of Applied Mathematics & Computational Neuroscience Center, University of Washington), *"Symmetry-Protected Lyapunov Neutral Modes in Equivariant Recurrent Networks"*, arXiv:2605.03338**v1**, submitted **Tue, 5 May 2026 03:59:56 UTC**, cs.NE + math.DS. **Preprint, not peer-reviewed; no venue in the comments field; only one version exists.**

### Q1.1 ⭐ Law or prediction? — **PREDICTION, empirically validated. No law.**

Evidence, in the paper's own words:

- **Abstract (verbatim):** "When this protection is explicitly broken, the formerly protected direction can acquire a pseudo-gap; **in our controlled breaking experiments this pseudo-gap predicts finite memory lifetime.**" — the hedge ("can acquire", "in our controlled breaking experiments") is the author's own.
- **Introduction (verbatim):** "We also quantify the complementary failure mode: **in the explicit breaking families tested here**, breaking the protecting symmetry moves a formerly protected zero exponent to a nonzero pseudo-gap that predicts finite memory lifetime and path-integration drift."
- **§6 Limitations (verbatim):** "A further extension would test approximately equivariant trained models by **measuring whether their finite memory lifetimes are predicted by the measured symmetry-breaking pseudo-gap, which would connect the exact theorem to imperfect learned dynamics without claiming exact protection**; see figure 4." — i.e. the author explicitly scopes the gap→lifetime link as *unproven outside the controlled families* and lists it as **future work**.
- **Complete equation inventory of the paper** (requested and returned as an exhaustive list): (1) `f(g·x) = D(g)_x f(x)`; (2) `E^G_x = T_x(G·x) = {ξ_M(x) : ξ ∈ 𝔤}`; (3) `Dϕ_t(x) ξ_M(x) = ξ_M(ϕ_t(x))`; (4) `E_eq = max_{x,g} ‖f(g·x) − D(g)_x f(x)‖ / (1 + ‖f(x)‖)`; (5) `λ̂_ξ(T) = (1/T) log(‖Dϕ_T(x) ξ_M(x)‖ / ‖ξ_M(x)‖)`. **None is a lifetime relation.**
- **Theorem 1 (the paper's only theorem-level claim, from the abstract, verbatim):** "For a finite-dimensional autonomous C¹ vector field equivariant under a Lie group G, we prove that any compact invariant set carrying a uniformly nondegenerate group-orbit bundle with stabilizer type H has, at points where the Lyapunov spectrum is defined, **at least dim(G/H) zero Lyapunov exponents tangent to the group orbit.**" The theorem is about **counting zeros**, not about lifetimes.
- **A formal definition of "pseudo-gap" is absent.** All seven occurrences of the string are prose (Intro ×2, §5.4 ×4, §6 ×1); none defines it. Operationally, from code, it is `gap = max(0.0, −λ)` where λ is the direct group-tangent exponent of eq. (5).

**Verdict: qualitative prediction + a high-correlation empirical validation on a controlled ensemble. Not a closed-form law, not a derived exponent, not a theorem.**

### Q1.2 Functional form / exponent / crossover / floor

The paper reports the *outcome* of a prediction it never writes down. The predictor is recoverable **only from the released code** (`source_scripts/exp31_learned_equivariant_path_integration.py`, verbatim):

```python
lam = direct_group_tangent_exponent(model, np.zeros(128 if not quick else 64), phi0=phi0, device=device)
gap = max(0.0, -lam)
predicted = float("inf")
if gap > 1e-12 and threshold < abs(phi0):
    predicted = -math.log(1.0 - threshold / abs(phi0)) / gap
```

- **Functional form:** `T_pred = −ln(1 − θ*/|φ₀|) / λ_gap`. With the file's constants `phi0 = 0.35`, `threshold = 0.2` (radians), the prefactor is `−ln(1 − 0.2/0.35) = −ln(3/7) ≈ **0.8473**`, so `T_pred ≈ 0.847 / λ_gap`.
- **Exponent: −1 in the gap, exactly.** This is the analytic inversion of `φ(t) = φ₀ e^{−λt}` hitting an absolute angular error `θ*`; it is a *definition of the measurement*, not a fitted law.
- **Fitting procedure: none.** Nothing is fitted. The paper reports a **Pearson correlation on log-lifetime** between the measured and the analytically-predicted lifetime, plus a **median ratio**: "pseudo-gap log-lifetime correlation 0.9999999886, uncensored fraction 1.0, and median measured/predicted lifetime ratio about 1.013, so the pseudo-gap result is a quantitative breaking consequence rather than an informal visual trend" (§5.4). Fig. 4A is a **log–log scatter of predicted vs measured** (from `conference_fix.py:plot_fig4_pseudogap`).
- **Crossover: none reported. Floor: none reported.** The only structure is **right-censoring**: `measure_lifetime` integrates under zero input until `|angle_wrap(pred − φ₀)| ≥ threshold`, returning `censored=True` at `max_steps` (`max_time = 800.0` quick / `1500.0` full, `dt = 0.1`). Censoring is reported as "uncensored fraction 1.0" for main Fig. 4 and **0.857** for the learned cell (Table A3), i.e. **14.3 % of the learned-cell rows never decayed and were dropped from the correlation** ("The correlation is computed across uncensored runs where `predicted_lifetime` is finite and positive").
- **Measured on:** main result = the explicit-breaking ensemble of Fig. 4, families **weak-axis, unit-axis, rotated-strong**, plus a "random anisotropic breaking ensemble"; second result = post-training breaking of the learned S¹ cell (Fig. A7, Table A3, corr **0.9991**).

**⚠ Reproducibility gap (verified):** the main Fig. 4 numbers come from `results/journal_full/exp19_pseudogap_lifetime/pseudogap_lifetime.csv`, and **neither the `exp19` script nor that CSV is in the public repo** (`source_scripts/` contains only `conference_fix.py`, `exp21`, `exp25`, `exp26`, `exp29`, `exp31`, `exp32`, `make_fig2_…`; `generated_assets/support/` contains only three autonomous-zero-diagnostic files). So the headline **0.9999999886** rests on a frozen table the reader cannot inspect, and the `T ∝ 1/λ` formula is single-sourced from the *learned-cell* script. **The ε values swept in Fig. 4 are stated nowhere in the paper and are not recoverable from the public repo** — declared unavailable.

### Q1.3 ⭐ The units question — **λ is 1/time; the paper never mentions curvature and performs no conversion**

- **Eq. (5)** defines the measured quantity as `λ̂_ξ(T) = (1/T) log(‖Dϕ_T(x) ξ_M(x)‖ / ‖ξ_M(x)‖)` — an exponential growth rate, **units 1/time**, in the paper's continuous-time flow `ϕ_t`.
- The code confirms the units operationally: `gap` is used as `1/gap` to produce a lifetime in the same time units as `step * model.dt` (`dt = 0.1`).
- **The words "curvature", "damping"/"damped", "underdamped", "oscillat*", "noise", "stochastic", "temperature", "anchor", "restor*", "Langevin", "energy" do not appear anywhere in the document** (main text, appendices, captions, tables). Verified with a **positive control** on the same query (the searcher correctly returned "unit-axis" from §5.4 and "Unitary evolution recurrent neural networks / Arjovsky et al. (2016)" from the references before answering), per the standing rule that negative sweeps must be positive-controlled.
- **Therefore: the paper contains no potential-curvature (1/time²) quantity and performs no conversion of any kind between a rate and a curvature.** The "units question" is not raised, answered, or even acknowledged by N1.

### Q1.4 Protocol and baselines (S¹ path integration)

Verbatim / from Appendix B + Tables A3–A5:

- **Task:** "velocity-input S¹ path-integration"; "Velocity sequences are sampled from Gaussian, piecewise-constant, and correlated random-walk processes, and targets are vector outputs (cos φ_t, sin φ_t)". Baselines are "initialized from phase cue (cos φ₀, sin φ₀)"; the velocity generator and phase cue are **shared across models**.
- **Equivariant cell:** `dz/dt = a(I,h,u) z + b(I,h,u) J z`, `dh/dt = g(I,h,u)`, with `I = ‖z‖²` — rotation of `z` commutes with the field for each scalar input `u`. A **"broken control"** adds a non-equivariant perturbation to `dz/dt`.
- **Seeds: 6.** "Training runs / evaluation rows: 120 / 1920"; five model families (equivariant, broken equivariant, GRU, LSTM, orthogonal RNN), **24 runs each**.
- **Training config (Table A4):** hidden size 16, batch 64, **120 training steps per run**, AdamW, grad clip 1.0, training horizons **32 and 64**, full-phase and restricted-phase initializations. Params / LR: equivariant 1237 @ 3e-3; broken 1237 @ 3e-3; GRU 994 @ 2e-3; LSTM 1346 @ 2e-3; orthogonal RNN 386 @ 2e-3.
- **Headline diagnostics (Table A3):** max trained exact step equivariance error **3.19×10⁻⁸**; mean zero-input direct group-tangent exponent **−2.28×10⁻⁶**; mean zero-input principal angle **2.97×10⁻² deg**; max finite-time principal angle **9.11 deg**; learned pseudo-gap log-lifetime correlation **0.9991**; uncensored fraction **0.857**.
- **Stronger-baseline check (§B.1, Table A5)** — circular RMSE at test horizon 256, speed scale 1.8 (3 seeds, hidden 32, 500 steps, exact matrix-exponential orthogonal constraint): equivariant reference **0.2554 ± 0.016** (full-phase); GRU orig **1.748 ± 0.023** → stronger **1.212 ± 0.026**; LSTM orig **1.730 ± 0.021** → stronger **1.156 ± 0.029** (≈**4.5×** the equivariant reference); orthogonal RNN orig **1.790 ± 0.008** → stronger **1.749 ± 0.006**. Restricted-phase: GRU 1.619, LSTM 1.678, ortho-RNN 1.789 under stronger budget.
- **What is claimed (verbatim, §6):** "the learned comparison now uses six seeds, but **it remains a bounded comparison under one matched protocol rather than a claim that unconstrained recurrent networks cannot learn path integration**"; "the empirical claim is about the tested equivariant inductive bias and not about universal incapacity of generic recurrent networks"; and from the Intro, "**The task evidence is deliberately weaker** … they are not used as proof of the theorem." The author also concedes the original curves "do not certify baseline convergence" (GRU/LSTM val losses still improving at endpoint) — which is why §B.1 exists.

### Q1.5 The controlled breaking experiments

- **What was broken:** exact equivariance of the vector field, via explicit additive perturbations. Three named families — **weak-axis, unit-axis, rotated-strong** (confirmed independently in `conference_fix.py`: `colors = {"weak_axis": …, "unit_axis": …, "rotated_strong": …}`) — plus a **"random anisotropic breaking ensemble."** Separately, Fig. A7 breaks the *trained* cell post-hoc.
- **How measured:** breaking magnitude ε → equivariance error `E_eq` (eq. 4) → direct group-tangent exponent (eq. 5) → `gap` → measured vs predicted lifetime.
- **Over what range: NOT STATED in the paper and not recoverable from the public repo** (ε values live in the withheld `exp19` CSV). Declared unavailable.
- **What was reported (verbatim, §5.4):** "The measured lifetime agrees with the predicted pseudo-gap lifetime across weak-axis, unit-axis, and rotated-strong perturbations"; "Increasing the breaking magnitude shortens the lifetime to angular threshold, which is the expected behavior in these controls when a formerly protected zero exponent moves away from zero"; "The random anisotropic breaking ensemble shows measured symmetry-direction exponents tracking perturbative predictions, with color indicating the corresponding equivariance error"; "…correlation 0.9999999886, uncensored fraction 1.0, and median measured/predicted lifetime ratio about 1.013, **so the pseudo-gap result is a quantitative breaking consequence rather than an informal visual trend.**"
- **Fig. 4 caption (verbatim):** "Symmetry breaking opens pseudo-gaps. (A) Measured lifetimes match predicted gap-controlled lifetimes. (B) Memory lifetime decreases as breaking magnitude ϵ increases. (C) In random anisotropic breaking, measured symmetry-direction exponents track perturbative predictions and scale with equivariance error. These panels support the pseudo-gap consequence for the explicit breaking families tested here."
- **Related control (Table A2):** an "autonomous-flow zero-exponent diagnostic" separates group-tangent zeros from the time-translation zero via `rank E^G` vs `rank [f, E^G]`; the "collapse counterexample" row (rank 1 vs 2) is "excluded by Assumption 1".

### Q1.6 Anchor / temperature / noise / damping / underdamped

**None of these appear in the paper at all** (see Q1.3 — positive-controlled absence sweep). N1 is a **deterministic, noiseless, undamped, autonomous C¹-flow** paper. There is no corrective/anchor mechanism, no stochastic dynamics, no thermal parameter, no second-order/oscillatory regime. Its only "corrective" content is the broken-control ablation. Its §6 explicitly restricts Theorem 1 to **autonomous** flows and calls the input-driven trained task "task-level evidence rather than theorem proof."

---

## Q2 — NeurReps 2026 Findings Track

Sources: `https://neurreps.org/` (the 2026 site; sections *Call for Papers*, *Proceedings Track*, *Extended Abstract Track*, *Findings Track*, *What We're Looking For*, *A Novel Findings Track*, *Novel Findings Advisory Board*, *Dual Submission Policy*, *Past Editions* — **no FAQ section exists on the page**), and the OpenReview venue group `NeurIPS.cc/2026/Workshop/NeurReps_Findings` (JSON via api2). Retrieved 2026-08-21.

### Q2.1 ⭐ Archival status — **UNSTATED. Not published, not declared non-archival. Unverifiable.**

This is a *contrastive* absence, which is what makes it strong: the same page states archival status **explicitly for the other two tracks and omits it for Findings**.

- Proceedings Track (verbatim): "Self-contained, highly-developed research papers. **Archivally published in a dedicated PMLR volume.** Double-blind review via OpenReview."
- Extended Abstract Track (verbatim): "Early-stage results, negative findings, opinion pieces, or novel datasets. **Non-archival — may be posted to arXiv.** Double-blind review via OpenReview."
- Findings Track (verbatim, complete): "**New this year: high-impact collaborative work between experimentalists and theorists, in any standard preprint format. Single-blind, editorially reviewed by an advisory panel of experts in the field.**" — **no archival sentence.**
- Dual Submission Policy (verbatim, complete): "**Papers in the Proceedings Track will be archivally published.** Thus, submissions containing content that has been published or is under review elsewhere must include at least 30% new, unpublished/unsubmitted material. Likewise, to publish a NeurReps paper in another venue down the line, authors must add at least 30% new material. **There are no restrictions on Extended Abstract submissions.**" — **the Findings Track is not mentioned in the dual-submission policy at all**, so even the *derived* answer ("does submitting bar later publication?") is undefined for it.
- A targeted sweep of the entire page for "archival / archivally / non-archival / PMLR / proceedings / arXiv / published" returned hits only under Proceedings, Extended Abstract, and Dual Submission Policy; **zero under Findings.**
- OpenReview group JSON (`api2.openreview.net/groups?id=NeurIPS.cc/2026/Workshop/NeurReps_Findings`): title "NeurIPS 2026 Workshop on Symmetry and Geometry in Neural Representations (Findings Track)", location Sydney, start Dec 11 2026, contact organizers@neurreps.org, decisions `Accept (Oral)` / `Accept (Poster)`, `public_submissions: false`. **No archival field, no page limit, no eligibility text.**
- No prior-edition precedent exists (Q2.4).

**Declared: archival status of the NeurReps 2026 Findings Track is not stated on any venue-owned surface reachable on 2026-08-21.** The only route to a definitive answer is emailing organizers@neurreps.org. I did not do so (read-only, and out of scope).

### Q2.2 Page limit — **none.**

Verbatim: "**no page limit**" (track card) and, from *A Novel Findings Track*: "The track is designed with **minimal barriers to entry**: **no page limits, and any standard preprint format is welcome.**" (Compare: Proceedings "9 pages, excl. refs + appendices"; Extended Abstract "4 pages, excl. refs + appendices".)

### Q2.3 Eligibility — **descriptive of what is sought, with an explicit lowered technical bar; not phrased as a hard gate.**

Full verbatim text of *A Novel Findings Track*:

> "NeurReps has long been a primary home for broad computational and theoretical neuroscience work outside the scope of traditional machine learning venues. To honor and extend NeurIPS's historic ties to systems neuroscience, the workshop is introducing a **Findings Track** for high-impact collaborative work between experimentalists and theorists — early versions of work of the caliber published in venues such as *Cell*, *Nature*, or *Science*, with the goal of early community exposure and dialogue between ML researchers and experimental neuroscientists.
>
> The track is designed with **minimal barriers to entry**: no page limits, and any standard preprint format is welcome. **No complex machine learning or deep learning is required — just some form of geometry, topology, or algebra.** For example, work such as Gardner et al.'s *Toroidal topology of population activity in grid cells* (*Nature*, 2022) would be a natural fit. Because lab and dataset identity are often inseparable from the work, submissions are **single-blind**. Contributors present as **posters**, with a subset selected for **spotlight talks** by an advisory board of experts in the field."

Note the exact grammar: the track is *"for"* collaborative experimentalist–theorist work — a statement of purpose and an exemplar ("would be a natural fit"), with **no eligibility clause, no "must", and no stated desk-reject criterion.** The page nowhere says a submission without an experimental collaborator is ineligible; it also nowhere says such a submission is welcome. **The site does not resolve this; I am not inferring it.** Note also the *editorial* review model: "Editorial review for the Findings Track is led by the following advisory board: Eva Dyer, Chethan Pandarinath, Alex Williams, Jonathan Pillow, Maneesh Sahani, Matthew Perich, Andreas Tolias" — an all-systems-neuroscience panel.

### Q2.4 Did the track exist in 2024/2025? — **No. New in 2026.**

- The site itself says "**New this year**" and titles the section "**A Novel Findings Track**".
- **2025 (4th edition, NeurIPS 2025, San Diego):** two tracks only — Proceedings (9 pp, PMLR) and Extended Abstract (4 pp); the workshop's own X account announced "**Two tracks: Proceedings (9 pages) and Extended Abstract (4 pages).**"
- **2024:** two tracks only — Proceedings (9 pp, PMLR) and Extended Abstract (4 pp, "not included in the PMLR volume, but authors could post to arXiv under the NeurReps index"); proceedings in **PMLR v228**.
- Consequently **nothing has ever been accepted into a Findings Track**; there is no precedent corpus to inspect. *(2025 track count: two independent sources — the workshop X account and the comp-neuro list CFP. 2024: CFP + PMLR volume. Confidence: high.)*

### Q2.5 Deadline and submission mechanism

- **Site (verbatim):** "Submission Deadline **August 24, 2026 · AoE**"; "Accept / Reject Notification **September 29, 2026**". One deadline for all three tracks.
- **OpenReview group JSON:** submission deadline "**Aug 25 2026 11:59AM UTC-0**". These are **consistent** (Aug 24 23:59 AoE = UTC−12 = Aug 25 11:59 UTC), not a discrepancy.
- **Mechanism:** OpenReview, dedicated per-track venue. Findings: `https://openreview.net/group?id=NeurIPS.cc/2026/Workshop/NeurReps_Findings` (link text "Submit on OpenReview →"). Workshop date **Dec 11 2026**, Sydney.
- **Profile rule (verbatim):** "All submitting authors need an OpenReview profile for the Proceedings and Extended Abstract tracks — creating one can take a few days, so please don't wait until the deadline." / "**For the Findings track, only the submitting author needs a profile; co-authors can be added by email.**"
- ⏰ **Timing fact (not a recommendation):** today is 2026-08-21; the deadline is **3 days out**.

---

## Confidence & gaps

| Item | Confidence | Basis |
|---|---|---|
| N1 has **no** closed-form gap→lifetime law in its text | **High** | Exhaustive equation inventory (5 eqns), exhaustive "lifetime"/"pseudo-gap" occurrence sweeps, §4/§5.4/§6/App. A/B/C read verbatim; author's own future-work framing in §6 |
| `T = −ln(1−θ*/|φ₀|)/λ`, exponent −1 | **High for exp31 (learned cell); medium for Fig. 4** | Verbatim code from the paper-linked repo; `exp19` (Fig. 4) script/CSV **not public** → single-sourced |
| Gap is a Lyapunov exponent, 1/time; no curvature anywhere | **High** | Eq. (5) + positive-controlled absence sweep |
| Absences (noise/damping/anchor/temperature/underdamped/energy) | **Medium-high** | Positive control passed, but the sweep ran through a summarising fetch layer, not a local grep |
| ε range of the breaking sweep | **Unrecoverable** | Not in paper; `exp19` data withheld |
| Findings Track archival status | **High that it is UNSTATED**; **zero** on what it actually is | Site sweep + OpenReview JSON + no FAQ + no prior edition |
| Findings eligibility hard-vs-soft | **Unresolved by the venue** | No "must"/desk-reject language exists either way |

**Methodological caveat (honesty):** arXiv's PDF text layer was not extractable by my fetch tool (it returned only the PDF skeleton). The full text was read via the **arXiv HTML v1 rendering** (`arxiv.org/html/2605.03338v1`) through ~10 targeted verbatim-extraction passes, each scoped to a section, table, or string search; long verbatim blocks (§6, "Claims and evidence", Tables 1/A1–A5, Fig. 4/A1–A12 captions) were returned intact and mutually consistent across passes. I did not have a local copy to grep. Two independent passes agree on every load-bearing number quoted above.

**What to search next (if the Advisor wants closure):** (i) email organizers@neurreps.org for the Findings archival status — it is the only remaining route; (ii) if the ε range matters, email the N1 author or check for a `results/journal_full` release in a later repo commit; (iii) re-check arXiv for a v2 of 2605.03338 before any citation ships (only v1 exists as of 2026-08-21).

---

## Reconciliation list (needs an owner, per §5 of the protocol)

1. **Any V2/CHLU text that describes N1 as containing a "lifetime law" or a "gap–lifetime relation" must be reworded** to "a qualitative prediction validated by a high correlation on a controlled breaking ensemble." N1 states no law.
2. **Any text asserting or denying that N1 handles units (1/time vs 1/time²)** must be updated to: *N1 never raises the question; "curvature" does not appear in the paper.*
3. **Any text that cites N1 as covering noise/damping/anchoring** must be corrected: N1 is deterministic, autonomous, noiseless, and undamped throughout.
4. **Any venue plan that assumes the Findings Track's archival status is known** must be re-gated: it is unstated at the source.

---

## Bibtex-ready refs

```bibtex
@misc{mo2026symmetryprotected,
  title        = {Symmetry-Protected Lyapunov Neutral Modes in Equivariant Recurrent Networks},
  author       = {Mo, Hanson Hanxuan},
  year         = {2026},
  month        = may,
  eprint       = {2605.03338},
  archivePrefix= {arXiv},
  primaryClass = {cs.NE},
  note         = {v1, submitted 5 May 2026; also math.DS. Preprint, not peer-reviewed.},
  url          = {https://arxiv.org/abs/2605.03338}
}

@misc{mo2026symmetryprotected_code,
  title        = {Code for "Symmetry-Protected Lyapunov Neutral Modes in Equivariant Recurrent Networks"},
  author       = {Mo, Hanson Hanxuan},
  year         = {2026},
  howpublished = {\url{https://github.com/NeuronalDynamics/Symmetry-Protected-Lyapunov-Neutral-Modes-in-Equivariant-Recurrent-Networks}},
  note         = {Retrieved 2026-08-21. exp19 (Fig. 4) script and CSV not included.}
}

@misc{neurreps2026cfp,
  title        = {NeurReps 2026 Call for Papers: Symmetry and Geometry in Neural Representations},
  author       = {{NeurReps Organizers}},
  year         = {2026},
  howpublished = {\url{https://neurreps.org/}},
  note         = {NeurIPS 2026 Workshop, Sydney, 11 Dec 2026. Retrieved 2026-08-21.
                  Findings Track archival status not stated.}
}

@misc{gardner2022toroidal,
  title   = {Toroidal topology of population activity in grid cells},
  author  = {Gardner, Richard J. and Hermansen, Erik and Pachitariu, Marius and Burak, Yoram
             and Baas, Nils A. and Dunn, Benjamin A. and Moser, May-Britt and Moser, Edvard I.},
  journal = {Nature},
  volume  = {602},
  pages   = {123--128},
  year    = {2022},
  note    = {Cited by the NeurReps 2026 site as the exemplar of a natural Findings-Track fit.}
}
```
*(Gardner et al. metadata is from general knowledge, cross-checked only against the NeurReps site's own citation string "Gardner et al.'s *Toroidal topology of population activity in grid cells* (*Nature*, 2022)" — volume/page numbers are **not** independently re-verified in this session; verify before shipping.)*

---

## Proposed handover updates (for the Hub)

- **N1 (arXiv:2605.03338) is now read in full, not abstract-only.** Verdict: **theorem about zero-exponent multiplicity + a qualitative pseudo-gap→lifetime prediction validated empirically (corr 0.9999999886, median ratio 1.013). No closed-form lifetime law, no formal definition of "pseudo-gap", no exponent derivation, no crossover, no floor.** The only lifetime formula is `T = −ln(1−θ*/|φ₀|)/λ` and it exists **only in the released code**, not in the paper.
- **The units question is not raised by N1 at all.** λ is a Lyapunov exponent (1/time, eq. 5); "curvature" and "1/time²" never appear; no conversion is performed.
- **N1 is deterministic/autonomous/noiseless/undamped** — no anchor, temperature, noise, damping, or underdamped regime anywhere. Its §6 restricts Theorem 1 to autonomous flows and calls the input-driven learned task "task-level evidence rather than theorem proof."
- **N1 reproducibility gap:** the headline Fig. 4 correlation depends on an `exp19` CSV that is **not in the public repo**; the ε sweep range is unrecoverable. The learned-cell replication (corr 0.9991) drops **14.3 %** of rows as right-censored.
- **NeurReps 2026 Findings Track: archival status UNSTATED at source** (site states it explicitly for the other two tracks, omits it for Findings; not mentioned in the dual-submission policy; absent from the OpenReview group JSON; no FAQ; no prior-edition precedent — the track is new in 2026). Under the standing rule this bars the track unless organizers@neurreps.org confirms.
- Findings facts: **no page limit**, any standard preprint format, **single-blind**, editorial review by a 7-member systems-neuro advisory board, posters + spotlight subset, **deadline Aug 24 2026 AoE (= Aug 25 11:59 UTC)**, notification Sep 29 2026, OpenReview `NeurIPS.cc/2026/Workshop/NeurReps_Findings`, only the submitting author needs an OpenReview profile. Eligibility language is **descriptive, not a stated hard gate** — the site does not resolve it either way.
- **Reconciliation list above has four items and needs an owner assigned at the review that accepts this report.**

## Flags

1. 🚩 **Load-bearing absence, positive-controlled but tool-mediated:** the "no lifetime law / no curvature / no noise" findings rest on string sweeps executed through a summarising fetch layer over arXiv HTML, not a local grep. Positive control passed ("unit-axis", "unitary" both retrieved), and the equation inventory (5 eqns) is corroborated by §4 and §5.4 read verbatim. Residual risk: low but nonzero.
2. 🚩 **Single-sourced:** the `T ∝ 1/λ` formula for the **main** Fig. 4 experiment. Verified only in `exp31` (the learned-cell path); `exp19` is withheld. I assume-but-cannot-prove Fig. 4 uses the same predictor.
3. 🚩 **Unrecoverable, declared not inferred:** (a) the ε range of the breaking sweep; (b) the archival status of the Findings Track; (c) whether the experimentalist–theorist framing is a hard eligibility gate.
4. 🚩 **Timing:** deadline is 2026-08-24 AoE — **3 days from today**. Reported as fact; the venue call is the Advisor's and the Head's.
5. 🚩 **PDF text extraction failed** (tool limitation); all quotes are from the arXiv HTML v1 rendering. If any quote will be reproduced verbatim in a paper, re-verify against the PDF locally.
6. 🚩 **N1 is a v1 preprint by a sole author with no stated venue** — cite as preprint; re-check for v2/publication before shipping.
