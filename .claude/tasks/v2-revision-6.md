# v2-revision-6 — paper-writer (V2 → v0.7: the Q11 re-centering + the colleague primer + the stale-delta fold)

**Commissioned by the Shorts Advisor (charter: `.claude/advisor-head-shorts-charter.md`, Addendum 16; Head rulings Q11 (Add.8), primer integration (Add.11), quality-first posture (Add.8 Q7), 2026-08-18).** Read `.claude/AGENT_PROTOCOL.md`, then this file. You edit only `papers/v2-short/` (draft.md canonical, draft.tex + CHANGELOG.md per house practice). ⛔ `.claude/colleague/main.tex` is NEVER edited — it is a source you adapt from.

**DIAL DECLARATION: none — revision pass; no new measurement; no laundering control applies.**

## Inputs (read in this order)
1. `papers/v2-short/draft.md` (v0.6) + CHANGELOG.
2. **Addendum 9 + `.claude/outputs/v2-colleague-physics-review.md`** — the primer's verified audit: riders R1–R6, symbol collisions, S1, integration Option 3.
3. `.claude/colleague/main.tex` — the primer source (25 lines, 2026-08-14).
4. `.claude/outputs/shorts-evidence-inventory.md` §2 (V2's delta map — ⚠ built at v2.11/N224, navigation not completeness) + `claims_matrix.md` CM-15 · CM-16a/CM-16b (⛔ never "CM-16" whole) · CM-17 · CM-21 + registry N46 · N149/N150.

## The changes (all Head-ruled; your judgment is in execution, not scope)

**1 — Q11 re-centering (Add.8 Q11: V5 owns the V-curve/critical-damping law).**
Abstract + contribution list lead with V2's distinctive ground: the mode-mass budget · the Mo head-to-head (Fig 2 stays the headline) · GMOR-proper · designed-vs-emergent + price-of-physics · the realization taxonomy. The critical-damping retention minimum is REMOVED from the contribution list and demoted to a supporting result (a §3 sub-result or appendix, your call) presented as a corollary of the budget — no headline framing, no "optimum" language in the abstract. ⛔ C-8 hermetic: no reference of any kind to any other short of ours; overlapping supporting content is fine (Q3), cross-citation is not.

**2 — The colleague primer enters as the OPENING APPENDIX (Add.11; the Head's shape: first section of the appendix block, a primer).**
Adapt the primer's text into a new first appendix ("An SO(2) symmetry-breaking primer for ML readers" or similar; reletter downstream appendices as needed). Mandatory, from the accepted theorist audit:
- **Symbol renames:** α→`s` (group parameter, F5 house style) · J→`X` (generator; J stays the Jacobian) · ω→`θ₀`/`ϑ` (vacuum angle) · H→`𝓗` (subgroup). Also: American spelling; the T1/T2/T3/T5 notation fixes (H ⊂ G as subgroup; "one NG mode per broken generator; the NG fields parametrize G/H"; group-vs-algebra phrasing; display the stationarity condition and evaluate Eq. 2 at q*).
- **S1 reword:** the closing Wigner–Weyl sentence becomes "does not *force* a zero spectral mass — the orbit argument degenerates (Xq* = 0) and gives no information about the eigenvalue" (the theorist's recommended fix). ⚠ FLAG in the CHANGELOG: this wording awaits the colleague's sign-off via the Head — mark the sentence with an inline comment in draft.md source.
- **The boxed scope note at the appendix head** (App-J house pattern), discharging R1–R4 in one place: designed-exact-SO(2)-only (N46/CM-16a rider verbatim) · T=0 scope with a pointer to the D_θ diffusion law (§2.1/App J; CM-16b) · the γ>0 latch clause (flat = neutrality, storage needs damping; theory-note Thm-latch) · the Mo (2026) citation with the sufficient-not-necessary pointer to §3.4.
- **The whitening sentence** (T4): bare vs mass-whitened Hessian, zero mode exact by Sylvester, pointer to §1 nomenclature. Plus the channel-vs-unit clarification ("two-dimensional latent space" = the channel plane of a dim-4 unit).
- **§2.1 remains the document's controlling SSB definition** (R6) — the primer defers to it explicitly and never replaces it.

**3 — The stale-delta fold.**
- **N149/N150 blast-radius sweep:** every lifetime/tilt/ε sentence in the draft checked; any pseudo-Goldstone-tilt or ε-lifetime wording carries the learned-store sign-refutation rider and the `τ_max = Γ/2α` ceiling where applicable. This is v0.6's biggest known gap (it predates both entries).
- **CM-21 completeness:** all four retired positioning claims (HiPPO/Kong/NTM/transformer-competitor) verified absent or properly retired in the text.
- **CAFE quote-restrictions** (w19+): verify nothing in the draft quotes restricted CAFE material.
- Sweep the final text against matrix §0.1–§0.14 (per-file greps only, positive controls, zero-hit list printed in the CHANGELOG entry).

**4 — The scale line (Add.8 Q5/Q9).** The future-work section gains the scope-choice sentence (scale results deliberately out of scope for this short) and may name specific scaling experiments as future work. ⛔ No C3-era number, no implication that a scale result exists.

**5 — The §A20.5 substrate-scope sentence** in the paper's own voice, once: "these laws govern the store; end-to-end performance additionally depends on the encoder, measured separately, φ-bytes ledgered."

**6 — Venue-class header** → NeurReps Extended-Abstract class: 4 pp main text excl. references+appendices, non-archival, double-blind. Main text must fit the 4-pp target — with the primer in the appendix this is achievable; if the re-centering leaves main text over budget, appendix-maximalism decides what moves (nothing is cut outright).

## Guards
- ⛔ Lexical false friends (Add.10 F3): V2's "friction-field composition/placement" negatives (N12–N15, App F) are NOT the demoted compositional family — do not rename or scope them.
- ⛔ CM-16a/CM-16b are always cited split, never as "CM-16".
- ⛔ "CLU-former" never appears; the CLU name-debut form ("the CLU, introduced as CHLU in Jawahar & Pierini (2026)", third person) is unchanged.
- Anonymization discipline unchanged; every quantitative statement keeps its flag-provenance row (App A updated for any moved result).

## Acceptance criteria
1. CHANGELOG entry lists every changed passage with its authority (ruling/registry row/report section); the S1 sign-off flag is present.
2. The primer appendix carries the boxed scope note with all four riders; all four symbol renames verified by grep (no stray α-as-angle, J-as-generator, ω-as-vacuum-angle, bare-H-as-subgroup in the appendix).
3. The contribution list contains no critical-damping/V-curve headline; Fig 2 remains the headline figure.
4. Never-quote sweep printed (per-file, positive-controlled).
5. Zero edits outside `papers/v2-short/`; `.claude/colleague/main.tex` byte-untouched.
