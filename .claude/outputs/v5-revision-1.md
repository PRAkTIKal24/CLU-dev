# v5-revision-1 — paper-writer report

**Task + acceptance criterion:** close all 14 MUST-FIX / 12 SHOULD-FIX / 6 NICE of `v5-referee-v02` plus the 7 reconciliation items of `v5-scope-scout` in `papers/v5-short/`, with zero new measurements, a resolving bibliography, correct compiled cross-references, and the 4 pp main-text limit held.
**Status: done.** V5 is at **v0.3**; `draft.md` (canonical), `draft.tex` (generated) and `draft.pdf` all rebuilt; `CHANGELOG.md` carries the one-line v0.3 entry.

**DIAL DECLARATION (echoed): none — revision pass; ⛔ zero new measurements. No number was changed, added, rounded or smoothed.** Every figure in this revision is a **replot of banked JSON** produced by the original harnesses (`outputs/t-lever-forgetting/s4b_jacobian.json`, `outputs/v5-gate/{e1c_vcurve,e1_jacobian,e0_geometry,r3main_results,r3t6_g50_results}.json`, `outputs/t-lever-forgetting/s2_dlaw_cells.json` …); the appendix figures are the **source scripts re-executed verbatim except for leaked labels and output paths**. Provenance for every plotted number is unchanged and still sits in App A.2/A.3/A.4.

> ## ⚠ DOWNSTREAM RECONCILIATION LIST — READ FIRST (needs a named owner)
> 1. **The N131 citation fence is still wrong in the matrix/registry.** The draft now prints the corrected form (Guo **§2, Eq. (1), the ε-only condition**, with the unnumbered (ε,δ) display after it) at all three sites, per the scout and my task's explicit instruction to follow the source over the stale fence. **Until the Hub amends N131, the matrix and the draft disagree** and a future compliance sweep will "correct" the draft back to the error. Owner: Hub.
> 2. **The `13.9×`-family never-quote list should gain the MF-8 formula**, not just the number: `γ_eff/(2−γ_eff)` as *the vault ratio* is a distinct, checkable error that survived two drafts. Proposed registry entry below.
> 3. **The V2↔V5 lockstep now has one deliberate divergence I could not resolve from V5's side:** V2's App K.4 instrument note says the Hessian counterpart is `μ² ≤ 2.4×10⁻¹⁵` (V2's checkpoints, "13–14 orders"); V5's own checkpoints give `|μ²| ≈ 2.9×10⁻¹⁶` and `≈12 orders` in `1−|λ_coset|`. I wrote V5's numbers, sourced to V5's own reports (C.1/C.2). **Both are right on their own instruments and both are now labelled by grid — but a reader with both papers sees two different "designed flatness" numbers.** Owner: Hub, at the next lockstep check.
> 4. **Internal report slugs still appear in the provenance tables** (`t-lever-forgetting`, `v5-gate`, `mia-decay-measurement`, `placement-landing`, `deletion-waitlist-stiffness`, `order-independent-placement`, `c2w10-lifecycle-mechanics`, `sleep-erosion-study`). SF-12's paths are gone; these slugs are the C-7 provenance convention and I left them, but they are the same *class* of internal identifier. **Head/Hub call before the supplement ships.** (I did remove the one internal *process*-document reference, `venue-follow-up` §3, from App H.)

---

## 1. What I did — MF/SF/NICE closure table (with locations)

*Section numbers below are the NEW numbering: Results = §2, Related work = §3, Limitations = §4.*

| item | closed where | note |
|---|---|---|
| **MF-1** Jacobian-vs-Hessian instrument note | §2.1 ("Two instrument caveats"), **C.3** (full form) | V2 K.4's note imported and re-based on V5's own numbers: the `1.7×10⁻¹²` endpoint is the **ring-profile probe floor**, the Hessian μ² of the same checkpoint is machine zero (`≈2.9×10⁻¹⁶`, C.1). Both numbers cross-pointed. |
| **MF-2** "two values of μ" | §1 contributions, §2.1 ¶1 | Printed as **"three regimes of one curve, evaluated at two values of μ, not two laws"** (CM-16b). The banned "three values of μ" appears nowhere. |
| **MF-3** Jacobian-vs-rollout | §2.1, **C.3**, Limitations **(iv)** | `233.6/653.3/249.0` vs `190/370/150`, `1.23–1.77×`, "19–43% faster", with the linear-response-vs-finite-amplitude-write reason and why the Jacobian is quoted. **Not** replaced by a promissory note; ME-1 is named only as an unrun future experiment. |
| **MF-4** asymptotic windows + exponent gap | §2.1 ¶1 (`γ<γ_crit/2`, `γ>2γ_crit`), §2.1 caveats, **B.6**, **C.3** | C.3 states both fit windows (designed: γ_crit/2 and 2γ_crit; emergent: a factor 2.5 either side of the refined argmin — read from the harness) and says plainly that both overdamped exponents exceed the continuum +1 on finite windows of a discrete map, that we do not claim they agree, and that C.5 forbids reading anything designed-vs-emergent off raw exponents. |
| **MF-5** headline figure | **Fig. 1** = `figs/fig1_collapse.png`; pseudo-Goldstone panel → **Fig. C.3** | New collapse: `n₁/₂/n₁/₂^min` vs `γ/γ_crit`, 5 designed radial + 3 emergent coset curves, ∓1 asymptotes, log-μ² colourbar from `10⁻¹²` to `2` with the flat coset marked at the `1.7×10⁻¹²` probe floor. "Cor-13" gone; "T5"/"T6" gone from Fig. D.1 with the D̂ vault quoted on the panel and the raw-FPT bars labelled raw. Also stripped: "F5 Cor-13" (fig B.3), "CM-16" (fig C.2), "Fig 1/2/4" suptitles. |
| **MF-6** References | **§References** (58 entries) | From the scout's verified records + the v2-cite-check carried entries; page-free per Add.20. Every body citation resolves (checked programmatically; the only unresolved token is the word "Aug" in a date). |
| **MF-7** cross-references | whole document | Nine `§3.5` danglers eliminated; sections renumbered so markdown and PDF agree (`\section{Results}` = §2 ⇒ all refs are §2.x); Limitations **(iv)** restored as a real limitation (the linear-response instrument); no `§2` gap. Compiled `\S` inventory: `§2` ×4 (3 of them Guo's own §2 + one source-report range), `§2.1` ×5, `§2.2` ×14, `§2.3` ×4, `§4` ×3 — all resolve. |
| **MF-8** D.1 arithmetic | **D.1** | `[γ_eff(2−γ)]/[γ(2−γ_eff)]`, evaluated at γ=0.05, γ_φ=0.5 (γ_eff=0.525) = `13.88×`. The wrong form appears nowhere. |
| **MF-9** TTL-flag laundering control | §2.3 ("The tighter laundering control fires"), **E.5(vi)** (new 3-row table) | `0.983` vs `1.000` (query, exact), `1.000` vs `1.000` (white-box), `0.559` vs `0.996` (σ_obs=0.1); framed as the referee asked — retrieval geometry is the honest survivor, not a preferred metric. |
| **MF-10** 2503.21536-as-CD | **App H(a)** and **K.2** | Both sentences deleted. Substrate is now Fischer & Igel **2010, 2011** + Nijkamp 2020; Toledo-Marín et al. 2025 is kept and described accurately (hierarchical feature learning, an *initialization* symmetry, CD/PCD only as sampling implementations) and explicitly labelled *not a precedent*. |
| **MF-11** k-regime clause | **App H(c)** | V2 v0.8's **printed** form, verbatim in substance: it does not assert where the mixing time sits, only that our sweep does not resolve it, so the finding is "frequency-decisive across the two chain lengths we run". |
| **MF-12** stale scout marker | **App H** closing | Replaced by the executed two-instrument search with its surface list and its explicit "absence over the surfaces listed, not proof of none" coverage statement. |
| **MF-13** abstract riders | **Abstract** | Vault: "on the designed register (3 seeds)". Deletion: three stated conditions verbatim + recency exclusion + the dim-3 / capacity-8–64 / no-learning scale. |
| **MF-14** self-containment + "companion" | §1, **A.0** | A.0 now derives the reduction in place (linearise at a critical point, diagonalise the mass-whitened Hessian, one 2×2 matrix per mode fixed by (εμ,γ), spectral radius = retention, complex↔real crossover at εμ≈γ/2 = the γ*≈2εμ minimum). §1 carries the one-sentence version. **"companion" is gone**; the note is cited as "(Anonymous, 2026)" with the sentence "*no result here depends on it*" — the paragraph survives the citation's deletion. |
| **SF-1** two flatness numbers | **C.2** | Labelled by grid: `1.7×10⁻¹⁴` is A.2's 22-point designed grid, `1.1×10⁻¹⁵` is A.3's 48-point designed control. |
| **SF-2** packing numbers | **E.3** | `61/64` admitted (σ=0), per-admitted `1.0000`, per-offered `0.9531`; ×1.05 → `64/64`, `1.0000`; refuse-and-relocate `43/64`, `0.6719`; churn `2.836` moves/delete at full load. Main text keeps the claim + pointer only (page budget). |
| **SF-3** \|c\| clause | §2.3 (in-sentence) and **E.5(v)** | E.5 carries the full form: controller-placed disk vs a max-radius ring, `0.500` at A=0.06 against `0.886`. |
| **SF-4** Fig. D.1 | `figs/fig2_vault.png` + caption | T5/T6 stripped; D̂ vault `107.77±4.78×` on the panel; bars relabelled "84×/86×/91× raw"; caption says the bars are boundary-layer biased and are not the quoted number. |
| **SF-5** five orphan PNGs | **Figs B.1, B.2, B.3, C.1, C.2** | All embedded with captions and C-2 labels; zero figures referenced by bare filename; zero orphan PNGs on disk. |
| **SF-6** window qualifiers | §2.1, B.6 | See MF-4. |
| **SF-7** venue-native §4 | **§3** (rewritten) + **K.2** (new opening subsection) | Contrast-not-competition register throughout: "We benchmark against none of them; the contrast is structural." Main text cites Zep, Generative Agents, Titans, Ghost Vectors; K.2 carries the full 13-work brief (MemGPT, Mem0, Infini-attention, MemoryBank, Expire-Span, ForgetEval, Memora, MemLeak, Agentic Unlearning). |
| **SF-8 / SF-9** §3.4 → App F, contribution (2) | **App F.0** (verbatim), §1 contributions | The §0.13 lifecycle wording, both riders and the declared-not-run label are preserved **verbatim** at the head of App F. Contribution (2) is now the **composition** claim (packing certificate + decaying content + delete/decay commutation + one energy function), with the exactness measurement as its consequence. |
| **SF-10 / R6** competing work | **E.7**, **K.2** | "a 2026 preprint", `arXiv:2603.15033`, Laguna et al. — **no presentation type**; the numbers are now forget-set MIA AUROC `51.4%/51.2%` and "its **average gap to the retrained oracle across four metrics** is 0.56±0.21". |
| **SF-11** falsifier | §2.1 (last sentence of the caveats paragraph) | "a stored direction that mixes normal modes, or is anharmonic at the write amplitude, has no single-mode reduction, so the curves would neither collapse nor put their minimum at 2εμ". |
| **SF-12** paths | A.5–A.9, E.3 | `chlu/…` → "a placement module (286 lines, pure numpy)"; `.claude/scratch/…` → "a standalone reference implementation (`pgcp.py`, …)"; `../CHLU-waitlist` / `../CHLU-c2w10` → "a separate worktree"; "main venv" → "shared virtualenv"; the cwd/PYTHONPATH note reworded. **Hashes kept.** Sweep: `chlu/`, `CHLU-`, `.claude` = 0 in both `draft.md` and `draft.tex`. |
| **N-1** | §2.3 | TTL radius quoted as `0.75–0.77`. |
| **N-2** | front matter | Marked "drafting furniture, not part of the submitted document"; it is stripped by the md→tex generator and never reaches the PDF. |
| **N-3** | Fig. 1, Fig. C.3 | y-axis is now `n₁/₂/n₁/₂^min`; C.3's is "n₁/₂ (steps), from the one-step Jacobian". |
| **N-4** | Fig. 1 | The collapse is by `n_min`, so the "3/3 seeds, one curve" claim is visually true. |
| **N-5** | §2.2 | The designed-symmetry precondition has its own bolded lead-in paragraph. |
| **N-6** | C.2 | Grid minimum `7.6×10⁻⁵–2.0×10⁻⁴` printed against a designed grid minimum of exactly 0. |

### Scout reconciliation items
- **R1 (Guo)** — corrected at **all three sites** (§3, E.7, K.2): §2, Eq. (1), the **ε-only** condition; the (ε,δ) relaxation is the unnumbered display after it; "not a numbered Definition, and we cite it that way". ⚠ See reconciliation item 1 above.
- **R2 (lineage)** — rewritten in **K.2** and **E.2**: Snyder (FOCS'77) → Sundar & Tarjan / Andersson & Ottmann; Micciancio = *oblivious*, explicitly not canonical; WHI/SHI = Naor & Teague; Hartline et al. for the equivalence (and "**their** theorem", not "his").
- **R3 (Mo)** — "**at least** dim(G/ℋ)" at both sites (§3, K.2).
- **R4 (HLW)** — K.2 now attributes conformal symplecticity to **McLachlan & Perlmutter 2001** and cites **HLW 2006, Ch. XII** for leapfrog/stability only.
- **R5 (uncited laws)** — GMOR cited in A.0; Mermin & Wagner 1966 + Coleman 1973 cited in App G's first sentence.
- **R6 (MUNKEY)** — see SF-10 row. No venue, no presentation type.
- **R7 (Titans)** — cited in §3 (main text) and K.2, in the contrast register.
- **Template note** — the NeurIPS 2026 style is **not installed** on this machine. The measured 4 pp is a **generic-`article` approximation** and the build note says so explicitly, adding that the NeurIPS text block is narrower, so the margin is thin.

### Additional fixes I made that were not on the list
- Two Buchbinder–Petrank sentences gained the **scope of the exponential separation** (weak-vs-strong HI, comparison-based, heaps and queues) — E.2 and K.2 — per the scout's §3.2 item 6 warning.
- **Hidaka & Minami 2020** author order corrected in K.2 (and the "two Noether charges" claim sourced to **2018** only, with the four-type classification attributed to 2020).
- SILO / PALL / Ticketed / MUSE / CURE4Rec gained author-year forms so every citation resolves against the References list.
- The one internal *process*-document reference (`venue-follow-up` §3) removed from App H's source line.
- "companion" removed from C.6's heading (benign use, but it takes the semantic sweep to a clean zero).

---

## 2. How I verified

**Build (both artifacts, this session, `tectonic`):**
- Full document: `Output written on draft.xdv (25 pages, 163448 bytes)`; **0 TeX errors**, 0 undefined references; 36 warnings, all `Underfull`/`Overfull \hbox` (31/5), expected in a generic `article` class.
- Main text alone (abstract + §1–§4, references and appendices excluded per the venue's own text): `Output written on maincount.xdv (4 pages, 29744 bytes)`. **Main text = exactly 4 pp**, 2,241 words. Page 4 is full to the last line — there is no slack in this class.
- ⚠ **Caveat, stated because it matters:** the venue template is the NeurIPS 2026 style, which is not on this machine. NeurIPS's text block is *narrower* than `article`+1in margins, so **4 pp here is not a guarantee of 4 pp there**. The working title also runs three lines; a real title buys back roughly two.

**Never-quote sweep — per-file, positive-controlled, run on `draft.md` AND `draft.tex`:**
- Instrument LIVE: 12 positive controls, **12/12 found in `draft.md`**; 11/12 in `draft.tex`, the twelfth (`read_hits`) present only in its LaTeX-escaped form `read\_hits` (verified separately, 2 occurrences).
- 41 forbidden patterns (the v0.2 list + 12 new v0.3 patterns for MF-5/MF-8/MF-10/MF-11/MF-12/R1/R2/R3/R6/SF-12/C-8). **3 hits in each file, all inspected and compliant:**
  1. + 2. `certified removal` ×2 — both are the *literature description* of Guo's ε-certified removal (the CM-22(m) ban is on claiming it for our mechanism; the draft denies it explicitly at three sites). Regex false positive.
  3. `independent of chain length` — present **because** claim (c) says it, and the mandatory k-regime scope clause now follows in the same sentence group. This is the MF-11 closure, detected by design.
- Targeted leak sweep, both files: `chlu/` 0 · `CHLU-` 0 · `.claude` 0 · `§3.5` 0 · `Cor-13` 0 · `CM-16` 0 · `T5`/`T6` (as labels) 0 · `venue-follow-up` 0 · `(oral)` 0 · `13.9`/`≈14×` 0.
- **Semantic hermeticity: 0.** `companion` 0 · `sibling` 0 · `our other` 0 · `another of our` 0 · `forthcoming` 0 · `in preparation` 0 · `the program` 0 · `our shorts` 0 · `concurrent submission` 0. `Anonymous` appears 3× — §1 (non-load-bearing pointer), the References entry, and the anonymization note. `CHLU` appears twice: the mandated continuity sentence and the J&P reference title.
- **Citation resolution check (programmatic):** every `(Surname …, YEAR)` token in the body matches a surname in the References list. 58 entries.

**Figures — regenerated, not hand-edited:**
- `fig1_collapse.png` — new script (`scratch/v5-revision-1/fig1_collapse.py`) reading `s4b_jacobian.json` (5 designed seeds × 23 γ) and `e1c_vcurve.json` (3 emergent seeds × 48 γ). Printed provenance: designed `μ²_rad = 0.670302, 0.770891, 1.190122, 1.092699, 1.347820`; emergent `μ²_soft = 5.449e-2, 2.029e-2, 5.132e-2`. All are table entries from the source reports; nothing was recomputed into a new claim.
- The other six figures come from the **source harness scripts re-executed** (`scratch/v5-revision-1/regen_figs.py`), patched only where a title/annotation leaked an internal label or where the output path changed. Every patch is an `assert`-guarded string replacement, so a silent no-op is impossible.

---

## 3. Findings / editorial notes for the Hub

1. **The 4 pp limit is now the binding constraint on this paper, and it bit hard.** The referee estimated the MUST-FIX additions at ≈0.17 pp; the actual cost of MF-1/3/4/9/13 **plus SF-2/SF-3/SF-7/SF-11** was ≈1.9 pp. Landing at 4 pp required the entire move-menu **and** roughly 4,500 characters of line-by-line compression. The full move-menu is spent: §3.4 is in App F, the attribution block is at its one-clause in-line form, the trilemma is at its corner sentence, and the gated-stiffness R₅₀ numbers are in E.6. **There is no more slack in this class.** If the NeurIPS style measures over 4 pp, the next cut has to come from a do-not-cut item, and that is a Head decision, not a writer's.
2. **What I deliberately did *not* cut**, because the referee's list is binding: N108's sentence (3 sites), the CM-25(f) verbatim, the BG attribution, the score sentence, the §A20.5 substrate sentence, the designed-symmetry precondition paragraph, and the `fdt`+Newtonian fine print beside the T>0 claim.
3. **Two mandated items were placed in the appendix under the referee's own option.** MF-1 and MF-3 both say "§3.1 **or** App C.3". The main text carries the claim halves (probe floor; 19–43% with the numbers), C.3 carries the full statements including the fit windows and the ring-ripple validation figure. If the Advisor wants the full form in the main text, it costs ≈0.15 pp and something else must leave.
4. **The vault figure's per-seed D̂ numbers do not exist in any report**, only the aggregate `107.77 ± 4.78×`. I therefore annotated the aggregate on the panel and labelled the per-seed bars as raw FPT, rather than inventing per-seed D̂ values. This is the honest reading of SF-4 and it is the only place where I could not follow the instruction literally.
5. **The eleven-decade span is now defensible but it is still the paper's softest headline.** With the instrument note in place a reviewer can see exactly what the endpoints are; what they will still notice is that the *emergent* arm spans under half a decade and the rest is the designed corner. ME-3 (a vault on an emergent checkpoint) and ME-1 (the rollout-validated curve) remain the two runs that would change that.
6. **Missing-experiment notes (unchanged, none improvised):** ME-1 rollout-validated V-curve (named in §4 and C.3 as unrun), ME-3 emergent-checkpoint vault, ME-4 amortized per-delete cost (named in §4, E.7, App J #9), ME-5 occupancy sweep (named in E.5's quote-the-curve rider), ME-6 deletion at 10³ items (named in §4). I did not cite `v5-vcurve-validation`: its report does not exist on disk and ME-1 is separately scoped, so per the task the caveat ships as a caveat.

### Proposed registry entries (Hub to rule)
- **Never-quote:** *the vault ratio written as `γ_eff/(2−γ_eff)`* — it evaluates to 0.356; the ratio is `[γ_eff(2−γ)]/[γ(2−γ_eff)]`. (The bare number 13.88 stays quotable as the refuted coupled-bath prediction.)
- **Never-quote:** *`1.7×10⁻¹²` as a spectral mass* — it is the ring-profile probe's resolution floor on the designed checkpoint. Always "probe floor", never "the designed μ²".
- **Standing instrument caveat (N51 class):** *every V-curve quote travels with "one-step Jacobian; the one direct-rollout cross-check disagrees by 19–43% on absolute step counts"* until ME-1 lands.
- **Amend N131:** Guo **§2**, Eq. (1), ε-only; the (ε,δ) relaxation is the unnumbered display after it. The ⛔"Def. 1/2" ban is correct and stays.

---

## 4. Git footprint

**None.** No tracked file was touched; all edits are under `.claude/papers/v5-short/` and `.claude/scratch/v5-revision-1/`, both gitignored. No branch, no commit.

**Files written:**
- `.claude/papers/v5-short/draft.md` (canonical, v0.3) · `draft.tex` (generated) · `draft.pdf` (25 pp) · `draft.log` · `CHANGELOG.md` (v0.3 entry appended)
- `.claude/papers/v5-short/figs/` — `fig1_collapse.png` (new), `figC_lambda_coset.png` (new), and `fig2_vault.png`, `figB_dlaw.png`, `figB_signflip.png`, `figB_massive_vs_flat.png`, `figC_Tstar.png`, `figC_register_capacity.png` regenerated. **`fig1_vcurve.png` deleted** — it carried the "Cor-13" leak and was no longer referenced.
- `.claude/scratch/v5-revision-1/` — `fig1_collapse.py`, `regen_figs.py`, `md2tex.py`, `maincount.py`, `maincount.{tex,pdf,log}`, `sweep.py`, `edit1.py`…`edit24.py` (every text change is a scripted, assert-guarded replacement — the full edit history is replayable), `draft_v02_backup.{md,tex}`.

---

## Open questions / follow-ups / risks

1. **Venue-class page count.** 4 pp is measured in generic `article`. Someone with the NeurIPS 2026 style must re-measure before freeze. My estimate is that the narrower text block costs ≈0.2–0.3 pp, i.e. **the paper is probably ~4.2 pp in the venue class today**. Options if so, in the order I would take them: (a) the real title (saves ~2 lines); (b) move the MF-1 probe-floor sentence wholly to C.3 (~2 lines); (c) drop the E.3 packing pointer from §2.3 (~1 line); (d) Head ruling on a do-not-cut item.
2. **N131 vs the draft** (reconciliation item 1) — needs a Hub owner this wave, or the next sweep reintroduces the error.
3. **Internal report slugs in App A** (reconciliation item 4) — Head/Hub call.
4. **I did not re-verify any citation record.** Every bibliography entry is transcribed from `v5-scope-scout` (fresh, 2026-08-19) or `v2-cite-check` (2026-08-18). Two entries are from the scout's *fallback* routes and are labelled as such there: Blelloch–Golovin–Vassilevska (DBLP API) and Snyder 1977 (taken from Blelloch–Golovin's reference list, DOI not independently verified). Andersson & Ottmann 1995 and Sundar & Tarjan 1990 are named in K.2/E.2 and listed in the References from the same secondary source — **flagged: these three have no primary-source verification and should be checked before camera-ready.**
5. **The "certified removal" sweep false positive** will recur for anyone running the same regex. If the Hub wants a clean zero, the pattern needs a negative lookbehind for "Guo".

## Proposed handover updates (for the Hub)

1. **V5 is at v0.3 (`v5-revision-1`).** All 14 MF, 12 SF and 6 NICE closed with locations (table above), plus the scout's 7 reconciliation items; zero new measurements; main text measured at **exactly 4 pp** in generic `article`, full document 25 pp, build clean.
2. **Both Add.21/Add.23 carry-item failures are demonstrably closed:** the 2503.21536-as-CD sentence is gone at both sites (Fischer & Igel 2010+2011 + Nijkamp 2020 in its place, Toledo-Marín described accurately) and the k-regime scope clause is in App H(c) in V2 v0.8's printed form.
3. **A bibliography now exists** (58 verified records) and every in-body citation resolves.
4. **Four proposed registry entries** are listed above (two never-quotes, one standing instrument caveat, one N131 amendment).
5. **The 4 pp budget is spent.** The referee's move-menu is fully consumed; any further mandated main-text addition displaces a do-not-cut item and needs a Head ruling.
6. **Three references lack primary verification** (Snyder 1977, Sundar & Tarjan 1990, Andersson & Ottmann 1995 — all from Blelloch–Golovin's reference list). A half-spoke closes this.
