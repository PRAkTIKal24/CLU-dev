# v5-rebuild — paper-writer report

Task + acceptance criterion: rebuild V5 ground-up to v0.2 (budget cube + V-curve headline + the R1 deletion estate + the shipped lifecycle) inside a **hard 4 pp** PALM SHORT-track main text, with the CM/never-quote discipline intact. **Status: done.**

**DIAL DECLARATION (echoed, protocol §7):** *none — rebuild pass; no new measurement; no laundering control applies.* No number in the draft was produced by me; every number is transcribed from a named output/registry row.

> ⚠ **DOWNSTREAM RECONCILIATION LIST — needs an owner (protocol §5 corollary). THREE items, all editorial, none numeric:**
> 1. **The 4 pp limit forced ~1,650 words out of the main text.** Nothing was deleted — all of it is banked in Appendix K (K.1 nomenclature/discipline, K.2 related work, K.3 learned-store refutations), A.0 (full Setup) and C.7 (the precondition subsection). If the venue template is denser than `article`/10pt/1in (e.g. two-column), the Hub may want a *re-expansion* pass, not another cut.
> 2. **CM-23(v)'s scoped-form long quote is now in Appendix E.2, not main text** — main text carries CM-25(f) (the full-scope form that *replaces* CM-23(v)'s qualifier) plus the store-level/no-(ε,δ) clause inline. The task permitted "and/or"; flagging the choice explicitly so a referee pass does not read it as an omission.
> 3. **The trilemma appears in main text in paraphrase + the verbatim corner sentence; the full CM-23(y) verbatim block is Appendix E.6.** Same rationale (page budget). If the Advisor wants CM-23(y) verbatim in main text it costs ~45 words and pushes to 5 pp unless something else leaves.

## What I did
- **Read first, in order:** `AGENT_PROTOCOL.md` → Positioning Charter (`philosophy-synthesis.md` L581–603, verified on disk TODAY: **C-1 is in its POST-REVERSAL form — no defensive audit paragraph**, and none exists in the draft) → `claims_matrix.md` §0.1–§0.14 / §1 / §2 (CM-16a/b, CM-22, CM-23(v)(y), CM-25(f)(g)) → `advisor-head-shorts-charter.md` §4 + Add.2/4/8/18/19/20 → task file.
- **Rebuilt `papers/v5-short/draft.md` from scratch (v0.1 backed up to `.claude/scratch/v5-rebuild/draft_v0.1_backup.md`).** New structure, one contribution arc, three contributions; the deletion estate and the lifecycle are new material that post-dated v0.1.
- **Generated `draft.tex` from the markdown** with a purpose-built converter (`.claude/scratch/v5-rebuild/md2tex.py`) so the two files cannot drift, and **compiled both** with `tectonic` (which *is* installed here — `pdflatex`/`xelatex`/`latexmk` are not).
- **Wrote the CHANGELOG v0.2 entry** with the rebuild map, the measured page count and the full main-text-vs-appendix cut map.
- **Ran the never-quote sweep** (`.claude/scratch/v5-rebuild/sweep.py`), per-file and positive-controlled.

## How I verified (commands + observed output)
| check | command | observed |
|---|---|---|
| main text ≤ 4 pp | `tectonic -X compile maincount.tex` (main text only, `article` 10pt, 1in margins, single column, headline figure included) | **`Output written on maincount.xdv (4 pages, 28920 bytes)`** — 2,585 md-words / 2,617 rendered words |
| full document builds | `tectonic -X compile draft.tex --keep-logs` | **`Output written on draft.xdv (17 pages, 123696 bytes)`**, `Writing draft.pdf (484.94 KiB)`, **0 errors** (over/underfull hbox warnings only, expected for a generic `article` class) |
| never-quote sweep | `python3 .claude/scratch/v5-rebuild/sweep.py` | **`TOTAL FORBIDDEN HITS: 0`**; positive controls `107.77` 8×, `Blelloch` 11×, `stops answering…` 3×, `read_hits` 2×, `0.5000` 3×, `Jawahar` 1× ⇒ **instrument LIVE** (not a silent-negative sweep) |
| mandatory-content checklist | 30-string per-file grep (see below) | **MISSING: 0** |
| C-8 hermetic | greps for `V1 short`/`V2 short`/`our other short`/`companion short`/`B′`/`bprime` | **0 occurrences each** |
| scope of edits | `git status --porcelain` | **0 lines** (no tracked file touched; everything under `.claude/`) |

**Mandatory-content checklist (all present, per-file counts):** name debut 1× · §A20.5 substrate-scope sentence 1× · score sentence 1× · CM-25(f) load range + three conditions 1× · BG attribution verbatim (both halves) 2× · N108 "stops answering before it stops leaking" 3× · §0.13 lifecycle wording 1× · "L4 labelled UNEXERCISED (0 refusals on the stream)" 1× · demotion rider 1× · `read_hits` rider 1× · "declared not-run" 3× · compute-adaptive-read corner 1× · `R₅₀` 1.135→0.771 1× · N129 oscillatory caveat 3× · N112 overflow clause 1× · `τ_max=Γ/2α` 6× · "refuted in sign" 5× · Guo "§3 Eq. (1)" 3× · flat-table trivial substitute 1× · Δ/ℓ_θ discipline 2× · fdt+Newtonian 5× · verification-vs-evidence labels 1× · scale-as-scope-choice 1× · title/author placeholders 1×/1× · PALM "including code" 2× · 107.77±4.78 7×.

## Findings / results (what the draft now is)

**Structure (main text, 4 pp).** Abstract · §1 Introduction (nomenclature + measurement discipline + 3 contributions + verification/evidence labelling + the §A20.5 sentence) · §3 Results with a compressed setup preamble: **§3.1 the damping optimum** (headline, Fig. 1: argmin `0.902±0.003×γ_crit`, slopes `−1.0020`/`+1.116`, 3/3 emergent seeds, μ² over 11 decades) · **§3.2** friction-preserves/temperature-erases (`D_θ` 1.0068±0.0219 over 25 cells; sign flip 10/10; 3.77±0.23×) + the vault (**107.77±4.78×**, D̂-estimator, pre-registered; scalar control 13.28±0.12×; ratio 8.11±0.37) + the designed-symmetry-precondition rider · **§3.3** the deletion estate · **§3.4** the lifecycle · §4 Related work (compressed) · §5 Limitations + scale + horizon.

**Evidence backing each main-text section (source report → section):**
- §3.1 — `t-lever-forgetting` §4 (5 designed seeds) + `v5-gate` §3.4/R1 (3 emergent + matched designed control); CM-16b. Provenance A.2/A.3.
- §3.2 — `t-lever-forgetting` §2–4 (diffusion law, sign flip, latch) + `v5-gate` R3 (vault, pre-registered); CM-16a is used **only** as the designed-only latch statement, always with §3.5's/§3.2's precondition rider. Provenance A.2/A.4.
- §3.3 — `order-independent-placement` (rule + theorems), `placement-landing` (shipped code + acceptance test), `deletion-waitlist-stiffness` (load sweep 0.29×–1.71×, gated-stiffness `R₅₀`), `mia-decay-measurement` (leakage curve, `R₅₀`, TTL comparator); CM-25(f), CM-23(v)/(y), CM-25(g), N99 w26 block, N108, N112, N118, N127–N131. Provenance A.5–A.8.
- §3.4 — `c2w10-lifecycle-mechanics`; §0.13 approved wording verbatim. Provenance A.9.
- §4 — related-work prose **lifted from the scout reports' own draft prose**: the history-independence/unlearning paragraph is `deletion-prior-art` §1.6's one-paragraph novelty statement (used verbatim in K.2, compressed in §4); the Goldstone/equivariant paragraphs are carried forward from the v0.1 draft's `scout-goldstone-positioning`-derived text.

**Appendices (supplementary, 13 pp).** A flag provenance **A.0–A.9** (A.0 = the banked full Setup; A.5–A.9 are new: canonical placement theory, shipped landing, MIA harness, waitlist/gate, lifecycle build — each with commit, seeds, store/controller/read config, venv+JAX version, PREREG existence) · B diffusion law + sign flip + latch + knob table · C emergent arm (N46), `T*≈3e-3`, the raw-exponent instrument warning, the two-observables caveat, C.7 banked · D vault in full (absorb-only algebra, refrigerator ladder, discriminator, `T=0` erasure, write attenuation, Fig. D.1) · **E the deletion estate in full** (placement rule; theorems as *instantiations* with the mandatory attribution; exactness at 0 tolerance + shipped landing + acceptance test; load sweep + overflow scope + the corrected harness-artefact cell; the leakage curve with the exact/resolution-limited split; trilemma + gated channel with both refuted repairs; citation/vocabulary discipline) · **F the lifecycle in full** (kill-conditions-first build order, 7-leg table with designed negatives *and* the mutation that makes each fail, L4-UNEXERCISED discussion, the build's owned reconciliations) · G Coleman/Mermin–Wagner + `fdt`/Newtonian scope · H erosion-as-restoration with the (a)–(d) ship rules preserved verbatim **plus** a new designed-vacuum-only rider · I the `T_φ` shredder horizon · **J negatives ledger, 15 entries** · **K banked main-text material**.

**Discipline decisions worth the Hub's eye:**
1. **No VALUE number anywhere in the lifecycle material.** Stream-level counts (promotions/demotions, routings, trash bytes) from `c2w10-lifecycle-mechanics` were deliberately **excluded** — §0.13 declares that substrate *never a claim venue*, so the leg table reports *landed / designed negative / can-fail mutation* only, and the L4 discussion is qualitative except for the approved "0 refusals on the stream".
2. **`0.99985` always carries its load** (8 offers into an 8-capacity store = 7/8 of the packing bound, with the 2/4/6-offer curve beside it) per the quote-the-curve rule.
3. **"Certified"** appears only (a) inside the approved denial *"we do not claim certified (ε,δ) unlearning"* and (b) describing the literature's notion (Guo §3 Eq. (1)). **"Unlearning"** likewise. **"Packing certificate"** is the approved CM-23(v) phrase and is untouched.
4. **Prohibition scaffolding was rewritten out of the paper.** The first assembled draft contained ⛔-marked "we do not write X" lines (an artefact of transcribing never-quote rules into prose); they are now positive statements — the sweep flagged them and they are gone (0 ⛔ glyphs remain in the draft).
5. **The ε-notation collision is stated in-paper** (integrator step vs symmetry-breaking tilt), because N149/N150's blast radius is expressed in a tilt-ε and V5's ε is `dt`.
6. **No cost claim** of any kind (N131/M1 treated as not-available) — stated as a limitation, with the `O(n)` rebuild fact given as the reason.

## Open editorial questions (for the Hub/Head)
1. **Venue template.** The 4 pp measurement is in `article`/10pt/1in/single-column with the headline figure. If PALM ships its own style file, re-measure before freeze: a tighter style buys ~0.3–0.5 pp (re-expand from Appendix K), a looser one costs the same (the next things I would cut are, in order: §4 to three sentences; the §3.3 contrast clause; the §3.2 latch sentence).
2. **Which deletion wording leads.** As flagged above: CM-25(f) leads and CM-23(v) is appendix. Confirm or reverse.
3. **Title.** Working title is deliberately long/descriptive: *"Forgetting You Can Budget, Delete and Schedule…"*. Workshop practice is shorter; a title pass at the end is owed (Charter C-10).
4. **Figure count.** Main text carries exactly one figure (the V-curve); the vault figure moved to D.1. If a second main-text figure is wanted, ~180 words must leave.
5. **`R₅₀` baseline pair.** Main text quotes both the gated (`1.135→0.771`) and baseline (`1.146→0.752`) contractions, and §3.3's leakage sentence quotes the baseline pair again as the TTL differentiator. A referee may read it as duplication; I kept both because CM-25(g) and CM-22(p) are separate approved wordings.
6. **Un-refereed.** V5 has never been refereed (Add.18 §3 pre-commits the pass); the mechanical precondition — a v0.2 CHANGELOG entry — now exists.

## Git footprint
**None.** No tracked file was created, modified or deleted; `git status --porcelain` returns 0 lines. All work is under `.claude/papers/v5-short/` (draft.md, draft.tex, draft.pdf, draft.log, CHANGELOG.md) and `.claude/scratch/v5-rebuild/` (converter, sweep, page-count harness, v0.1 backups). No branch, no commit — as scoped.

## Proposed handover updates (for the Hub)
- **`papers/v5-short/` is at v0.2** — a rebuild, 4 pp main text (measured, not estimated) + 13 pp appendices, `draft.pdf` built; the V5 referee pass's mechanical precondition (a v0.2 CHANGELOG entry) is satisfied.
- **V5 now carries the R1 deletion estate and the C2W10 lifecycle** — any future edit to those sections must re-check §0.13 (lifecycle wording + riders), CM-25(f), CM-23(y), N108/N112/N118/N127–N131 and the BG attribution, all of which are load-bearing verbatim text.
- **A reusable artefact exists:** `.claude/scratch/v5-rebuild/md2tex.py` (markdown→LaTeX for these drafts, math/code-safe) and `.claude/scratch/v5-rebuild/sweep.py` (parameterised never-quote sweep with positive controls). Both are drafting infrastructure the other shorts could reuse; the Hub may want them promoted to a shared location.
- **`tectonic` is available on this machine** — the standing "no TeX toolchain, drafts are unbuilt" note in the other shorts' build notes is stale and can be retired.
