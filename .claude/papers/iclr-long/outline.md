# ICLR 2027 long — outline stub (PLANNING; no prose yet)

> Spine + evidence map + gap list live in `.claude/outputs/iclr-long-skeleton.md`. This is the section stub only.
> Title: `[WORKING TITLE: …]` (workshopped at the end). Authors: `[AUTHORS PLACEHOLDER]`.
> Naming: continuity sentence in §2 — "the CLU, introduced as CHLU in Jawahar & Pierini (2026)". Nomenclature: inertial **M** vs spectral **μ**, never bare "mass".
> Citations hermetic (C-8/M1): J&P 2026 + the F5 note (third person) only.

## THESIS (forks on `g7b-torus-voraus`; see skeleton §1)
- **Recommended:** conservation-by-construction is a usable deep-network substrate — stability, interference-firewall, priced communication, certified test-time compute, and O(1)-memory training all *compose* and hold on real robot time-series, **given the symmetry is designed in, not awaited.**
- **Fallback (voraus ties/loses):** G7c "Pareto-not-podium: a physically-motivated memory with set-at-write-time lifetimes and an exact decay law, Pareto-competitive" + designed-precondition gains headline weight. [⛔ the earlier wording "certified, predictable-lifetime memory" is FORBIDDEN — CM-22(m): *certified* = (ε,δ)-indistinguishability from retraining (Guo et al., ICML 2020, §3 Eq. 1), which we do not supply; substitute per CM-22(m).]

## SECTIONS (main, ~8pp)
1. Introduction — thesis; contributions on p1; own the "what the physics buys / does not" composition up front; name headline figure. [C-1, C-3]
2. The CLU primitive and the CLU-Net — unit, lattice, conserved charges; continuity sentence; M/μ; F5 as apparatus.
3. Guarantees that compose — BIBO + latch [CM-1/CM-7]; Noether (1−γ)ⁿ; reversible O(1) [CM-13, STRUCTURAL].
4. **Interference firewall at scale (HEADLINE)** — degree-bounded vs width-linear [CM-9], N≤16, 12 seeds; parameter-separation-not-physics owned in-text. **Fig 1.**
5. The priced channel — sync∝κ^−1/2, n₁/₂∝κ^−1 [CM-10], N∈{2,4,8,16}, both topologies, 5 seeds, pre-registered; designed verification at N≤8 [C-2].
6. Certified test-time compute — paid-access certificate [CM-12, VER, oracle-scope inline]; calibrated escalatable gate [CM-2]; one-hop edge not energy [CM-7]. [CM-3 forbidden; C-6 fine print inline.]
7. **Real industrial data — torus-CLU vs baseline floors on voraus-AD (accept-maker)** — identical episode-AUROC protocol [CM-3 bound]. **PENDING (g7b-torus-voraus).** Fig 2 (fork-dependent).
8. Related work — lifted from scout reports (skeleton §5).
9. Limitations + the designed-symmetry precondition (Hyp-3) — owned scope.

## APPENDICES (C-10 maximalism)
- A Flag-provenance tables (C-7) — all §s.
- B Banding [CM-11] + mass-lr doctrine grid [CM-5, N7].
- C Block-monolith + coordination controls [CM-9 tail]; metric discipline (R not NTK cosine).
- D Full regime map / cost story [CM-8] + noise wall [N37].
- E Certificate derivations — squeeze-MH [CM-14], BIBO battery, analytic checks [C-6].
- F Reversible-O(1) full tables + γ>0 horizon [CM-13].
- G Erosion + anchor cure [CM-6].
- H Prominent negatives registry [C-9]: N7, N37, N46/CM-16a, erosion, retries-null, energy≈margin, pricing pointwise 47–56%, voraus smoke below baseline.
- I [PENDING] full 130ch voraus floors + topology-match control; TEP secondary.
- J [optional, NMI-adjacent] XY/KT designed-coupling reduction [xy-1d-control] — designed-precondition evidence, cross-ref from §9.

## HEADLINE FIGURE
Fig 1 = interference scaling (degree-bounded vs width-linear), from V3 short `figures/fig1_scaling_curve.png`. Reassign to voraus per-category panel IFF g7b lands a win.

## OPEN (drive experiment priority — skeleton §4)
- G1 real-data (g7b-torus-voraus, unlaunched, trending negative) · G1b full floors (CSF3) · G4 reversible in trainer + GPU [UNASSIGNED] · G5 scale beyond N≤16 + depth [UNASSIGNED] · G6 learned + deep-TSAD baselines [UNASSIGNED cluster] · G7 learned entrance-steering · G8 noise-wall.
