# ICLR pivot — Head ↔ Pivot Advisor decisions log

**Opened 2026-09-05.** Living record of the pivot conversation. Two sections only: what is decided, what is open. Short by design; append dated deltas, don't rewrite.

---

## A. DECIDED

### A1. Target and framing
- **An ICLR main-track long paper needs a SOTA result.** Head's call, reaffirmed after the Advisor argued against a SOTA chase on gauged CAMELS.
- **Two venues, both kept:**
  1. **Ungauged-basin streamflow (CAMELS, prediction-in-ungauged-basins split).** The SOTA metric (median NSE on withheld basins) and the vision sentence ("unseen point triggers the wells nearest it") are the same measurement. Lookup tripwire already cleared on the harder gauged variant; loader spec exists; nothing built yet.
  2. **HEPA benchmark suite, long-horizon.** Frozen HEPA encoder (JEPA-style, embedding per historical window, horizon-conditioned predictor) in front of CLU; ~10 datasets (BATADAL, N-CMAPSS, MSR, traffic, water flow, …); headline metrics are observable-space task metrics (e.g. h-AUROC), never latent error alone.
- **Dropped for now:** continual learning (time); gauged-CAMELS SOTA; N-CMAPSS RUL as a long-horizon task (only the local vicinity of the fault matters).

### A2. Comparison rules
- **Rerun everything, always.** ~5 GPU-h per dataset for all models × 5 seeds makes it cheap. HEPA is printed as one row among externally cited baselines, never as the headline reference (it is a non-archival workshop paper on arXiv → "beats our own preprint" is self-comparison).
- Published numbers transfer only under an unchanged protocol. Long-horizon rows are always reruns.
- Every CLU number carries: matched-byte lookup (kNN over frozen target-encoder latents — free from cached embeddings), store-removed control, ≥3 seeds, the curve over horizon not the endpoint.
- Point-adjust anomaly scoring is inadmissible for a headline row. N-CMAPSS numbers only on corrected labels and original NASA files.

### A3. Long-horizon protocol design principles
- Long memory must be the only route to the score. Two structural forms: **open-loop simulation** (emit H steps from inputs with no target feedback; horizon is a conditioning input so H is a sweep) and **regime recurrence** (slice error by distance-to-last-similar-regime).
- Prognostics variant where memory matters is **cross-unit**: persistent store from other units, tested on unseen units (same shape as ungauged basins).
- **Gate every dataset×horizon pair** with two CPU tripwires (matched-byte exemplar; best trivial baseline such as climatology/cycle-count) before it enters the suite. Pairs where either comes within a few percent of the learned model are dropped and reported.

### A4. Architecture / training pipeline
- **Three stages.** (i) Pretrain HEPA encoder on its own objective, freeze. (ii) Pretrain addressing + read-out with a **store-only objective, no encoder→head bypass** — the store cannot be routed around because no other route exists. (iii) Unfreeze a light task head with bypass allowed; store-removed control must still show a drop.
- **Frozen encoder for the headline arm.** Joint fine-tuning is a secondary arm, labelled.
- **Wells = groups of encoder feature maps (coordinate axes); modes = recurring value patterns on those axes learned from training samples.** A sample latent is never stored; only per-well modes are. Reconstruction composes; unseen combinations are the test. This is also what defeats the "CLU learns to copy the encoder" failure and the "persistent store scales with data" concern (footprint = K wells × M modes × width, independent of dataset size).
- **Well substrate = feature maps of a layer before the final flat embedding** (penultimate by expectation). Two guards: choose the layer by measured block-diagonality of channel correlations, not by position; pool time before grouping or declare the byte cost. *(High-level agreement; precision deferred until other components are fixed.)*
- **Grouping arms, all run:** (1) fixed correlation-clustered partition — the control; (2) learned rotation then partition; (3) learned soft channel-to-well assignment (closest to the vision, closest to the old addressing failure; pre-register its failure threshold). Falsifier: random groupings of the same K.
- **Addressing parameters come from a small head on the encoder's penultimate features,** trained in stage (ii). Outputs are structured and low-dimensional (which wells fire, coordinates within each), not a free vector. Unfreezing the encoder's last layer is a secondary arm.
- **Controller: redesign as a small transformer/attention block over well states, with the rule that its output is actions on the store (write/skip, merge, lifetime, decay) and never features to the head.** Atoms/landscape dynamics are NOT replaced by a transformer — that is the primitive and the physics constants depend on it. Smarter reads (attention over modes) are allowed.
- Store init: condensed training embeddings (k-means-style, to budget) vs random init — both run; if random reaches the same place, init isn't doing the work.

### A5. Paper shape (≈9-page main text)
- Intro/positioning ~1.5 pp · method ~2.5 pp incl. one figure that makes the store legible in 30 s · experiments ~4 pp · limitations 0.5. Experiments = one headline table across datasets · one mechanism figure (store-removed + error-vs-horizon with kNN underneath) · half a page of isolating toys (delete, lifetimes, composition). Everything else appendix.
- Method section must position against product quantisation, VQ latents and modern Hopfield memories; differentiators (continuous navigable landscape, per-mode decay/lifetime/exact delete, composition on unseen combinations) must be measured, not asserted.

### A6. Cost estimates (unmeasured; firm up with a one-fold / one-dataset pilot)
| Venue | Shape | A100-h |
|---|---|---|
| Ungauged CAMELS, 2 pilot-sized bad attempts + 1 full cell (10 folds × 3 seeds × 5 arms) + 8-seed ensemble top-up | | ~2,600 (1,500–4,000) |
| HEPA suite, 10 datasets, 3 attempts + horizon sweep + 3 grouping arms | rivals ≈50 h/cell, CLU assumed 3–5× | ~1,000 |
- First output of any pilot is the store-removed delta. Inside seed noise ⇒ stop, spend on addressing.

### A7. Launched / in flight
- **HEPA benchmark scout** — launch block handed to the Head 2026-09-05; writes `.claude/outputs/hepa-benchmark-scout.md`. Extends the two prior scouts (`scout-cafe-hepa.md`, `scout-hepa-predictor.md`). HEPA = arXiv:2605.11130.

---

## B. OPEN (discussed, not decided)

- **B1.** Exact layer and K for the well substrate; time-pooling vs keeping time as a second index; byte budget for the penultimate-layer write. *Waits on the scout's encoder shapes.*
- **B2.** Which of the ~10 HEPA datasets survive the long-horizon tripwires; which long-horizon form (open-loop vs regime-recurrence) each gets.
- **B3.** Store-only objective for stage (ii): exact loss and the pre-registered "addressing failed" threshold.
- **B4.** Controller redesign spec: action set, block size, and its ablation. Which failure it targets (consolidation vs addressing) must be named first.
- **B5.** CAMELS ungauged: k-fold count (published used 12), rival set (LSTM reference + GRU byte-matched + one SSM), whether it runs in parallel with the HEPA suite or after.
- **B6.** Whether target-encoder latents are cached per dataset (makes kNN control and condensed init free). *Scout item.*
- **B7.** Which HEPA datasets used point-adjust and which have exogenous-driven vs autoregressive targets. *Scout item.*
- **B8.** Whether HEPA's encoder last layer unfreezes in any headline arm (currently: secondary arm only).
- **B9.** Sequencing vs the existing text ladder and the queued CAMELS loader task.

---

## C. Standing rules carried into every decision above
Pre-register the falsifier · build the thing that kills the idea first · ≥3 seeds · every claim with its control · report the curve · nothing quoted without an artifact on disk · Head launches everything.

---

## ⟲ 2026-09-07 — HEPA benchmark scout landed (`.claude/outputs/hepa-benchmark-scout.md`)

### Facts that change the plan
- **Suite = 14 datasets / 11 domains, one metric (h-AUROC, HEPA's own), horizon is a conditioning input (K=150 C-MAPSS/TEP, 200 elsewhere).** No N-CMAPSS, no MSR, no traffic in HEPA. The datasets are C-MAPSS FD001–4, SMAP, PSM, MBA (ECG), GECCO, BATADAL, TEP, ETTm1, Weather, Beijing-AQ, VIX.
- **HEPA's margins are thin:** wins 10/14 but only 3 by ≥0.05 (C-MAPSS-3, C-MAPSS-4, GECCO); 4 losses (SMAP, MBA, BATADAL, ETTm1); 2 exact ties counted as wins (Beijing-AQ, VIX). MAE is the second-strongest arm (4 wins).
- **The encoder is small, not massive:** 2-layer causal transformer, d=256, 32 tokens (512-step context, patch 16), ~850K params. Whole apparatus 2.16M because the target encoder is a *separate*, jointly-trained clone. The "penultimate feature maps" are the (32 tokens × 256 dims) grid before last-token pooling.
- **Every HEPA arm sees at most 512 steps of history.** Anything older is invisible to HEPA and to all four rerun baselines.
- **Compute is negligible** (<2 h on one A10G for 14 datasets × 5 seeds). **Data acquisition is the cost:** HEPA redistributes nothing; licences NOT STATED for nearly all sources; three loaders (SMAP, TEP, VIX) carry event definitions the authors call placeholder; VIX's event threshold is fit on the test split (leak).
- **Confounds to fix in any rerun:** horizon conditioning differs by arm (HEPA scalar-concat 198K head vs baselines learned-embedding 264K head); two inconsistent protocol files in the repo (epochs/seeds); C-MAPSS uses a train-engine-only id split with the official test set discarded, and HEPA loses badly on published RUL RMSE (28.5 vs 12.2).
- **h-AUROC uses no point-adjust** (good). PA-F1 is point-adjusted and secondary only.
- **HEPA code is public under CC BY-NC-SA 4.0.** CHLU is Apache-2.0 ⇒ do not vendor HEPA code into CHLU; run it as a separate dependency/repo.

### Decided in light of the scout
- **A1 amended: the HEPA suite is the long-horizon *capability* section, not the SOTA headline.** h-AUROC on HEPA's own event definitions is an internal benchmark to an ICLR reviewer, whatever the margin. The recognised-metric SOTA claim stays with **ungauged CAMELS**, which is now the primary venue for the headline. *(Advisor recommendation; Head to confirm or override.)*
- **Long-horizon on the HEPA suite is defined as history beyond the 512-step window,** not a longer prediction horizon. That is the structural gap in every existing arm and the natural job for a persistent store. Report h-AUROC sliced by the age of the decisive information.
- **Matched capacity = the 850K context encoder + store bytes,** stated explicitly; never the 2.16M figure.
- **Well substrate is the 32×256 token grid.** K from correlation clustering over the 256 dims on cached training activations; the 32-token time axis is either pooled or kept as a second index — 8K floats per write either way, cheap at this scale.
- **Reruns match the horizon-conditioning mechanism across all arms** and cite one protocol file.
- **Dataset triage:** drop VIX (test-split leak); hold SMAP and TEP until their event definitions are repaired; C-MAPSS results are reported on the id-split with the split named, never against published RUL tables.

### Newly open
- **B10.** Does the Head accept CAMELS-ungauged as the SOTA headline with the HEPA suite as the capability section? *(Advisor recommends yes.)*
- **B11.** Which of the 11 admissible datasets have events that depend on history older than 512 steps? Needs a per-dataset CPU probe (HEPA at 512 vs a longer-context variant) before the persistent-store claim is built on any of them. Candidates by mechanism: GECCO, BATADAL, Beijing-AQ, Weather, C-MAPSS.
- **B12.** The "encoder emits addressing parameters" idea (3b) assumed a large encoder; at 850K the addressing head carries the load. Decide whether to scale the encoder (we rerun everything anyway) or accept a heavier addressing head.
- **B13.** Data acquisition and licence clearance for 11 sources — owner and sequencing.
- **B1 partially closed** (substrate shape known); layer choice still measured, not assumed.
- **A7:** scout DONE. Prior scout errors corrected in its §2 (target encoder is separate; code is public; 0.918-vs-0.81 resolved as early-stopping vs fixed-epoch).

### ⟲ 2026-09-07 (later) — Head rulings
- **B10 CLOSED: yes for now** — ungauged CAMELS is the SOTA headline; HEPA suite is the capability section.
- **B13 CLOSED — licensing is the Head's job, not the program's.** All sources are public under permissive licences. Strike every licence line above; never raise it again.
- **B12: scaling the encoder is allowed,** subject to a speed rule: results must come fast enough for multiple iterations at ≥3 seeds per model. Encoder is shared infrastructure across all arms, never the variable.
- **B11 → task scoped for a lead hub** (dataset triage for the long-horizon section); block handed to the Head 2026-09-07.
- **New: fair-fight rules for baselines** (A2 extension, discussed 2026-09-07):
  - Same frozen encoder, same task head and horizon-conditioning mechanism, same data, seeds, epoch/early-stopping policy for every arm. Only the memory component varies.
  - **All arms get access to the same history**; they differ in how they compress it. Recurrent/SSM rivals run stateful across windows; the attention rival gets the longest raw window that fits the byte budget; the exemplar store gets the budget in raw exemplars.
  - Matched at **inference state bytes** (store vs KV cache vs hidden state), parameters roughly matched, dtype as deployed.
  - Rival families that must both be present: **"see more context"** (long-context attention / raw-history buffer) and **"carry state"** (GRU/LSTM, Mamba-2 or GDN-2 head). The incumbent HEPA predictor head is a third row.
  - Equal tuning budget per arm (same number of trials on each arm's key knob), reported; library defaults are never "the baseline".
  - Reference-only rows (no fairness claim): HEPA's paper numbers, Chronos-2.

---

## ⟲ 2026-09-07 — long-horizon triage returned: the HEPA long-horizon framing is DEAD

**Result:** 0 KEEP / 1 HOLD (GECCO) / 10 DROP. No HEPA-suite dataset needs history older than 512 steps. Every measurable context curve **declines monotonically** (ETTm1 .811→.741→.637 · Beijing-AQ .758→.723 · PSM .530→.372 · Weather .769→.649). C-MAPSS drops on the data audit alone (<1.1 % of events have >512 steps of own-entity history); MBA/BATADAL are untestable (max reachable context 512 and 100); GECCO reaches 8192 but its test split holds one event — a power failure, not a negative.
Pre-registered before any data was touched; the falsifier was "flat" and reality was worse than flat. **Clean, decisive negative. The Advisor recommended this venue framing on 2026-09-05; the triage killed it. Not defended.**

**A1 amended again: the HEPA suite is NOT a venue for this paper.** It survives only as a possible negative-results appendix ("where a persistent store has no job"). B2 and B11 are closed by measurement.

### ⛔ The larger finding, and the one that needs adjudication: the HEPA suite fails our own tripwire
- **FD001:** matched-byte kNN, 98,304 B (residual-stream budget), 431 exemplars, 56-dim per-channel summary descriptor (mean/std/last/slope), payload = time-to-event → **h-AUROC 0.9777 ± 0.0006 (3 seeds)** vs **HEPA 0.920** re-run on the same split. **+0.058 to the classical store.**
- Audited by the Advisor against the artifact: engine-id split, store built from train windows only, k chosen on val, test never used for selection, budget utilisation 0.9996, nulls clean (shuffle .517, permutation 5.2σ, prevalence exactly .500). **No leak found. The result is real.**
- Trivial `last_event_age` (rank by steps since the record began — no model, no encoder, no window) scores **0.713**, confirming that a large share of the FD001 signal is elapsed running time.
- **GECCO:** a 24-bin time-of-day table scores **0.818**, equalling the best published baseline (criterion-2 failure — a trivial baseline at the frontier).
- ⚠ **Scope:** the tripwire was run on **2 datasets, both failed**. The other 12 are unprobed. Do not state "the suite fails" until they are run — state "2 of 2 probed".
- ⭐ **HEPA reproduces:** FD001 = 0.920 (3 seeds) vs its stated 0.918. Comparisons are against a run, not a README.

### Consequences
- **CAMELS-ungauged is now the only live venue.** A second leg is needed or the paper is single-venue.
- The tripwire finding is a **benchmark-critique result about the Head's own benchmark**. Whether it is written up, and where, is the Head's call alone; the Advisor does not route it.

### Newly open
- **B14.** Second venue for the paper. Advisor recommends **Caravan** (the global CAMELS extension — same task family, cross-region transfer, directly tests "unseen point → nearest wells" at greater diversity), conditional on the matched-byte tripwire. Alternatives: screen from scratch (6 of 7 died last time); or go single-venue with a deeper mechanism section.
- **B15.** Whether to run the tripwire across the remaining 12 HEPA datasets — cheap, CPU-only, and it decides whether the finding is "two datasets" or "a suite".
- **B16.** ⛔ **CSF3 was unreachable during the triage** (runs went to local CPU/Metal; all declines are within-device, so internally valid). Every cost estimate in §A6 assumes CSF3. **Restore or re-price.**
- **Housekeeping:** the 3 orphan `run_hepa` PIDs (4325, 5711, 6597) are **already gone** — verified, no action. The consolidated triage doc was not written; artifacts are complete under `.claude/outputs/hepa-longhorizon-triage/`.

### ⟲ 2026-09-07 — competitor set REFRAMED (Head): CLU is a latent module, not an end-to-end model
- **Reference row ≠ competitor row.** Reference = the field's end-to-end SOTA (CAMELS-PUB: the LSTM ensemble); printed once, it is the bar for the word "SOTA." Competitors = **same frozen encoder, same task head, swap the middle module**; byte-matched at the module, equal tuning budget, ≥3 seeds.
- **Slot competitors (pending Head confirmation):** (1) none — encoder + head directly (= the store-removed control; run first) · (2) carry-state latent dynamics — GRU or Mamba-2 in the slot · (3) memory module — modern Hopfield or Titans-style · (4) discrete structure — VQ/FSQ codebook (also retires the SC-JEPA collision) · (5) retrieval — matched-byte kNN (the tripwire) · (6) Hamiltonian/symplectic latent dynamics, mandatory only if the paper claims "physically consistent."
- **Capabilities count as comparisons only when a competitor is scored on them:** delete vs kNN-delete and an unlearning baseline · navigation vs encoder-space and codebook interpolation · composition vs codebook product and Hopfield · lifetimes vs a fixed-window buffer.
- **Open (B17):** the noun. "Latent dynamics model" ⇒ future-latent prediction is the primary metric and rows 2 & 6 are unavoidable. "Structured memory" ⇒ rows 3–5 carry the weight. Head to pick; the noun picks the reviewers.
- The single-domain / general-claim exposure and the "encoder does all the work" exposure are unchanged by the reframe.

---

## ⟲ 2026-09-07 (later) — PIVOT OF THE HEADLINE: generative CLU with calibrated refusal

**Head's proposal, developed with the Advisor.** The paper is no longer "CLU scores well on a benchmark." It is: *a latent primitive that composes unseen-but-valid configurations and refuses impossible ones.*

### The claim structure
- ⛔ **Do not compete on generation quality** (diffusion wins at any scale we can afford). Compete on **compositional correctness + calibrated abstention**.
- **The money figure — a 2×2 that no existing model passes by construction:** (A) seen combinations → generate · (B) **unseen but valid** → generate · (C) **impossible** → refuse. A trivial OOD detector fails B; an ordinary generator fails C. B-vs-C is the entire contribution.
- **Metrics, all established, none invented:** per-factor accuracy of generated samples under a probe trained on real data (A, B) · zero-shot AUROC separating C from A∪B, **no tuning on invalid examples** · **risk–coverage curves + AURC** from the selective-prediction literature.
- **Two-tier refusal claim, pre-registered as such:**
  - **Tier 1 (safe):** refusal from the learned energy/barrier is calibrated and beats post-hoc thresholds on rival generators.
  - **Tier 2 (the ICLR-grade result):** the refusal boundary is **predicted by the measured reach law with no fitting on invalid examples** — predicted-then-measured, the same pattern as the capacity and decay constants.
  - ⛔ Tier 2 is a **hypothesis, not a result**. Falsifier filed before running: if the reach-derived predicate does not track actual validity, report Tier 1 and state that Tier 2 failed.

### Paper spine (replaces §A5's ordering)
1. Long-horizon spatio-temporal ML lacks benchmarks with known compositional structure — **evidence is our own triage** (14 datasets, none needs >512 steps, monotonic declines; 2 of 2 probed fall to a matched-byte lookup).
2. Root cause: real data carries no factor labels and no held-out-combination splits.
3. A primitive with explicit, addressable, factorised latent structure can construct such data **and know which combinations are valid**.
4. Theory: landscape, reach, causal structure.
5. Evidence: compositional generation + calibrated abstention (**headline**) · physical-consistency generation on a simulated system (**mechanism**) · CAMELS synthetic-basin transfer (**real data**).
6. Negatives reported, not buried: text, HEPA suite.

### CAMELS REFRAMED — ⛔ the NSE-SOTA chase is dropped
- New form: generate discharge for an unseen (catchment attributes × forcing) combination = *"simulate a sensor at an unseen location."*
- ⭐ **Validity is objectively computable: mass balance** (discharge ≤ precipitation + storage change). A hard physical predicate makes refusal checkable on real data — no human judgement.
- Downstream test is established: **train an LSTM on synthetic basins, test on real held-out basins.** Hydrology has an acknowledged ungauged-data-scarcity problem, so this contributes to that field rather than adding a marginal NSE row.
- This supersedes B5 and B14 in their SOTA-NSE form. Caravan is no longer needed as a second SOTA venue.

### Competitor set (Head confirmed 5 of 6; the Hamiltonian/symplectic row is CUT for time — it is the CLU's own basis)
Adapted to generation: VAE/diffusion decoder with likelihood-threshold refusal · EBM with energy threshold · VQ/FSQ codebook composition · kNN retrieval + threshold · store-removed control. Same frozen encoder, **one frozen decoder shared by every arm**, byte-matched at the module, equal tuning budget, ≥3 seeds.

### Risks named
- ⛔⛔ **The dangerous prior work is compositional energy-based generation** (composing energy functions at inference to produce novel combinations — an established line). Differentiator would have to be: we *learn* the factorised store rather than composing separately pretrained models, and refusal comes from geometry rather than a threshold. **This is the gate; it could sink the headline.**
- **Synthetic-benchmark circularity:** never claim the generated data is a benchmark others should adopt. It is a controlled testbed with known held-out structure. The value is the control, not the realism.
- **Decoder discipline:** a strong decoder hides store failures, a weak one makes every arm look bad. Frozen and shared.
- **Store-removed control still binds:** remove the store, the generator must stop composing.
- **Sample fidelity is a stated limitation up front**, not a reviewer's discovery.

### Cost
64×64 factor-structured data, small conv encoder/decoder ⇒ ~10 min per run. 5 arms × 5 seeds × 3 datasets × 4 ablations ≈ **50 GPU-h for a complete cell**; iteration in hours. Meets the Head's speed rule.

### Newly open
- **B18.** Novelty scout on compositional energy-based generation and abstention/selective generation — **the gate before anything is built.** Block offered, not yet written.
- **B19.** Which datasets supply the (A/B/C) split: factor-structured image data for the headline; which simulated physical system for the mechanism section (validity = a conservation law).
- **B20.** Where "impossible" comes from in each dataset — structural rules vs physical constraints. Must be objective and computable, never a human judgement.
- **B17 (the noun) is effectively answered:** the paper's noun is a **generative latent primitive**; future-latent prediction is no longer the primary metric.

### ⟲ 2026-09-07 — CAMELS/Caravan feasibility gate returned: NEITHER, with a cheap repair
- **Caravan: no benchmark protocol exists** (no split, no metric convention, no baselines, not even in its own paper) ⇒ cannot carry any claim. Closed.
- **CAMELS-US PUB: tripwire FIRED (dies = true).** A 1.4 MB untrained nearest-neighbour donor store scores median NSE 0.637 ungauged — beats the National Water Model (0.577) and the AlphaEarth baseline (0.612), ties calibrated SAC-SMA (0.645). Mechanism: PUB holds basins out at random ⇒ 71 % of "ungauged" basins keep a gauged neighbour within 50 km; flow-copy NSE 0.82 under 10 km, −0.16 beyond 100 km. **The venue scores gauge density, not hydrology.** Controls all held (loader reproduces 12/12 published rows, worst miss 0.0014; 690 leak assertions clean; climatology-only arm 0.10 ⇒ signal is the concurrent donor observation).
- **Reference corrections banked:** PUB frontier = **0.7858 (Li et al. 2025)**, not the stale 0.69; MPR+mHM 0.53 struck (in-sample, different task).
- **Repair ADOPTED: PUR (spatially blocked ungauged regions), ~0.701 frontier, same Zenodo record, harness exists.** Pre-registered prediction carried: spatial blocking removes the neighbour, the donor arm collapses toward −0.157, PUR admissible where PUB is not. ⛔ If the donor arm survives PUR too, the CAMELS family is OUT and the mass-balance validity leg moves to a different physical system. **PUR is the one remaining venue gate.**
- ⭐ **This is the second benchmark-critique result of the week** (HEPA suite → lookup; PUB → donor copy). Both feed the paper's motivation: spatio-temporal benchmarks reward proximity/lookup, not understanding. Two measured demonstrations now, not an assertion.
- **NEW BINDING DESIGN RULE (applies to our own headline, every modality):** every (B) unseen-but-valid split must be protected against the donor arm in disguise — spatial blocking on spatial data; on factor data, held-out combinations verified genuinely absent from training (not merely rare) and the probe able to distinguish. **B-vs-A separation is a measured property of the split, never an assumption.**
- B14 closed (Caravan). B5 superseded. Newly open: **B21** — PUR tripwire run (pre-registered, cheap); **B22** — fallback physical system for mass-balance validity if PUR falls.

### ⟲ 2026-09-07 — generation pipeline grounded in the existing codebase (Advisor audit)
- **Exists and load-bearing for the generation line:** the factored store (shared wells, items = well-subsets, items never stored, combinatorial unseen-combination guarantee — the 0.86 result's vehicle) · PCD/Langevin "dreaming" generation with temperature schedules (validated on MNIST) · non-learned write-time admission control (spacing gate with refuse-and-relocate + bounded-damage certificate; "refusal is a correct controller output" is existing doctrine) · exact delete + lifetimes · gradient shield · measured capacity/reach/decay constants.
- **Pipeline:** frozen encoder → grouping/addressing head → wells×modes factored store (admission-gated writes) → conditional dreaming (clamp addressed wells, relax the rest) → frozen shared decoder. Refusal = read-time analogue of the write certificates (tier 1 energy barrier; tier 2 reach predicate).
- **To build (only four):** standard conv decoder (days) · conditional/clamped dreaming (modest integrator extension) · read-time refusal certificate (assembly — quantities already exposed) · the encoder-side grouping/addressing head (**the one real build; carries the known 24 % addressing risk; already decided as needed**).
- **Deferred from paper 1:** the transformer controller — the hand-coded gates make the stronger claim ("refusal is derived, not trained") and save weeks. B4 deferred.
- **Second showcase adopted into the claim set: generative unlearning** — byte-exact deletion of a concept's modes ⇒ the model provably cannot generate it, neighbours intact; vs approximate diffusion-unlearning baselines. Near-free given exact delete. Claim suite: **compose · refuse · delete · enrol-without-forgetting · attribute** — no baseline generator has more than one.

---

## ⟲ 2026-09-09 — novelty scout returned (`.claude/outputs/generative-novelty-scout.md`): BOTH HEADLINE CLAIMS UNCLAIMED. B18 closed.

### Verdicts
- **V1 (B-vs-C): NOT FOUND.** Nearest three each break one leg — Diverging Flows (arXiv:2602.13061; trains on PGD-synthesised negatives, cannot separate unseen-valid from impossible) · Latent Safety Filters (RSS 2025, arXiv:2502.00935; dynamics-derived refusal but over *actions*, needs labelled failures — the mechanism's nearest sibling, must-cite) · Open-World CZSL (CVPR 2021; infeasibility without invalid examples but discriminative). *"The accept/reject asymmetry between novel-valid and impossible is unformulated as a task"* — we define the task.
- **V2 (exact generative unlearning): claimed once** — CDM (arXiv:2308.01937, preprint), shard-dropping only. All single-model erasure is approximate and **broken by 4 attack papers** (7 methods circumvented zero-training, ICLR 2024; concept resurgence under benign fine-tuning). Surgical mode-level deletion with neighbours provably intact = **unclaimed**; the field's own survey lists formal guarantees as an open problem.

### Positioning line (ADOPTED verbatim)
Novelty is NOT compositional energy generation (Du 2020 / COMET / LACE own it), NOT concept deletion (crowded, broken), NOT editable memory (RDM/Larimar/Memory Layers). It IS: **(a)** the accept-unseen-valid / refuse-impossible asymmetry from one dynamics-derived criterion with **zero invalid-example supervision**; **(b)** an enumerable wells×modes store making deletion and provenance **structural**.

### Dataset slate (proposed, B19 largely closed)
1. **CLEVR-CoGenT** — the standard compositional split; meets Du/COMET on their ground.
2. **Shapes3D + dSprites, Montero (ICLR 2021) recombination-to-element splits** — the hard bar; the field's negative result lives here; a win is the extraordinary claim and carries the full control stack.
3. **C-queries from the physical-AI impossibility taxonomies** (WorldModelBench's 5 violation types incl. mass conservation; IPV-Bench's 14 categories) — ready-made vocabularies, world-model lens satisfied.
4. **UnlearnCanvas** (NeurIPS 2024 D&B) — the only erasure dataset with factor structure; bridges composition + deletion.
5. Conservation-law sim (mechanism) + PUR (real data) as before.
- ⭐ **Census finding, now a measured literature fact:** the four fronts share almost no datasets; **no existing dataset supports all five capabilities** ⇒ "we construct the testbed" is evidenced, not asserted. A per-capability venue split must be explicitly defended in the paper.

### New binding cautions
- ⛔ **LACE (arXiv:2110.10873) is unread and load-bearing** (latent EBM on a frozen generator, zero-shot unseen combos). No novelty sentence until read in full. (B23)
- ⛔ **Sampler discipline:** Reduce-Reuse-Recycle (ICML 2023) — compositional failures are the *sampler's* fault; Soiffer 2026 — score error dominates off-support. The conditional-dreaming build spec must declare its sampler and address Metropolis correction. (Binding on the engineer's task file.)
- ⛔ **Zero invalid-example supervision is the differentiator** — no synthesised negatives anywhere, not even threshold tuning. One tuned threshold forfeits claim (a).
- Must-cite set for the method section: Du 2020 · COMET · LACE · Composable/RRR diffusion · Wiedemer 2023 (the compositional-generalisation theorem) · Latent Safety Filters · Diverging Flows · OW-CZSL · CDM + the 4 attack papers.

### ⟲ 2026-09-09 — Head rulings + PUR gate CLEARED
- **Dataset slate CONFIRMED by the Head** (CLEVR-CoGenT · Shapes3D/dSprites-Montero · physical-AI impossibility taxonomies for C-queries · UnlearnCanvas · conservation-law sim · PUR). B19 closed.
- **B21 CLOSED — PUR is admissible: 0 of 32 tripwires fired.** Store 0.274 vs weakest reference 0.546 (−50 % rel). The pre-registered prediction held: spatial blocking (median nearest neighbour 265 km, zero <10 km) collapses the donor arm 0.64 → 0.11 while references retain 0.82–0.89. PUR partition is pinned from the authors' own code. B22 (fallback sim) moot.
- ⭐ **Third benchmark-design finding of the pivot** (HEPA→lookup · PUB→donor copy, standalone and CLU-independent · PUR admissible). The paper's motivation section now has three measured demonstrations.
- ⚠ **Carried caveat, verbatim spirit:** admissibility is a precondition, not a result — references sit at 0.578–0.701 vs classical floor 0.274; a CLU arm must be competitive in the upper band to matter. PUB frontier correction (0.786, not 0.69) already banked.
- **LACE full read: Advisor executing now (B23).** Next artifact after that: the engineer build task (decoder · conditional dreaming with sampler ruling · read-time certificate · addressing head).

### ⟲ 2026-09-09 — B23 CLOSED: LACE read in full (Advisor, primary source arXiv:2110.10873v2 + abstract page)
- **LACE = attribute classifiers on a frozen StyleGAN's latents**; energy = classifier logits + Gaussian prior; ODE sampling; AND/OR/NOT compose classifier energies; needs full attribute labels (50k CIFAR / 10k FFHQ pairs). Zero-shot unseen-combination ACC 0.935–0.982 vs StyleFlow 0.679–0.694, up to 1024².
- **Confirmed ABSENT in LACE:** refusal, invalid-combination detection, deletion, any persistent/enumerable concept store. **Both headline claims survive.**
- ⛔ **New never-quote: capability 1 (compose unseen-valid) is NEVER presented as novel on its own** — LACE owned it in 2021. It is table stakes; the claims are the asymmetry (a) and the structural store (b).
- ⭐ **LACE is adopted as a named competitor arm** (replaces the vague "EBM + threshold" row): same frozen backbone, classifier-energy composition, energy-threshold refusal as the canonical post-hoc baseline. Structural contrast for the method section: LACE's energy is derived from label supervision; ours is a learned store written from data — which is why deletion/provenance/dynamics-derived barrier exist for us and cannot for LACE.
- **Build task `gen-clu-phase1` handed to the Head 2026-09-09** (engineer, via hub): loaders w/ Montero splits · frozen encoder + cached features · one shared frozen decoder · correlation-clustered wells + store-only stage-(ii) training · clamped-well conditional dreaming with the sampler ruling (declare sampler, Metropolis stance, audit traces) · read-time certificate with zero invalid-example supervision (thresholds from valid data only) · first A/B/C cell on Shapes3D with pre-registered kill rules (store-removed matches on B ⇒ fail · not separable from kNN ⇒ fail · certificate AUROC ≤ 0.5+spread ⇒ fail; failures return to Head+Advisor).

### ⟲ 2026-09-09 — D3 (encoder pretext) RULED
- **Reconstruction pretext is PRIMARY, mandatory.** A factor-classification pretext collapses the adopted LACE contrast ("their energy from label supervision; our store learned from data") and lets reviewers attribute the semantic organisation to labels one stage upstream. Also matches the Montero bar (their negative result is on reconstruction-trained models — same setting makes our win extraordinary).
- **Classification pretext retained as a labelled DIAGNOSTIC arm only** (never a headline row): fault isolation for kill-rule failures — recon fails + class passes ⇒ features at fault; both fail ⇒ store/addressing at fault.
- The evaluation probe classifier still trains on factor labels — measurement, not the generative path.

### ⟲ 2026-09-09 — phase-1 task concerns ruled (engineer, via Head)
- **C set: RULED — defined in factor space** (two values of one factor; out-of-range value), fixed before grouping runs, identical for every arm. Well-space mapping of the same queries reported post-grouping as a grouping diagnostic only. (Same lesson as the donor arm: query sets must be arm-independent.)
- **Decoder ceiling: RULED — promoted to a GATE.** Probe accuracy on decoded *real* cached held-out features = the B ceiling. Pre-registered floor: ceiling on B ≥ 0.8× its A equivalent, else the phase stops before arms run. Kill rule (i) read relative to the ceiling. Assertion required: the decoder never trains on held-out-combination images (evaluation-only use).
- Third concern announced but not yet received.
- **Third concern RULED — LACE-threshold on C is NOT a phase-1 kill** (label-supervised vs our zero-label certificate; phase-1 kills test mechanism existence, not margins). The margin is pre-registered and reported from phase 1, and **binds at paper time** via the tier-1 claim (must beat post-hoc thresholds by submission). **New kill rule (iv):** certificate loses the paired C-vs-(A∪B) AUROC to the *kNN distance-threshold* arm (same supervision class) beyond seed spread ⇒ phase fails — naive distance should fail B-refusal on the paired metric, so losing to it means the derived criterion adds nothing. Build note: request-embedding convention declared once for all refusal arms (composed per-factor prototypes for kNN; composed classifier energies for LACE).

### ⟲ 2026-09-09 — phase-1 hub questions ruled
- **Refusal metric: PRIMARY = AUROC(B unseen-valid vs C impossible).** Kill rules (iii)/(iv) bind on it. C-vs-A and C-vs-(A∪B) reported alongside. C set sized comparably to B, pre-registered, sampled from the factor-space definition.
- **Unseen set widening: Advisor recommends YES as a labelled SECONDARY split** (Montero's 72 = headline, protocol untouched; our extra held-out combos reported separately for statistical power; verified-absent-by-assertion). Head to confirm.
- **Backup push: internal (code+state) fork only** — standing rule; never the code-only base, never origin.
- **Confirmed:** decoder gate (≥ 0.8× floor, stop-and-return) · store-deletion check before arms (kill rule (i), early).
- **Encoder/LACE trade-off RULED:** our choices never weaken a competitor — shared feature width so LACE whitens natively, or a declared PCA projection for LACE reported both ways.
- **Prereg sign-off:** Head + Advisor sign point predictions and kill thresholds before the cell switch.

### ⟲ 2026-09-10 — the matched-seen control finding; kill rule (i) re-read; decoder gate FIRES on the compositional term
- **Finding (spoke, unrequested, exemplary):** Montero's held-out cell (all three hues > 0.8) kills photometric contrast. Against photometrically matched *seen* images, `scale` is 0.708 held-out vs 0.703 seen (rendering artefact, not composition); `shape` is the only genuinely compositional factor: 0.876 → 0.500. Four of six factors sit at ~1.0 and carry no compositional information.
- **RULED — option 1:** kill rule (i) reads on the compositional term (`shape` primary; `scale` normalised against matched-seen); aggregate reported alongside; the matched-seen control is a permanent companion row. Pre-arm, pre-cell, filed as irrevocable. Option 3 (renormalise everything) rejected.
- ⛔ **Consequence:** the decoder gate must read on the same term ⇒ shape ceiling 0.500 / 0.876 = **0.57× < 0.8× floor — the gate FIRES.** The decoder cannot render held-out shapes from real features (the capped-B failure, exactly as predicted). **Order:** (1) improve the decoder, bounded (train-split only, ≤2 variants, ~1 day), re-probe; (2) if the ceiling doesn't move, pre-register the floor at the measured ceiling with reason, and the widened secondary split becomes MANDATORY.
- **Widened secondary split: Advisor now says NECESSARY, not optional** — 72 items × 3 seeds in a 0.25–0.50 band cannot separate arms at reviewer-acceptable confidence. Awaiting the Head's word.
- **2026-09-10 — Head confirmed: gate FIRED.** `scale` vs matched-seen 1.007 (photometric attribution vindicated); `shape` 0.571 vs matched-seen, 0.653 vs random-A; floor 0.8. **Phase-level STOP on the compositional term.**
- **Next bounded pass (ruled):** (0) attribute the cap first — linear probe for `shape` on encoder *features* (held-out vs matched-seen); ~0.57 ⇒ encoder never carries it (fix upstream; D3 diagnostic arm is the tool); high ⇒ decoder at fault. (1) ≤2 variants on the blamed side, train-split only, frozen after, ~1 day; re-probe against 0.8× on `shape`. (2) If still below: floor pre-registered at the measured ceiling with attribution stated; widened secondary split carries the statistics; Shapes3D compositional claim scoped to the band.
- Side-note parked: Montero's negative result on this cell may be partly a contrast artefact — appendix line later, not chased now.
- **2026-09-10 — mechanism clarified (Head Q):** B (recombination-to-element) never requires a new mode — only novel co-activation of existing modes across wells. New factor *values* (extrapolation) would need new modes: not a phase-1 claim. Well assignment = grouping head, once, per channel (structural). Mode assignment = landscape geometry + admission radius (in-basin ⇒ same mode; beyond d_safe ⇒ new mode; between ⇒ refuse-and-relocate); no learned classifier.
- ⭐ **Implication for the STOP:** a decoder that can't render held-out shape from *real* features means hue has leaked into the shape channels — an entanglement failure in the encoder/grouping, not a decoder failure. **Attribution probe sharpened to PER-WELL:** predict shape from the shape-well's sub-vector alone, held-out vs matched-seen. Collapse ⇒ fix is the learned-rotation grouping arm or a better-factorising pretext, not decoder variants.
- **2026-09-10 — three Head Qs ruled/clarified:** (1) composition uses the **k-particle launch** (learned multiplicity + stopping rule + weighted counting code), never single-particle; read-time addressing is learned in the launch head (correction: write-time placement is geometry). (2) ⛔ **Refusal must be a positive certificate computed on the clamped request, never inferred from a failed read** — addressing failure masquerading as refusal = the trivial-detector failure; "certificate valid / read failed" is its own reported row. (3) The decoder absorbs *consistent* channel mixing, not *combination-dependent* leakage; the learned-rotation grouping arm exists to buy factorisation the encoder lacks.
- **2026-09-10 — FUTURE (B24): multi-layer feature trajectories (L−2 → L−1 → L) as the written item.** Advisor read: richer than the final point, *not* richer than the earliest layer (frozen encoder ⇒ later layers are deterministic functions); the gain is structure, not information. Prior trajectory-write/read negative exists in the codebase — read its cause of death before designing. Costs: bytes scale with layers × spatial size (lookup gets the same raw material); per-layer projections = 3× learned addressing. **Cash-in NOW:** run the per-well probe on all three layers in the attribution pass — if L−2 is less entangled for shape, the substrate becomes "least-entangled layer per well," which addresses the STOP with no redesign and gives the multi-layer idea its first evidence. Not a paper-1 feature.
- **B24 refined (Head, 2026-09-10):** the cross-layer trajectory makes the kept/discarded partition *explicit* (path direction = what the encoder deemed irrelevant — under a task-trained encoder that's task-irrelevant *signal*, i.e. generative factors). Prior trajectory negative does NOT transfer (it unfolded one settled point; this has distinct content per point). New risk: per-layer projections + path-following reads (never quantise the continuous coordinate).
- **Entanglement quantification RULED:** primary = per-well probe transfer (seen→held-out, matched-seen-normalised; the drop is the leakage), per layer; companions DCI/MIG; label-free = post-clustering block-diagonality. **Forcing disentanglement — supervision discipline:** label-supervised loss inadmissible for the headline (re-opens D3); unsupervised TC/decorrelation penalty at the penultimate (or one layer earlier) admissible; weak supervision as a labelled arm. **Order:** learned-rotation grouping (post-hoc, encoder untouched) → unsupervised penalty → weak-supervision arm. Full labels never.
- **2026-09-10 — PAPER-1 EXPERIMENT ADDED: the factorisation ladder** (mechanism section, Shapes3D, ~10 GPU-h). Rungs R0 fixed grouping · R1 learned rotation · R2 unsupervised TC/decorrelation penalty at penultimate · R3 same one layer earlier · R4 weak supervision (labelled arm) · R5 full-label pretext (diagnostic ceiling, never headline); random-grouping control + matched-byte kNN per rung. X = per-well probe-transfer leakage (matched-seen-normalised); Y = B composition (fraction of rung's decoder ceiling), AUROC(B vs C), deletion locality. Prereg: P1 monotone; P2 label-free rungs ≥ X% of R5; P3 deletion locality least leakage-sensitive. KILL: capabilities don't track leakage ⇒ mechanism section rewritten. Only R0–R3 headline-eligible. Converts the STOP into a section and pre-empts "conveniently factorised encoder." **Trajectory representation banked to paper 2 (B24).**

### ⟲ 2026-09-11 — decoder gate UN-FIRED; store proven load-bearing; the one-class catch
- **Decoder gate clears:** `shape` 0.5926 → 0.9398 held-out, ratio 0.996 vs matched-seen, 3/3 seeds. **Two pre-registered expectations falsified** (decoder-only wouldn't reach 0.8; object-weighted loss = the clue — it lost at 0.537; the null-control variant, plain capacity + longer schedule, won). Both to the negatives registry. Lesson: shipped decoder was under-resourced, not mis-objectived.
- ⭐⭐ **STORE IS LOAD-BEARING, STRUCTURALLY:** deleting it moves the read by +0.4752 vs 0.01 near-zero line (475× the historic <0.001-bit failure). **A bypass is inexpressible** — the read takes no features; the only encoder→decoder channel is 30 bits. Goes in the method section verbatim; 30-bit width = declared design constant. Correlation grouping beats random +0.20 on `shape` (81× seed spread) = first evidence wells carry factor structure; = ladder R0 point.
- Widened split: assertion fires on all 13 cells; Montero re-emits bit-identically. Mandatory stands.
- ⛔ **The catch:** Montero's held-out cell is `shape == cylinder` — one realised class ⇒ held-out `shape` accuracy is a *detection* rate (majority 1.0) while the seen normaliser is 3-way discrimination. The 0.59→0.94 lift survives (same probe, same set); the ratio-to-ceiling that kill rule (i) reads does not.
- **RULED:** (1) interim one-class matched-seen normaliser + widened split structurally (hub's proposal, confirmed); never drop `shape`. (2) **Kill rule (i) reads within-corner discrimination** — held-out cylinders vs matched-seen non-cylinders in the same colours, binary AUROC / balanced accuracy — until multi-shape held-out cells make it moot. (3) **Ladder x-axis redefined:** leakage = 1 − balanced-accuracy transfer (held-out ∪ matched-seen-other-shapes, same corner) ÷ seen-corner balanced accuracy; never a ratio of raw accuracies. (4) Re-score spoke approved (R0 under new decoder + corrected normaliser); **add:** re-run the decoder ceiling gate on the widened split's multi-shape cells (0.94 is detection, discrimination unproven).
- Outstanding: backup push (internal-only, ruled) · launcher live smoke before any arm.

### ⟲ 2026-09-11 — analyst re-score accepted; ladder sequencing + kill-rule statistic RULED
- **Gate verdict HOLDS under balanced accuracy:** V1 0.9414 ± 0.0179 (3/3 clear 0.8); baseline 0.7191 (0/3). Old held-out `shape` accuracy = exactly this metric's TPR (0.5926 / 0.9398 verbatim); the correction added the TNR column: baseline called 15.4 % of seen non-cylinders "cylinder", V1 5.7 %. **The lift was real discrimination.**
- **R0 read margin decoder-independent:** +0.1504 before and after the decoder swap (both arms lift +0.0110/+0.0111).
- ⛔ **CORRECTION TO 2026-09-11 ENTRY ABOVE:** "wells carry factor structure" is established on the **READ axis only** (+0.2008, 12× spread). On the **leakage axis it is NOT established** (+0.0147 ± 0.0771, sign-inconsistent across seeds). Pooled leakage 2.26× spread — directionally right, cannot carry a headline. Degeneracy corrected in ledger: −198.0 worst case, 30.6 % NaN (spoke reported −12.9).
- **RULED — ladder sequenced AFTER `gen-bridge-wide`** (all four shapes held out ⇒ four-class discrimination both sides; free; third independent route to the structural fix). Option 3 (switch x-axis to MIG/DCI-C because they separate) REJECTED — metric-shopping; MIG/DCI-C stay pre-registered companions. **Fallback pre-registered:** if widened-split shape leakage separates groupings by < ~3× spread, P1 reads off pooled leakage + companions and shape-leakage is declared underpowered, in those words.
- **RULED — kill rule (i) reads BALANCED ACCURACY; AUROC mandatory companion** (0.9886 vs 0.9414 shows the loosening a silent swap would buy).
- **30-bit store ceiling:** in-corner read 0.6866 vs real-feature ceiling 0.9414 — declared companion to the decoder ceiling in the cell; bounds all arms on B. ⛔ Channel width NOT touched in phase 1 (it is why bypass is inexpressible); capacity-vs-bypass-risk = a separate future axis, pre-registered as such.
- **2026-09-11 (later) — widened-split results:** the gate-failure residue reduces to one confusable pair (cube↔cylinder) under low contrast, isolated to a single cell (best wrong class 0.517 = coin flip). 8/13 cells at ~1.0 ceiling — headroom for arms (store in-corner 0.6866), not a resolution problem. **RULED:** shared-corner cells `montero` + `wide-000` (each holds out the other's positives; negatives realise only 2 of 3 other shapes) are quoted as a **two-class corner, never pooled**; pooled headline on the other 11 cells. Residue finding = the appendix line for the Montero-critique note. Cluster smoke runs before the cell.
- **2026-09-11 — RULED: R0 re-fitted on the wide feature cache; the ladder runs entirely on the wide configuration** (~1200 Adam steps, 3 seeds; no encoder/decoder/K change). Riders: (1) re-derive the correlation grouping on the wide train cache and report agreement with the Montero grouping (adjusted Rand) — K fixed; (2) Montero R0 = literature link in the compositional table, OFF the ladder figure; (3) wide-R0 vs Montero-R0 reported side by side once as a cache-consistency check — divergence beyond seed spread on the read margin ⇒ stop before R1 (plumbing fault). R1 waits on the re-fit + agreement number.

### ⟲ 2026-09-12 — four spokes: dreaming FALSIFIED, certificate is the first headline-grade number, C-set giveaway caught
- **`gen-dream-sampler` — falsifier FIRED (contrast +0.0095 ± 0.0082 vs pre-registered +0.03).** Coupling-off control: gain +0.0200 where the clamp provably cannot reach ⇒ steering-attributable = zero. The chain samples a mode; the clamp doesn't steer. **RULED: clamp-and-relax conditional dreaming is RETIRED as the composition mechanism (it was the Advisor's addition).** Composition = **k-particle launch read over the requested modes → code → frozen decoder** (the validated mechanism). Dreaming survives only as within-mode diversity sampling, off the critical path. Partial-request completion banked as a secondary capability, not scored in phase 1. Item 4 (addressing clamps 4.42/5 wells ⇒ relaxed set degenerate) reaches the same conclusion independently.
- **`gen-certificate` — TIER-1 REFUSAL STANDS:** barrier 0.9146 vs naive control 0.5630 after the giveaway is removed (+0.3516, 9.5× spread); attribution names the culprit factor 0.797/1.000 vs chance 0.167; zero-invalid-supervision audit clean (0 C ids across 15 fits / 78,848 ids). On the contaminated C set the margin was +0.077 (equivocal) — the control did its job.
- ⛔ **C-set giveaway (hub's own spec):** all 36 `duplicate_factor` records carry 7 terms vs 6 for A/B — a term-counting function scores AUROC 1.0 on that half, 0.75 pooled. Fix A7 shipped (valid requests with a redundant term, built from valid data only; supervision claim untouched). **RULED:** A7 approved; baselines' C numbers RE-SCORED not re-run. **New standing rule: every C set carries a pre-registered no-model detector control** (request-structure-only function must score ≤ 0.5 + spread) before any arm is scored.
- **`gen-r0-wide-refit`:** read margins agree wide vs Montero at 0.70× spread — no plumbing fault; wide margins +0.1410 aggregate / +0.1813 shape. But **the leakage axis inverts sign across configurations** (6-class numerator vs 15-class denominator; balanced accuracy normalises within a class count) and the pre-registered fallback TRIGGERED (shape leakage 1.56× spread). Hub's and Advisor's expectation that the wide split would restore resolution: **wrong, recorded.**
- **RULED — ladder x-axis promoted to DCI-C**, chance-corrected transfer (BA − 1/c)/(1 − 1/c) as companion. Justification is **configuration-invariance** (same features, different partition ⇒ same value), a criterion independent of rung results; leakage fails it structurally, DCI-C passes. Stated in the paper. **Standing rule (third like-for-like strike):** every ladder/cell metric gets a pre-registered invariance check before use. R1 unblocks on DCI-C passing it.
- **Grouping re-derived per seed** (labels index each seed's own encoder channels); only K, n_modes, D, store_bytes portable. Cost logged: ceiling 0.5579 vs 0.6208 on seed 1.
- **ARI rider RETIRED** — premise wrong (cross-encoder ARI bounded by encoder reproducibility ~0.06; measured 0.115 > Montero's own cross-seed). The plumbing check that holds is the read-margin agreement.
- Smaller, ruled: launcher gets a `certificate` stage · `data_root` ambiguity must FAIL LOUDLY (silent C-skip = not-run reported as null) · cluster smoke still gates the cell.
- `gen-baselines`: 3 seeds finished, report pending; C numbers to be re-scored under A7.
- **2026-09-13 — Head confirmed:** phase-1 prereg signed after smoke + A7 re-score; dreaming off the paper-1 critical path.
- **RULED — kNN byte advantage: declared-and-kept** as the harder-for-us over-budget row (CAMELS precedent), declared in the table row itself. **Trigger:** kNN beats or ties CLU on any headline cell ⇒ matched re-run at 524,288 mandatory before that cell is quoted.
- **RULED — no-model detector control becomes a SUITE TEST**, run on the registered query-set artifact (every future C set inherits it), using every field except factor values, asserting AUROC ≤ 0.5 + declared tolerance vs A∪B, failing loudly. Silent C-set skip (`data_root`) = test failure, not a warning.

### ⟲ 2026-09-13 — baselines report: kill rule (ii) in danger; three rulings
- **`knn_proto`** (per-(factor,value) mean of train features, composed additively, decoded; no training/energy/sampler; **in budget**, unbounded 2048-float read channel) reaches in-corner `shape` BA **0.8884 ± 0.0088** — +0.2018 above our 0.6866 store ceiling (23× spread), 94.4 % of decoder ceiling. **Channel-matched kNN at 25–42 bits: 0.5206 (~chance).**
- **RULED — kill rule (ii) reads the BYTE-MATCHED unbounded-channel arm, as pre-registered.** Reading channel-matched would be a post-baseline amendment in the kinder direction — the first such; rejected. **Scope clarified now, before the CLU arm runs:** (ii) firing kills the *composition claim*, not the phase — composition was already table stakes (LACE); refusal (iii/iv) and structural deletion are not conditioned on (ii). If it fires: reportable task finding — Shapes3D recombination-to-element is solved by additive prototypes given decent features. **Both rows in the paper:** byte-matched (prototypes beat our ceiling) and channel-matched (30-bit store 0.6866 vs same-width lookup at chance = bits-efficiency statement, presented as such, not a rescue).
- **New pre-registered arm: CLU-wide** — decode from the store's composed per-well mode vector at `knn_proto`'s channel width; store→decoder only (bypass guarantee intact). Honest like-for-like; if it ties `knn_proto`, composed modes are prototype means with machinery and we say so. Sits beside (ii), does not replace it.
- **RULED — kill rule (iv) reads the ORACLE bound (0.8373 ± 0.0109), not deployable (0.7998 ± 0.0443)** — valid-only statistics cannot select for detection power; surrogate invalid data breaks the supervision-class match; the lower number is a false pass.
- **RULED — LACE whitening exactness GATES its configuration:** non-exact seed ⇒ fall back to nearest exact configuration or run the general form with the prior score; both reported, stronger quoted. Never-weaken-a-competitor covers our selection code.
- **Reporting rule:** C results always split by kind (out-of-range / duplicate-factor); pooled as companion only. kNN 0.8045/0.3907, LACE 0.3971/0.9991 — neither covers both; our certificate 0.797/1.000 does. ⚠ Not quotable until rivals are re-scored under A7.

### ⟲ 2026-09-13 — THREAD HANDOVER. Wave-5 relay (main @ 5fee3f4) received, NOT yet answered.
Handover written to `.claude/advisor-pivot-handover.md` (boot line inside). The new Advisor thread answers Q1–Q4 of the wave-5 relay first; outgoing leanings recorded there, not here.

### ⟲ 2026-09-13 — wave-5 relay ANSWERED; pivot re-aimed at the full CLU (Head ruled)
- **Q1 RULED:** the 0.86 exact-set was **per full set** (whole subset recovered), precondition "handed correct addresses". Shapes3D comparable = **0.510 per K-tuple**, not 0.871 per well; the gap is addressing. Hub verifies the unit against the artifact row before quoting anywhere; never mix units in one comparison.
- **Q2 RULED:** composition **released as a claim**. 9-of-72 reported as a measured **commitment collapse** (not capacity — 30 bits ≫ 72 codes). Kill rule (ii) fired as pre-registered (CLU-wide −0.1400, 0/3). Disk-only rider: check whether the 9 codes coincide with seen configurations (retrieval-to-nearest-well vs composition) — one line in the write-up either way.
- **Q3 RULED:** ladder confirmed — wide config, R0 re-fitted, grouping per seed, DCI-C axis, MIG companion. Sign convention pinned in the prereg: corr − rand, positive = good. DCI-C qualified on the invariance criterion alone, never on having diagnosed the other metrics' failures.
- **Q4 RULED:** escalate if a rival survives A7. Pre-registered now: a surviving rival's margin sets the refusal comparison (not naive 0.5630); if the barrier does not clear it, refusal tier 1 drops to companion. **Already fired in part:** kNN 0.8045 ± 0.1227 ties the certificate 0.797 on out_of_range ⇒ refusal claim = *one dynamics-derived criterion covers both kinds with zero invalid supervision, where each rival covers one*; per-kind reporting; the out_of_range result is a **tie**, never a kNN win or loss. Not quotable until the duplicate_factor wiring spoke lands.
- **Q4 follow-ups (Advisor recommendation; Head has not contested):** (a) LACE fallback (nearest exact whitening config, or general form with prior score; both reported, stronger quoted) is **mandatory on the 2 non-exact seeds before any LACE number is quoted** — 0.4694 is a provisional lower bound; quoting as-is would be weakening a competitor by our own choice. (b) Composition-cell matched re-run trigger **discharged by proof in the table row** (prototype store size = factors × values × width, already under budget ⇒ re-run returns the same number); moot anyway once composition is released.
- ⭐ **HEAD RULED — the pivot re-aims at the full CLU.** Advisor's honest read, accepted by the Head: phase 1 is a mechanism study; the wave as scoped tests features in isolation on a simple problem and produces no compositional-generation result. Advisor names its own error: leaning to park commitment in phase 2 was wrong.
  1. **Commitment collapse = wave PRIMARY.** Pre-registered falsifier: distinct codes per 72 requests vs stopping rule / multiplicity / launch temperature. Pass ≈ ≥60/72 distinct **and** store→decoder arm within spread of knn_proto. Fail ⇒ **new approaches, not abandonment** (Head: keep attacking — k-particles, particle traversal, etc.).
  2. **Generation harness built NOW, in parallel:** request → launch read → codes → decoder → image → external classifier on held-out combinations; rivals LACE, energy-based composition, knn_proto; the venue's protocol. This is the headline table; nothing in the pipeline produced it before.
  3. **Encoder unfreeze with the store in the loop enters the plan now** (bypass guarantee kept; this is the "full CLU"). Cluster jobs; needs the Head's 2FA step.
  4. **Navigation** (well-to-well traversal, external classifier along the path — the vision's untested core claim) queued directly after 1–2.
  5. **Ladder R1–R5 = mechanism appendix, background.** Must not consume the wave.
- **Honesty lines for the paper:** store-load-bearing is by construction (30-bit channel) ⇒ method section, not headline. Deletion is a differentiator only through the dynamics (knn_proto can drop a mean too). Real-data (PUR / long-horizon time series) leg follows item 3.

### ⟲ 2026-09-14 — waves 6–7 sitrep received; four decisions RULED; composition NOT abandoned
- **Sitrep facts (hub, W6–W7):** commitment collapse FAILED as pre-registered, 3/3 seeds (~9 distinct codes; 5× width ⇒ 70 distinct but scale still wrong ⇒ not a width problem). Loss profile confirmed three ways: the 30-bit channel carries colour + shape, discards scale + orientation, everywhere. Headline generation table exists: knn_proto 0.91 / CLU 0.66 / store-deleted 0.50 (external classifier; agrees with old measure within 0.006). Rivals fair, worst delta 0.0 across 30 numbers ⇒ refusal QUOTABLE (kNN catches out_of_range, LACE catches duplicate_factor, ours both). dSprites 0.9616 (3 seeds) — claim it supports and its knn_proto row still to be named. Navigation NOT RUN (endpoints irreproducible), not "wrong idea" — hub to reclassify. Cluster: upload script had two bugs (cluster ran 6-day-old code); JAX inits on A100; smoke still not green; certificate stage never executed. Full suite owed.
- **RULED (Head):** (1) `clu-dev` backup now, via the existing push runbook. (2) Loss-profile spoke YES — attribution first: probe scale/orientation at encoder features → per-well sub-vector → store code; falsifier: encoder ≥~0.9 & code ≈ chance ⇒ store grouping/quantisation at fault (local fix admissible); encoder lacks it ⇒ store innocent, unfreeze is the only lever. (3) Full suite at a quiet moment, HEAD recorded before/after. (4) Encoder unfreeze BUILDS NOW with a local tiny-scale smoke as acceptance; cluster run gated on the green smoke; Head clears the 2FA login.
- ⭐ **RULED (Head): composition is NOT given up.** Attack it with a free hand to add or remove features. ⛔ **Every new feature is brainstormed Head + Advisor before anything is built.** Analysis-only spokes (attribution) may proceed; new mechanisms wait for the brainstorm ruling.
- Hub corrections relayed: navigation → not-run; dSprites needs its claim + tripwire row; headline-table caption carries the loss profile.

### ⟲ 2026-09-14 — composition brainstorm: candidate list + Head's lever RULED
- **Diagnosis (Advisor, accepted):** wells ≠ factors; knn_proto wins because one-mean-per-factor-value is additive in feature space; our store would do the same if wells = factors and modes = values. Width ruled out (5× ⇒ distinct but wrong). Attribution spoke decides world A (encoder carries scale/orientation, store drops it) vs world B (encoder never had it).
- **Candidates tabled:** (1) factor-aligned wells without labels (learned rotation / decorrelation from the ladder, judged on the loss profile) · (2) continuous landscape-derived coordinate per well for continuous factors (changes the 30-bit constant; bypass re-proved) · (3) finer admission radius · (4) write earlier layers / trajectory · (5) encoder unfreeze (ruled) · (6) hybrid read (risk: baseline with machinery) · (7) weakly supervised grouping (diagnostic ceiling only) · **(8) HEAD'S LEVER: self-supervised encoder** — reconstruction family only (AE/MAE headline; disentangling VAE as declared comparison arm; ⛔ augmentation-contrastive excluded: it trains invariance to exactly scale/orientation).
- **RULED (Head):** candidate 8 enters as stated; runs BEHIND the attribution spoke; encoder stays frozen after SSL training for phase 1, store-in-the-loop unfreeze follows as its own attributable step. Hub to confirm what the current encoder was trained on (label exposure upstream of the "zero supervision" claim).
- **Falsifier for 8:** scale/orientation probe on new features ≥ ~0.9 on seen data, else swap within the reconstruction family before touching the store; then the same probe on the store code decides whether 1/2 are still needed. All phase-1 numbers re-run on the new cache (refusal re-measured, never assumed). knn_proto gets the same features.

### ⟲ 2026-09-14 — ⛔ the phase-1 store is a product quantiser, not the CLU; Head RULES "both, dynamics-first"
- **Advisor finding (verified in shipped code on main):** the compose path = correlation-partitioned wells × per-well k-means codebook (soft→hard, commitment pull) + a count-table address book (per factor VALUE mode histograms, naive-Bayes product, argmax per well) + codebook copy-back read + frozen decoder. No landscape, no particles, no settling, no inter-well coupling (declared unfitted/undeployed), no decay on that path. "k-particle launch read" in the compose module names an argmax over the count table — ⛔ **claims-integrity risk: the mechanism the paper would describe is not the one the code runs.** The dynamical compose (clamp-and-relax dreaming) was retired when its falsifier fired; the quantiser is what remained.
- **Consequences named:** (a) knn_proto and our store draw on the same per-value statistics of the same features; ours quantises ⇒ the composition claim was unwinnable by construction (mirror of a test that cannot fail). (b) The loss profile is k-means on reconstruction MSE keeping high-variance directions (colour, shape) and discarding low-variance ones (scale, orientation). (c) 9-of-72 = flat count-table rows for scale. (d) Navigation is meaningless on a codebook. (e) Deletion = zeroing a codebook entry — not a differentiator vs knn_proto.
- **Address book stays:** labels ground the words at the interface only; every arm pays it identically; an unsupervised encoder changes nothing on that path.
- ⭐ **RULED (Head): option 3 — BOTH. Quantiser kept as the write-side organiser; the READ becomes dynamical, and the programme's focus is the dynamics from here on.** Request → address-book clamp on the addressed wells → launch particles → settle in a per-well landscape with a FITTED inter-well coupling → decode the SETTLED (continuous) state via the store→decoder path at prototype width. **Rollouts at 1000 steps, not 4000** (Head, in the interest of time); rider (Advisor): a chain still descending at 1000 steps is reported NOT CONVERGED, never as a null. Bypass guarantee re-proved: the settled state is a function of request + store only (the request-only assertion extends to the launch).
- **Brainstorm gate stands:** the hub brings the dynamical-read DESIGN (spec, falsifiers, cost, artifact paths) to Head + Advisor before building. Candidates 1/2 fold into it (factor-aligned wells; continuous settled coordinate comes free).
- Encoder track unchanged: attribution spoke first; reconstruction encoder (candidate 8) behind it; frozen for phase 1; store-in-the-loop unfreeze after.
