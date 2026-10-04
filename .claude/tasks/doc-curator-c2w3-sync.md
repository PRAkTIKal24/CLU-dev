# doc-curator-c2w3-sync — fold C2W3, file the three errata, keep the registries CURRENT

**Campaign 2, wave C2W4. Agent:** doc-curator. **NO WORKTREE, NO CODE.** Writes only to `.claude/**`.
Charter **ADDENDUM 3 §A15** (*"Plus the standing curator pass — registries stay current, the three
errata fold in"*). **Spawns immediately, early in the wave** — a paper wave cannot cite stale registries,
and `bprime-draft` reads what you write.

**Read first:** `.claude/AGENT_PROTOCOL.md`; `.claude/advisor-head-c2-charter.md` **ADDENDUM 3 in full
(§A12 the adjudication · §A13 the claim architecture v3 · §A14 the eight rulings · §A15)**, plus
**ADDENDUM 2 §A7–§A11** for what C2W3 was measuring against; the **`2026-07-31 (later still)` `[C2W3]`
§10 entry** in `.claude/handover_context.md` — ⭐ **its RECONCILIATIONS-OWED table (8 items) and its
NOT-RUN list are your work order**; `.claude/outputs/{bprime-fb4-gate,route3-stage1-plus-2x2,
bprime-theory,bprime-fb1-recon,doc-curator-c1w27-c2w1-sync}.md`; and, **as they land this wave**,
`.claude/outputs/{bprime-rivals,bprime-c6,harness-debt}.md`.

⭐ **STARTING STATE — the registries are CURRENT for the first time in four waves** (your predecessor's
three passes; the consolidated dated never-quote list lives in **`claims_matrix.md` §0**). ⭐ **Your job
is to keep them that way, not to rebuild them.** ⚠ **`claims_matrix.md` v2.5 / v2.6 / v2.7 are all still
marked "pending Hub confirmation" — the curator does not self-confirm. Flag them for the Hub to confirm
together at this wave's review; do not confirm them yourself.**

---

## 0. ⭐ P0 — THE THREE LIVE ERRATA. File these first; everything downstream quotes them.

### E1 — the byte law is **24/28**, not 28/28
- **The published sentence** *"verified to 1e-9 in all 28 cells"* is wrong: the **published closed form**
  matches in **24 of 28** cells. The **corrected law `ratio = [A(D+2) + d]/(d+m)` is exact in all 28** in
  rational arithmetic (0 ulp).
- **Where it bites:** the four `manifold` (`n_spectator = 1`) cells measure **52.00×** against a
  published **43.33×** (**+8.667, 20 %**); the floor **RISES** from a printed **2.00×** to **2.40×** at
  `n_spec = 1` (the `n_spec = 0` floor **2.20×** and the measured min **2.28×** are unchanged). Shell
  floors are **2.40 / 2.60×** (`×9/8` surcharge on the atom term, `+1/(D+2) = 12.5 %`).
- ⭐ **The theorem STANDS and the error is CONSERVATIVE** — the store costs *more* relative to the table
  than we published, **so no claim was inflated** — and **`PREREG-Bprime.md` §7's reuse licence STANDS**.
- ⛔ **HUB RULING, BINDING: `PREREG-Bprime.md` IS NOT EDITED.** A pre-registration whose text is revised
  after the fact stops being one. ⭐ **File a dated erratum BESIDE it** —
  `.claude/outputs/track2-admissibility/ERRATA-Bprime.md`, pointing AT the prereg, naming the wrong
  sentence, the corrected law, the affected cells, and the direction of the error.
- **Sites to correct** (reconciliation 2): `.claude/outputs/memory-gym-v0.md` §2/§3.1 ·
  `.claude/outputs/track2-admissibility.md` §2 · **charter §A2.3** (⚠ **the charter is the ADVISOR's —
  it is already annotated in place by the Advisor; verify, do not edit**) · every registry occurrence.
  ⚠ `harness-debt` fixes the **code** this wave and publishes its diff — **use their landed numbers, not
  my summary, if the two ever differ.**

### E2 — MUNKEY's venue
⛔ **MUNKEY (arXiv:2603.15033) is an ICLR-2026 *workshop* paper (oral), NOT ICML 2026.** arXiv v3 carries
an **empty comments field**; the authors' own group page says ICLR-2026 workshop (oral) and an
independent secondary **disagrees on WHICH workshop** ⇒ ⛔ **the workshop's identity is QUARANTINED —
cite it as "an ICLR-2026 workshop paper (oral)" and name no workshop.** Its v3 self-describes as **"a
memory-augmented transformer"**, not a ViT classifier.
- ⭐ **The narrowing itself STANDS**: unlearning-by-design with **MIA-AUROC → 0.5 by design** — *our own
  instrument* — but **not exact** (gap to retraining **0.56 ± 0.21**).
- **Sites** (reconciliation 3): **charter §A9.9** (Advisor's — verify only) · `PREREG-Bprime.md` §8
  (⛔ **not edited — goes in the ERRATA file**) · `.claude/outputs/track2-admissibility.md` §3.3 and its
  never-quote 13 · C2W3 task files · **≥4 documents total; sweep for all of them.**

### E3 — monitor #6's "27 post-repair" is PROVISIONAL
The C2W2 repair is **half-landed**: the loss-slope dead-band shipped, the **`+eps_acq` half did not**, so
the theorist's two predicted **recovered false negatives** never materialised. **`harness-debt` lands the
missing half this wave and publishes a one-time re-score diff.**
- ⛔ **Until that diff lands, "27 post-repair" carries a PROVISIONAL qualifier and is never-quote-adjacent.**
  ⭐ **After it lands, file the corrected count WITH its pre/post diff** (this is your explicit assignment
  in reconciliation 4: *"have the curator file the pre/post counts"*).
- ⚠ Also standing: **"58 trips" is never quoted without *"pre-repair"***, and **the artefact count is
  31 of 58, not 29.**

## 1. P0 — the remaining owed reconciliations you own
| # | item | your action |
|---|---|---|
| 5 | **`to_race_cell` (Route 2) silently dropped the mandatory trajectory launder** — every C2W2 `route2` race cell reported `fired = False` **by absence, not by measurement**. Fixed in C2W3. ⚠ **It changes no C2W2 dividend** (the gate arithmetic never read that field) | ⭐ **File the C2W2 Route-2 `trajectory_launder` column as UNMEASURED, not clear.** That distinction is the entire entry — an absent measurement recorded as a passing control is the failure mode this registry exists to catch |
| 7 | **Six rival papers absent from our registry, incl. Gated DeltaNet-2 (arXiv:2605.22791)**, which **§A14.2 rules SUPERSEDES GDN as the delta-rule reference arm** | File all six. ⭐ **Gated DeltaNet-2 first and flagged — `bprime-rivals` is building against it THIS WAVE and needs the entry** |
| 8 | ⛔ **The substitute-audit *idea* is NOT ours in general form** — it is the partial-input / trivial-baseline audit tradition (**Poliak et al. 2018; Feng, Wallace & Boyd-Graber, ACL 2019**) | File it as a **positioning constraint**, and note its owner is `bprime-draft` (*"whoever drafts B′"*). ⭐ **Also file `bprime-fb1-recon`'s narrowed P4** — the audit-at-equal-bits discipline **is** standard outside the family (learned Bloom filters / learned indexes / SOSD = **our methodological ancestry, cited not suppressed**), and a **token-matched trivial control** was published in LLM-agent memory evaluation (arXiv:2607.21962). ⭐ The surviving claim: *seven independent groups built the adjacent instrument and none closed the loop* |
| — | **R3 (fb4): the CLU reads BELOW its own blank store on `recency` even post-fix** (0.4769 vs 0.5463) — *"possibly nobody, the family is struck"* | ⭐ **File it anyway, as a negative, with the family's struck status attached.** A struck family's anomaly is exactly what gets rediscovered in two waves if it is not written down |

## 2. P1 — fold C2W3 into the registries
`negative_results.md` (N-numbered, tiered), `claims_matrix.md` (roll the version, with the pending-confirmation
note preserved), `research_roadmap.md` (per-R statuses), `.claude/outputs/HEP_primers.md` and
`philosophy-synthesis.md` (⟲ dated addenda) per each doc's own protocol.

**The C2W3 findings that must land** (each with its scope clause — a finding without its scope is how a
retraction is born):
- ⭐ **FB4 = ◐ PARTIAL, does NOT fire; sole survivor `aggregate`.** `S(f)`: overload **1.0000** ·
  aggregate **0.5068** · recency **1.0000** · manifold **1.0000**. ⭐ *"Everything is at ceiling" is
  FALSE — but on three of four families something costing ≤4 B sits at the metric's exact maximum, and
  it is NEVER the CLU.* On `overload` the CLU reads **0.9722 ± 0.0139**, below three readers of a table
  costing **1/478th** of its bytes. ⚠ Carry the engineer's **unresolved objection**: the rule cannot
  distinguish *"the family is substitutable"* from *"the family's **anchor** is substitutable"*.
- ⭐ **The generalisable construction rule:** *"the answer is provably not in the table"* is **the only
  property that has survived a +0 B audit in two waves** (C2W1 0-for-4; C2W3 1-of-4). ⭐ File §A3.7's
  four family-design rules as a **standing** construction standard — `orgdiv-prereg` is building the cat
  test to them this wave.
- ⛔⛔ **§A9.4 `unlock = true` (18 clearing rows, all `overload`) AND §A9.5 FIRES AND OVERRIDES ⇒ NO
  STAGE 2.** The per-slot matched-bytes table reproduces **37/38 slots**, **18/18 clearing slots**,
  **zero read-beats**. ⭐ **File the MECHANISM, not just the verdict:** the flow map is contractive
  within-item and separated across-item (Fisher **6.3 → 1.47e7**) ⇒ per-slot content is a function of
  item identity ⇒ table-expressible. **The kill is a consequence of the medium being in the productive
  regime, not of it being broken.** ⚠ **And file §A14.1's SCOPING: the kill is for INFERENCE-READ claims
  only; Route 3's training-time machinery stays LIVE as tier-ii tooling.** Filing the kill without its
  scope would close a direction the Head kept open.
- ⭐ **Slot count buys NO per-item capacity** — measured rank **13 = 3d+1** at `d=4` for every `S ≥ 2`
  (**8** at `S=1`, **4 = d** under the shipped read) ⇒ **any Route-3 claim can only ever be CROSS-ITEM.**
- ⭐ **§A8.1 CONFIRMED at one integrator step** (`t = 0.05`): `D = 0.559` vs a store-deleted launder of
  **0.268** and a launch-noise floor of **0.154** — **`∇V` IS the store and it enters immediately.**
  ⛔ **§A8.1's q/p split is REFUTED as stated, and the reason is exact:** at the first slots q and p
  carry **identical information to four decimals**, because the shipped read launches at `p₀ = 0` with
  `q₀`'s payload channels zeroed, so Verlet gives `q₁ − q₀ = dt·p₁/m` — a positive multiple, hence
  rank-identical. ⭐ **Where it holds is the ADDRESS block**: position's address block is **pure query at
  ALL t** (dividend exactly **0.000** at every slot); momentum's address block is **destroyed** by the
  store (0.319–0.556 vs 1.000). *(Declared post-hoc secondary diagnostic — file it with that label.)*
- ⭐ **The 2×2:** shell and path costs **compose sub-additively**, interaction **−0.1250**;
  `gauss×endpoint` **−0.0278 ± 0.0139** and `gauss×path@0.3` **−0.2917 ± 0.0636** reproduce C2W2
  **digit-for-digit** across a third branch; `shell×endpoint` **−0.0833 ± 0.0417**; the previously unrun
  `shell×path@0.3` **−0.4722 ± 0.0278** is **the worst cell** ⇒ *"an objective that can see a path makes
  the designed degeneracy pay"* is **REFUTED IN SIGN**. ⭐ The path term moves the learned shell radius
  **0.49854 → 0.50252 (+0.0040, 16 SE of the endpoint arm's spread — it IS visible)** but that is **12×
  below** the registered bar while costing **0.209 of dividend** and **0.9–1.1 of `λ_min`** ⇒ **~50×
  more effective at damaging the write than at shaping the design.**
- ⭐ **The theorist's set:** T1 (byte floor + **compression and byte-exact deletion are the SAME trade**,
  `S* = (D+2)A_tot/m` — **7** at `A_tot=1`, **2387** at the shipped anchor; deletable fraction
  **≤0.042 %** at `r=1`) · T2 (Prop D2a with its four hypotheses; **(H3) is the only drop that opens a
  channel without degrading the store**) · ⭐⭐ **T3 as ONE statement, both directions** (`e^{−C}` law;
  **retro-explains w19/N61 with zero fitted parameters** — the C1 γ-scan to within 0.92–8.9× over five
  decades; at C1's 3000-step probe the address gradient is **10^−33.4**, **26 orders below float32
  eps**) · T4 (seven protocol caveats, **each with a domain**) · T5 (**third-party store attribution**
  is the one coupling a per-slot table cannot express, suppressed as `exp(−½(d/s)²)`: **7.0e-4** at the
  designed gate · **0.111** in the soft-certificate region · **0.69–0.80** at the rig actually run — a
  **1089×** span). ⚠ **File the theorist's §9 DECLARED-NOT-DERIVED list as caveats**, especially that
  **`s` for a learned multi-atom well is an unsolved modelling question that gates the transfer of every
  domain statement**, and that **T5.4's coupling list is NOT proven exhaustive**. ⚠ `bprime-c6` measures
  the coupling curve **this wave** — fold their number when it lands.
- ⭐ **The instrument items:** `store_write_mask_factory` landed (⭐ **and the engineer corrected the task
  file's own premise** — the real hazard is a leaf **inside** `.learned` that is not
  `centers`/`log_width`/`amp`, not a leaf outside it) · **SC-1…SC-7 landed default-off**, and the SC-1
  demonstration **reproduces the `d_safe_override` hack's admissions bit-for-bit** (⭐ *the override WAS
  the soft certificate, undeclared*) at **6.0× the declared violation budget with nothing saying so** ·
  **monitor #3's correlation leg RETIRED** (15-cell diff, every `full`/`launder`/`dividend`
  bit-identical, exactly two monitors changing, **zero trips added**; the 9 `TRIP → clear` cells all had
  the fire-rate leg in band) · ⭐ **`AttentionPsi` QUARANTINED — it RAISES, it does not warn** · ⛔ **the
  trajectory launder FIRED on 2 of 24 cells** (`manifold/shell/path@0.3`, seeds 0 and 2 — `q0_only =
  full` **exactly**), both already inadmissible, 22/24 clear — **and this is how reconciliation 1 was
  found.**
- ⭐ **Standing doctrine adopted from the wave:** ⭐⭐ ***build the kill-condition BEFORE the thing it can
  kill*** — §A9.5 moved into stage 1 killed a stage-2 build for hours of measurement. File it as a
  program-wide rule beside *"quote the curve, not the endpoint"*.
- ⛔ **The honest cost:** **two of three families cannot be written admissibly at all**
  (`aggregate`/`manifold` **0/3** after escalation; endpoint loss **0.24–0.38** against a **0.05**
  tolerance), independently reproducing C2W2's D5 — **the plateau is the atom budget's expressivity
  floor, not an optimisation-budget artefact.** ⭐ **Every Route-3 verdict rests on ONE family.**

## 3. P1 — the claim architecture v3, filed as the governing frame
§A13 **supersedes the claim framing of §2.1/§A5 wherever they conflict** (the intervention doc's
epistemics are unchanged). File, with the supersession dated and the superseded form kept as **that
entry's own history, never as a live fact** (your predecessor's rule — it resolved all five predicted
contradictions correctly and it is the standard):
- **the three tiers and their OWN controls** — tier i the settle-deleted launder · tier ii the
  **organizer swap** · tier iii the **system-level swap**; ⭐ **"controls move UP a level, they never
  relax"**;
- ⭐ **the mis-specification stated plainly** — the settle-deleted launder tests inference-time dynamics
  vs a table **given the organization**; it has **never** tested physics-trained organization vs
  non-physics-trained organization. **A CLU that organizes well and then reads like a cheap table scores
  dividend ≈0 BY DESIGN — and that is a feature (inference-cost win), not a failure.**
- ⛔⛔ **"CLU-former" is a PLACEHOLDER NAME (Head ruling) — flag it in the registry as
  NEVER-BAKE-INTO-A-DRAFT**, and make that flag impossible to miss;
- **§A14.8: gyms are HEAVILY demoted** — designed families **retire as claim venues** (FB4 killed 3 of
  4) and remain **only as regression instruments for the collapse modes**;
- **§A12's wave-accounting fact, recorded plainly: C2W3 closed REVIEWED-BUT-INCOMPLETE** because
  `bprime-rivals` was released and never spawned.

## 4. P2 — the never-quote fold
Extend **`claims_matrix.md` §0**'s consolidated dated list with C2W3's and C2W4's additions. ⭐ **One
list, one home, dated** — the C2W3 pass established that and it must not fragment again (`rival-recon`'s
additions had never been filed anywhere and lived only in that report). Add at minimum: the three errata
(E1/E2/E3) · **any C2W3-or-later cell as a byte-matched dividend** (min ratio anywhere **17.11×**) · any
**`AttentionPsi`** trajectory number · **`sep/2`** as a certified inradius · **`λ_min > 0`** as
certifying a nonempty basin (**0.000** at `λ_min = +0.910`) · ⭐ **any tier-ii or tier-iii claim sourced
from tier-i evidence** · ⭐ **"CLU-former"**.

## 5. Method (carried from the pass that worked)
- **Pass A** — read the wave's outputs and §10; extract candidate entries with their scope clauses.
- **Pass B** — **hunt contradictions against what is already filed.** Your predecessor **predicted five
  and found all five**, and **looked for a sixth and did not find it** — ⭐ **report the negative search
  too**; "I looked and it wasn't there" is evidence.
- **Pass C** — resolve each **the right way: the corrected form is the entry, the superseded form is
  that entry's own history, never both as live facts.**
- ⚠ **Placeholders with named owners are acceptable** where an item genuinely is not resolvable yet
  (your predecessor left two, correctly). ⛔ **A placeholder without an owner is not.**
- ⚠ **You do not self-confirm a `claims_matrix` version, and you do not edit
  `.claude/handover_context.md`** (protocol §2 — that is the Hub's). Proposed handover updates go in a
  `## Proposed handover updates` section of your report.
- ⛔ **You do not edit `PREREG-Bprime.md`** (Hub ruling — errata beside it, never inside it) and
  ⛔ **you do not edit the charter** (`advisor-head-c2-charter.md` is the Advisor's; §A2.3 and §A9.9 are
  already annotated in place — **verify and report, do not edit**).

## 6. Sequencing (you run early, and some inputs land after you start)
Spawn immediately and do **§0 + §1 first** — `bprime-draft` and `bprime-rivals` both consume E1/E2 and
reconciliation 7. `harness-debt` (E3's diff) and `bprime-c6` (the coupling curve, the corrected `B`) land
mid-wave. ⭐ **If a required input has not landed when you reach it, file the entry as a
PLACEHOLDER-WITH-NAMED-OWNER and say so in your report's first 10 lines** — do not stall, and do not
guess the number.

## 7. Output — `.claude/outputs/doc-curator-c2w3-sync.md`, protocol §5 format
- **What landed in each registry**, per document, per section;
- ⭐ **`.claude/outputs/track2-admissibility/ERRATA-Bprime.md`** — the dated errata file beside the
  untouched prereg (E1 + E2), created;
- **the contradiction hunt's results — including the contradictions you looked for and did NOT find**;
- **every placeholder you left, with its named owner**;
- **the `claims_matrix` version roll**, flagged **pending Hub confirmation** together with v2.5/v2.6/v2.7;
- **your reconciliation list in the FIRST 10 LINES** if you produce one;
- **`## Proposed handover updates`** for the Hub.
- **Git footprint:** none — say so explicitly.
