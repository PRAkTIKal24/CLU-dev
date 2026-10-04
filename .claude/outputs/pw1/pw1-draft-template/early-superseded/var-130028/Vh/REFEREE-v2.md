# REFEREE-v2 — referee report on `DRAFT-v2.tex` (w17 `referee-3`)

**Refereed file:** `~/clu/outputs/w17-paper/DRAFT-v2.tex`, the un-paraphrased draft, snapshot SHA-256 `edd813758f23db02…`,
2,151 lines (copy: `~/clu/outputs/w17-referee-3/DRAFT-v2.snapshot.tex`). **Line numbers below are lines of that snapshot.**
**Checked against:** the evidence base `~/clu/evidence/outputs/` (w ≤ 14 only, per AMENDMENT-1) and `VENUE.md` (ICLR 2027 main track).
**Compile check:** `pdflatex` ×2 on a scratch copy, TeX Live 2026, exit 0, 31 pages, no undefined references.
**Reviewer model:** a hostile but fair ICLR area reviewer who knows the unlearning and compositional-generation literatures,
reads the main text and Tables 1–6 closely, and skims the appendix.

---

## 0. Summary of the submission, as a reviewer would write it

The paper proposes the "Naive CLU", a label-free store of per-well PCA subspaces over a frozen encoder's latents. A
request-indexed tensor "address book" reads it and a frozen decoder renders the result. Because the store keeps the full
covariance, rows can be deleted by downdating the statistic and re-eigendecomposing. The main result is on Shapes3D at
K\* = 32 wells with 3 seeds. The store generates held-out factor combinations far better than ~131k-parameter MLP, LSTM,
transformer and continuous-MLP rivals (≈32 and ≈67 distinct valid novel tuples per corner, against 0–15), and all four
landscape controls collapse to 0. Deletion is exact at store level on one seed, and at 4.08–4.09× a 524 kB reference in
bytes. On a colour maze with an exact judge, deletion transfers (residual ~1e-16), but composition does not transfer
through the store. A plan-space "merge" verb lifts every arm, and it lifts a matched-bytes lookup most, so the paper
reports it as an interface result. Theory and limitations sections close the paper, and several rows are marked as
pending placeholders.

## 1. What I verified, and how

- **Every cell of Table 1 (`tab:t7`, L338–354)** against `w12b-nr6-and-kstar/NR6-T7.md`: all 3-seed per-arm values, the
  mean ± sd (32.350 ± 0.815, 67.177 ± 1.320), all paired Δ, the controls and 109/109. **All match.**
- **Table 2 (`tab:ledger-kstar`)** against `w13-seeds-and-tables/TABLES.md` T7 (seed 0) and `w12b…/FINDINGS.md`. They match.
- **Table 3 (`tab:ladder`)**: the store rows match `w12b…/KSTAR-LADDER.md` and the rival rows match TABLES.md T8. **Matches
  per cell, but see M2.** The control reading of 0.0083 is printed as the source prints it.
- **Table 4 (`tab:del`)** against TABLES.md T9: all cells match. Fit-pool size derived from T9 (see M1).
- **Table 7 (`tab:verb`)** and the whole of §4.3 against `w14-read-verb/FINAL-REPORT.md`. It matches, including
  +0.8299 vs +0.4749, the non-collapsing controls and the lookup at 0.9648 / 0.9646 / 0.9808.
- **Table 6 (`tab:maze-del`)** matches TABLES.md l.360 (7.522 / 8.826 / 7.818 e-16). *Upstream note:* `w14-read-verb/FINAL-REPORT.md`
  l.297 quotes seed 2 as "8.2e-16", which contradicts TABLES.md's 7.818e-16. The draft uses the tabulated value, which is
  correct. The FINAL-REPORT figure must never be copied in.
- **Positioning counts** (L145–157) match `_thinkpad/w8/w8-novelty/NOVELTY.md` Addendum B (4,050; 3,038; 759; 254; 10 hits
  and 0.3%; 0 of 3,038; 4 of 6; the orthogonal query → 0).
- **Spot checks** (clue, ablations, encoder-adapt): −39.959 / −46.918 / −39.061, 81,312, 63.612, 29.717, −21.0816, 18.56,
  +0.5688→+1.3303. All found in their cited files.
- **Provenance:** draft-v2's own checker reports 1,824 tokens and 0 misses by string. I agree with its caveat that string
  presence is not cell identity. My cell-level checks above found **no transcription error**. The defects below are about
  **framing and omission, not wrong numbers**.
- **No w15/w16 number** appears anywhere, and every SRC path is w ≤ 14. ✔

---

## 2. MUST-FIX (a real reviewer would score the paper down on each of these)

### M1. The headline rivals were trained on one-eighth of the store's data. This is undisclosed and changes the reading of Table 1.
- **Where:** abstract L100–103; contribution 1 L192–195; §2 "Rivals: naive against naive" L280–283; Table 1 caption L328–335;
  App. B L1026–1029.
- **Defect:** "naive vs naive" is matched on read-path **parameters** only. The rivals in Table 1 are the
  `w12-deletion-baselines` checkpoints, and their fit pool is **16,384 rows**. `w12-deletion-baselines/FINDINGS.md` l.119 and
  l.324 say: *"`--n-fit 16384` is the declared BUILD/SMOKE scale"*. The refits were run "on the same 16,384 fit rows"
  (`w12b-nr6-and-kstar/FINDINGS.md` l.222; draft L1029). The deletable store's fit pool is **131,072 rows**: TABLES.md T9's
  vacuous scope refits "131,072" rows, and red = 11,044 rows = 0.084259 of the pool. The paper never prints the store's pool
  size beside the rivals'. The main-text reader therefore sees an 8× data advantage presented as a parameter-fair
  comparison. The only data-matched learned comparison is the 1× MLP of App. C, "fitted from the same 32,768 fit rows as
  the store's content" (L1107). Against that MLP the margin is **+3.117 to +17.875**, not +39 to +49 (Table `tab:1x`).
- **Minimal fix:**
  1. Print both fit-pool sizes (store 131,072, rivals 16,384) in the Table 1 caption and in §2.
  2. Add one main-text sentence reporting the 1× data-matched margin (`tab:1x`, smallest per-seed +3.117; 3-seed means
     +4.356 at n = 2 and +7.978 at n = 6), labelled as the conservative comparison.
  3. Drop "naive against naive" as a fairness claim unless data is matched too.
  4. In the abstract, replace "beating every read-path-matched learned rival" with wording that names the data asymmetry,
     or lead with the 1× comparison.

  No new run is needed.

### M2. The same rival families score 6–15× apart on the same corners in two main-text tables, and the paper never says why.
- **Where:** Table 1 (L345–351) against Table 3 (L422–426). App. A L1076–1078 makes it worse.
- **Defect:** on the 60 pair corners, the table floor is **identical** in both tables (5.8333 / 3.4833 / 6.1833). The MLP,
  however, reads **2.9333 / 3.4333 / 5.2667** in Table 1 and **17.0167 / 22.0000 / 23.3333** in Table 3. On six-factor
  corners, the transformer reads **1.9184 / 0.6327 / 4.7347** in Table 1 and **28.8000 / 33.9000 / 51.9917** in Table 3.
  Table 3's n ≥ 3 corners contain Table 1's 49, and L1076 says the 49-corner prefix "reproduce[s] the shipped rows … 0.0000
  for all five rivals". The earlier cell's 178,952-parameter MLP reads 25.68–29.13 / 53.47–59.86 (`tab:ablations`), which is
  within 2–10 tuples of the store. Two rival sets therefore sit in main-text tables at an order of magnitude apart on
  identical corners. A reader will conclude that the Table 1 rivals are badly trained. The only caveat (L450–452) calls the
  **stronger** set "pre-repair", which implies it is the weaker one. That reverses the direction.
- **Minimal fix:** add one paragraph in §3 (with SRC) naming both rival sets, their fit pools and schedules, and their
  source (`w12-deletion-baselines` vs `w12-composition-ladder` / spoke-A). State that the ladder's rivals are the stronger
  set on the shared corners. Report the store's margin against the **strongest** available rival on each corner set
  (Table 3 already gives it: e.g. n = 2, store 31.800 / 32.500 / 33.633 vs best rival 19.4167 / 22.0000 / 24.2500). If the
  gap cannot be explained from artefacts, say so in a `\TODO` and do not lead with Table 1's +39 to +49 margins.

### M3. The definition of exact deletion in the introduction claims more than the store delivers. The address book is fitted with the deleted rows and is never refit.
- **Where:** §1.2 L180–182 (*"the deletable object equals … the object fitted without those rows"*); §2 L245–258; §3.3 L477–480
  and L524–526.
- **Defect:** the source definition (TABLES.md l.26–33, `/exactness_definition`) includes the bullet *"⛔ NOT a byte-level
  claim about the address book — the basis-change carry rewrites the book and is declared separately"*. The draft's
  quoted scope box (L253–257) drops that bullet. §1.2 then defines exactness on "the deletable object", and the address
  book is part 3 of that object's own four-part ledger. The address book is fitted on the full pool and only
  **rotated** at delete time (L248, L522–523: "rewrites 1,015,808 of the address book's 1,060,864 bytes"). The object after
  deletion is therefore **not** the object fitted without the rows. The paper's own evidence shows this matters: the probe
  finds the value identity *"lives in the address book"* (L526). The companion law finds that 83.5% of the post-delete
  vs never-inserted difference comes from *"the address book and coupling table [being] fitted on the pool"* (L2052;
  `w10-physics-section-evidence/EVIDENCE.md`). A privacy-minded reviewer will call "exact deletion" misleading here.
- **Minimal fix:**
  1. Rewrite L180–182 to say "the store **content** (atoms and statistic) equals…".
  2. Add the missing bullet to the scope box verbatim.
  3. Declare the address book outside the exact guarantee, as the encoder and decoder already are, in §2 and in the
     abstract ("Deletion is store-scoped and exact" → "exact on the store content; the address book is carried, not
     refit").
  4. Add the address book to the ethics statement's scoping sentence (L970–973).

### M4. The positioning claim that the intersection is empty is contradicted by known work that is neither cited nor distinguished.
- **Where:** abstract L90–93 (*"finds no generative model that deletes training rows exactly"*); §1.1 L145–162; L155–156 (the
  orthogonal query *compositional ∧ generative ∧ exact deletion* "returns zero works"); §1.2 L182–183 (*"a learned generator …
  can delete only by retraining"*).
- **Defect:**
  - **SISA** (Bourtoule et al., 2021, S&P) gives exact unlearning for any learned model, generators included, by retraining
    one shard. The citation scout already lists it (`CITATIONS.md` l.67).
  - **Compartmentalized Diffusion Models** (Golatkar, Achille, Swaminathan & Soatto, *"Training Data Protection with
    Compositional Diffusion Models"*, arXiv:2308.01937) trains diffusion models on separate data sources and composes them
    at inference. It advertises selective forgetting by dropping a source's model. Its title alone answers the paper's
    own orthogonal query.
  - Neither work is in the draft, and CDM is not in `CITATIONS.md`.
  - "Compositional" in CDM means composing data-source models, not factor values. That distinction saves the claim, but only
    if the paper makes it.

  As written, the abstract's sentence is false for a reviewer who knows SISA, and the orthogonal-query sentence invites a
  one-line rebuttal.
- **Minimal fix:**
  1. Narrow the claim everywhere to *factor-compositional* generation (held-out combinations of factor values) with exact
     deletion **that reads no data and retrains nothing**.
  2. Add a paragraph citing SISA and CDM and distinguishing both: sharded retrain vs data-free downdate; composing
     data-source models vs composing factor values.
  3. Rephrase L182–183 as "…can delete exactly only by retraining (the whole model, or a shard of it)".
  4. Hand CDM to cite-tagger.

  The screen's scope caveats (L160–162) stay.

### M5. Environments share tables, against the paper's own rule and COMMON.md.
- **Where:** L669–670 states *"no two environments ever share a table"*. Then:
  - **Table `tab:req`** (L1608–1630) prints CMAZE-2 and CMAZE-2b side by side.
  - **Table `tab:cmazet`** (L1931–1958) prints CMAZE-T and CMAZE-O in one table.
  - `fig_maze_requestlayer` (L1584) plots both CMAZE-2 and CMAZE-2b.
- **Defect:** COMMON.md says ⛔ *"CMAZE-1 / CMAZE-2 / CMAZE-2b / CMAZE-T never appear in one table."* The caption's
  "before/after, no result row" argument is a special pleading that the paper's own sentence does not allow.
- **Minimal fix:** split `tab:req` into two single-environment tables, "CMAZE-2 request-layer audit" and "CMAZE-2b
  request-layer audit", and put the before/after contrast in prose. Split `tab:cmazet` into `tab:cmazet` and `tab:cmazeo`;
  the label already exists at L1942. Ask `plots` to split fig 5b into two panels-as-figures, or drop it.

### M6. Some tables have no seed-count column, although the paper says every table has one.
- **Where:** L295 asserts *"Seed count is a column of every table."* No seeds column appears in:
  - `tab:ledger-kstar`, `tab:maze-del`, `tab:verb`, `tab:cmaze2`, `tab:cmaze1`, `tab:req`
  - `tab:rivals-del`, `tab:unlearn`, `tab:k5`, `tab:1x`, `tab:encloop`, `tab:crr`

  Most carry the count only in the caption or as per-seed columns.
- **Minimal fix:** add a `seeds` column to each of them. Values are already in each caption, and the change is mechanical.

### M7. Venue compliance. Each item alone risks desk rejection (VENUE.md §§0, 1, 4, 7).
- **(a) AI use statement is a `\TODO` (L963–966).** This section is **required**. The draft's own `\TODO` text already
  concedes that required categories apply: implementing methods, designing experiments, interpreting results. Without an
  accurate statement, a Code-of-Ethics desk rejection is possible. *Fix:* the Head writes it before any submission. This
  is not a drafting item.
- **(b) Main text is 12 pp against a strict 9 pp limit.** *Fix:* the Head's trim pass. This is flagged as blocking, not as
  this wave's job.
- **(c) No bibliography and no `\cite{}`.** Related work names about 20 works in prose only (L127–170). *Fix:* cite-tagger,
  plus the SISA and CDM entries from M4.
- **(d) The abstract deadline (Sep 18, 2026 AOE) has passed** (VENUE.md §0). If no abstract was registered, this paper
  cannot go to ICLR 2027 at all, whatever the verdict below says.
- **(e) Internal process language renders in the PDF** and reads as an automated-pipeline artefact, not as a paper:
  - "named in the task spine … this spoke does not divide them" (L622–625)
  - "dropped from paper rows by ruling" (L541, L1362)
  - "the Head to confirm" (L963)
  - "wave 16" / "wave 18 regenerates" (L463, L1600, L1604, L1922, L1933)
  - "charter controls" and "BAR 1–5", never defined for an outside reader (L195, L286, L805, L1646–1654)
  - Appendix "Corrections absorbed" (L2124–2148), which corrects "earlier write-ups" the reader has never seen

  None of these breaks anonymity by name; I found no author, institution, URL or machine name in the rendered text. They
  do signal the pipeline, which is relevant under the AI policy. *Fix:* strip them. Define "charter controls" once as
  "pre-declared landscape controls". Rename BAR k to "pre-registered criterion k". Move App. "Corrections absorbed" out of
  the submission.
- **(f)** At submission, **zero** `\PLACEHOLDER` and **zero** `\TODO` may remain (29 + 4 at present).

---

## 3. SHOULD-FIX

- **S1. Seed count of the Shapes deletion claim missing from the abstract and contribution 2** (L104–107, L199–205). Table 4 is
  **n = 1**. The probe ("changes what the store holds but not what it answers") is also seed 0 only. *Fix:* add "(one seed)"
  to both, as the table already does.
- **S2. "Composition does not transfer through the store" is asymmetric in its cross-maze argument** (L813–826). The lookup is
  correctly printed as the **within-maze** floor that cannot transfer across mazes, and +0.83 vs +0.47 is printed. ✔ But the
  store is **also** evaluated within-maze, on the training mazes' requests, and its own cross-maze transfer is not
  measured. *Fix:* one sentence saying the store is not tested across mazes either, so "the lookup cannot transfer" is a
  property of the floor and gives the store no advantage.
- **S3. The floor is called a "comparator"** (L320–322: *"smallest paired delta against any comparator … is +39.266, against the
  table floor"*; `tab:t7` lists the table among "arm / comparator" rows). *Fix:* "…against any learned rival is +39.716
  (LSTM, seed 2); against the printed table floor, +39.266". Both values are in NR6-T7.md. Also put the table floor below a
  rule in Table 1, as the controls already are.
- **S4. Byte figures without their dtype or convention.** Atoms and address book in `tab:ledger-kstar` and `tab:maze-del` (L702)
  name no dtype; only the statistic is named float64. The composition-only store's 712,072–719,352 B (L618) names no
  convention. *Fix:* name float32 or float64 per part in the captions. The comment block (L22–32) records the
  one-convention choice, and the bracket is printed in App. A L1091. ✔ Keep that.
- **S5. The crossover-seeds placeholder misstates what can happen** (L1203–1205: *"the selected K\* may change"*). Under the
  frozen rule, K\* = max{K : v(5) − mean v(K) ≤ 0.02}. K = 32 already has its 3 seeds and passes (−0.033758; `tab:kcurve`), so
  the middle rows cannot change the max. *Fix:* "…the curve's shape and minimum may change, and a crossover above 0.10 may
  appear; K\* = 32 cannot change under the frozen rule". This is honest in both directions and keeps the negative.
- **S6. The LACE/PoE placeholder has no reading for the case where they win** (L355, L654–657). The placeholder correctly says
  they "may beat the store", but nothing says what contribution 1 becomes if they do. *Fix:* one sentence in §1.3 stating
  that if LACE or PoE out-compose the store, the contribution is "composition **and** data-free exact deletion on one
  object", not "the best composer". The paper should survive that outcome in writing now.
- **S7. `fig_ladder` panel (b) puts the App. C stress comparison into the main text by figure while the prose defers it**
  (L463–467; draft-v2 flag 1). `plots/FIGURES.md` also records that the 1× learner is **not** the strongest learned arm at
  n = 3. Seed 0 is beaten by the 524,288 B LSTM and seed 2 by the transformer. *Fix:* after M1, keep panel (b) and caption it
  as the data-matched comparison, or swap it for margin-vs-strongest-rival once that artefact exists (`\TODO`).
- **S8. No generative-model baseline in any main-text table.** Every rival is a small regressor from request to latent,
  decoded by the same frozen decoder. A reviewer will ask for a conditional VAE, GAN or diffusion model on the same frozen
  latent. LACE/PoE (placeholders) partly answer this. *Fix:* state the choice and its reason in §2, and list the missing
  baseline in §5.1 as a limitation. Do not imply it was covered.
- **S9. The metric is unusual and under-motivated for an ICLR audience.** "Distinct valid novel tuples per corner", with ceilings
  of 36 and 72, carries all the weight. *Fix:* three sentences in §2 plus a small worked example, and a judge-accuracy
  column in Table 1 (TABLES.md T7 has it: store 0.9012, MLP 0.0945, …). The judge-accuracy column also makes the rival
  collapse visible and honest.
- **S10. Readability and definitions.** The paper never expands "CLU". "Well", "corner", "rung", "address book", "carry",
  "hole" and "landscape" are used before or without definition, and there is no overview figure of the object. *Fix:*
  expand the acronym, add a one-paragraph glossary in §2, and add a schematic (store → read → frozen decoder; statistic →
  downdate).
- **S11. The rank/width warning is out of the main text.** T7's source prints rank/width 0.9688, above the 0.938 at which the
  maze store was decorative, with the verdict "only the control decides". It sits in App. D only (L1156–1160). *Fix:* one
  clause in §3.1, because the main text's claim rests on that control.
- **S12. Maze verb disclosures are appendix-only** (L1656–1662): the seen row is a lookup verdict, and the budget condition
  departs from the gate's primary. *Fix:* one sentence in §4.3, because the verb is a headline result.

## 4. NICE-TO-HAVE

- **N1.** The ablation paragraph (L617–618): the composition-only store at 1.358–1.372× is bit-identical in composition to the single
  design at K = 5 (23.9–24.9×). Deletion therefore costs roughly 17–18× the bytes at K = 5. Say so plainly: it is the most
  honest one-line price of exactness the paper has. Do not compute the ratio yourselves; the "~1/7×" `\TODO` at L622 stays a
  `\TODO` until an artefact prints it.
- **N2.** The `\TODO` at L447 (paired K\* ladder margin vs read-path rivals, with CI) is zero-run arithmetic on per-corner
  files. Wave 18 should produce the artefact. Until then Table 3's "exceeds every rival's printed score" is a statement
  about means only, and the text should say that.
- **N3.** The intro's "Thickness" paragraph (L216–222) is good and honest. Keep it through the trim.
- **N4.** "TensorFaces" appears only as a citation of prior work (L128), not as a description of this method. ✔ Keep it that way.
- **N5.** "9/9 seeds": not present. "No changes after deletion": not present. "Light and data-free": present (L477). Frozen
  encoder and decoder declared outside the deletable object: present (L260). ✔
- **N6.** Upstream consistency: `w14-read-verb/FINAL-REPORT.md` l.297 quotes seed-2 residual 8.2e-16, and TABLES.md gives
  7.818e-16. The draft is right. Flag the upstream file for the evidence maintainers.

---

## 5. Hunt-list checklist (task file)

| hunt item | finding |
|---|---|
| every claim cross-checked; unlocated number = must-fix | **No unlocated number.** Cell-level checks passed (§1). The 4 `\TODO`s are honest gaps. |
| verb written as store | **No.** §4.3 is correctly an INTERFACE claim, +0.8299 vs +0.4749 printed, controls' non-collapse printed. ✔ |
| store row where controls did not collapse | **No.** Maze: no store row (L808, L855–859). ✔ |
| lookup as rival | Mostly ✔. One slip in wording (S3). |
| missing baselines / naive-vs-naive | **M1, M2** (data asymmetry; inconsistent rival sets); S8 (no generative baseline). Lookup floor printed ✔. Inflated 3.13–3.26 M rival not printed as a row ✔. |
| scope creep | **M3** (address book), **M4** (positioning), S1, S2. |
| bytes without convention | S4. Convention chosen and recorded in a LaTeX comment (L22–32); bracket printed (L1091). ✔ |
| seed counts missing from tables | **M6.** |
| two mazes in one table | **M5.** Result tables are clean. The two audit and placeholder tables are not. |
| ladder n ≥ 3 caveat | **Present and correctly worded** (caption L409–410, paragraph L432–441, with "only n = 2 vs n = 6" and r = −0.96…−0.98). ✔ |
| venue compliance | **M7** (AI statement, pages, bib, deadline, process language). Ethics ✔; reproducibility ✔ (code archive `\TODO`); author block anonymous ✔; natbib or ICLR bst wired ✔; appendix after the bibliography ✔. |
| placeholders honest; does the paper stand if they are negative? | See §6. |

## 6. The placeholder test: does the paper stand if every w15/w16 row lands negative?

**Yes, with the three fixes S1, S5 and S6.** All 29 placeholders name their row. Each states that it may land as a negative,
and none carries a number or a direction. The main claims are T7 at K\* (3 seeds), the ladder, seed-0 deletion, maze
deletion transfer, and the verb as an interface result. All of them rest on w ≤ 14 evidence. §4.4 (L840–844) already writes
the maze section's shape for the case where no world-model row lands, and that is exactly what a reviewer needs to see.
Row by row:
- `single-design-seeds` negative: the Shapes deletion claim falls back to seed 0 plus the three-seed maze deletion. That is
  honest only if the abstract already says n = 1 (**S1**).
- `lace-poe-rows` wins: contribution 1 needs a sentence it does not yet have (**S6**).
- `crossover-seeds`: the placeholder over-states its own risk (**S5**).
- `coverage-control`, `bar-d-score`, `maze-deletion-2b`, harness smokes and every w16 cell: each is an appendix row or has a
  stated fallback. The paper stands.

**No placeholder presumes a win.** One placeholder is framed around a win ("w16 fallback: if neither … yields a win",
L1965), but it states its own negative.

---

## 7. SIMULATED VERDICT

**Score: 3 / 10 (reject) as submitted. Confidence 4/5.**
*(After M1–M6 are fixed, and assuming M7 is resolved by the Head, I would expect 5 (marginally below threshold) from this
reviewer. The ceiling is set by S8 and by the single synthetic image environment.)*

**Rationale, paragraph 1 (strengths and what I believe).** The paper is unusually careful about what it does not claim.
It defines exactness before any table. It prints a four-part byte ledger and never nets it. It gates every store row on
pre-declared landscape controls that are measured, not assumed. It reports its maze composition result as an interface
property, because a matched-bytes lookup benefits more than the store does. Few submissions show that kind of
self-refutation. The core mechanism is simple and correct: downdate the untruncated statistic, then re-eigendecompose, so
no truncation error enters the subtraction. The maze deletion residuals (~8e-16, projector 0.000, three seeds, replicated
from a clean checkout) are convincing as store-level exactness. The zero-dynamics identity and its failure conditions are
stated precisely, and every number I checked traces to a recorded artefact.

**Rationale, paragraph 2 (why I reject).** The headline composition result does not survive a close reading of its own
tables. Table 1's rivals score 0–15 tuples per corner. The same families on the same corners score 17–52 in Table 3, and an
earlier 179k-parameter MLP comes within 2–10 tuples of the store. The Table 1 rivals were trained on 16,384 rows, a scale
their own build notes call "BUILD/SMOKE", while the store was fitted on 131,072 rows. The one data-matched learner in the
paper shrinks the margin from about +40 to +3–+18. Separately, the "exact deletion" in the introduction is defined on the
whole deletable object, but the address book that carries the deleted value's identity is fitted with the deleted rows
and only rotated afterwards. The paper's own probe shows the model still answers the deleted value. The novelty claim that
"compositional ∧ generative ∧ exact deletion" is empty overlooks SISA-style exact unlearning and a 2023 compositional
diffusion model built for selective forgetting. The evaluation is one synthetic image dataset at three seeds; deletion on
that dataset is one seed, and no generative baseline appears. What remains is a store with exactly deletable content,
competitive composition against a fair learner, and a well-argued negative on mazes. That could be a good workshop paper,
or a main-track paper once M1–M4 are fixed and one real generative baseline is added. It is not yet the claim the
abstract makes.

---

## 8. What this referee did NOT do (DECLARED NOT-DONE)
- **No scored run and no compute.** The only command beyond reading was one `pdflatex` compile of a scratch copy. No number
  was re-derived. The fit-pool figure in M1 is read from TABLES.md T9 ("131,072" rows, fraction 0.084259) and from
  `w12-deletion-baselines/FINDINGS.md` ("16,384"). It is not computed.
- **I did not read the style-paraphrase version** (`DRAFT-v2-style.tex`). The task says to referee the un-paraphrased draft.
- **I did not open the figure PDFs.** Figures were checked against `figures/FIGURES.md` only.
- **I verified one external paper** (Golatkar et al., arXiv:2308.01937) by a web search for its title and abstract. I did not
  read the full text, so the M4 distinction (data-source composition vs factor composition) rests on the abstract.
- **I did not edit `DRAFT-v2.tex` or any other spoke's file.** The only shared file written is this one.
- **I did not check the draft after the snapshot.** If cite-tagger has since inserted `\cite{}`, the line numbers may drift by
  a few lines. The snapshot SHA pins what was refereed.
